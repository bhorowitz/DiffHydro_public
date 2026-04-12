"""
physical_pm_force.py
====================
Non-cosmological (physical-coordinate) PM gravity force for DiffHydro.

Units throughout:
  length  : kpc
  mass    : Msun
  time    : Myr
  velocity: kpc/Myr
  G       : 4.498e-12 kpc^3 Msun^-1 Myr^-2

The force class is compatible with the DiffHydro `hydro` solver interface:
    force(i_step, U, params, dt) -> (U_new, params_out)

DM state lives in params["dm"]:
    params["dm"]["pos"]   : (N_par, 3) float32, positions in kpc in [0, L_box)
    params["dm"]["vel"]   : (N_par, 3) float32, velocities in kpc/Myr
    params["dm"]["m_par"] : float, mass per particle in Msun

Gas state U has shape (n_cons, N, N, N) in physical units:
    U[0] = rho  [Msun/kpc^3]
    U[1:4] = rho*v  [Msun/(kpc^2 Myr)]
    U[4] = E_total  [Msun/(kpc Myr^2)]

Conventions (Stage 0 document):
  - Positions are PROPER (physical) coordinates in kpc
  - Velocities are PROPER peculiar velocities in kpc/Myr
  - Density is PHYSICAL mass density in Msun/kpc^3
  - No scale factor, no comoving coordinates
  - Gravity via periodic PM (FFT Poisson)
  - G is explicit (no implicit normalization by background density)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# -------------------------------------------------------------------------
# Physical constants
# -------------------------------------------------------------------------
G_PHYS = 4.498e-12  # kpc^3 Msun^-1 Myr^-2


# -------------------------------------------------------------------------
# CIC deposit/sample helpers
# -------------------------------------------------------------------------

def _cic_paint(mesh, positions, weight):
    """CIC mass deposition.

    Parameters
    ----------
    mesh : (N, N, N) float32, initially zeros
    positions : (N_par, 3) float32, in grid cells [0, N)
    weight : float scalar or (N_par,) float32, per-particle weight (Msun)

    Returns
    -------
    (N, N, N) float32 with deposited weights
    """
    N = mesh.shape[0]
    pos_exp = jnp.expand_dims(positions, axis=1)           # (N_par, 1, 3)
    floor_pos = jnp.floor(pos_exp)                          # (N_par, 1, 3)
    offsets = jnp.array(
        [[0,0,0],[1,0,0],[0,1,0],[0,0,1],
         [1,1,0],[1,0,1],[0,1,1],[1,1,1]],
        dtype=jnp.float32,
    )[None, :, :]                                           # (1, 8, 3)

    neigh = floor_pos + offsets                             # (N_par, 8, 3)
    kernel = 1.0 - jnp.abs(pos_exp - neigh)               # (N_par, 8, 3)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]  # (N_par, 8)

    w = jnp.asarray(weight, dtype=jnp.float32)
    if w.ndim == 1:
        kernel = kernel * w[:, None]
    else:
        kernel = kernel * w  # scalar broadcast

    neigh_int = jnp.mod(
        neigh.astype(jnp.int32),
        jnp.array([N, N, N], dtype=jnp.int32),
    )                                                       # (N_par, 8, 3)

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2),
    )
    return jax.lax.scatter_add(mesh, neigh_int, kernel, dnums)


def _cic_sample(field, positions):
    """CIC trilinear interpolation from grid to particle positions.

    Parameters
    ----------
    field : (N, N, N) float32
    positions : (N_par, 3) float32, in grid cells [0, N)

    Returns
    -------
    (N_par,) float32 interpolated values
    """
    N = field.shape[0]
    pos_exp = jnp.expand_dims(positions, axis=1)           # (N_par, 1, 3)
    floor_pos = jnp.floor(pos_exp)                          # (N_par, 1, 3)
    offsets = jnp.array(
        [[0,0,0],[1,0,0],[0,1,0],[0,0,1],
         [1,1,0],[1,0,1],[0,1,1],[1,1,1]],
        dtype=jnp.float32,
    )[None, :, :]                                           # (1, 8, 3)

    neigh = floor_pos + offsets                             # (N_par, 8, 3)
    kernel = 1.0 - jnp.abs(pos_exp - neigh)               # (N_par, 8, 3)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]  # (N_par, 8)

    neigh_int = jnp.mod(
        neigh.astype(jnp.int32),
        jnp.array([N, N, N], dtype=jnp.int32),
    )                                                       # (N_par, 8, 3)

    field_vals = field[
        neigh_int[..., 0],
        neigh_int[..., 1],
        neigh_int[..., 2],
    ]                                                       # (N_par, 8)
    return jnp.sum(kernel * field_vals, axis=1)             # (N_par,)


# -------------------------------------------------------------------------
# FFT Poisson solver (physical kpc units)
# -------------------------------------------------------------------------

def _poisson_accel(rho_phys, kx_r, ky_r, kz_r, k2_r, G, subtract_mean, mesh_shape):
    """Compute acceleration from density via FFT Poisson solve.

    Parameters
    ----------
    rho_phys : (N, N, N) float32, density in Msun/kpc^3
    kx_r, ky_r, kz_r, k2_r : precomputed k-grids in rad/kpc (half-spectrum)
    G : float, gravitational constant in kpc^3 Msun^-1 Myr^-2
    subtract_mean : bool, whether to subtract mean density
    mesh_shape : tuple (N, N, N)

    Returns
    -------
    ax, ay, az : (N, N, N) float32, acceleration in kpc/Myr^2
    """
    rho = jnp.asarray(rho_phys, dtype=jnp.float32)
    if subtract_mean:
        rho = rho - jnp.mean(rho)

    rho_hat = jnp.fft.rfftn(rho)
    denom = jnp.where(k2_r == 0.0, jnp.asarray(1.0, dtype=jnp.float32), k2_r)
    # Poisson: nabla^2 Phi = 4*pi*G*rho  =>  Phi_hat = -4*pi*G*rho_hat/k^2
    phi_hat = (-4.0 * jnp.pi * G) * rho_hat / denom
    phi_hat = jnp.where(k2_r == 0.0, jnp.asarray(0.0, dtype=phi_hat.dtype), phi_hat)

    # a = -grad(Phi)  =>  a_x_hat = i*kx*Phi_hat  (note sign: a = -grad Phi, grad = ik)
    # a_x_hat = -i*kx*Phi_hat... wait:
    # grad Phi_hat = i*k * Phi_hat  in the convention F(f)(k) = integral f(x) exp(-ikx) dx
    # So a = -grad Phi = -i*k*Phi_hat
    ax = jnp.fft.irfftn(
        (-1j * kx_r) * phi_hat, s=mesh_shape
    ).real.astype(jnp.float32)
    ay = jnp.fft.irfftn(
        (-1j * ky_r) * phi_hat, s=mesh_shape
    ).real.astype(jnp.float32)
    az = jnp.fft.irfftn(
        (-1j * kz_r) * phi_hat, s=mesh_shape
    ).real.astype(jnp.float32)
    return ax, ay, az


# -------------------------------------------------------------------------
# Main force class
# -------------------------------------------------------------------------

class PhysicalDMGravityForce:
    """Non-cosmological PM gravity for DM particles + gas grid.

    Compatible with DiffHydro's force interface. Works in physical units
    (kpc, Msun, Myr) with explicit gravitational constant G.

    Implements KDK (leapfrog) integration for DM particles.
    Applies matching gravity kick to gas grid.

    Parameters
    ----------
    N_grid : int
        Number of grid cells along each axis.
    L_box_kpc : float
        Physical box side length in kpc.
    G : float, optional
        Gravitational constant. Default: 4.498e-12 kpc^3 Msun^-1 Myr^-2.
    subtract_mean : bool
        Subtract mean density before solving Poisson. Should be True for
        isolated halos in periodic boxes to avoid spurious uniform field.
    eps : float
        Floor for density to avoid divisions by zero.
    cfl_ff : float
        Safety factor for free-fall timestep limiter.
    include_gas_in_gravity : bool
        Whether to include gas density in the Poisson source. Default False
        (Stage 1: DM-only gravity).
    gas_density_unit : float
        Conversion factor: rho_code * gas_density_unit = Msun/kpc^3.
        Default 1.0 (gas already in Msun/kpc^3).
    """

    def __init__(
        self,
        N_grid: int,
        L_box_kpc: float,
        *,
        G: float = G_PHYS,
        subtract_mean: bool = True,
        eps: float = 1e-20,
        cfl_ff: float = 0.3,
        include_gas_in_gravity: bool = False,
        gas_density_unit: float = 1.0,
        static_density_field: np.ndarray | None = None,
    ):
        self.N = int(N_grid)
        self.L = float(L_box_kpc)
        self.dx = self.L / self.N          # kpc per cell
        self.G = float(G)
        self.subtract_mean = subtract_mean
        self.eps = float(eps)
        self.cfl_ff = float(cfl_ff)
        self.include_gas = include_gas_in_gravity
        self.gas_unit = float(gas_density_unit)
        self.mesh_shape = (self.N, self.N, self.N)
        if static_density_field is None:
            self.static_density = None
        else:
            arr = np.asarray(static_density_field, dtype=np.float32)
            if arr.shape != self.mesh_shape:
                raise ValueError(
                    f"static_density_field shape {arr.shape} does not match mesh {self.mesh_shape}"
                )
            self.static_density = jnp.asarray(arr, dtype=jnp.float32)

        # Precompute k-grids in rad/kpc (half-spectrum along last axis)
        kx = jnp.fft.fftfreq(self.N, d=self.dx) * (2.0 * jnp.pi)
        ky = jnp.fft.fftfreq(self.N, d=self.dx) * (2.0 * jnp.pi)
        kz = jnp.fft.rfftfreq(self.N, d=self.dx) * (2.0 * jnp.pi)
        self.kx_r, self.ky_r, self.kz_r = jnp.meshgrid(
            kx.astype(jnp.float32),
            ky.astype(jnp.float32),
            kz.astype(jnp.float32),
            indexing="ij",
        )
        self.k2_r = (self.kx_r**2 + self.ky_r**2 + self.kz_r**2).astype(jnp.float32)

    # ------------------------------------------------------------------
    # DiffHydro force interface
    # ------------------------------------------------------------------

    def timestep(self, U):
        """Free-fall CFL limiter based on peak DM+gas density."""
        rho = jnp.maximum(jnp.asarray(U[0], dtype=jnp.float32), self.eps)
        rho_max = jnp.max(rho)
        t_ff = self.cfl_ff * jnp.sqrt(
            3.0 * jnp.pi / (32.0 * self.G * rho_max + self.eps)
        )
        return t_ff

    def force(self, i_step, U, params, dt):
        """KDK leapfrog step.

        Half-kick old forces → drift → recompute forces → half-kick new.

        Parameters
        ----------
        i_step : int (unused)
        U : (n_cons, N, N, N) gas state
        params : dict with params["dm"] containing pos, vel, m_par
        dt : float, time step in Myr

        Returns
        -------
        U_new : updated gas state
        params_out : updated params with new dm pos/vel
        """
        dt = jnp.maximum(jnp.asarray(dt, dtype=jnp.float32), 0.0)
        dt_half = 0.5 * dt

        dm = params.get("dm", None)
        if dm is None:
            return U, params

        pos = jnp.asarray(dm["pos"], dtype=jnp.float32)    # (N_par, 3) kpc
        vel = jnp.asarray(dm["vel"], dtype=jnp.float32)    # (N_par, 3) kpc/Myr
        m_par = jnp.asarray(dm["m_par"], dtype=jnp.float32)  # Msun per particle

        # ----- compute source density -----
        rho_dm_old = self._deposit_density(pos, m_par)

        if self.include_gas:
            rho_gas = jnp.asarray(U[0], dtype=jnp.float32) * self.gas_unit
            rho_src_old = rho_dm_old + rho_gas
        else:
            rho_src_old = rho_dm_old
        if self.static_density is not None:
            rho_src_old = rho_src_old + self.static_density

        # ----- old force -----
        ax_old, ay_old, az_old = _poisson_accel(
            rho_src_old,
            self.kx_r, self.ky_r, self.kz_r, self.k2_r,
            self.G, self.subtract_mean, self.mesh_shape,
        )

        # Sample acceleration at particle positions
        pos_grid_old = self._pos_to_grid(pos)
        a_par_old = self._sample_accel_at_par(pos_grid_old, ax_old, ay_old, az_old)

        # ----- half-kick -----
        vel_half = vel + dt_half * a_par_old

        # ----- drift -----
        pos_new = jnp.mod(pos + dt * vel_half, self.L)

        # ----- new force -----
        rho_dm_new = self._deposit_density(pos_new, m_par)

        if self.include_gas:
            rho_src_new = rho_dm_new + rho_gas
        else:
            rho_src_new = rho_dm_new
        if self.static_density is not None:
            rho_src_new = rho_src_new + self.static_density

        ax_new, ay_new, az_new = _poisson_accel(
            rho_src_new,
            self.kx_r, self.ky_r, self.kz_r, self.k2_r,
            self.G, self.subtract_mean, self.mesh_shape,
        )

        pos_grid_new = self._pos_to_grid(pos_new)
        a_par_new = self._sample_accel_at_par(pos_grid_new, ax_new, ay_new, az_new)

        # ----- second half-kick -----
        vel_new = vel_half + dt_half * a_par_new

        # ----- gas kick (DM gravity acting on gas) -----
        rho_g = jnp.maximum(jnp.asarray(U[0], dtype=jnp.float32), self.eps)
        # Convert acceleration to gas-code units (gas in Msun/kpc^3 → a in kpc/Myr^2)
        # Gas receives full kick in one shot (no KDK split; the hydro solver handles that)
        U_new = _kick_gas(U, ax_new, ay_new, az_new, rho_g, dt, self.eps)

        params_out = dict(params)
        dm_out = dict(dm)
        dm_out["pos"] = pos_new
        dm_out["vel"] = vel_new
        params_out["dm"] = dm_out

        return U_new, params_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pos_to_grid(self, pos_kpc):
        """Convert kpc positions to fractional grid cells [0, N)."""
        return jnp.mod(pos_kpc / self.dx, float(self.N))

    def _deposit_density(self, pos_kpc, m_par):
        """CIC deposit → density in Msun/kpc^3."""
        pos_grid = self._pos_to_grid(pos_kpc)
        mesh = jnp.zeros(self.mesh_shape, dtype=jnp.float32)
        mesh = _cic_paint(mesh, pos_grid, m_par)
        return mesh / (self.dx ** 3)  # Msun/kpc^3

    def _sample_accel_at_par(self, pos_grid, ax, ay, az):
        """CIC interpolate acceleration fields at particle positions.

        Returns (N_par, 3) acceleration in kpc/Myr^2.
        """
        a_x = _cic_sample(ax, pos_grid)
        a_y = _cic_sample(ay, pos_grid)
        a_z = _cic_sample(az, pos_grid)
        return jnp.stack([a_x, a_y, a_z], axis=-1)

    # ------------------------------------------------------------------
    # Public utilities (not used inside force() loop)
    # ------------------------------------------------------------------

    def compute_accel_field(self, pos_kpc, m_par):
        """Compute (ax, ay, az) grid fields in kpc/Myr^2 for diagnostics."""
        rho = self._deposit_density(pos_kpc, m_par)
        if self.static_density is not None:
            rho = rho + self.static_density
        ax, ay, az = _poisson_accel(
            rho,
            self.kx_r, self.ky_r, self.kz_r, self.k2_r,
            self.G, self.subtract_mean, self.mesh_shape,
        )
        return ax, ay, az

    def compute_potential_field(self, pos_kpc, m_par):
        """Compute Phi grid field in kpc^2/Myr^2 for diagnostics."""
        rho = self._deposit_density(pos_kpc, m_par)
        if self.static_density is not None:
            rho = rho + self.static_density
        if self.subtract_mean:
            rho = rho - jnp.mean(rho)
        rho_hat = jnp.fft.rfftn(rho)
        denom = jnp.where(self.k2_r == 0.0, 1.0, self.k2_r)
        phi_hat = (-4.0 * jnp.pi * self.G) * rho_hat / denom
        phi_hat = jnp.where(self.k2_r == 0.0, 0.0, phi_hat)
        phi = jnp.fft.irfftn(phi_hat, s=self.mesh_shape).real
        return phi.astype(jnp.float32)


# -------------------------------------------------------------------------
# Gas kick helper (standalone so it can be jit-compiled easily)
# -------------------------------------------------------------------------

def _kick_gas(U, ax, ay, az, rho_g, dt, eps=1e-20):
    """Apply gravitational kick to gas conservative variables.

    ax, ay, az : (N, N, N) float32, acceleration in kpc/Myr^2
    rho_g : (N, N, N) float32, gas density in Msun/kpc^3
    dt : float, Myr
    """
    i_rho, i_mx, i_my, i_mz, i_E = 0, 1, 2, 3, 4
    rho_safe = jnp.maximum(rho_g, eps)

    mx_old = U[i_mx]
    my_old = U[i_my]
    mz_old = U[i_mz]
    E_old  = U[i_E]

    dmx = rho_g * ax * dt
    dmy = rho_g * ay * dt
    dmz = rho_g * az * dt

    mx_new = mx_old + dmx
    my_new = my_old + dmy
    mz_new = mz_old + dmz

    kin_old = 0.5 * (mx_old**2 + my_old**2 + mz_old**2) / rho_safe
    kin_new = 0.5 * (mx_new**2 + my_new**2 + mz_new**2) / rho_safe
    eint = jnp.maximum(E_old - kin_old, eps)
    E_new = jnp.maximum(eint + kin_new, kin_new + eps)

    U_new = U
    U_new = U_new.at[i_mx].set(mx_new)
    U_new = U_new.at[i_my].set(my_new)
    U_new = U_new.at[i_mz].set(mz_new)
    U_new = U_new.at[i_E].set(E_new)
    return U_new


# -------------------------------------------------------------------------
# Analytic NFW helpers (physical units: kpc, Msun)
# -------------------------------------------------------------------------

def nfw_density(r, rho_s, a):
    """NFW density profile: rho_s / ((r/a)(1+r/a)^2) [Msun/kpc^3]."""
    x = r / a
    return rho_s / (x * (1.0 + x) ** 2)


def nfw_enclosed_mass(r, rho_s, a):
    """Enclosed mass M(<r) for NFW [Msun]."""
    x = r / a
    return 4.0 * np.pi * rho_s * a**3 * (np.log(1.0 + x) - x / (1.0 + x))


def nfw_acceleration(r, rho_s, a, G=G_PHYS):
    """Radial acceleration magnitude |g(r)| = G M(<r)/r^2 [kpc/Myr^2]."""
    M = nfw_enclosed_mass(r, rho_s, a)
    return G * M / r**2


def nfw_circular_velocity(r, rho_s, a, G=G_PHYS):
    """Circular velocity v_c(r) = sqrt(G M(<r)/r) [kpc/Myr]."""
    M = nfw_enclosed_mass(r, rho_s, a)
    return np.sqrt(G * M / r)
