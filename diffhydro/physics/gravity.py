# gravity_rfft.py

import jax
import jax.numpy as jnp
from jax import lax
from .multigrid_gs import poisson_multigrid 
from diffhydro.physics import multigrid_gs as mg

# ---------------------------
# Custom-VJP spectral gravity using rFFT
# ---------------------------

@jax.custom_vjp
def gravity_accel_rfft(rho, kx_r, ky_r, kz_r, k2_r, G, subtract_mean):
    """Return (ax, ay, az) from rho via rFFT-based Poisson solve.

    Args:
      rho: real array, shape (nx, ny, nz)  [float32]
      kx_r, ky_r, kz_r, k2_r: precomputed real k-grids for rFFT with shape (nx, ny, nz_r)
      G: scalar
      subtract_mean: boolean (can be traced)

    Returns:
      ax, ay, az: real arrays (nx, ny, nz) [float32]
    """
    subtract_mean = jnp.asarray(subtract_mean, dtype=bool)
    rho = jnp.asarray(rho, jnp.float32)
    mean_rho = jnp.mean(rho)
    rho_src  = rho - jnp.where(subtract_mean, mean_rho, jnp.asarray(0.0, rho.dtype))

    # rFFT forward (half-spectrum along last axis)
    rho_hat = jnp.fft.rfftn(rho_src)  # complex64, shape (nx, ny, nz_r)

    denom   = jnp.where(k2_r == 0.0, jnp.asarray(1.0, jnp.float32), k2_r)  # fp32 to avoid c128
    C       = (-4.0 * jnp.pi * G)
    phi_hat = (C * rho_hat) / denom
    phi_hat = jnp.where(k2_r == 0.0, jnp.asarray(0.0, phi_hat.dtype), phi_hat)

    # a = -∇Φ = -i k Φ̂  (still in half-spectrum, then irfftn to real space)
    ax = -jnp.fft.irfftn(1j * kx_r * phi_hat, s=rho.shape).real
    ay = -jnp.fft.irfftn(1j * ky_r * phi_hat, s=rho.shape).real
    az = -jnp.fft.irfftn(1j * kz_r * phi_hat, s=rho.shape).real
    return ax.astype(jnp.float32), ay.astype(jnp.float32), az.astype(jnp.float32)

def _grav_rfft_fwd(rho, kx_r, ky_r, kz_r, k2_r, G, subtract_mean):
    out = gravity_accel_rfft(rho, kx_r, ky_r, kz_r, k2_r, G, subtract_mean)
    # Save only tiny scalars/refs (no big activations)
    return out, (kx_r, ky_r, kz_r, k2_r, G, subtract_mean, rho.shape)

def _grav_rfft_bwd(res, g_out):
    kx_r, ky_r, kz_r, k2_r, G, subtract_mean, rho_shape = res
    g_ax, g_ay, g_az = g_out

    # Correct adjoint of: ax = -irfftn(1j * kx * phi_hat).real
    # => g_phi_hat = (+1j * kx) * rfftn(g_ax)
    g_phi_hat  = (1j * kx_r) * jnp.fft.rfftn(g_ax)  # ✅ Fixed sign
    g_phi_hat += (1j * ky_r) * jnp.fft.rfftn(g_ay)  # ✅ Fixed sign
    g_phi_hat += (1j * kz_r) * jnp.fft.rfftn(g_az)  # ✅ Fixed sign

    C     = (-4.0 * jnp.pi * G)
    denom = jnp.where(k2_r == 0.0, jnp.asarray(1.0, g_phi_hat.real.dtype), k2_r)
    g_rho_hat = g_phi_hat * (C / denom)
    g_rho_hat = jnp.where(k2_r == 0.0, jnp.asarray(0.0, g_rho_hat.dtype), g_rho_hat)

    # Back to real space
    g_rho = jnp.fft.irfftn(g_rho_hat, s=rho_shape).real.astype(jnp.float32)

    # If subtract_mean was applied in forward, project out mean in adjoint
    subtract_mean = jnp.asarray(subtract_mean, dtype=bool)
    g_rho = g_rho - jnp.where(subtract_mean, jnp.mean(g_rho), jnp.asarray(0.0, g_rho.dtype))

    zeros = lambda x: jnp.zeros_like(x) if isinstance(x, jnp.ndarray) else 0.0
    return (g_rho, zeros(kx_r), zeros(ky_r), zeros(kz_r), zeros(k2_r), 0.0, None)

gravity_accel_rfft.defvjp(_grav_rfft_fwd, _grav_rfft_bwd)


# ---------------------------
# Force class using rFFT
# ---------------------------

class FFTSelfGravityForce:
    def __init__(self, equation_manager, G=1.0, subtract_mean=True, eps=1e-20,
                 cfl_ff=0.5, cfl_acc=0.5, dx=1.0):
        self.eq = equation_manager
        self.G = float(G)
        self.subtract_mean = bool(subtract_mean)
        self.eps = float(eps)
        self.cfl_ff = float(cfl_ff)
        self.cfl_acc = float(cfl_acc)
        self.dx = float(dx)

        self.i_rho = self.eq.mass_ids
        self.i_mx, self.i_my, self.i_mz = self.eq.vel_ids
        self.i_E   = self.eq.energy_ids

        # ----- Precompute k-grids for rFFT (half-spectrum on last axis) -----
        nx, ny, nz = self.eq.mesh_shape
        # Full-grid spacing doesn't matter if you keep consistent units; match your old code:
        kx = jnp.fft.fftfreq(nx, d=1.0) * (2.0 * jnp.pi)
        ky = jnp.fft.fftfreq(ny, d=1.0) * (2.0 * jnp.pi)
        kz_r = jnp.fft.rfftfreq(nz, d=1.0) * (2.0 * jnp.pi)        # half-spectrum last axis

        kx = kx.astype(jnp.float32)
        ky = ky.astype(jnp.float32)
        kz_r = kz_r.astype(jnp.float32)

        # Meshgrid in half-spectrum shape: (nx, ny, nz_r)
        self.kx_r, self.ky_r, self.kz_r = jnp.meshgrid(kx, ky, kz_r, indexing="ij")
        self.k2_r = (self.kx_r*self.kx_r + self.ky_r*self.ky_r + self.kz_r*self.kz_r).astype(jnp.float32)

    # Optional: cheaper timestep (no FFT) — you can keep yours if you prefer
    def timestep(self, U):
        U = lax.stop_gradient(U)
        rho = jnp.maximum(jnp.asarray(U[self.i_rho], jnp.float32), self.eps)
        rho_max = jnp.max(rho)
        t_ff = self.cfl_ff * jnp.sqrt(3.0*jnp.pi / (32.0*self.G*rho_max + self.eps))
        return t_ff

    def force(self, i, U, params, dt):
        """Apply one gravity kick over dt to CONS state U and return updated U."""
        dt  = jnp.maximum(jnp.asarray(dt), 0.0)
        rho = jnp.asarray(U[self.i_rho], jnp.float32)

        # Spectral gravity via rFFT + custom VJP (memory-light)
        ax, ay, az = gravity_accel_rfft(
            rho, self.kx_r, self.ky_r, self.kz_r, self.k2_r, self.G, self.subtract_mean
        )

        # Momentum update: m += ρ a dt
        U_new = U
        U_new = U_new.at[self.i_mx].add(rho * ax * dt)
        if rho.ndim >= 2:
            U_new = U_new.at[self.i_my].add(rho * ay * dt)
        if rho.ndim >= 3:
            U_new = U_new.at[self.i_mz].add(rho * az * dt)

        # Energy update: E += ρ (v · a) dt, with v = m / ρ
        rho_safe = jnp.maximum(rho, self.eps)
        ux = U[self.i_mx] / rho_safe
        uy = U[self.i_my] / rho_safe if rho.ndim >= 2 else 0.0
        uz = U[self.i_mz] / rho_safe if rho.ndim >= 3 else 0.0
        power = rho * (ux * ax + uy * ay + uz * az)
        U_new = U_new.at[self.i_E].add(power * dt)

        return U_new


def _infer_levels(shape: tuple[int, ...]) -> int:
    """
    Max V-cycle depth supported by even-sized periodic restriction.
    Stops before any axis would drop below 2 cells.
    """
    dims = list(shape)
    levels = 0
    while all(d % 2 == 0 and d >= 4 for d in dims):
        levels += 1
        dims = [d // 2 for d in dims]
    return max(levels, 1)


def _grad_centered_periodic(phi: jnp.ndarray, dx: float):
    """-∇phi via centered differences with periodic wrap (1D/2D/3D)."""
    nd = phi.ndim
    dphidx = (jnp.roll(phi, -1, 0) - jnp.roll(phi, 1, 0)) / (2.0 * dx)
    dphidy = (jnp.roll(phi, -1, 1) - jnp.roll(phi, 1, 1)) / (2.0 * dx) if nd >= 2 else 0.0
    dphidz = (jnp.roll(phi, -1, 2) - jnp.roll(phi, 1, 2)) / (2.0 * dx) if nd >= 3 else 0.0
    return (-dphidx, -dphidy, -dphidz)


class MGSelfGravityForce:
    """
    Periodic self-gravity via multigrid Poisson solve.

    - Zero-mean RHS for solvability on a torus: F = 4πG (ρ - ⟨ρ⟩).
    - Uses your poisson_multigrid(F, U0, l, v1, v2, mu, iter_cycle, eps, h).
    - No evolving state kept on `self` (jit- and scan-friendly).
    """

    def __init__(
        self,
        equation_manager,
        *,
        G: float = 1.0,
        dx: float = 1.0,
        subtract_mean: bool = True,
        eps: float = 1e-20,
        # CFL helpers (we keep timestep cheap; no gravity FFTs here)
        cfl_ff: float = 0.5,
        # Multigrid settings (safe defaults; override as needed)
        l: int | None = None,
        v1: int = 2,
        v2: int = 2,
        mu: int = 1,
        iter_cycle: int = 2,
        mg_tol: float = 1e-6,
        checkpoint_mg: bool = False,
        adjoint_grad: bool = True #so far seems to work well
    ):
        self.eq = equation_manager
        self.phi0 = jnp.zeros(self.eq.mesh_shape, dtype=jnp.float32)
        self.G = float(G)
        self.dx = float(dx)
        self.subtract_mean = bool(subtract_mean)
        self.eps = float(eps)
        self.cfl_ff = float(cfl_ff)

        # equation_manager indices
        self.i_rho = self.eq.mass_ids
        self.i_mx, self.i_my, self.i_mz = self.eq.vel_ids
        self.i_E = self.eq.energy_ids

        # Multigrid params
        mesh_shape = tuple(self.eq.mesh_shape)
        self.l = _infer_levels(mesh_shape) if l is None else int(l)
        self.v1 = int(v1)
        self.v2 = int(v2)
        self.mu = int(mu)
        self.iter_cycle = int(iter_cycle)
        self.mg_tol = float(mg_tol)
        self.checkpoint_mg = bool(checkpoint_mg)

    
        if adjoint_grad:
            self._mg_solve = mg.make_poisson_mg_solver(
                l=self.l,
                v1=self.v1,
                v2=self.v2,
                mu=self.mu,
                iter_cycle=self.iter_cycle,
                eps=self.mg_tol,
                h=self.dx,
            )
        else:
            # Wrap MG solve with remat if requested (cuts peak VJP memory)
            def _solve(F, U0):
                return poisson_multigrid(
                    F=F, U=U0,
                    l=self.l, v1=self.v1, v2=self.v2, mu=self.mu,
                    iter_cycle=self.iter_cycle, eps=self.mg_tol, h=self.dx
                )
            self._mg_solve = jax.checkpoint(_solve) if self.checkpoint_mg else _solve
            
    # --- Optional: cheap dt (free-fall only; no gravity solve here) ---
    def timestep(self, U: jnp.ndarray) -> jnp.ndarray:
        U = lax.stop_gradient(U)
        rho = jnp.maximum(jnp.asarray(U[self.i_rho], jnp.float32), self.eps)
        rho_max = jnp.max(rho)
        t_ff = self.cfl_ff * jnp.sqrt(3.0 * jnp.pi / (32.0 * self.G * rho_max + self.eps))
        return t_ff

    # --- Force application: one kick over dt ---
    def force(self, i: int, U: jnp.ndarray, params, dt: jnp.ndarray) -> jnp.ndarray:
        dt = jnp.maximum(jnp.asarray(dt), 0.0)

        # Density (fp32 to keep MG & grads light)
        rho = jnp.asarray(U[self.i_rho], jnp.float32)

        # Periodic solvability: subtract mean if requested
        mean_rho = jnp.mean(rho)
        rho_src = rho - jnp.where(self.subtract_mean, mean_rho, jnp.asarray(0.0, rho.dtype))

        # Poisson RHS and initial guess
        F = (4.0 * jnp.pi * self.G) * rho_src
        
        if self.phi0 is None:
            phi0 = jnp.zeros_like(F, dtype=jnp.float32)
        else:
            phi0 = self.phi0
            
        # Solve A phi = F with multigrid
        phi = self._mg_solve(F, phi0)

        # Acceleration = -∇φ (periodic centered)
        ax, ay, az = _grad_centered_periodic(phi, self.dx)

        # Momentum update: m += ρ a dt
        U_new = U
        U_new = U_new.at[self.i_mx].add(rho * ax * dt)
        if rho.ndim >= 2:
            U_new = U_new.at[self.i_my].add(rho * ay * dt)
        if rho.ndim >= 3:
            U_new = U_new.at[self.i_mz].add(rho * az * dt)

        # Energy update: E += ρ (v · a) dt, with v = m / ρ
        rho_safe = jnp.maximum(rho, self.eps)
        ux = U[self.i_mx] / rho_safe
        uy = U[self.i_my] / rho_safe if rho.ndim >= 2 else 0.0
        uz = U[self.i_mz] / rho_safe if rho.ndim >= 3 else 0.0
        power = rho * (ux * ax + uy * ay + uz * az)
        U_new = U_new.at[self.i_E].add(power * dt)

        return U_new