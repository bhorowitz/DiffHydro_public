"""Cosmology-aware force terms for supercomoving DiffHydro runs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .background import LCDMBackground

try:
    from jaxpm.pm import cic_paint as jpm_cic_paint
    from jaxpm.pm import pm_forces as jpm_pm_forces

    _HAS_JAXPM = True
except Exception:  # pragma: no cover - optional dependency
    jpm_cic_paint = None
    jpm_pm_forces = None
    _HAS_JAXPM = False


class BackgroundExpansionForce:
    """Force that evolves only the scale factor a in params."""

    def __init__(
        self,
        background: LCDMBackground,
        *,
        a_init: float = 1.0,
        max_dtau: float = 1.0e10,
    ):
        self.background = background
        self.a_init = float(a_init)
        self.max_dtau = float(max_dtau)

    def timestep(self, U):
        del U
        return jnp.asarray(self.max_dtau, dtype=jnp.float32)

    def force(self, i, U, params, dt):
        del i
        params_out = dict(params)
        dt = jnp.asarray(dt)
        a = jnp.asarray(params_out.get("a", self.a_init), dtype=dt.dtype)
        a_new = a + self.background.da_dtau(a) * dt
        params_out["a"] = a_new
        return U, params_out


def _cic_paint_local(mesh, positions, weight=None):
    """Simple local CIC painting fallback when jaxpm is unavailable."""
    positions = jnp.expand_dims(positions, axis=1)
    floor = jnp.floor(positions)
    offsets = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=positions.dtype,
    )[None, :, :]

    neigh = floor + offsets
    kernel = 1.0 - jnp.abs(positions - neigh)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    if weight is not None:
        kernel = kernel * jnp.expand_dims(jnp.asarray(weight), -1)

    neigh = jnp.mod(neigh.astype(jnp.int32), jnp.asarray(mesh.shape, dtype=jnp.int32))

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2),
    )
    return jax.lax.scatter_add(mesh, neigh, kernel, dnums)


class JaxPMCoupledGravityForce:
    """Single PM solve coupling gas+DM with KDK particle updates."""

    def __init__(
        self,
        equation_manager,
        *,
        mesh_shape=None,
        subtract_mean: bool = True,
        use_jaxpm: bool = True,
        dm_particle_mass: float = 1.0,
        dm_drift_factor: float = 1.0,
        dm_kick_factor: float = 1.0,
        gas_kick_factor: float | None = None,
        eps: float = 1.0e-20,
        cfl_ff: float = 0.5,
    ):
        self.eq = equation_manager
        self.mesh_shape = tuple(mesh_shape if mesh_shape is not None else equation_manager.mesh_shape)
        self.subtract_mean = bool(subtract_mean)
        self.use_jaxpm = bool(use_jaxpm and _HAS_JAXPM)
        self.dm_particle_mass = float(dm_particle_mass)
        self.dm_drift_factor = float(dm_drift_factor)
        self.dm_kick_factor = float(dm_kick_factor)
        self.gas_kick_factor = None if gas_kick_factor is None else float(gas_kick_factor)
        self.eps = float(eps)
        self.cfl_ff = float(cfl_ff)

        self.i_rho = self.eq.mass_ids
        self.i_mx, self.i_my, self.i_mz = self.eq.vel_ids
        self.i_E = self.eq.energy_ids

        nx, ny, nz = self.mesh_shape
        gx = jnp.arange(nx, dtype=jnp.float32) + 0.5
        gy = jnp.arange(ny, dtype=jnp.float32) + 0.5
        gz = jnp.arange(nz, dtype=jnp.float32) + 0.5
        xx, yy, zz = jnp.meshgrid(gx, gy, gz, indexing="ij")
        self._gas_cell_positions = jnp.stack([xx, yy, zz], axis=-1).reshape((-1, 3))

        kx = jnp.fft.fftfreq(nx, d=1.0) * (2.0 * jnp.pi)
        ky = jnp.fft.fftfreq(ny, d=1.0) * (2.0 * jnp.pi)
        kz = jnp.fft.rfftfreq(nz, d=1.0) * (2.0 * jnp.pi)
        self._kx_r, self._ky_r, self._kz_r = jnp.meshgrid(kx, ky, kz, indexing="ij")
        self._k2_r = self._kx_r**2 + self._ky_r**2 + self._kz_r**2

    def timestep(self, U):
        rho = jnp.maximum(jnp.asarray(U[self.i_rho], dtype=jnp.float32), self.eps)
        rho_max = jnp.max(rho)
        # Free-fall-style limiter in code units.
        return self.cfl_ff / jnp.sqrt(rho_max + self.eps)

    def _deposit_dm(self, positions, weight):
        mesh = jnp.zeros(self.mesh_shape, dtype=jnp.float32)
        if self.use_jaxpm:
            return jpm_cic_paint(mesh, positions, weight=weight)
        return _cic_paint_local(mesh, positions, weight=weight)

    def _mesh_acceleration_fallback(self, delta):
        delta_hat = jnp.fft.rfftn(delta)
        denom = jnp.where(self._k2_r == 0.0, 1.0, self._k2_r)
        phi_hat = -delta_hat / denom
        phi_hat = jnp.where(self._k2_r == 0.0, 0.0, phi_hat)

        ax = -jnp.fft.irfftn(1j * self._kx_r * phi_hat, s=self.mesh_shape).real
        ay = -jnp.fft.irfftn(1j * self._ky_r * phi_hat, s=self.mesh_shape).real
        az = -jnp.fft.irfftn(1j * self._kz_r * phi_hat, s=self.mesh_shape).real
        return jnp.stack([ax, ay, az], axis=-1)

    def _sample_forces(self, positions, delta):
        if self.use_jaxpm:
            return jpm_pm_forces(positions, mesh_shape=self.mesh_shape, delta=delta, r_split=0.0)

        acc = self._mesh_acceleration_fallback(delta)
        idx = jnp.mod(jnp.floor(positions).astype(jnp.int32), jnp.asarray(self.mesh_shape, dtype=jnp.int32))
        return acc[idx[:, 0], idx[:, 1], idx[:, 2]]

    def _as_particle_weight(self, positions, weight):
        w = jnp.asarray(weight, dtype=jnp.float32)
        if w.ndim == 0:
            return jnp.ones((positions.shape[0],), dtype=jnp.float32) * w
        return w

    def _make_delta(self, rho_gas, dm_positions=None, dm_weight=None):
        total_rho = rho_gas
        if dm_positions is not None:
            total_rho = total_rho + self._deposit_dm(dm_positions, dm_weight)

        mean_total = jnp.mean(total_rho)
        source = total_rho - jnp.where(self.subtract_mean, mean_total, 0.0)
        return source / (jnp.maximum(mean_total, self.eps))

    def _apply_gas_kick(self, U, rho_gas, accel_field, dt, kick_factor):
        ax = kick_factor * accel_field[..., 0]
        ay = kick_factor * accel_field[..., 1]
        az = kick_factor * accel_field[..., 2]

        rho_safe = jnp.maximum(rho_gas, self.eps)
        # Protect kinetic energy calculation denominator from underflow
        # when density drops below numerical precision
        rho_kin = jnp.maximum(rho_safe, 1.0e-10)
        mx_old = U[self.i_mx]
        my_old = U[self.i_my]
        mz_old = U[self.i_mz]
        E_old = U[self.i_E]

        dmx = rho_gas * ax * dt
        dmy = rho_gas * ay * dt
        dmz = rho_gas * az * dt

        mx_new = mx_old + dmx
        my_new = my_old + dmy
        mz_new = mz_old + dmz

        kin_old = 0.5 * (mx_old * mx_old + my_old * my_old + mz_old * mz_old) / rho_kin
        kin_new = 0.5 * (mx_new * mx_new + my_new * my_new + mz_new * mz_new) / rho_kin
        eint = jnp.maximum(E_old - kin_old, self.eps)
        
        # CRITICAL: Protect against kinetic energy exceeding total energy.
        # When large gas_kick_factor is used, momentum changes can be large,
        # leading to kin_new > eint, which would make E_new < kin_new (impossible).
        # The fix: ensure total energy is at least kin_new + eps to maintain
        # numerical stability, but don't artificially cap kinetic energy.
        # This preserves momentum and prevents negative internal energy issues.
        E_new = jnp.maximum(eint + kin_new, kin_new + self.eps)

        U_new = U
        U_new = U_new.at[self.i_mx].set(mx_new)
        U_new = U_new.at[self.i_my].set(my_new)
        U_new = U_new.at[self.i_mz].set(mz_new)
        U_new = U_new.at[self.i_E].set(E_new)
        return U_new

    def force(self, i, U_gas, params, dtau):
        del i
        dtau = jnp.maximum(jnp.asarray(dtau), 0.0)
        dtau_half = 0.5 * dtau

        rho_gas = jnp.asarray(U_gas[self.i_rho], dtype=jnp.float32)

        params_out = dict(params)
        # Get scale factor at START of step for consistent calculations
        a_start = jnp.asarray(params_out.get("a", 1.0), dtype=jnp.float32)
        
        dm_params = params.get("dm", None)
        dm_positions = None
        dm_mom = None
        dm_weight = None
        drift_factor = self.dm_drift_factor
        kick_factor = self.dm_kick_factor
        gas_kick_factor = self.gas_kick_factor
        kick_prefactor = None
        gas_kick_prefactor = None

        if dm_params is not None and "x" in dm_params:
            dm_positions = jnp.asarray(dm_params["x"], dtype=jnp.float32)
            dm_mom = jnp.asarray(dm_params.get("p_or_v", jnp.zeros_like(dm_positions)), dtype=jnp.float32)
            dm_weight = self._as_particle_weight(
                dm_positions, dm_params.get("mass", self.dm_particle_mass)
            )
            drift_factor = jnp.asarray(dm_params.get("drift_factor", drift_factor), dtype=jnp.float32)
            kick_prefactor = dm_params.get("kick_prefactor", None)
            if "kick_factor" in dm_params:
                kick_factor = jnp.asarray(dm_params["kick_factor"], dtype=jnp.float32)
            elif kick_prefactor is not None:
                kick_factor = jnp.asarray(kick_prefactor, dtype=jnp.float32) * a_start
            else:
                kick_factor = jnp.asarray(kick_factor, dtype=jnp.float32)
            gas_kick_factor = dm_params.get("gas_kick_factor", gas_kick_factor)
            gas_kick_prefactor = dm_params.get("gas_kick_prefactor", None)
            if gas_kick_factor is None and gas_kick_prefactor is not None:
                gas_kick_factor = jnp.asarray(gas_kick_prefactor, dtype=jnp.float32) * a_start
            if gas_kick_factor is None:
                gas_kick_factor = drift_factor * kick_factor

        if gas_kick_factor is None:
            gas_kick_factor = jnp.asarray(1.0, dtype=jnp.float32)
        else:
            gas_kick_factor = jnp.asarray(gas_kick_factor, dtype=jnp.float32)

        # 1) old-force solve and first half-kick
        delta_old = self._make_delta(rho_gas, dm_positions, dm_weight)
        gas_accel_old = self._sample_forces(self._gas_cell_positions, delta_old).reshape(
            self.mesh_shape + (3,)
        )
        U_half = self._apply_gas_kick(U_gas, rho_gas, gas_accel_old, dtau_half, gas_kick_factor)

        if dm_positions is not None:
            dm_accel_old = self._sample_forces(dm_positions, delta_old)
            dm_mom_half = dm_mom + dtau_half * kick_factor * dm_accel_old

            # 2) drift
            mesh_shape = jnp.asarray(self.mesh_shape, dtype=dm_positions.dtype)
            dm_positions_drift = jnp.mod(dm_positions + dtau * drift_factor * dm_mom_half, mesh_shape)

            # 3) new-force solve and second half-kick
            delta_new = self._make_delta(rho_gas, dm_positions_drift, dm_weight)
            gas_accel_new = self._sample_forces(self._gas_cell_positions, delta_new).reshape(
                self.mesh_shape + (3,)
            )
            U_new = self._apply_gas_kick(U_half, rho_gas, gas_accel_new, dtau_half, gas_kick_factor)

            dm_accel_new = self._sample_forces(dm_positions_drift, delta_new)
            dm_mom_new = dm_mom_half + dtau_half * kick_factor * dm_accel_new

            dm_out = dict(dm_params)
            dm_out["x"] = dm_positions_drift
            dm_out["p_or_v"] = dm_mom_new
            params_out["dm"] = dm_out
        else:
            U_new = U_half

        return U_new, params_out
