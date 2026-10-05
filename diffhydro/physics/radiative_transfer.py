"""
Radiative transfer module (stub).

Placeholder functions for future radiative transfer physics.
All methods currently return stub values.
"""

import os
from turtle import up


import jax
import jax.numpy as jnp
from ..utils.debug_checks import _check_finite,_check_all_float_variables
from ..units import CodeUnits, UnitParser, from_code, to_code
from diffhydro.equationmanager_radiative_transf_no_chat import EquationManager as EquationManager_RT
from diffhydro.units.convert import temperature_code_from_Prho


class RadiativeTransfer:
    """Placeholder for radiative transfer physics (not implemented)."""

    def __init__(self, eq=None):
        self.eq = eq
        self.active = False
        self.stub_mode = True

    def get_radiation_energy(self, temperature, density):
        """Stub: returns zeros."""
        return jnp.zeros_like(temperature)

    def get_radiation_pressure(self, temperature, density):
        """Stub: returns zeros."""
        return jnp.zeros_like(temperature)

    def apply_radiative_coupling(self, primitives, conservatives, dt):
        """Stub: returns conservatives unchanged."""
        return conservatives

    def compute_optical_depth(self, density, temperature):
        """Stub: returns ones."""
        return jnp.ones_like(density)


def get_radiation_temperature(Ey, eq=None):
    """Stub: returns zeros."""
    return jnp.zeros_like(Ey)


def compute_heating_cooling(primitives, conservatives, dt, eq=None):
    """Stub: returns zero source terms."""
    return jnp.zeros_like(primitives)


def apply_flux_limiting(fluxes):
    """Stub: returns fluxes unchanged."""
    return fluxes


RADIATIVE_CONFIG = {
    "active": False,
    "lte_mode": False,
    "diffusion_limit": False,
    "stefan_boltzmann": 5.67e-5,
    "radiation_constant": 8.0e-1,
}

#new version 
class StellarRadiationForce:
    """
    Radiative source term from stellar populations.
    Updates the E_gamma field based on stellar mass, age and metallicity.
    """

    def __init__(
        self,
        escape_fraction=0.1,
        stellar_spectrum_func=None,
        dx=1.0,
        injection_mode="physical",
        stromgren_rate=1e-7,
        gaussian_star=True,
        injection_geometry="3D",
        injection_momentum=False,
        momentum_only=False,
        eq=None,
        debug=False,
        one_injection=False,
        # --- new beam parameters ---
        beam_axis=0,          # 0=x, 1=y, 2=z
        beam_sign=+1,         # +1 or -1
        beam_length_cells=8,  # number of beam cells
        beam_sigma=3.0,       # Gaussian spreading
        beam_reduced_flux=0.95,  # |F|/(c*E) max
        beam_momentum_scaling="legacy_c2_source2",
        cu=None,
        parser=None,
        chemistry=False,
        
    ):
        self.escape_fraction = escape_fraction
        self.stellar_spectrum_func = stellar_spectrum_func
        self.dx = dx
        self.injection_mode = injection_mode
        self.stromgren_rate = stromgren_rate
        self.gaussian_star = gaussian_star
        self.injection_geometry = injection_geometry
        self.injection_momentum = injection_momentum
        self.momentum_only = momentum_only
        self.chemistry = chemistry
        self.debug = debug
        self.light_speed = eq.light_speed if eq is not None else 1.0
        self.mesh_shape = eq.mesh_shape if eq is not None else (100, 100, 100)
        self.one_injection = one_injection
        #beam
        self.beam_axis        = beam_axis
        self.beam_sign        = beam_sign
        self.beam_length_cells= beam_length_cells
        self.beam_reduced_flux= beam_reduced_flux
        self.beam_momentum_scaling = beam_momentum_scaling
        self.cu = cu
        self.parser = parser or UnitParser()
        self.sol = None
        self.eq = eq
        self.debug_dt_sum_code = 0.0
        self.debug_dt_sum_phys = 0.0
        self.debug_step_count = 0

    def set_code_units(self, cu: CodeUnits, parser: UnitParser | None = None):
        """Configure the unit system used by the radiative source helpers."""
        self.cu = cu
        self.parser = parser or self.parser or UnitParser()

    def convert_physical_to_code(self, value, dimension: str):
        """Convert a physical quantity into code units using the configured CodeUnits."""
        if self.cu is None:
            raise ValueError("CodeUnits not configured.")
        return to_code(value, dimension, self.cu, self.parser)

    def convert_code_to_physical(self, value, dimension: str, out_unit: str | None = None):
        """Convert a code-unit quantity back to a physical value wrapper."""
        if self.cu is None:
            raise ValueError("CodeUnits not configured.")
        unit = out_unit or self.parser.default_cgs_unit(dimension)
        return from_code(value, dimension, self.cu, unit, self.parser)
    
    def get_temp_code(self, T_phys):
        """Convert a physical temperature (float in Kelvin, or a unit string
        like '5000 K') into code units. Does NOT read from environment."""
        if isinstance(T_phys, str):
            temp_q = self.parser.parse(T_phys, expected_dim="temperature")
            T_phys_K = temp_q.cgs_value
        else:
            T_phys_K = float(T_phys)  # assumed already in Kelvin (cgs)
        T_code = T_phys_K / self.cu.Temp_cgs
        return T_code    
    
    def get_stellar_emission(self, star_age, star_metallicity):
        if self.stellar_spectrum_func is not None:
            return self.stellar_spectrum_func(star_age, star_metallicity)
        age_factor = jnp.exp(-star_age / 10.0)
        Z_factor = jnp.maximum(star_metallicity, 1e-4)
        return age_factor * Z_factor

    def timestep(self, sol):
        # print("timestepblast")
        return 1e30
    
    def debug_grid_stats(self, sol, eq, label, ax):
        E = sol[0]#jnp.maximum(sol[0], eq.eps)
        Fx = sol[1]
        Fy = sol[2]
        Fz = sol[3]

        Fmag = jnp.sqrt(Fx**2 + Fy**2 + Fz**2)

        mask = sol[0] > eq.eps
        mask_ratio = E > eq.eps

        ratio = jnp.where(
            mask_ratio,
            Fmag / jnp.maximum(self.light_speed * E, 1e-30),
            0.0,
        )

        n_active = jnp.sum(mask)
        n_ratio_active = jnp.sum(mask_ratio)

        E_masked     = jnp.where(mask, E, 0.0)
        Fx_masked    = jnp.where(mask, Fx, 0.0)
        Fy_masked    = jnp.where(mask, Fy, 0.0)
        Fz_masked    = jnp.where(mask, Fz, 0.0)
        Fmag_masked  = jnp.where(mask, Fmag, 0.0)
        ratio_masked = jnp.where(mask_ratio, ratio, 0.0)

        E_min_vis     = jnp.min(jnp.where(mask, E, jnp.inf))
        Fx_min_vis    = jnp.min(jnp.where(mask, Fx, jnp.inf))
        Fy_min_vis    = jnp.min(jnp.where(mask, Fy, jnp.inf))
        Fz_min_vis    = jnp.min(jnp.where(mask, Fz, jnp.inf))
        Fmag_min_vis  = jnp.min(jnp.where(mask, Fmag, jnp.inf))
        ratio_min_vis = jnp.min(jnp.where(mask_ratio, ratio, jnp.inf))

        E_max_vis     = jnp.max(jnp.where(mask, E, -jnp.inf))
        Fx_max_vis    = jnp.max(jnp.where(mask, Fx, -jnp.inf))
        Fy_max_vis    = jnp.max(jnp.where(mask, Fy, -jnp.inf))
        Fz_max_vis    = jnp.max(jnp.where(mask, Fz, -jnp.inf))
        Fmag_max_vis  = jnp.max(jnp.where(mask, Fmag, -jnp.inf))
        ratio_max_vis = jnp.max(jnp.where(mask_ratio, ratio, -jnp.inf))

        denom = jnp.maximum(n_active, 1)
        denom_ratio = jnp.maximum(n_ratio_active, 1)

        E_mean_vis     = jnp.sum(E_masked) / denom
        Fx_mean_vis    = jnp.sum(Fx_masked) / denom
        Fy_mean_vis    = jnp.sum(Fy_masked) / denom
        Fz_mean_vis    = jnp.sum(Fz_masked) / denom
        Fmag_mean_vis  = jnp.sum(Fmag_masked) / denom
        ratio_mean_vis = jnp.sum(ratio_masked) / denom_ratio

        any_active = jnp.any(mask)
        any_ratio_active = jnp.any(mask_ratio)

        jax.debug.print(
        """
        [{label}] ax={ax}
        shape = {shape}
        n_active = {n_active}
        n_ratio_active = {n_ratio_active}
        any_active = {any_active}
        any_ratio_active = {any_ratio_active}

        E on active cells:
        mean={E_mean} min={E_min} max={E_max}
        any nan? {E_nan}
        any inf? {E_inf}
        any nonfinite? {E_bad}

        Fx on active cells:
        mean={Fx_mean} min={Fx_min} max={Fx_max}

        Fy on active cells:
        mean={Fy_mean} min={Fy_min} max={Fy_max}

        Fz on active cells:
        mean={Fz_mean} min={Fz_min} max={Fz_max}

        |F| on active cells:
        mean={F_mean} min={F_min} max={F_max}
        any nan? {F_nan}
        any inf? {F_inf}
        any nonfinite? {F_bad}

        |F|/(c E) on cells where E > eps:
        mean={r_mean} min={r_min} max={r_max}
        any > 1 ? {r_gt1}

        ratio slice:
        {ratio_slice}
        """,
        label=label,
        ax=ax,
        shape=sol.shape,
        n_active=n_active,
        n_ratio_active=n_ratio_active,
        any_active=any_active,
        any_ratio_active=any_ratio_active,

        E_mean=E_mean_vis,
        E_min=jnp.where(any_active, E_min_vis, 0.0),
        E_max=jnp.where(any_active, E_max_vis, 0.0),
        E_nan=jnp.any(jnp.isnan(jnp.where(mask, E, jnp.nan))),
        E_inf=jnp.any(jnp.isinf(jnp.where(mask, E, 0.0))),
        E_bad=jnp.any(~jnp.isfinite(jnp.where(mask, E, 0.0))),

        Fx_mean=Fx_mean_vis,
        Fx_min=jnp.where(any_active, Fx_min_vis, 0.0),
        Fx_max=jnp.where(any_active, Fx_max_vis, 0.0),

        Fy_mean=Fy_mean_vis,
        Fy_min=jnp.where(any_active, Fy_min_vis, 0.0),
        Fy_max=jnp.where(any_active, Fy_max_vis, 0.0),

        Fz_mean=Fz_mean_vis,
        Fz_min=jnp.where(any_active, Fz_min_vis, 0.0),
        Fz_max=jnp.where(any_active, Fz_max_vis, 0.0),

        F_mean=Fmag_mean_vis,
        F_min=jnp.where(any_active, Fmag_min_vis, 0.0),
        F_max=jnp.where(any_active, Fmag_max_vis, 0.0),
        F_nan=jnp.any(jnp.isnan(jnp.where(mask, Fmag, jnp.nan))),
        F_inf=jnp.any(jnp.isinf(jnp.where(mask, Fmag, 0.0))),
        F_bad=jnp.any(~jnp.isfinite(jnp.where(mask, Fmag, 0.0))),

        r_mean=ratio_mean_vis,
        r_min=jnp.where(any_ratio_active, ratio_min_vis, 0.0),
        r_max=jnp.where(any_ratio_active, ratio_max_vis, 0.0),
        r_gt1=jnp.any(jnp.where(mask_ratio, ratio > 1.0 + 1e-6, False)),

        ratio_slice=jnp.where(mask_ratio[0:20, 40:60, 50], ratio[0:20, 40:60, 50], 0.0),
        ordered=True,
    )

    def _clip_indices_2d(self, x0, y0, z0, di2, dj2):
        xi = x0 + di2
        yi = y0 + dj2
        zi = jnp.full(di2.shape, z0, dtype=jnp.int32)
        valid = (
            (xi >= 0) & (xi < self.mesh_shape[0]) &
            (yi >= 0) & (yi < self.mesh_shape[1]) &
            (zi >= 0) & (zi < self.mesh_shape[2])
        )
        return xi, yi, zi, valid

    def _clip_indices_3d(self, x0, y0, z0, di3, dj3, dk3):
        xi = x0 + di3
        yi = y0 + dj3
        zi = z0 + dk3
        valid = (
            (xi >= 0) & (xi < self.mesh_shape[0]) &
            (yi >= 0) & (yi < self.mesh_shape[1]) &
            (zi >= 0) & (zi < self.mesh_shape[2])
        )
        return xi, yi, zi, valid

    def _normalized_weights_2d(self, offsets, sigma, valid):
        di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
        weights2 = jnp.exp(-(di2**2 + dj2**2) / (2 * sigma**2))
        weights2 = jnp.where(valid, weights2, 0.0)
        weights2 = weights2 / (jnp.sum(weights2) + 1e-30)
        return di2, dj2, weights2

    def _normalized_weights_3d(self, offsets, sigma, valid):
        offsets = offsets#*2.5
        di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
        weights3 = jnp.exp(-(di3**2 + dj3**2 + dk3**2) / (2 * sigma**2))
        weights3 = jnp.where(valid, weights3, 0.0)
        weights3 = weights3 / (jnp.sum(weights3) + 1e-30)
        return di3, dj3, dk3, weights3

    def _beam_momentum_factor(self, source, weights):
        if self.beam_momentum_scaling == "legacy_c2_source2":
            return source * (self.light_speed ** 2) * weights
        elif self.beam_momentum_scaling == "physical":
            return self.beam_reduced_flux * self.light_speed * source * weights
        else:
            raise ValueError(f"Unknown beam_momentum_scaling: {self.beam_momentum_scaling}")

    # ─────────────── Injection énergie ───────────────

    def _inject_energy_beam_x(self, sol, x0, y0, z0, source, sigma, beam_len):
        s = jnp.arange(0, beam_len, dtype=jnp.int32)
        xi = x0 + s
        yi = jnp.full_like(xi, y0)
        zi = jnp.full_like(xi, z0)
        valid = (
            (xi >= 0) & (xi < self.mesh_shape[0]) &
            (yi >= 0) & (yi < self.mesh_shape[1]) &
            (zi >= 0) & (zi < self.mesh_shape[2])
        )
        s_float = s.astype(jnp.float64)
        weights = jnp.exp(- (s_float**2) / (2.0 * float(sigma)**2))
        weights = jnp.where(valid, weights, 0.0)
        weights = weights / (jnp.sum(weights) + 1e-30)
        if self.debug:
            self.debug_grid_stats(sol, self.eq, "dans sol energy injection beam", 0)
        return sol.at[0, xi, yi, zi].add(source * weights)

    def _inject_energy_2d(self, sol, x0, y0, z0, source, offsets, sigma):
        if self.debug:
            self.debug_grid_stats(sol, self.eq, "before clip sol energy injection 2D", 0)
        di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
        xi, yi, zi, valid = self._clip_indices_2d(x0, y0, z0, di2, dj2)
        _, _, weights2 = self._normalized_weights_2d(offsets, sigma, valid)
        if self.debug:
            self.debug_grid_stats(sol, self.eq, "sol energy injection 2D", 0)
        return sol.at[0, xi, yi, zi].add(source * weights2)

    def _inject_energy_3d(self, sol, x0, y0, z0, source, offsets, sigma):
        di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
        xi, yi, zi, valid = self._clip_indices_3d(x0, y0, z0, di3, dj3, dk3)
        _, _, _, weights3 = self._normalized_weights_3d(offsets, sigma, valid)
        return sol.at[0, xi, yi, zi].add(source * weights3)

    # ─────────────── Injection momentum ───────────────

    def _inject_momentum_beam_x(self, sol, x0, y0, z0, source, sigma, beam_len):
        s = jnp.arange(0, beam_len, dtype=jnp.int32)
        xi = x0 + s
        yi = jnp.full_like(xi, y0)
        zi = jnp.full_like(xi, z0)
        valid = (
            (xi >= 0) & (xi < self.mesh_shape[0]) &
            (yi >= 0) & (yi < self.mesh_shape[1]) &
            (zi >= 0) & (zi < self.mesh_shape[2])
        )
        s_float = s.astype(jnp.float64)
        weights = jnp.exp(- (s_float**2) / (2.0 * float(sigma)**2))
        weights = jnp.where(valid, weights, 0.0)
        weights = weights / (jnp.sum(weights) + 1e-30)
        fx_inj = self._beam_momentum_factor(source, weights)
        if self.debug:
            jax.debug.print("Injecting momentum beam at x=[{}, {}], y={}, z={}", xi[0], xi[-1], yi[0], zi[0])
            jax.debug.print("Momentum source: {}, weights sum: {}", source, jnp.sum(weights))
            jax.debug.print("Fx injection profile: {}", fx_inj)
            jax.debug.print("test c^2: {}", fx_inj / (source**2 + 1e-30) / weights)
            jax.debug.print("Sol[1] after injection: {}", sol[1, xi, yi, zi])
            jax.debug.print("max E = {}", jnp.max(sol[0]))
            jax.debug.print("any nan E = {}", jnp.any(jnp.isnan(sol[0])))
            jax.debug.print("any inf E = {}", jnp.any(jnp.isinf(sol[0])))
        sol = sol.at[1, xi, yi, zi].add(fx_inj)
        sol = sol.at[2, xi, yi, zi].add(jnp.zeros_like(0))
        sol = sol.at[3, xi, yi, zi].add(jnp.zeros_like(0))
        return weights, sol

    def _inject_momentum_x_2d(self, sol, x0, y0, z0, source, offsets, sigma):
        di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
        xi, yi, zi, valid = self._clip_indices_2d(x0, y0, z0, di2, dj2)
        _, _, weights2 = self._normalized_weights_2d(offsets, sigma, valid)
        fx_inj = self._beam_momentum_factor(source, weights2)
        sol = sol.at[1, xi, yi, zi].add(fx_inj)
        sol = sol.at[2, xi, yi, zi].add(jnp.zeros_like(0))
        sol = sol.at[3, xi, yi, zi].add(jnp.zeros_like(0))
        if self.debug:
            E = sol[0, xi, yi, zi]
            jax.debug.print("Injecting momentum 2D at x=[{}, {}], y={}, z={}", xi[0, 0], xi[-1, -1], y0, z0)
            jax.debug.print("Momentum source: {}, weights sum: {}", source, jnp.sum(weights2))
            jax.debug.print("test c^2: {}", fx_inj / (source**2 + 1e-30) / weights2)
            jax.debug.print("max E = {}", jnp.max(E))
            jax.debug.print("any nan E = {}", jnp.any(jnp.isnan(E)))
            jax.debug.print("any inf E = {}", jnp.any(jnp.isinf(E)))
        return weights2, sol

    def _inject_momentum_x_3d(self, sol, x0, y0, z0, source, offsets, sigma):
        di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
        xi, yi, zi, valid = self._clip_indices_3d(x0, y0, z0, di3, dj3, dk3)
        _, _, _, weights3 = self._normalized_weights_3d(offsets, sigma, valid)
        fx_inj = self._beam_momentum_factor(source, weights3)
        sol = sol.at[1, xi, yi, zi].add(fx_inj)
        sol = sol.at[2, xi, yi, zi].add(jnp.zeros_like(fx_inj))#modify the geometry of the beam there
        sol = sol.at[3, xi, yi, zi].add(jnp.zeros_like(fx_inj))
        return sol

    def _inject_momentum_radial_2d(self, sol, x0, y0, z0, source, offsets, sigma):
        """Radial (isotropic point-source) momentum injection in the (x, z=z0) plane.
        Unlike a beam, F points outward from the source at each offset cell,
        matching an isotropically emitting star rather than a collimated jet."""
        di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
        xi, yi, zi, valid = self._clip_indices_2d(x0, y0, z0, di2, dj2)
        _, _, weights2 = self._normalized_weights_2d(offsets, sigma, valid)

        r = jnp.sqrt(di2**2 + dj2**2)
        inv_r = jnp.where(r > 0, 1.0 / r, 0.0)   # avoid 0/0 at the source cell
        ux = di2 * inv_r
        uy = dj2 * inv_r

        f_mag = self._beam_momentum_factor(source, weights2)
        fx_inj = f_mag * ux
        fy_inj = f_mag * uy

        sol = sol.at[1, xi, yi, zi].add(fx_inj)
        sol = sol.at[2, xi, yi, zi].add(fy_inj)
        sol = sol.at[3, xi, yi, zi].add(jnp.zeros_like(fx_inj))

        if self.debug:
            E = sol[0, xi, yi, zi]
            jax.debug.print("Injecting radial momentum 2D at x0={}, y0={}, z0={}", x0, y0, z0)
            jax.debug.print("Momentum source: {}, weights sum: {}", source, jnp.sum(weights2))
            jax.debug.print("|F| test c^2: {}", f_mag / (source**2 + 1e-30) / weights2)
            jax.debug.print("max E = {}", jnp.max(E))
            jax.debug.print("any nan E = {}", jnp.any(jnp.isnan(E)))
            jax.debug.print("any inf E = {}", jnp.any(jnp.isinf(E)))

        return weights2, sol


    def _inject_momentum_radial_3d(self, sol, x0, y0, z0, source, offsets, sigma):
        """Radial (isotropic point-source) momentum injection in full 3D."""
        di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
        xi, yi, zi, valid = self._clip_indices_3d(x0, y0, z0, di3, dj3, dk3)
        _, _, _, weights3 = self._normalized_weights_3d(offsets, sigma, valid)

        r = jnp.sqrt(di3**2 + dj3**2 + dk3**2)
        inv_r = jnp.where(r > 0, 1.0 / r, 0.0)
        ux = di3 * inv_r
        uy = dj3 * inv_r
        uz = dk3 * inv_r

        f_mag = self._beam_momentum_factor(source, weights3)
        fx_inj = f_mag * ux
        fy_inj = f_mag * uy
        fz_inj = f_mag * uz

        sol = sol.at[1, xi, yi, zi].add(fx_inj)
        sol = sol.at[2, xi, yi, zi].add(fy_inj)
        sol = sol.at[3, xi, yi, zi].add(fz_inj)

        if self.debug:
            jax.debug.print("Injecting radial momentum 3D at x0={}, y0={}, z0={}", x0, y0, z0)
            jax.debug.print("Momentum source: {}, weights sum: {}", source, jnp.sum(weights3))
            jax.debug.print("|F| max: {}", jnp.max(jnp.sqrt(fx_inj**2 + fy_inj**2 + fz_inj**2)))

        return weights3, sol
    def _clip_to_m1_cone(self, sol):
        c = self.light_speed
        E  = sol[0]
        Fx = sol[1]; Fy = sol[2]; Fz = sol[3]
        Fnorm = jnp.sqrt(Fx**2 + Fy**2 + Fz**2 + 1e-30)
        Fmax  = self.beam_reduced_flux * c * E
        scale = jnp.minimum(1.0, Fmax / (Fnorm + 1e-30))
        sol = sol.at[1].set(Fx * scale)
        sol = sol.at[2].set(Fy * scale)
        sol = sol.at[3].set(Fz * scale)
        return sol
    
    def _debug_stromgren_units_jax(self, dt, per_star_source):
        rate_code = self.get_N_gamma_stromgen_sphere()

        if self.cu is None:
            jax.debug.print(
                """
                [stromgren debug]
                dt_code = {dt_code}
                rate_code = {rate_code}
                per_star_source_code = {src_code}
                """,
                dt_code=dt,
                rate_code=rate_code,
                src_code=per_star_source,
                ordered=True,
            )
            return

        T_cgs = jnp.asarray(self.cu.T_cgs, dtype=dt.dtype)
        photon_rate_cgs = jnp.asarray(self.cu.PhotonRate_cgs, dtype=dt.dtype)

        dt_phys_s = dt * T_cgs
        rate_phys = rate_code * photon_rate_cgs
        injected_phys = rate_phys * dt_phys_s
        sum_dt_phys = jnp.sum(dt_phys_s)

        jax.debug.print(
            """
            [stromgren debug]
            dt_code = {dt_code}
            dt_code_brut = {dt_code_brut}
            dt_phys_s = {dt_phys_s}

            stromgren_injection_code = {rate_code}
            stromgren_rate_code_brut = {rate_code_brut}
            stromgren_rate_phys_ph_per_s = {rate_phys}

            per_star_source_code = {src_code}
            per_star_source_code_brut = {src_code_brut}
            per_star_source_phys_photons = {src_phys}
            evolve_dt = {sum_dt_phys}
            """,
            dt_code=dt,
            dt_code_brut=dt,
            dt_phys_s=dt_phys_s,
            rate_code=rate_code,
            rate_code_brut=photon_rate_cgs,
            rate_phys=rate_phys,
            src_code=per_star_source,
            src_code_brut=photon_rate_cgs,
            src_phys=injected_phys,
            sum_dt_phys=sum_dt_phys,
            ordered=True,
        )



    def force(self, i, sol, params, dt):
        if "star_masses" not in params or params["star_masses"] is None:
            return sol, params
        if self.debug:
            self.debug_grid_stats(sol, self.eq, "sol before injection", 0)

        star_masses        = jnp.asarray(params["star_masses"])
        star_ages_old      = jnp.asarray(params["star_ages"])
        star_ages_new      = star_ages_old + dt
        star_metallicities = jnp.asarray(params["star_metallicities"])
        # beam_sign

        if self.injection_mode == "stromgren":
            per_star_source = (self.get_N_gamma_stromgen_sphere()* dt + self.get_N_chemistry(
              rho_gas=sol[4], sigma_HI=self.sigma_HI, caseA=self.caseA, caseB=self.caseB, 
              fraction_HI=sol[8], T=self.temperature
                ) )
            if self.debug:
                self._debug_stromgren_units_jax(dt, per_star_source)
        elif self.injection_mode == "physical":
            per_star_source = self.get_N_gamma(
                star_masses, star_ages_old, star_ages_new, star_metallicities, sol
            ) * dt
            
        else:
            raise ValueError(f"Unknown injection_mode: {self.injection_mode}")

        if self.one_injection:
            inject_now      = jnp.equal(i, 0)
            per_star_source = jnp.where(inject_now, per_star_source, 0.0)

        if "star_positions" not in params or params["star_positions"] is None:
            star_positions = jnp.asarray(
                [[self.mesh_shape[0] // 2, self.mesh_shape[1] // 2, self.mesh_shape[2] // 2]],
                dtype=jnp.int32,
            )
            if jnp.ndim(per_star_source) == 0:
                per_star_source = jnp.asarray([per_star_source])
            else:
                per_star_source = jnp.asarray([jnp.sum(per_star_source)])
        else:
            star_positions = jnp.asarray(params["star_positions"], dtype=jnp.int32)
            if jnp.ndim(per_star_source) == 0:
                per_star_source = jnp.full((star_positions.shape[0],), per_star_source)

        ix = star_positions[:, 0]
        iy = star_positions[:, 1]
        iz = star_positions[:, 2]

        sigma    = max(1, round(self.mesh_shape[0] // 100))
        offsets  = jnp.arange(-3 * sigma, 3 * sigma + 1)
        beam_len = int(self.beam_length_cells)

        # ── Photon injection ──
        if self.momentum_only == False:
            if self.gaussian_star == False:
                for s in range(star_positions.shape[0]):
                    x0, y0, z0 = ix[s], iy[s], iz[s]
                    x_safe = jnp.clip(x0, 0, self.mesh_shape[0] - 1)
                    y_safe = jnp.clip(y0, 0, self.mesh_shape[1] - 1)
                    z_safe = jnp.clip(z0, 0, self.mesh_shape[2] - 1)

                    in_bounds = (
                        (0 <= x0) & (x0 < self.mesh_shape[0]) &
                        (0 <= y0) & (y0 < self.mesh_shape[1]) &
                        (0 <= z0) & (z0 < self.mesh_shape[2]))
                    sol = sol.at[0, x_safe, y_safe, z_safe].add(jnp.where(in_bounds, per_star_source[s], 0.0))
            else:
                for s in range(star_positions.shape[0]):
                    x0, y0, z0 = ix[s], iy[s], iz[s]
                    if self.injection_geometry == "2D":
                        sol = self._inject_energy_2d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                    elif self.injection_geometry == "3D":
                        sol = self._inject_energy_3d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                    elif self.injection_geometry == "beam_x":
                        sol = self._inject_energy_beam_x(sol, x0, y0, z0, per_star_source[s], sigma, beam_len)
                    else:
                        raise ValueError(f"Unknown injection_geometry: {self.injection_geometry}")

        # ── Momentum injection ──
        if self.injection_momentum:
            if self.gaussian_star == False:
                for s in range(star_positions.shape[0]):
                    x0, y0, z0 = ix[s], iy[s], iz[s]
                    x_safe = jnp.clip(x0, 0, self.mesh_shape[0] - 1)
                    y_safe = jnp.clip(y0, 0, self.mesh_shape[1] - 1)
                    z_safe = jnp.clip(z0, 0, self.mesh_shape[2] - 1)

                    in_bounds = (
                        (0 <= x0) & (x0 < self.mesh_shape[0]) &
                        (0 <= y0) & (y0 < self.mesh_shape[1]) &
                        (0 <= z0) & (z0 < self.mesh_shape[2]))
                    sol = sol.at[0, x_safe, y_safe, z_safe].add(jnp.where(in_bounds, per_star_source[s], 0.0))
            else:
                for s in range(star_positions.shape[0]):
                    x0, y0, z0 = ix[s], iy[s], iz[s]
                    if self.chemistry == False: 
                        if self.injection_geometry == "2D":
                            wdbg, sol = self._inject_momentum_x_2d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                            if self.debug:
                                bad_E = jnp.any(~jnp.isfinite(sol[0]))
                                bad_F = jnp.any(~jnp.isfinite(sol[1:]))
                                jax.debug.print(
                                    "NaN/Inf after injection? Egamma={E_bad}, Fgamma={F_bad}",
                                    E_bad=bad_E, F_bad=bad_F,
                                )
                                jax.debug.print("sum weight = 1? {}", wdbg.sum())
                        elif self.injection_geometry == "3D":
                            sol = self._inject_momentum_x_3d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                        elif self.injection_geometry == "beam_x":
                            wdbg, sol = self._inject_momentum_beam_x(sol, x0, y0, z0, per_star_source[s], sigma, beam_len)
                            if self.debug:
                                self.debug_grid_stats(sol, self.eq, "after clip sol momentum injection beam", 0)
                        else:
                            raise ValueError(f"Unknown injection_geometry: {self.injection_geometry}")
                        
                    else:
                        if self.injection_geometry == "2D":
                            wdbg, sol = self._inject_momentum_radial_2d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                        elif self.injection_geometry == "3D":
                            wdbg, sol = self._inject_momentum_radial_3d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                        elif self.injection_geometry == "beam_x":
                            wdbg, sol = self._inject_momentum_beam_x(sol, x0, y0, z0, per_star_source[s], sigma, beam_len)
        # Safety limiter
        sol = self._clip_to_m1_cone(sol)

        if self.debug:
            self._debug_after_clip(sol, i)   # <-- voir note ci-dessous

        params_out              = dict(params)
        params_out["star_ages"] = star_ages_new
        self.sol = sol
        return sol, params_out


    def get_N_gamma(self, star_masses, star_ages_old, star_ages_new, star_metallicities, sol):
        emission_old  = self.get_stellar_emission(star_ages_old, star_metallicities)
        emission_new  = self.get_stellar_emission(star_ages_new, star_metallicities)
        delta_emission = emission_new - emission_old
        cell_volume   = self.dx ** (sol.ndim - 1)
        return (star_masses * delta_emission) * self.escape_fraction / cell_volume

    def get_N_gamma_stromgen_sphere(self):
        # stromgren_rate est un TAUX de photons (photons par unite de temps code),
        # alors que sol[0] est une DENSITE (photons par unite de volume code) :
        # il faut donc etaler le taux sur le volume de cellule dx**ndim, avec
        # ndim = nombre d'axes spatiaux (len(mesh_shape), comme sol.ndim - 1).
        cell_volume = self.dx ** len(self.mesh_shape)
        return self.stromgren_rate / cell_volume

    def get_temp_code(self, cu, sol):
        rho_code = sol[4]
        p_code   = sol[9]
        T_code_field = temperature_code_from_Prho(p_code, rho_code, cu)
        return T_code_field
    
    def molar_mass_to_code(M_g_per_mol: float, cu) -> float:
        """Convert a molar mass (g/mol) to code units (code-mass/mol).
        N_avogadro is NOT rescaled: mol is a pure count, not a length/mass/time
        dimension, so cu.N_avogadro_cgs stays valid unchanged in code units."""
        return M_g_per_mol / cu.M_cgs
    
    def rate_coeff_to_code(value_cgs: float, cu) -> float:
        """Convert a cm^3 s^-1 rate coefficient (e.g. recombination rate alpha_B)
        to code units, using only L_cgs and T_cgs from CodeUnits."""
        unit_scale = cu.L_cgs**3 / cu.T_cgs
        return value_cgs / unit_scale
    
    def caseA(self, T):
        T = 1
        return self.rate_coeff_to_code(4.2e-13, self.cu) * T #conversion to code units needed
    
    def caseB(self, T):
        return self.rate_coeff_to_code(2.6e-13, self.cu) * (T / 1e4) ** -0.7 #conversion to code units needed

    def get_number_density_gas(self, rho_gas, N_avogadro=6.022e23,M_molaire=2.3e-26): #conversion to code units needed
        return rho_gas  / (self.molar_mass_to_code(M_molaire, self.cu) * N_avogadro) # waiting for Enrico check
    
    def get_sigma_HI(self, T):
        return  #conversion to code units needed
    
    def get_N_chemistry(self, rho_gas ,sigma_HI, caseA, caseB, fraction_HI, T ):
        Nstar_plus_Nrec = - self.get_N_gamma_stromgen_sphere() * self.get_number_density_gas(rho_gas) * sigma_HI * self.light_speed * (1-fraction_HI) 
        + ((caseA(self,T) - caseB(self,T)) * self.get_number_density_gas(rho_gas)**2 * fraction_HI*(1-fraction_HI) )
        return Nstar_plus_Nrec

    def get_flux_source_decay(self, rho_gas, sigma_HI, fraction_HI, dt):
        """
        Implicit (exact) decay factor for the radiative flux absorption sink,
        symmetric to the energy sink term but with the extra factor of c:
            dF/dt = -c * n_HI * sigma_HI * F
        Solved analytically over dt to remain stable for stiff opacities
        (avoids explicit update F += -c*kappa*F*dt blowing up when
        c*kappa*dt >> 1).
        """
        n = self.get_number_density_gas(rho_gas)
        n_HI = n * fraction_HI
        kappa = n_HI * sigma_HI                     # code units: 1 / length
        decay = jnp.exp(-kappa * self.light_speed * dt)
        return decay


    def apply_flux_chemistry_sink(self, sol, rho_gas, sigma_HI, fraction_HI, dt):
        """Apply the M1 flux absorption sink to Fx, Fy, Fz (sol[1:4])."""
        decay = self.get_flux_source_decay(rho_gas, sigma_HI, fraction_HI, dt)
        sol = sol.at[1].multiply(decay)
        sol = sol.at[2].multiply(decay)
        sol = sol.at[3].multiply(decay)
        return sol