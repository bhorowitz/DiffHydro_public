
"""
Radiative transfer module (stub).

Placeholder functions for future radiative transfer physics.
All methods currently return stub values.
"""

import os

import jax
import jax.numpy as jnp
from ..utils.debug_checks import _check_finite, _check_all_float_variables
from ..units import CodeUnits, UnitParser, from_code, to_code
from diffhydro.equationmanager_radiative_transf_no_chat import EquationManager as EquationManager_RT
from diffhydro.units.convert import temperature_code_from_Prho
from . import hydrogen_chemistry as hchem
from .hydrogen_chemistry import HydrogenStateView


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


# new version
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
        hydro_eq=None,
        debug=False,
        one_injection=False,
        # --- beam parameters ---
        beam_axis=0,             # 0=x, 1=y, 2=z
        beam_sign=+1,            # +1 or -1
        beam_length_cells=8,     # number of beam cells
        beam_sigma=3.0,          # Gaussian spreading
        beam_reduced_flux=0.95,  # |F|/(c*E) max
        beam_momentum_scaling="legacy_c2_source2",
        cu=None,
        parser=None,
        chemistry=False,
        # --- chemistry defaults (used only if injection_mode == "stromgren"
        #     and chemistry-related fields are not supplied through sol) ---
        default_sigma_HI_cgs=hchem.SIGMA_HI_0_CGS,  # cm^2, HI photoionization x-section at threshold
        default_temperature_K=1e4,      # K, fallback gas temperature
        # --- chemistry conventions ---
        frequency=hchem.NU_HI_CGS,      # single photon group frequency [Hz]
        chemistry_case="A",             # must match HydrogenIonizationForce
        b_rec=None,                     # None -> 1.0 for case A, 0.0 for case B
        xHII_weighted=False,            # True if sol[idx_xHII] stores N_gamma * x_HII
        X_H=1.0,                        # hydrogen mass fraction (pure H here)
        chem_max_frac=0.9,              # positivity limiter: |dN| <= 0.9 N
        photon_sink_mode="explicit_limited",  # or "exponential" (exact)
        expansion_factor=1.0,           # 'a' in the Compton term, fixed to 1
    ):
        self.eps = 1e-30
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
        # beam
        self.beam_axis = beam_axis
        self.beam_sign = beam_sign
        self.beam_length_cells = beam_length_cells
        self.beam_sigma = beam_sigma
        self.beam_reduced_flux = beam_reduced_flux
        self.beam_momentum_scaling = beam_momentum_scaling
        self.cu = cu
        self.parser = parser or UnitParser()
        self.sol = None
        self.eq = eq
        self.hydro_eq = hydro_eq
        self.debug_dt_sum_code = 0.0
        self.debug_dt_sum_phys = 0.0
        self.debug_step_count = 0
        # chemistry fallback defaults
        self.default_sigma_HI_cgs = default_sigma_HI_cgs
        self.default_temperature_K = default_temperature_K
        # chemistry conventions
        self.frequency = float(frequency)
        self.chemistry_case = str(chemistry_case).upper()
        self.b_rec = (1.0 if self.chemistry_case == "A" else 0.0) if b_rec is None else float(b_rec)
        self.xHII_weighted = bool(xHII_weighted)
        self.X_H = float(X_H)
        self.chem_max_frac = float(chem_max_frac)
        self.photon_sink_mode = str(photon_sink_mode)
        self.expansion_factor = float(expansion_factor)
        # cgs cross-section of the single transported group; with
        # frequency == NU_HI_CGS this is sigma_0 = 6.35e-18 cm^2.
        self.sigma_HI_cgs = float(
            hchem.sigma_HI_powerlaw_cgs(nu=self.frequency, nu0=hchem.NU_HI_CGS)
        )

        # ── Layout of the combined sol array (RT | hydro | chemistry) ──

        # Absolute indices into the combined sol tensor. The radiative state
        # keeps only the photon moments; x_HII now belongs to the gas block.
        self.n_rt_cons = eq.n_cons if eq is not None else 4
        self.n_hydro_cons = hydro_eq.n_cons if hydro_eq is not None else 0

        if self.n_hydro_cons > 0:
            self.idx_rho = self.n_rt_cons
            self.idx_mom = slice(self.n_rt_cons + 1, self.n_rt_cons + 4)
            self.idx_pressure = self.n_rt_cons + 4 if self.n_hydro_cons >= 5 else self.idx_rho
            self.idx_energy = self.idx_pressure
            self.idx_xHII = self.n_rt_cons + getattr(hydro_eq, "xHII_id", self.n_hydro_cons - 1)
        else:
            self.idx_rho = 0
            self.idx_mom = slice(1, 4)
            self.idx_energy = 0
            self.idx_pressure = 0
            self.idx_xHII = 0

        # ── Unit-safe view of the CONSERVATIVE state used by the chemistry ──
        # This is the only object allowed to translate between the solver's
        # code units and the cgs rate coefficients of hydrogen_chemistry.
        self.view = None
        if self.cu is not None and self.n_hydro_cons > 0:
            self.view = HydrogenStateView(
                cu=self.cu,
                gamma=getattr(hydro_eq, "gamma", 5.0 / 3.0),
                idx_N=0,
                idx_F=(1, 2, 3),
                idx_xHII=self.idx_xHII,
                idx_rho=self.idx_rho,
                idx_mom=tuple(range(self.idx_mom.start, self.idx_mom.stop)),
                idx_Etot=self.idx_energy,
                xHII_weight_idx=self.idx_rho,
                X_H=self.X_H,
                light_speed_code=self.light_speed,
            )
        if self.chemistry and self.view is None:
            raise ValueError(
                "chemistry=True requires both `cu` (CodeUnits) and `hydro_eq`: "
                "the hydrogen chemistry needs rho, E_tot and x_HII from the "
                "combined state, plus the unit system to convert to cgs."
            )

    def set_code_units(self, cu: CodeUnits, parser: "UnitParser | None" = None):
        """Configure the unit system used by the radiative source helpers."""
        self.cu = cu
        self.parser = parser or self.parser or UnitParser()

    def convert_physical_to_code(self, value, dimension: str):
        """Convert a physical quantity into code units using the configured CodeUnits."""
        if self.cu is None:
            raise ValueError("CodeUnits not configured.")
        return to_code(value, dimension, self.cu, self.parser)

    def convert_code_to_physical(self, value, dimension: str, out_unit: "str | None" = None):
        """Convert a code-unit quantity back to a physical value wrapper."""
        if self.cu is None:
            raise ValueError("CodeUnits not configured.")
        unit = out_unit or self.parser.default_cgs_unit(dimension)
        return from_code(value, dimension, self.cu, unit, self.parser)

    def temp_phys_to_code(self, T_phys):
        """Convert a physical temperature (float in Kelvin, or a unit string
        like '5000 K') into code units. Does NOT read from environment."""
        if isinstance(T_phys, str):
            temp_q = self.parser.parse(T_phys, expected_dim="temperature")
            T_phys_K = temp_q.cgs_value
        else:
            T_phys_K = float(T_phys)  # assumed already in Kelvin (cgs)
        return T_phys_K / self.cu.Temp_cgs

    def get_temperature_K(self, sol):
        """Gas temperature [K] from the CONSERVATIVE state.

        ``sol[self.idx_pressure]`` is NOT the pressure: forces are applied
        by ``hydro._hydrostep`` on the conservative array, so that slot
        holds the TOTAL energy E_tot = rho e + 0.5 rho v^2. The pressure is
        recovered as p = (gamma - 1)(E_tot - 0.5 rho v^2) and the
        temperature uses n_tot = n_H (1 + x_HII) for a pure-H plasma.
        """
        return self.view.temperature_K(sol)

    def get_temp_code(self, sol):
        """Same as :meth:`get_temperature_K`, expressed in code temperature
        units (kept for backward compatibility with existing call sites)."""
        return self.get_temperature_K(sol) / self.cu.Temp_cgs

    def get_stellar_emission(self, star_age, star_metallicity):
        if self.stellar_spectrum_func is not None:
            return self.stellar_spectrum_func(star_age, star_metallicity)
        age_factor = jnp.exp(-star_age / 10.0)
        Z_factor = jnp.maximum(star_metallicity, 1e-4)
        return age_factor * Z_factor

    def timestep(self, sol):
        return 1e30

    def debug_grid_stats(self, sol, eq, label, ax):
        E = sol[0]
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

        E_masked = jnp.where(mask, E, 0.0)
        Fx_masked = jnp.where(mask, Fx, 0.0)
        Fy_masked = jnp.where(mask, Fy, 0.0)
        Fz_masked = jnp.where(mask, Fz, 0.0)
        Fmag_masked = jnp.where(mask, Fmag, 0.0)
        ratio_masked = jnp.where(mask_ratio, ratio, 0.0)

        E_min_vis = jnp.min(jnp.where(mask, E, jnp.inf))
        Fx_min_vis = jnp.min(jnp.where(mask, Fx, jnp.inf))
        Fy_min_vis = jnp.min(jnp.where(mask, Fy, jnp.inf))
        Fz_min_vis = jnp.min(jnp.where(mask, Fz, jnp.inf))
        Fmag_min_vis = jnp.min(jnp.where(mask, Fmag, jnp.inf))
        ratio_min_vis = jnp.min(jnp.where(mask_ratio, ratio, jnp.inf))

        E_max_vis = jnp.max(jnp.where(mask, E, -jnp.inf))
        Fx_max_vis = jnp.max(jnp.where(mask, Fx, -jnp.inf))
        Fy_max_vis = jnp.max(jnp.where(mask, Fy, -jnp.inf))
        Fz_max_vis = jnp.max(jnp.where(mask, Fz, -jnp.inf))
        Fmag_max_vis = jnp.max(jnp.where(mask, Fmag, -jnp.inf))
        ratio_max_vis = jnp.max(jnp.where(mask_ratio, ratio, -jnp.inf))

        denom = jnp.maximum(n_active, 1)
        denom_ratio = jnp.maximum(n_ratio_active, 1)

        E_mean_vis = jnp.sum(E_masked) / denom
        Fx_mean_vis = jnp.sum(Fx_masked) / denom
        Fy_mean_vis = jnp.sum(Fy_masked) / denom
        Fz_mean_vis = jnp.sum(Fz_masked) / denom
        Fmag_mean_vis = jnp.sum(Fmag_masked) / denom
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

    # ─────────────── Energy injection ───────────────

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
        weights = jnp.exp(-(s_float**2) / (2.0 * float(sigma)**2))
        weights = jnp.where(valid, weights, 0.0)
        weights = weights / (jnp.sum(weights) + 1e-30)
        if self.debug:
            self.debug_grid_stats(sol, self.eq, "energy injection beam (before)", 0)
        return sol.at[0, xi, yi, zi].add(source * weights)

    def _inject_energy_2d(self, sol, x0, y0, z0, source, offsets, sigma):
        if self.debug:
            self.debug_grid_stats(sol, self.eq, "energy injection 2D (before clip)", 0)
        di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
        xi, yi, zi, valid = self._clip_indices_2d(x0, y0, z0, di2, dj2)
        _, _, weights2 = self._normalized_weights_2d(offsets, sigma, valid)
        if self.debug:
            self.debug_grid_stats(sol, self.eq, "energy injection 2D", 0)
        return sol.at[0, xi, yi, zi].add(source * weights2)

    def _inject_energy_3d(self, sol, x0, y0, z0, source, offsets, sigma):
        di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
        xi, yi, zi, valid = self._clip_indices_3d(x0, y0, z0, di3, dj3, dk3)
        _, _, _, weights3 = self._normalized_weights_3d(offsets, sigma, valid)
        return sol.at[0, xi, yi, zi].add(source * weights3)

    # ─────────────── Momentum injection ───────────────

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
        weights = jnp.exp(-(s_float**2) / (2.0 * float(sigma)**2))
        weights = jnp.where(valid, weights, 0.0)
        weights = weights / (jnp.sum(weights) + 1e-30)
        fx_inj = self._beam_momentum_factor(source, weights)
        if self.debug:
            jax.debug.print("Injecting momentum beam at x=[{}, {}], y={}, z={}", xi[0], xi[-1], yi[0], zi[0])
            jax.debug.print("Momentum source: {}, weights sum: {}", source, jnp.sum(weights))
            jax.debug.print("Fx injection profile: {}", fx_inj)
            jax.debug.print("Sol[1] after injection: {}", sol[1, xi, yi, zi])
            jax.debug.print("max E = {}", jnp.max(sol[0]))
            jax.debug.print("any nan E = {}", jnp.any(jnp.isnan(sol[0])))
            jax.debug.print("any inf E = {}", jnp.any(jnp.isinf(sol[0])))
        sol = sol.at[1, xi, yi, zi].add(fx_inj)
        sol = sol.at[2, xi, yi, zi].add(jnp.zeros_like(fx_inj))
        sol = sol.at[3, xi, yi, zi].add(jnp.zeros_like(fx_inj))
        return weights, sol

    def _inject_momentum_x_2d(self, sol, x0, y0, z0, source, offsets, sigma):
        di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
        xi, yi, zi, valid = self._clip_indices_2d(x0, y0, z0, di2, dj2)
        _, _, weights2 = self._normalized_weights_2d(offsets, sigma, valid)
        fx_inj = self._beam_momentum_factor(source, weights2)
        sol = sol.at[1, xi, yi, zi].add(fx_inj)
        sol = sol.at[2, xi, yi, zi].add(jnp.zeros_like(fx_inj))
        sol = sol.at[3, xi, yi, zi].add(jnp.zeros_like(fx_inj))
        if self.debug:
            E = sol[0, xi, yi, zi]
            jax.debug.print("Injecting momentum 2D at x=[{}, {}], y={}, z={}", xi[0, 0], xi[-1, -1], y0, z0)
            jax.debug.print("Momentum source: {}, weights sum: {}", source, jnp.sum(weights2))
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
        sol = sol.at[2, xi, yi, zi].add(jnp.zeros_like(fx_inj))
        sol = sol.at[3, xi, yi, zi].add(jnp.zeros_like(fx_inj))
        return weights3, sol

    def _inject_momentum_radial_2d(self, sol, x0, y0, z0, source, offsets, sigma):
        """Radial (isotropic point-source) momentum injection in the (x, z=z0) plane.
        Unlike a beam, F points outward from the source at each offset cell,
        matching an isotropically emitting star rather than a collimated jet."""
        di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
        xi, yi, zi, valid = self._clip_indices_2d(x0, y0, z0, di2, dj2)
        _, _, weights2 = self._normalized_weights_2d(offsets, sigma, valid)

        r = jnp.sqrt(di2**2 + dj2**2)
        inv_r = jnp.where(r > 0, 1.0 / r, 0.0)  # avoid 0/0 at the source cell
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
        """Renormalize F_gamma onto the M1 causality cone |F| <= f_max c N.

        This is the flux renormalization requested for the chemistry: after
        an absorption sink (or an injection) the direction of F is kept and
        only its magnitude is rescaled by (f_max c N) / |F| when |F| exceeds
        the cone, which is the condition for the M1 Eddington factor
        chi(f) to remain defined (f = |F|/(cN) must be in [0, 1]).
        """
        Fx, Fy, Fz = hchem.limit_m1_flux_cone(
            sol[0], sol[1], sol[2], sol[3],
            c=self.light_speed, f_max=self.beam_reduced_flux, eps=1e-30,
        )
        sol = sol.at[1].set(Fx)
        sol = sol.at[2].set(Fy)
        sol = sol.at[3].set(Fz)
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
        # rate_code is a photon density rate [ph / code-volume / code-time];
        # CodeUnits has no PhotonRate_cgs attribute, the physical rate is
        # rate_code * cell_volume_code / T_cgs.
        cell_volume_code = self.dx ** len(self.mesh_shape)
        photon_rate_cgs = jnp.asarray(cell_volume_code / self.cu.T_cgs, dtype=dt.dtype)

        dt_phys_s = dt * T_cgs
        rate_phys = rate_code * photon_rate_cgs
        injected_phys = rate_phys * dt_phys_s
        sum_dt_phys = jnp.sum(dt_phys_s)

        jax.debug.print(
            """
            [stromgren debug]
            dt_code = {dt_code}
            dt_phys_s = {dt_phys_s}

            stromgren_injection_code = {rate_code}
            stromgren_rate_phys_ph_per_s = {rate_phys}

            per_star_source_code = {src_code}
            per_star_source_phys_photons = {src_phys}
            evolve_dt = {sum_dt_phys}
            """,
            dt_code=dt,
            dt_phys_s=dt_phys_s,
            rate_code=rate_code,
            rate_phys=rate_phys,
            src_code=per_star_source,
            src_phys=injected_phys,
            sum_dt_phys=sum_dt_phys,
            ordered=True,
        )

    def force(self, i, sol, params, dt):
        if "star_masses" not in params or params["star_masses"] is None:
            # No stars: no injection, but the chemistry sink still applies
            # to whatever radiation is already in the box.
            if self.chemistry == True:
                sol = self.apply_photon_chemistry_sink(sol, dt)
                sol = self.apply_flux_chemistry_sink(sol, dt)
            return sol, params
        if self.debug:
            self.debug_grid_stats(sol, self.eq, "sol before injection", 0)

        star_masses = jnp.asarray(params["star_masses"])
        star_ages_old = jnp.asarray(params["star_ages"])
        star_ages_new = star_ages_old + dt
        star_metallicities = jnp.asarray(params["star_metallicities"])

        # ── Chemistry sink/source on the radiation field (eq. 25' and 26') ──
        # Applied BEFORE the stellar injection so that the freshly injected
        # photons are not immediately absorbed within the same half step,
        # and with the positivity limiter |Delta N| <= 0.9 N.
        if self.chemistry == True:
            sol = self.apply_photon_chemistry_sink(sol, dt)
            sol = self.apply_flux_chemistry_sink(sol, dt)
            if self.debug:  # forcer temporairement
                jax.debug.print(
                "DEBUG_CHEM sigma_HI_cgs={} n_HI_max={} N_gamma_max={} chem_term_min={} chem_term_max={}",
                self.sigma_HI_cgs,
                jnp.max(self.get_number_density_HI(sol)),
                jnp.max(self.view.photon_density_cgs(sol)),
                jnp.min(self.get_N_chemistry(sol)),
                jnp.max(self.get_N_chemistry(sol)),
                )
        if self.injection_mode == "stromgren":
            per_star_source = self.get_N_gamma_stromgen_sphere() * dt
            if self.debug:
                self._debug_stromgren_units_jax(dt, per_star_source)

        elif self.injection_mode == "physical":
            per_star_source = self.get_N_gamma(
                star_masses, star_ages_old, star_ages_new, star_metallicities, sol
            )
        else:
            raise ValueError(f"Unknown injection_mode: {self.injection_mode}")

        if self.one_injection:
            inject_now = jnp.equal(i, 0)
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

        sigma = max(1, round(self.mesh_shape[0] // 100))
        offsets = jnp.arange(-3 * sigma, 3 * sigma + 1)
        beam_len = int(self.beam_length_cells)

        # ── Photon injection ──
        if not self.momentum_only:
            if not self.gaussian_star:
                for s in range(star_positions.shape[0]):
                    x0, y0, z0 = ix[s], iy[s], iz[s]
                    x_safe = jnp.clip(x0, 0, self.mesh_shape[0] - 1)
                    y_safe = jnp.clip(y0, 0, self.mesh_shape[1] - 1)
                    z_safe = jnp.clip(z0, 0, self.mesh_shape[2] - 1)

                    in_bounds = (
                        (0 <= x0) & (x0 < self.mesh_shape[0]) &
                        (0 <= y0) & (y0 < self.mesh_shape[1]) &
                        (0 <= z0) & (z0 < self.mesh_shape[2])
                    )
                    sol = sol.at[0, x_safe, y_safe, z_safe].add(jnp.where(in_bounds, per_star_source[s], 0.0))
            else:
                for s in range(star_positions.shape[0]):
                    x0, y0, z0 = ix[s], iy[s], iz[s]
                    if self.injection_geometry in ("2D", "radial_2D"):
                        sol = self._inject_energy_2d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                    elif self.injection_geometry in ("3D", "radial_3D"):
                        sol = self._inject_energy_3d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                    elif self.injection_geometry == "beam_x":
                        sol = self._inject_energy_beam_x(sol, x0, y0, z0, per_star_source[s], sigma, beam_len)
                    else:
                        raise ValueError(f"Unknown injection_geometry: {self.injection_geometry}")

        # ── Momentum injection ──
        if self.injection_momentum:
            if not self.gaussian_star:
                for s in range(star_positions.shape[0]):
                    x0, y0, z0 = ix[s], iy[s], iz[s]
                    x_safe = jnp.clip(x0, 0, self.mesh_shape[0] - 1)
                    y_safe = jnp.clip(y0, 0, self.mesh_shape[1] - 1)
                    z_safe = jnp.clip(z0, 0, self.mesh_shape[2] - 1)

                    in_bounds = (
                        (0 <= x0) & (x0 < self.mesh_shape[0]) &
                        (0 <= y0) & (y0 < self.mesh_shape[1]) &
                        (0 <= z0) & (z0 < self.mesh_shape[2])
                    )
                    sol = sol.at[0, x_safe, y_safe, z_safe].add(jnp.where(in_bounds, per_star_source[s], 0.0))
            else:
                # NOTE: this branch used to be guarded by `if not
                # self.chemistry`, so enabling chemistry silently disabled
                # the whole momentum (F_gamma) injection and replaced it by
                # a second call to the flux sink. The sink now lives at the
                # top of force(), and the injection always runs.
                for s in range(star_positions.shape[0]):
                    x0, y0, z0 = ix[s], iy[s], iz[s]
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
                        # WARNING: "_x_3d" injects F along +x ONLY, so a
                        # "stromgren" point source is in fact a collimated
                        # source. Use injection_geometry="radial_3D" for a
                        # genuinely isotropic star.
                        wdbg, sol = self._inject_momentum_x_3d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                    elif self.injection_geometry == "radial_2D":
                        wdbg, sol = self._inject_momentum_radial_2d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                    elif self.injection_geometry == "radial_3D":
                        wdbg, sol = self._inject_momentum_radial_3d(sol, x0, y0, z0, per_star_source[s], offsets, sigma)
                    elif self.injection_geometry == "beam_x":
                        wdbg, sol = self._inject_momentum_beam_x(sol, x0, y0, z0, per_star_source[s], sigma, beam_len)
                        if self.debug:
                            self.debug_grid_stats(sol, self.eq, "momentum injection beam (after)", 0)
                    else:
                        raise ValueError(f"Unknown injection_geometry: {self.injection_geometry}")

        # Safety limiter: enforce the M1 causality cone |F| <= beam_reduced_flux * c * E
        sol = self._clip_to_m1_cone(sol)

        if self.debug:
            self.debug_grid_stats(sol, self.eq, "sol after injection + clip", 0)

        params_out = dict(params)
        params_out["star_ages"] = star_ages_new
        self.sol = sol
        return sol, params_out

    def get_N_gamma(self, star_masses, star_ages_old, star_ages_new, star_metallicities, sol):
        emission_old = self.get_stellar_emission(star_ages_old, star_metallicities)
        emission_new = self.get_stellar_emission(star_ages_new, star_metallicities)
        delta_emission = emission_new - emission_old
        cell_volume = self.dx ** (sol.ndim - 1)
        return (star_masses * delta_emission) * self.escape_fraction / cell_volume

    def get_N_gamma_stromgen_sphere(self):
        # stromgren_rate is a photon RATE (photons per unit code time), while
        # sol[0] is a DENSITY (photons per unit code volume): the rate must be
        # spread over the cell volume dx**ndim, where ndim is the number of
        # spatial axes (len(mesh_shape), same as sol.ndim - 1).
        cell_volume = self.dx ** len(self.mesh_shape)
        return self.stromgren_rate / cell_volume

    def molar_mass_to_code(self, M_g_per_mol: float, cu) -> float:
        """Convert a molar mass (g/mol) to code units (code-mass/mol).
        N_avogadro is NOT rescaled: mol is a pure count, not a length/mass/time
        dimension, so N_avogadro stays valid unchanged in code units."""
        return M_g_per_mol / cu.M_cgs

    def rate_coeff_to_code(self, value_cgs: float, cu) -> float:
        """Convert a cm^3 s^-1 rate coefficient (e.g. recombination rate alpha_B)
        to code units, using only L_cgs and T_cgs from CodeUnits."""
        unit_scale = cu.L_cgs**3 / cu.T_cgs
        return value_cgs / unit_scale

    def _lambda_HI(self, T_K):
        """Hui & Gnedin (1997) dimensionless variable for HI recombination fits.
        T_K must be in Kelvin."""
        return 315614.0 / jnp.maximum(T_K, self.eps)

    def caseA(self, T_K):
        """Case A recombination coefficient alpha^A_HII [cm^3 s^-1].

        Hui & Gnedin (1997). NOTE: the argument is now in KELVIN, not in
        code temperature units -- the whole chemistry is evaluated in cgs
        and converted once, in HydrogenStateView.
        """
        return hchem.alpha_A_HII_cgs(T_K)

    def caseB(self, T_K):
        """Case B recombination coefficient alpha^B_HII [cm^3 s^-1] (Kelvin in)."""
        return hchem.alpha_B_HII_cgs(T_K)

    def get_number_density_gas(self, rho_gas, proton_mass_cgs=hchem.MH_CGS,
                               mean_molecular_weight=None):
        """Hydrogen nucleus number density n_H [cm^-3] from rho in CODE units.

            n_H = rho_code * rho_cgs * X_H / m_H

        Previously this divided by ``cu.mu = 0.61``, the mean molecular
        weight of a fully ionized primordial gas: for a pure-hydrogen run
        that overestimates n_H by a factor 1/0.61 = 1.64 and therefore
        n_HI, n_HII, n_e, every recombination rate and the whole cooling
        function. For hydrogen nuclei the correct divisor is m_H alone.
        """
        if self.cu is None:
            raise ValueError("CodeUnits not configured.")
        mu = 1.0 if mean_molecular_weight is None else mean_molecular_weight
        return jnp.asarray(rho_gas) * self.cu.rho_cgs * self.X_H / (mu * proton_mass_cgs)

    def get_number_density_HI(self, sol):
        """n_HI [cm^-3], reading x_HII through the state view (so the
        conservative/primitive convention is handled in one place)."""
        _, n_HI, _, _ = self.view.number_densities_cgs(sol)
        return n_HI


    # def get_sigma_HI(self, energies, T):
    #     """HI photoionization cross-section, converted to code units (length^2).
    #     Uses a fixed threshold cross-section (cm^2) independent of T by default;
    #     override default_sigma_HI_cgs at construction time if a T-dependent
    #     cross-section is required."""
    #     L_cgs = getattr(self.cu, "L_cgs", 1.0)
    #     x = energies /13.6
    #     y = jnp.sqrt(x**2 - 0)
    #     sigma_0 = 5.475e-14  # cm^2
    #     P = 2.963
    #     ya = 32.88
    #     energy_0 = 0.4298
    #     energy_ionization = 13.6
    #     if energies >= energy_ionization:
    #         sigma_mono_lambda = sigma_0 * (x - 1)**2 * (((energy_ionization / energy_0)**2 + 0)**0.5)**(0.5*P-5.5) / (1 + ((((energy_ionization / energy_0 )**2)**0.5) / ya)**0.5  )**P
    #     else:
    #         sigma_mono_lambda = 0.0
    #     return sigma_mono_lambda / L_cgs**2
    def get_sigma_HI(self, frequency=None):
        """HI photoionization cross-section [cm^2] for the transported group.

            sigma_HI(nu) = sigma_HI(nu_0) * (nu / nu_0)^-3,  nu >= nu_0

        with sigma_HI(nu_0) = 6.35e-18 cm^2 obtained from the Verner et al.
        (1996) fit evaluated at 13.6 eV.

        Previous version returned ``5.475e-14 / L_cgs^2``. 5.475e-14 is the
        Verner *fit parameter* sigma_0, not the threshold cross-section:
        it overestimated sigma_HI by a factor ~8600, i.e. the gas was
        optically thick over ~1e-4 of the intended distance. The extra
        1/L_cgs^2 also made the result depend on the arbitrary code length
        unit; the chemistry is now done entirely in cgs.
        """
        nu = self.frequency if frequency is None else frequency
        return hchem.sigma_HI_powerlaw_cgs(nu=nu, nu0=hchem.NU_HI_CGS)

    def get_N_chemistry(self, sol, caseA=None, caseB=None, T=None):
        """dN_gamma/dt in cm^-3 s^-1 (eq. 25'), from the CONSERVATIVE state.

            dN/dt = -n_HI sigma_HI c N
                    + b_rec (alpha^A_HII - alpha^B_HII) n_HII n_e

        The second term is the diffuse recombination photons: b_rec = 1 in
        case A (they are put back into the radiation field, which is what
        the ionization equation assumes when it uses alpha^A) and b_rec = 0
        in case B (on-the-spot: they are assumed absorbed locally).

        ``caseA`` / ``caseB`` / ``T`` are accepted for backward
        compatibility; if given, T MUST be in Kelvin.
        """
        view = self.view
        x = view.xHII(sol)
        T_K = view.temperature_K(sol, x) if T is None else T
        _, n_HI, n_HII, n_e = view.number_densities_cgs(sol, x)
        N_cgs = view.photon_density_cgs(sol)

        alpha_A = hchem.alpha_A_HII_cgs(T_K) if caseA is None else caseA(T_K)
        alpha_B = hchem.alpha_B_HII_cgs(T_K) if caseB is None else caseB(T_K)

        ionization_loss = N_cgs * n_HI * self.sigma_HI_cgs * view.c_interaction_cgs
        recombination_gain = self.b_rec * (alpha_A - alpha_B) * n_HII * n_e
        return -ionization_loss + recombination_gain

    # def apply_photon_chemistry_sink(self, sol, dt):
    #     """Update N_gamma with the absorption sink / recombination source.

    #     ``photon_sink_mode``:

    #     * ``"explicit_limited"`` (default, the requested scheme)
    #           Delta N = dN/dt Delta t, clipped to -max_frac * N when
    #           negative, so N stays strictly positive whatever the value of
    #           n_HI sigma c dt. Beware: when n_HI sigma c dt >> 1 the
    #           limiter under-absorbs (it removes 90 % where the exact
    #           solution removes ~100 %), which makes an optically thick
    #           ionization front propagate slightly too fast.
    #     * ``"exponential"``
    #           exact solution of dN/dt = -k N + S over dt,
    #           N <- N e^{-k dt} + (S/k)(1 - e^{-k dt}).
    #           No limiter needed, positivity guaranteed, correct in the
    #           stiff limit. Use it if the front speed matters.
    #     """
    #     view = self.view
    #     dt_s = view.dt_cgs(dt) #version chat 
    #     N_cgs = view.photon_density_cgs(sol)

    #     if self.photon_sink_mode == "exponential":
    #         x = view.xHII(sol)
    #         T_K = view.temperature_K(sol, x)
    #         _, n_HI, n_HII, n_e = view.number_densities_cgs(sol, x)
    #         k = n_HI * self.sigma_HI_cgs * view.c_interaction_cgs     # s^-1
    #         S = self.b_rec * (hchem.alpha_A_HII_cgs(T_K)
    #                           - hchem.alpha_B_HII_cgs(T_K)) * n_HII * n_e
    #         decay = -k * dt_s
    #         k_safe = jnp.maximum(k, hchem.tiny_like(sol))
    #         N_new = N_cgs * decay + S 
    #         # k -> 0 limit: N + S dt
    #         # N_new = jnp.where(k * dt_s < 1e-8, N_cgs + S * dt_s, N_new)
    #         dN_cgs = N_new - N_cgs
    #     else:
    #         rate_cgs = self.get_N_chemistry(sol) * dt_s
    #         dN_cgs = hchem.limited_explicit_update(
    #             N_cgs, rate_cgs, dt_s, max_frac=self.chem_max_frac
    #         )
    #     return view.add_photons_cgs(sol, dN_cgs)
    def apply_photon_chemistry_sink(self, sol, dt):
        """Update N_gamma with the absorption sink / recombination source.

        ``photon_sink_mode``:

        * ``"explicit_limited"`` (default, the requested scheme)
            Delta N = dN/dt Delta t, clipped to -max_frac * N when
            negative, so N stays strictly positive whatever the value of
            n_HI sigma c dt. Beware: when n_HI sigma c dt >> 1 the
            limiter under-absorbs (it removes 90 % where the exact
            solution removes ~100 %), which makes an optically thick
            ionization front propagate slightly too fast.
        * ``"exponential"``
            exact solution of dN/dt = -k N + S over dt,
            N <- N e^{-k dt} + (S/k)(1 - e^{-k dt}).
            No limiter needed, positivity guaranteed, correct in the
            stiff limit. Use it if the front speed matters.
        """
        view = self.view
        dt_s = view.dt_cgs(dt)
        N_cgs = view.photon_density_cgs(sol)

        if self.photon_sink_mode == "exponential":
            x = view.xHII(sol)
            T_K = view.temperature_K(sol, x)
            _, n_HI, n_HII, n_e = view.number_densities_cgs(sol, x)
            k = n_HI * self.sigma_HI_cgs * view.c_interaction_cgs     # s^-1
            S = self.b_rec * (hchem.alpha_A_HII_cgs(T_K)
                            - hchem.alpha_B_HII_cgs(T_K)) * n_HII * n_e

            k_safe = jnp.maximum(k, hchem.tiny_like(sol))
            decay = jnp.exp(-k * dt_s)

            # Exact solution of dN/dt = -k N + S over dt:
            # N_new = N_cgs * e^{-k dt} + (S / k) * (1 - e^{-k dt})
            N_new = N_cgs * decay + (S / k_safe) * (1.0 - decay)

            # k -> 0 limit: dN/dt = S, so N_new = N_cgs + S * dt (no decay)
            N_new = jnp.where(k * dt_s < 1e-8, N_cgs + S * dt_s, N_new)

            dN_cgs = N_new - N_cgs
        else:
            rate_cgs = self.get_N_chemistry(sol) * dt_s
            dN_cgs = hchem.limited_explicit_update(
                N_cgs, rate_cgs, dt_s, max_frac=self.chem_max_frac
            )
        return view.add_photons_cgs(sol, dN_cgs)
    
    def get_flux_source_decay(self, sol, dt):
        """Exponent of the exact decay factor for the flux absorption sink.

            dF/dt = -n_HI sigma_HI c F   ->   F(t+dt) = F(t) exp(-kappa c dt)

        Returns the (negative, dimensionless) exponent -kappa c dt, where
        kappa = n_HI sigma_HI [cm^-1] and everything is evaluated in cgs.
        """
        n_HI = self.get_number_density_HI(sol)
        kappa = n_HI * self.sigma_HI_cgs                # cm^-1
        return -kappa * self.view.c_interaction_cgs * self.view.dt_cgs(dt)

    def apply_flux_chemistry_sink(self, sol, dt):
        """Apply the M1 flux absorption sink to Fx, Fy, Fz (sol[1:4]).

        The previous version did ``sol.at[1].add(decay)``: it ADDED a
        dimensionless number to the flux instead of scaling it, which
        (a) has the wrong units, (b) added the same value to Fx, Fy and Fz
        and so destroyed the isotropy of a point source, and (c) could
        drive F negative. The exact solution of dF/dt = -kappa c F is a
        MULTIPLICATIVE exponential decay, which is also unconditionally
        stable for stiff opacities.

        The M1 causality cone |F| <= f_max c N is re-imposed afterwards,
        because N and F are damped by different amounts once the photon
        update is limited.
        """
        decay = self.get_flux_source_decay(sol, dt)
        sol = sol.at[1].multiply(decay)
        sol = sol.at[2].multiply(decay)
        sol = sol.at[3].multiply(decay)
        return self._clip_to_m1_cone(sol)
