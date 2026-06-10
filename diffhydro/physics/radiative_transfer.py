"""
Radiative transfer module (stub).

Placeholder functions for future radiative transfer physics.
All methods currently return stub values.
"""

import jax
import jax.numpy as jnp
from ..utils.debug_checks import _check_finite,_check_all_float_variables
from diffhydro.equationmanager_radiative_transf_no_chat import EquationManager as EquationManager_RT

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
        # --- nouveaux paramètres beam ---
        beam_axis=0,          # 0=x, 1=y, 2=z
        beam_sign=+1,         # +1 ou -1
        beam_length_cells=8,  # nb de cellules du faisceau
        beam_sigma=3.0,       # étalement gaussien
        beam_reduced_flux=0.95,  # |F|/(c*E) max
        
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
        self.debug = debug
        self.light_speed = eq.light_speed if eq is not None else 1.0
        self.mesh_shape = eq.mesh_shape if eq is not None else (100, 100, 100)
        self.one_injection = one_injection
        #beam
        self.beam_axis        = beam_axis
        self.beam_sign        = beam_sign
        self.beam_length_cells= beam_length_cells
        self.beam_reduced_flux= beam_reduced_flux
        self.sol = None
        self.eq = eq

    def get_stellar_emission(self, star_age, star_metallicity):
        if self.stellar_spectrum_func is not None:
            return self.stellar_spectrum_func(star_age, star_metallicity)
        age_factor = jnp.exp(-star_age / 10.0)
        Z_factor = jnp.maximum(star_metallicity, 1e-4)
        return age_factor * Z_factor

    def timestep(self, sol):
        print("timestepblast")
        return 1e30
    
    def debug_grid_stats(self, sol, eq, label, ax):
        E = jnp.maximum(sol[0], eq.eps)
        Fx = sol[1]
        Fy = sol[2]
        Fz = sol[3]

        Fmag = jnp.sqrt(Fx**2 + Fy**2 + Fz**2)
        ratio = Fmag / jnp.maximum(E**2, 1e-30)

        jax.debug.print(
            """
            [{label}] ax={ax}
            shape = {shape}

            E:
            {label} mean={E_mean} {label} E min={E_min} max={E_max}
            any nan? {E_nan}
            any inf? {E_inf}
            any nonfinite? {E_bad}

            Fx:
            {label} mean={Fx_mean} {label} Fx min={Fx_min} max={Fx_max}
            Fy:
            {label} mean={Fy_mean} {label} Fy min={Fy_min} max={Fy_max}
            Fz:
            {label} mean={Fz_mean} {label} Fz min={Fz_min} max={Fz_max}

            |F|:
            mean={F_mean} min={F_min} max={F_max}
            any nan? {F_nan}
            any inf? {F_inf}
            any nonfinite? {F_bad}

            |F|/(E^2):
            mean={r_mean} min={r_min} max={r_max}
            any > 1 ? {r_gt1}
            tableau {ratio}
            """,
            label=label,
            ax=ax,
            shape=sol.shape,

            E_mean=jnp.mean(E),
            E_min=jnp.min(E),
            E_max=jnp.max(E),
            E_nan=jnp.any(jnp.isnan(E)),
            E_inf=jnp.any(jnp.isinf(E)),
            E_bad=jnp.any(~jnp.isfinite(E)),

            Fx_mean=jnp.mean(Fx),
            Fx_min=jnp.min(Fx),
            Fx_max=jnp.max(Fx),

            Fy_mean=jnp.mean(Fy),
            Fy_min=jnp.min(Fy),
            Fy_max=jnp.max(Fy),

            Fz_mean=jnp.mean(Fz),
            Fz_min=jnp.min(Fz),
            Fz_max=jnp.max(Fz),

            F_mean=jnp.mean(Fmag),
            F_min=jnp.min(Fmag),
            F_max=jnp.max(Fmag),
            F_nan=jnp.any(jnp.isnan(Fmag)),
            F_inf=jnp.any(jnp.isinf(Fmag)),
            F_bad=jnp.any(~jnp.isfinite(Fmag)),

            r_mean=jnp.mean(ratio),
            r_min=jnp.min(ratio),
            r_max=jnp.max(ratio),
            r_gt1=jnp.any(ratio > 1.0 + 1e-6),
            ratio=ratio[0:20,40:60,50],
            ordered=True,
        )
    # def force(self, i, sol, params, dt):
    #     if self.one_injection:
    #         inject_now = jnp.equal(i, 0)
    #     else:
    #         inject_now = True

    #     amp = self.stromgren_rate * dt

    #     ix = self.mesh_shape[0] // 2
    #     iy = self.mesh_shape[1] // 2
    #     iz = self.mesh_shape[2] // 2

    #     e_inj = jnp.where(inject_now, amp, 0.0)
    #     fx_inj = jnp.where(inject_now, amp * self.light_speed**(-2), 0.0)

    #     if not self.momentum_only:
    #         sol = sol.at[0, ix, iy, iz].add(e_inj)

    #     if self.injection_momentum:
    #         sol = sol.at[1, ix, iy, iz].add(fx_inj)

    #     jax.debug.print("i = {}", i)
    #     jax.debug.print("E injected at center = {}", sol[0, ix, iy, iz])
    #     jax.debug.print("Fx injected at center = {}", sol[1, ix, iy, iz])

    #     params_out = dict(params)
    #     if "star_ages" in params and params["star_ages"] is not None:
    #         params_out["star_ages"] = jnp.asarray(params["star_ages"]) + dt

    #     return sol, params_out
    # def force(self, i, sol, params, dt):
    #     if "star_masses" not in params or params["star_masses"] is None:
    #         return sol, params

    #     star_masses        = jnp.asarray(params["star_masses"])
    #     star_ages_old      = jnp.asarray(params["star_ages"])
    #     star_ages_new      = star_ages_old + dt
    #     star_metallicities = jnp.asarray(params["star_metallicities"])

    #     if self.injection_mode == "stromgren":
    #         per_star_source = self.get_N_gamma_stromgen_sphere() * dt
    #     elif self.injection_mode == "physical":
    #         per_star_source = self.get_N_gamma(
    #             star_masses, star_ages_old, star_ages_new, star_metallicities, sol
    #         ) * dt
    #     else:
    #         raise ValueError(f"Unknown injection_mode: {self.injection_mode}")

    #     if self.one_injection:
    #         inject_now      = jnp.equal(i, 0)
    #         per_star_source = jnp.where(inject_now, per_star_source, 0.0)

    #     # ── Injection de photons (désactivée si momentum_only=True) ──────────
    #     if self.momentum_only == False:
    #         if "star_positions" not in params or params["star_positions"] is None:
    #             sol = sol.at[0,
    #                          self.mesh_shape[0] // 2,
    #                          self.mesh_shape[1] // 2,
    #                          self.mesh_shape[2] // 2].add(jnp.sum(per_star_source))
    #         else:
    #             star_positions = jnp.asarray(params["star_positions"], dtype=jnp.int32)
    #             if jnp.ndim(per_star_source) == 0:
    #                 per_star_source = jnp.full((star_positions.shape[0],), per_star_source)

    #             ix = star_positions[:, 0]
    #             iy = star_positions[:, 1]
    #             iz = star_positions[:, 2]

    #             if not self.gaussian_star:
    #                 sol = sol.at[0, ix, iy, iz].add(per_star_source)
    #             else:
    #                 sigma = max(1,round(self.mesh_shape[0] // 100))  #ajouter des rounds pour des cas comme 199 ou 150 
    #                 # offsets = jnp.arange(
    #                 #     -round(5 * self.mesh_shape[0] // 100),
    #                 #      round(5 * self.mesh_shape[0] // 100) + 1
    #                 # )
    #                 offsets = jnp.arange(-3 * sigma, 3 * sigma + 1)


    #                 if self.injection_geometry == "2D":
    #                     di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
    #                     weights2 = jnp.exp(-(di2**2 + dj2**2) / (2 * sigma**2))
    #                     weights2 = weights2 / weights2.sum()
    #                     for s in range(star_positions.shape[0]):
    #                         sol = sol.at[0, ix[s], iy[s] + di2, iz[s] + dj2].add(per_star_source[s] * weights2)
    #                         # sol = sol.at[1, ix[s], iy[s] + di2, iz[s] + dj2].add(per_star_source[s] * weights2 *self.light_speed**(-2))
    #                         # sol = sol.at[2:3, ix[s], iy[s] + di2, iz[s] + dj2].add(0)
    #                 elif self.injection_geometry == "3D":
    #                     di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
    #                     weights3 = jnp.exp(-(di3**2 + dj3**2 + dk3**2) / (2 * sigma**2))
    #                     weights3 = weights3 / weights3.sum()
    #                     for s in range(star_positions.shape[0]):
    #                         sol = sol.at[0, ix[s] + di3, iy[s] + dj3, iz[s] + dk3].add(
    #                             per_star_source[s] * weights3)
                            
    #                 # ── Injection de momentum ─────────────────────────────────────────────
    #                 # ── Injection de momentum orientée +x ────────────────────────────────
    #                 # if self.injection_momentum:
    #                 #     for s in range(star_positions.shape[0]):
    #                 #         if self.injection_geometry == "2D":
    #                 #             di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")

    #                 #             weights2 = jnp.exp(-(di2**2 + dj2**2) / (2 * sigma**2))
    #                 #             mask_x = (di2 >= 0).astype(weights2.dtype)
    #                 #             weights_dir = weights2 * mask_x
    #                 #             weights_dir = weights_dir / (weights_dir.sum() + 1e-30)

    #                 #             sol = sol.at[1, ix[s], iy[s] + di2, iz[s] + dj2].add(
    #                 #             per_star_source[s] * weights_dir * self.light_speed**(-2))

    #                 # elif self.injection_geometry == "3D":
    #                 #     di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")

    #                 #     weights3 = jnp.exp(-(di3**2 + dj3**2 + dk3**2) / (2 * sigma**2))
    #                 #     mask_x = (di3 >= 0).astype(weights3.dtype)
    #                 #     weights_dir = weights3 * mask_x
    #                 #     weights_dir = weights_dir / (weights_dir.sum() + 1e-30)

    #                 #     sol = sol.at[1, ix[s] + di3, iy[s] + dj3, iz[s] + dk3].add(
    #                 #     per_star_source[s] * weights_dir * self.light_speed**(-2))


    #                 if self.injection_momentum:
    #                     for s in range(star_positions.shape[0]):
    #                         if self.injection_geometry == "2D":
    #                             di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
    #                             weights2 = jnp.exp(-(di2**2 + dj2**2) / (2 * sigma**2))
    #                             weights2 = weights2 / (weights2.sum() + 1e-30)
    #                             sol = sol.at[1, ix[s], iy[s] + di2, iz[s] + dj2].add(per_star_source[s] * weights2 * self.light_speed**(-1))
    #                             sol = sol.at[2, ix[s], iy[s] + di2, iz[s] + dj2].add(0.0)
    #                             sol = sol.at[3, ix[s], iy[s] + di2, iz[s] + dj2].add(0.0)
    #                         elif self.injection_geometry == "3D":
    #                             di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
    #                             weights3 = jnp.exp(-(di3**2 + dj3**2 + dk3**2) / (2 * sigma**2))
    #                             mask_x = (di3 >= 0).astype(weights3.dtype)
    #                             weights_dir = weights3 * mask_x
    #                             weights_dir = weights_dir / (weights_dir.sum() + 1e-30)
    #                             sol = sol.at[1, ix[s] + di3, iy[s] + dj3, iz[s] + dk3].add(per_star_source[s] * weights_dir * self.light_speed**(-2))

    #                     #old version non functionnal
    #                     # xi           = jnp.arange(43, 57)
    #                     # total_source = jnp.sum(per_star_source)
    #                     # sol          = sol.at[1:3, xi, 0, xi].add(
    #                     #     total_source / len(xi)  
                    
    #                     #     )#self.light_speed**2 *
    #     # ── Debug ─────────────────────────────────────────────────────────────
    #     if self.debug:
    #         z_idx        = self.mesh_shape[2] // 2
    #         z_slice      = sol[0, :, :, z_idx]
    #         nonzero_count = jnp.count_nonzero(z_slice)
    #         coords_xy    = jnp.argwhere(z_slice != 0, size=z_slice.size, fill_value=-1)
    #         x_idx        = coords_xy[:, 0]
    #         y_idx        = coords_xy[:, 1]
    #         valid        = x_idx >= 0
    #         vals         = jnp.where(valid, z_slice[x_idx, y_idx], 0.0)

    #         log_threshold   = -17
    #         log_abs_vals    = jnp.where(valid, jnp.log(vals + 1e-300), jnp.inf)
    #         below_log_mask  = valid & (log_abs_vals < log_threshold)
    #         n_below_log     = jnp.count_nonzero(below_log_mask)
    #         x_below_min     = jnp.min(jnp.where(below_log_mask, x_idx, 1000))
    #         x_below_max     = jnp.max(jnp.where(below_log_mask, x_idx, -1))
    #         y_below_min     = jnp.min(jnp.where(below_log_mask, y_idx, 1000))
    #         y_below_max     = jnp.max(jnp.where(below_log_mask, y_idx, -1))
    #         x_below_size    = jnp.maximum(x_below_max - x_below_min + 1, 0)
    #         y_below_size    = jnp.maximum(y_below_max - y_below_min + 1, 0)

    #         x_min  = jnp.min(jnp.where(valid, x_idx, 1000))
    #         x_max  = jnp.max(jnp.where(valid, x_idx, -1))
    #         y_min  = jnp.min(jnp.where(valid, y_idx, 1000))
    #         y_max  = jnp.max(jnp.where(valid, y_idx, -1))
    #         x_size = jnp.maximum(x_max - x_min + 1, 0)
    #         y_size = jnp.maximum(y_max - y_min + 1, 0)

    #         slice_coord = self.mesh_shape[0] // 2
    #         line_x = sol[0, :, slice_coord, z_idx]
    #         line_y = sol[0, slice_coord, :, z_idx]
    #         line_z = sol[0, slice_coord, slice_coord, :]

    #         jax.debug.print("Line x at y=50, z=50: {line}", line=line_x)
    #         jax.debug.print("Line y at x=50, z=50: {line}", line=line_y)
    #         jax.debug.print("Line z at x=50, y=50: {line}", line=line_z)

    #         log_threshold = -11
    #         log_x = jnp.argwhere(jnp.log10(line_x + 1e-300) < log_threshold, size=line_x.size, fill_value=-1)[:, 0]
    #         log_y = jnp.argwhere(jnp.log10(line_y + 1e-300) < log_threshold, size=line_y.size, fill_value=-1)[:, 0]
    #         log_z = jnp.argwhere(jnp.log10(line_z + 1e-300) < log_threshold, size=line_z.size, fill_value=-1)[:, 0]

    #         non_zero_x = jnp.argwhere(line_x != 0, size=line_x.size, fill_value=-1)[:, 0]
    #         non_zero_y = jnp.argwhere(line_y != 0, size=line_y.size, fill_value=-1)[:, 0]
    #         non_zero_z = jnp.argwhere(line_z != 0, size=line_z.size, fill_value=-1)[:, 0]

    #         jax.debug.print("Non-zero x for y=50, z=50: {x}", x=non_zero_x)
    #         jax.debug.print("Non-zero y for x=50, z=50: {y}", y=non_zero_y)
    #         jax.debug.print("Non-zero z for x=50, y=50: {z}", z=non_zero_z)
    #         jax.debug.print("Log x for y=50, z=50: {x}", x=log_x)
    #         jax.debug.print("Log y for x=50, z=50: {y}", y=log_y)
    #         jax.debug.print("Log z for x=50, y=50: {z}", z=log_z)
    #         jax.debug.print("\n=== Timestep {} === Non-zero on [0,:,:,{}] ===", i, z_idx)
    #         jax.debug.print("Non-zero count: {}", nonzero_count)
    #         jax.debug.print("X range: [{}, {}] (size: {})", x_min.astype(jnp.int32), x_max.astype(jnp.int32), x_size.astype(jnp.int32))
    #         jax.debug.print("Y range: [{}, {}] (size: {})", y_min.astype(jnp.int32), y_max.astype(jnp.int32), y_size.astype(jnp.int32))
    #         jax.debug.print("Count log(|value|) < threshold: {}", n_below_log)
    #         jax.debug.print("X where log < threshold: [{}, {}] (size: {})", x_below_min.astype(jnp.int32), x_below_max.astype(jnp.int32), x_below_size.astype(jnp.int32))
    #         jax.debug.print("Y where log < threshold: [{}, {}] (size: {})", y_below_min.astype(jnp.int32), y_below_max.astype(jnp.int32), y_below_size.astype(jnp.int32))
    #         xmid = self.mesh_shape[0] // 2
    #         ymid = self.mesh_shape[1] // 2
    #         zmid = self.mesh_shape[2] // 2
    #         Fx_line = sol[1, :, ymid, zmid]
    #         jax.debug.print("Fx line: {}", Fx_line)
    #         jax.debug.print("sum Fx(x>x0) = {}", jnp.sum(Fx_line[xmid+1:]))
    #         jax.debug.print("sum Fx(x<x0) = {}", jnp.sum(Fx_line[:xmid]))
    #         x0 = 0
    #         Fx_line = sol[1, :, ymid, zmid]
    #         jax.debug.print("Fx line at y=50,z=50: {}", Fx_line)
    #         jax.debug.print("Fx at injection point x=0: {}", Fx_line[x0])
    #         jax.debug.print("sum Fx for x>0: {}", jnp.sum(Fx_line[x0+1:]))

    #         jax.debug.print("E_gamma min = {}", jnp.min(sol[0]))
    #         jax.debug.print("E_gamma max = {}", jnp.max(sol[0]))
    #         jax.debug.print("|F| max = {}", jnp.max(jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2)))
    #         jax.debug.print("f_max = {}", jnp.max(
    #             jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2) /
    #             jnp.where(sol[0] > 0, sol[0] * self.light_speed, 1e-30)))
    #         Fy = sol[1] 
    #         Fx = sol[2]  # ou l'index correspondant à Fx dans ta convention
    #         Fz = sol[3]  # idem
    #         jax.debug.print("sum Fy =", jnp.sum(Fy))
    #         jax.debug.print("sum |Fx| =", jnp.sum(jnp.abs(Fx)))
    #         jax.debug.print("sum |Fz| =", jnp.sum(jnp.abs(Fz)))
    #         Fy_tot = jnp.sum(sol[1])
    #         Fx_tot = jnp.sum(sol[2])
    #         Fz_tot = jnp.sum(sol[3])

    #         jax.debug.print("Fy max:", jnp.max(jnp.abs(Fy)))
    #         jax.debug.print("Fx max:", jnp.max(jnp.abs(Fx)))
    #         jax.debug.print("Fz max:", jnp.max(jnp.abs(Fz)))
    #         jax.debug.print("Fy/Fx:", jnp.max(jnp.abs(Fy)) / (jnp.max(jnp.abs(Fx)) + 1e-30))
    #         jax.debug.print("Fy/Fz:", jnp.max(jnp.abs(Fy)) / (jnp.max(jnp.abs(Fz)) + 1e-30))
    #         angle_y = jnp.arctan2(jnp.sqrt(Fx_tot**2 + Fz_tot**2), Fy_tot)
    #         jax.debug.print("angle vs y =", angle_y)
    #     # check sol after injection
    #     _check_finite("sol after injection", sol)

    #     params_out               = dict(params)
    #     params_out["star_ages"]  = star_ages_new
    #     return sol, params_out

    def force(self, i, sol, params, dt):
        if "star_masses" not in params or params["star_masses"] is None:
            return sol, params
        self.debug_grid_stats(sol, self.eq, "sol before injection", 0)
        star_masses        = jnp.asarray(params["star_masses"])
        star_ages_old      = jnp.asarray(params["star_ages"])
        star_ages_new      = star_ages_old + dt
        star_metallicities = jnp.asarray(params["star_metallicities"])

        if self.injection_mode == "stromgren":
            per_star_source = self.get_N_gamma_stromgen_sphere() * dt
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

        sigma = max(1, round(self.mesh_shape[0] // 100))
        offsets = jnp.arange(-3 * sigma, 3 * sigma + 1)

        # Longueur du faisceau en nombre de cellules (côté +x)
        beam_len = int(self.beam_length_cells)

        def _inject_energy_beam_x(sol, x0, y0, z0, source):
            """
            Dépose l'énergie Egamma le long de +x dans un petit faisceau 1D.
            """
            s = jnp.arange(0, beam_len, dtype=jnp.int32)
            xi = x0 + s
            yi = jnp.full_like(xi, y0)
            zi = jnp.full_like(xi, z0)

            valid = (
                (xi >= 0) & (xi < self.mesh_shape[0]) &
                (yi >= 0) & (yi < self.mesh_shape[1]) &
                (zi >= 0) & (zi < self.mesh_shape[2])
            )

            s_float = s.astype(jnp.float32)
            weights = jnp.exp(- (s_float**2) / (2.0 * float(sigma)**2))
            weights = jnp.where(valid, weights, 0.0)
            weights = weights / (jnp.sum(weights) + 1e-30)
            self.debug_grid_stats(sol, self.eq, "dans sol energy injection beam", 0)
            return sol.at[0, xi, yi, zi].add(source * weights)

        def _inject_momentum_beam_x(sol, x0, y0, z0, source):
            """
            Injecte seulement Fx (>0) le long de +x, en suivant le même faisceau 1D.
            """
            s = jnp.arange(0, beam_len, dtype=jnp.int32)
            xi = x0 + s
            yi = jnp.full_like(xi, y0)
            zi = jnp.full_like(xi, z0)

            valid = (
                (xi >= 0) & (xi < self.mesh_shape[0]) &
                (yi >= 0) & (yi < self.mesh_shape[1]) &
                (zi >= 0) & (zi < self.mesh_shape[2])
            )

            s_float = s.astype(jnp.float32) #conversion type
            weights = jnp.exp(- (s_float**2) / (2.0 * float(sigma)**2))
            weights = jnp.where(valid, weights, 0.0)
            weights = weights / (jnp.sum(weights) + 1e-30)

            # même échelle que ton _inject_momentum_x_3d
            fx_inj = source**2 * (self.light_speed ** (2)) * weights
            if self.debug:
                jax.debug.print("Injecting momentum beam at x=[{}, {}], y={}, z={}", xi[0], xi[-1], yi[0], zi[0])
                jax.debug.print("Momentum source: {}, weights sum: {}", source, jnp.sum(weights))
                jax.debug.print("Fx injection profile: {}", fx_inj)
                jax.debug.print("Fy injection profile: {} (should be 0)", jnp.zeros_like(fx_inj))
                jax.debug.print("Fz injection profile: {} (should be 0)", jnp.zeros_like(fx_inj))
                jax.debug.print("test c^2: {}",fx_inj / (source**2 + 1e-30)/weights)
                jax.debug.print("Sol[1] after injection: {}", sol[1, xi, yi, zi])
                jax.debug.print("Sol[2] after injection: {} (should be 0)", sol[2, xi, yi, zi])
                jax.debug.print("Sol[3] after injection: {} (should be 0)", sol[3, xi, yi, zi])
                jax.debug.print("max E = {}", jnp.max(sol[0]))
                jax.debug.print("max |F| = {}", jnp.max(jnp.abs(sol[1])))
                jax.debug.print("min E = {}", jnp.min(sol[0]))
                jax.debug.print("min |F| = {}", jnp.min(jnp.abs(sol[1])))
                jax.debug.print("any nan E = {}", jnp.any(jnp.isnan(sol[0])))
                jax.debug.print("any inf E = {}", jnp.any(jnp.isinf(sol[0])))
                # _check_float_status("F_gamma before injection", fx_inj)

            sol = sol.at[1, xi, yi, zi].add(fx_inj)            # F_x
            sol = sol.at[2, xi, yi, zi].add(jnp.zeros_like(0)) # F_y
            sol = sol.at[3, xi, yi, zi].add(jnp.zeros_like(0)) # F_z
            return weights,sol

        def _clip_to_m1_cone(sol):
            """
            Force partout |F| <= beam_reduced_flux * c * E pour stabilité M1.
            """
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


        def _clip_indices_2d(x0, y0, z0, di2, dj2):
            xi = jnp.full(di2.shape, x0, dtype=jnp.int32)
            yi = y0 + di2
            zi = z0 + dj2

            valid = (
                (xi >= 0) & (xi < self.mesh_shape[0]) &
                (yi >= 0) & (yi < self.mesh_shape[1]) &
                (zi >= 0) & (zi < self.mesh_shape[2])
            )
            return xi, yi, zi, valid

        def _clip_indices_3d(x0, y0, z0, di3, dj3, dk3):
            xi = x0 + di3
            yi = y0 + dj3
            zi = z0 + dk3

            valid = (
                (xi >= 0) & (xi < self.mesh_shape[0]) &
                (yi >= 0) & (yi < self.mesh_shape[1]) &
                (zi >= 0) & (zi < self.mesh_shape[2])
            )
            return xi, yi, zi, valid

        def _normalized_weights_2d(valid):
            di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
            weights2 = jnp.exp(-(di2**2 + dj2**2) / (2 * sigma**2))
            weights2 = jnp.where(valid, weights2, 0.0)
            weights2 = weights2 / (jnp.sum(weights2) + 1e-30)
            return di2, dj2, weights2

        def _normalized_weights_3d(valid):
            di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
            weights3 = jnp.exp(-(di3**2 + dj3**2 + dk3**2) / (2 * sigma**2))
            weights3 = jnp.where(valid, weights3, 0.0)
            weights3 = weights3 / (jnp.sum(weights3) + 1e-30)
            return di3, dj3, dk3, weights3

        def _inject_energy_2d(sol, x0, y0, z0, source):
            self.debug_grid_stats(sol, self.eq, "before clip sol energy injection 2D", 0)
            di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
            xi, yi, zi, valid = _clip_indices_2d(x0, y0, z0, di2, dj2)
            _, _, weights2 = _normalized_weights_2d(valid)
            self.debug_grid_stats(sol, self.eq, "sol energy injection 2D", 0)
            return sol.at[0, xi, yi, zi].add(source * weights2)

        def _inject_energy_3d(sol, x0, y0, z0, source):
            di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
            xi, yi, zi, valid = _clip_indices_3d(x0, y0, z0, di3, dj3, dk3)
            _, _, _, weights3 = _normalized_weights_3d(valid)
            return sol.at[0, xi, yi, zi].add(source * weights3)

        def _inject_momentum_x_2d(sol, x0, y0, z0, source):
            di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
            xi, yi, zi, valid = _clip_indices_2d(x0, y0, z0, di2, dj2)
            _, _, weights2 = _normalized_weights_2d(valid)
            fx_inj = source**2 * (self.light_speed ** (2)) * weights2
            sol = sol.at[1, xi, yi, zi].add(fx_inj)
            sol = sol.at[2, xi, yi, zi].add(jnp.zeros_like(0))
            sol = sol.at[3, xi, yi, zi].add(jnp.zeros_like(0))
            if self.debug:
                E = sol[0, xi, yi, zi]
                Fx = sol[1, xi, yi, zi]
                jax.debug.print("Injecting momentum beam at x=[{}, {}], y={}, z={}", xi[0], xi[-1], yi[0], zi[0])
                jax.debug.print("Momentum source: {}, weights sum: {}", source, jnp.sum(weights2))
                jax.debug.print("Fx injection profile: {}", fx_inj)
                jax.debug.print("Fy injection profile: {} (should be 0)", jnp.zeros_like(fx_inj))
                jax.debug.print("Fz injection profile: {} (should be 0)", jnp.zeros_like(fx_inj))
                jax.debug.print("test c^2: {}",fx_inj / (source**2 + 1e-30)/weights2)
                jax.debug.print("Sol[1] after injection: {}", sol[1, xi, yi, zi])
                jax.debug.print("Sol[2] after injection: {} (should be 0)", sol[2, xi, yi, zi])
                jax.debug.print("Sol[3] after injection: {} (should be 0)", sol[3, xi, yi, zi])
                jax.debug.print("max E = {}", jnp.max(E))
                jax.debug.print("max |F| = {}", jnp.max(jnp.abs(Fx)))
                jax.debug.print("min E = {}", jnp.min(E))
                jax.debug.print("min |F| = {}", jnp.min(jnp.abs(Fx)))
                jax.debug.print("any nan E = {}", jnp.any(jnp.isnan(E)))
                jax.debug.print("any inf E = {}", jnp.any(jnp.isinf(E))) 
                # _check_float_status("F_gamma before injection", fx_inj)
            return weights2,sol

        def _inject_momentum_x_3d(sol, x0, y0, z0, source):
            di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
            xi, yi, zi, valid = _clip_indices_3d(x0, y0, z0, di3, dj3, dk3)

            _, _, _, weights3 = _normalized_weights_3d(valid)
            fx_inj = source**2 * (self.light_speed ** (2)) * weights3

            sol = sol.at[1, xi, yi, zi].add(fx_inj)
            sol = sol.at[2, xi, yi, zi].add(jnp.zeros_like(fx_inj))
            sol = sol.at[3, xi, yi, zi].add(jnp.zeros_like(fx_inj))
            return sol

        # ── Injection de photons (inchangée dans l'esprit) ──────────────────────
        if self.momentum_only == False:
            if not self.gaussian_star:
                for s in range(star_positions.shape[0]):
                    x0 = ix[s]
                    y0 = iy[s]
                    z0 = iz[s]
                    if (
                        (0 <= x0 < self.mesh_shape[0]) and
                        (0 <= y0 < self.mesh_shape[1]) and
                        (0 <= z0 < self.mesh_shape[2])
                    ):
                        sol = sol.at[0, x0, y0, z0].add(per_star_source[s])
            else:
                for s in range(star_positions.shape[0]):
                    x0 = ix[s]
                    y0 = iy[s]
                    z0 = iz[s]

                    if self.injection_geometry == "2D":
                        sol = _inject_energy_2d(sol, x0, y0, z0, per_star_source[s])
                    elif self.injection_geometry == "3D":
                        sol = _inject_energy_3d(sol, x0, y0, z0, per_star_source[s])
                    elif self.injection_geometry == "beam_x":
                        sol = jnp.abs(_inject_energy_beam_x(sol, x0, y0, z0, per_star_source[s]))
                    else:
                        raise ValueError(f"Unknown injection_geometry: {self.injection_geometry}")

        # ── Injection de momentum en x activée si injection_momentum=True ────────
        if self.injection_momentum:
            if not self.gaussian_star:
                for s in range(star_positions.shape[0]):
                    x0 = ix[s]
                    y0 = iy[s]
                    z0 = iz[s]
                    if (
                        (0 <= x0 < self.mesh_shape[0]) and
                        (0 <= y0 < self.mesh_shape[1]) and
                        (0 <= z0 < self.mesh_shape[2])
                    ):
                        fx_inj = per_star_source[s] * (self.light_speed ** 2)
                        sol = sol.at[1, x0, y0, z0].add(fx_inj)
                        sol = sol.at[2, x0, y0, z0].add(0.0)
                        sol = sol.at[3, x0, y0, z0].add(0.0)
            else:
                for s in range(star_positions.shape[0]):
                    x0 = ix[s]
                    y0 = iy[s]
                    z0 = iz[s]

                    if self.injection_geometry == "2D":
                        wheight_2D_debug,sol = _inject_momentum_x_2d(sol, x0, y0, z0, per_star_source[s])
                        if self.debug:
                            bad_E = jnp.any(~jnp.isfinite(sol[0]))
                            bad_F = jnp.any(~jnp.isfinite(sol[1:]))
                            jax.debug.print(
                                "NaN/Inf after injection? Egamma={E_bad}, Fgamma={F_bad}",
                                E_bad=bad_E,
                                F_bad=bad_F,)
                            jax.debug.print("sum weight = 1? {}", wheight_2D_debug.sum())
                    elif self.injection_geometry == "3D":
                        sol = _inject_momentum_x_3d(sol, x0, y0, z0, per_star_source[s])
                    elif self.injection_geometry == "beam_x":
                        wheight_2D_debug,sol = _inject_momentum_beam_x(sol, x0, y0, z0, per_star_source[s])
                        self.debug_grid_stats(sol, self.eq, "after clip sol momentum injection beam", 0)
                    else:
                        raise ValueError(f"Unknown injection_geometry: {self.injection_geometry}")
            
        # Clip M1 pour éviter |F| > f_max c E
        # sol = _clip_to_m1_cone(sol)
            
        # ── Debug ─────────────────────────────────────────────────────────────
        if self.debug:
            jax.debug.print(
                "After clip: f_max = {}",
                jnp.max(
                    jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2)
                    / jnp.where(sol[0] > 0, sol[0] * self.light_speed, 1e-30)
                ),
            )
            z_idx        = self.mesh_shape[2] // 2
            z_slice      = sol[0, :, :, z_idx]
            nonzero_count = jnp.count_nonzero(z_slice)
            coords_xy    = jnp.argwhere(z_slice != 0, size=z_slice.size, fill_value=-1)
            x_idx        = coords_xy[:, 0]
            y_idx        = coords_xy[:, 1]
            valid        = x_idx >= 0
            vals         = jnp.where(valid, z_slice[x_idx, y_idx], 0.0)

            log_threshold   = -17
            log_abs_vals    = jnp.where(valid, jnp.log(vals + 1e-300), jnp.inf)
            below_log_mask  = valid & (log_abs_vals < log_threshold)
            n_below_log     = jnp.count_nonzero(below_log_mask)
            x_below_min     = jnp.min(jnp.where(below_log_mask, x_idx, 1000))
            x_below_max     = jnp.max(jnp.where(below_log_mask, x_idx, -1))
            y_below_min     = jnp.min(jnp.where(below_log_mask, y_idx, 1000))
            y_below_max     = jnp.max(jnp.where(below_log_mask, y_idx, -1))
            x_below_size    = jnp.maximum(x_below_max - x_below_min + 1, 0)
            y_below_size    = jnp.maximum(y_below_max - y_below_min + 1, 0)

            x_min  = jnp.min(jnp.where(valid, x_idx, 1000))
            x_max  = jnp.max(jnp.where(valid, x_idx, -1))
            y_min  = jnp.min(jnp.where(valid, y_idx, 1000))
            y_max  = jnp.max(jnp.where(valid, y_idx, -1))
            x_size = jnp.maximum(x_max - x_min + 1, 0)
            y_size = jnp.maximum(y_max - y_min + 1, 0)

            slice_coord = self.mesh_shape[0] // 2
            line_x = sol[0, :, slice_coord, z_idx]
            line_y = sol[0, slice_coord, :, z_idx]
            line_z = sol[0, slice_coord, slice_coord, :]

            jax.debug.print("Line x at y=50, z=50: {line}", line=line_x)
            jax.debug.print("Line y at x=50, z=50: {line}", line=line_y)
            jax.debug.print("Line z at x=50, y=50: {line}", line=line_z)

            log_threshold = -11
            log_x = jnp.argwhere(jnp.log10(line_x + 1e-300) < log_threshold, size=line_x.size, fill_value=-1)[:, 0]
            log_y = jnp.argwhere(jnp.log10(line_y + 1e-300) < log_threshold, size=line_y.size, fill_value=-1)[:, 0]
            log_z = jnp.argwhere(jnp.log10(line_z + 1e-300) < log_threshold, size=line_z.size, fill_value=-1)[:, 0]

            non_zero_x = jnp.argwhere(line_x != 0, size=line_x.size, fill_value=-1)[:, 0]
            non_zero_y = jnp.argwhere(line_y != 0, size=line_y.size, fill_value=-1)[:, 0]
            non_zero_z = jnp.argwhere(line_z != 0, size=line_z.size, fill_value=-1)[:, 0]

            jax.debug.print("Non-zero x for y=50, z=50: {x}", x=non_zero_x)
            jax.debug.print("Non-zero y for x=50, z=50: {y}", y=non_zero_y)
            jax.debug.print("Non-zero z for x=50, y=50: {z}", z=non_zero_z)
            jax.debug.print("Log x for y=50, z=50: {x}", x=log_x)
            jax.debug.print("Log y for x=50, z=50: {y}", y=log_y)
            jax.debug.print("Log z for x=50, y=50: {z}", z=log_z)
            jax.debug.print("\n=== Timestep {} === Non-zero on [0,:,:,{}] ===", i, z_idx)
            jax.debug.print("Non-zero count: {}", nonzero_count)
            jax.debug.print(
                "X range: [{}, {}] (size: {})",
                x_min.astype(jnp.int32),
                x_max.astype(jnp.int32),
                x_size.astype(jnp.int32),
            )
            jax.debug.print(
                "Y range: [{}, {}] (size: {})",
                y_min.astype(jnp.int32),
                y_max.astype(jnp.int32),
                y_size.astype(jnp.int32),
            )
            jax.debug.print("Count log(|value|) < threshold: {}", n_below_log)
            jax.debug.print(
                "X where log < threshold: [{}, {}] (size: {})",
                x_below_min.astype(jnp.int32),
                x_below_max.astype(jnp.int32),
                x_below_size.astype(jnp.int32),
            )
            jax.debug.print(
                "Y where log < threshold: [{}, {}] (size: {})",
                y_below_min.astype(jnp.int32),
                y_below_max.astype(jnp.int32),
                y_below_size.astype(jnp.int32),
            )

            xmid = self.mesh_shape[0] // 2
            ymid = self.mesh_shape[1] // 2
            zmid = self.mesh_shape[2] // 2
            jax.debug.print("Midpoint coordinates: x={}, y={}, z={}, x0={}", xmid, ymid, zmid, x0)
            # print("Midpoint coordinates: x={}, y={}, z={}, x0={}".format(xmid, ymid, zmid, x0))
            Fx_line = sol[1, :, ymid, zmid]
            jax.debug.print("Fx line: {}", Fx_line)
            jax.debug.print("Fx line at y=50,z=50: {}", Fx_line)
            jax.debug.print("sum Fx(x>x0) = {}", jnp.sum(Fx_line[xmid+1:]))
            jax.debug.print("sum Fx(x<x0) = {}", jnp.sum(Fx_line[:xmid]))
            x0 = 0
            # print("Midpoint coordinates: x={}, y={}, z={}, x0={}".format(xmid, ymid, zmid, x0))
            
            jax.debug.print("Fx at injection point x=0: {}", Fx_line[x0])
            jax.debug.print("sum Fx for x>0: {}", jnp.sum(Fx_line[x0+1:]))

            jax.debug.print("E_gamma min = {}", jnp.min(sol[0]))
            jax.debug.print("E_gamma max = {}", jnp.max(sol[0]))
            jax.debug.print(
                "|F| max = {}",
                jnp.max(jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2)),
            )
            jax.debug.print(
                "f_max = {}",
                jnp.max(
                    jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2)
                    / jnp.where(sol[0] > 0, sol[0] * self.light_speed, 1e-30)
                ),
            )

            Fy = sol[2]
            Fx = sol[1]
            Fz = sol[3]

            jax.debug.print("sum Fy = {}", jnp.sum(Fy))
            jax.debug.print("sum |Fx| = {}", jnp.sum(jnp.abs(Fx)))
            jax.debug.print("sum |Fz| = {}", jnp.sum(jnp.abs(Fz)))

            Fy_tot = jnp.sum(sol[2])
            Fx_tot = jnp.sum(sol[1])
            Fz_tot = jnp.sum(sol[3])

            jax.debug.print("Fy max: {}", jnp.max(jnp.abs(Fy)))
            jax.debug.print("Fx max: {}", jnp.max(jnp.abs(Fx)))
            jax.debug.print("Fz max: {}", jnp.max(jnp.abs(Fz)))
            jax.debug.print(
                "Fy/Fx: {}",
                jnp.max(jnp.abs(Fy)) / (jnp.max(jnp.abs(Fx)) + 1e-30),
            )
            jax.debug.print(
                "Fy/Fz: {}",
                jnp.max(jnp.abs(Fy)) / (jnp.max(jnp.abs(Fz)) + 1e-30),
            )
            angle_y = jnp.arctan2(jnp.sqrt(Fx_tot**2 + Fz_tot**2), Fy_tot)
            jax.debug.print("angle vs y = {}", angle_y)

            jax.debug.print("sum E_gamma = {}", jnp.sum(sol[0]))
            jax.debug.print("sum |F| = {}", jnp.sum(jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2)))

            Fmag = jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2)
            E_safe = jnp.where(sol[0] > 0, sol[0], 1e-30)
            ratio_cell = Fmag / E_safe
            ratio_cell_c = ratio_cell / self.light_speed

            jax.debug.print("max |F|/E (cell) = {}", jnp.max(ratio_cell))
            jax.debug.print("max |F|/(c E) (cell) = {}", jnp.max(ratio_cell_c))
            jax.debug.print("any cell |F|/E > c? {}", jnp.any(ratio_cell > self.light_speed))
            jax.debug.print(
                "any cell |F|/E > beam_reduced_flux*c? {}",
                jnp.any(ratio_cell > self.beam_reduced_flux * self.light_speed),
            )
            

            jax.debug.print("sum E_gamma = {}", jnp.sum(sol[0]))
            jax.debug.print("sum |F| = {}", jnp.sum(jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2)))

            # ── Vérification c² sur le faisceau uniquement ───────────────────────
            # Masque des cellules actives (E_gamma > seuil)
            mask_active = sol[0] > 1e-30

            E_active = jnp.where(mask_active, sol[0], 0.0)
            Fmag = jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2)
            F_active = jnp.where(mask_active, Fmag, 0.0)

            E_sum_active = jnp.sum(E_active)
            F_sum_active = jnp.sum(F_active)

            # Ratio |F| / E sur le faisceau
            ratio_active = F_sum_active / (E_sum_active + 1e-30)
            c_2_active = ratio_active / self.light_speed**2

            # Ratio global (sur tout le domaine) pour comparaison
            E_sum_tot = jnp.sum(sol[0])
            F_sum_tot = jnp.sum(Fmag)
            ratio_tot = F_sum_tot / (E_sum_tot + 1e-30)
            c_2_tot = ratio_tot / self.light_speed**2

            jax.debug.print("\n=== Vérification c² sur le faisceau ===")
            jax.debug.print("E_sum(faisceau) = {}", E_sum_active)
            jax.debug.print("|F|_sum(faisceau) = {}", F_sum_active)
            jax.debug.print("|F|/E(faisceau) = {}", ratio_active)
            jax.debug.print("|F|/E / c²(faisceau) = {}", c_2_active)
            jax.debug.print("|F|/E = c² ? faisceau = {}", jnp.isclose(ratio_active, self.light_speed**2))

            jax.debug.print("\n=== Vérification c² sur tout le domaine ===")
            jax.debug.print("E_sum(tout) = {}", E_sum_tot)
            jax.debug.print("|F|_sum(tout) = {}", F_sum_tot)
            jax.debug.print("|F|/E(tout) = {}", ratio_tot)
            jax.debug.print("|F|/E / c²(tout) = {}", c_2_tot)
            jax.debug.print("|F|/E = c² ? tout = {}", jnp.isclose(ratio_tot, self.light_speed**2))

            # Affiche aussi le nombre de cellules actives
            n_active = jnp.sum(mask_active.astype(jnp.int32))
            jax.debug.print("\nNombre de cellules actives (E > 1e-30) = {}", n_active)

            _check_all_float_variables(
                                        sol=sol,
                                        E_gamma=sol[0],
                                        Fx=sol[1],
                                        Fy=sol[2],
                                        Fz=sol[3],
                                    )
        
        self.debug_grid_stats(sol, self.eq, "fin sol energy injection 2D", 0)
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
        return self.stromgren_rate
    
    