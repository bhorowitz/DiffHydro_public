"""
Radiative transfer module (stub).

Placeholder functions for future radiative transfer physics.
All methods currently return stub values.
"""

import jax
import jax.numpy as jnp
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

    def get_stellar_emission(self, star_age, star_metallicity):
        if self.stellar_spectrum_func is not None:
            return self.stellar_spectrum_func(star_age, star_metallicity)
        age_factor = jnp.exp(-star_age / 10.0)
        Z_factor = jnp.maximum(star_metallicity, 1e-4)
        return age_factor * Z_factor

    def timestep(self, sol):
        print("timestepblast")
        return 1e30
    
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
    def force(self, i, sol, params, dt):
        if "star_masses" not in params or params["star_masses"] is None:
            return sol, params

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

        # ── Injection de photons (désactivée si momentum_only=True) ──────────
        if self.momentum_only == False:
            if "star_positions" not in params or params["star_positions"] is None:
                sol = sol.at[0,
                             self.mesh_shape[0] // 2,
                             self.mesh_shape[1] // 2,
                             self.mesh_shape[2] // 2].add(jnp.sum(per_star_source))
            else:
                star_positions = jnp.asarray(params["star_positions"], dtype=jnp.int32)
                if jnp.ndim(per_star_source) == 0:
                    per_star_source = jnp.full((star_positions.shape[0],), per_star_source)

                ix = star_positions[:, 0]
                iy = star_positions[:, 1]
                iz = star_positions[:, 2]

                if not self.gaussian_star:
                    sol = sol.at[0, ix, iy, iz].add(per_star_source)
                else:
                    sigma = max(1,round(self.mesh_shape[0] // 100))  #ajouter des rounds pour des cas comme 199 ou 150 
                    # offsets = jnp.arange(
                    #     -round(5 * self.mesh_shape[0] // 100),
                    #      round(5 * self.mesh_shape[0] // 100) + 1
                    # )
                    offsets = jnp.arange(-3 * sigma, 3 * sigma + 1)
                    if self.injection_geometry == "2D":
                        di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
                        weights2 = jnp.exp(-(di2**2 + dj2**2) / (2 * sigma**2))
                        weights2 = weights2 / weights2.sum()
                        for s in range(star_positions.shape[0]):
                            sol = sol.at[0, ix[s], iy[s] + di2, iz[s] + dj2].add(per_star_source[s] * weights2)
                            # sol = sol.at[1, ix[s], iy[s] + di2, iz[s] + dj2].add(per_star_source[s] * weights2 *self.light_speed**(-2))
                            # sol = sol.at[2:3, ix[s], iy[s] + di2, iz[s] + dj2].add(0)
                    elif self.injection_geometry == "3D":
                        di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
                        weights3 = jnp.exp(-(di3**2 + dj3**2 + dk3**2) / (2 * sigma**2))
                        weights3 = weights3 / weights3.sum()

                        for s in range(star_positions.shape[0]):
                            sol = sol.at[0, ix[s] + di3, iy[s] + dj3, iz[s] + dk3].add(
                                per_star_source[s] * weights3)
                            
                    # ── Injection de momentum ─────────────────────────────────────────────
                    # ── Injection de momentum orientée +x ────────────────────────────────
                    # if self.injection_momentum:
                    #     for s in range(star_positions.shape[0]):
                    #         if self.injection_geometry == "2D":
                    #             di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")

                    #             weights2 = jnp.exp(-(di2**2 + dj2**2) / (2 * sigma**2))
                    #             mask_x = (di2 >= 0).astype(weights2.dtype)
                    #             weights_dir = weights2 * mask_x
                    #             weights_dir = weights_dir / (weights_dir.sum() + 1e-30)

                    #             sol = sol.at[1, ix[s], iy[s] + di2, iz[s] + dj2].add(
                    #             per_star_source[s] * weights_dir * self.light_speed**(-2))

                    # elif self.injection_geometry == "3D":
                    #     di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")

                    #     weights3 = jnp.exp(-(di3**2 + dj3**2 + dk3**2) / (2 * sigma**2))
                    #     mask_x = (di3 >= 0).astype(weights3.dtype)
                    #     weights_dir = weights3 * mask_x
                    #     weights_dir = weights_dir / (weights_dir.sum() + 1e-30)

                    #     sol = sol.at[1, ix[s] + di3, iy[s] + dj3, iz[s] + dk3].add(
                    #     per_star_source[s] * weights_dir * self.light_speed**(-2))
                    if self.injection_momentum:
                        for s in range(star_positions.shape[0]):
                            if self.injection_geometry == "2D":
                                di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
                                weights2 = jnp.exp(-(di2**2 + dj2**2) / (2 * sigma**2))
                                mask_x = (di2 >= 0).astype(weights2.dtype)
                                weights_dir = weights2 * mask_x
                                weights_dir = weights_dir / (weights_dir.sum() + 1e-30)
                                sol = sol.at[1, ix[s], iy[s] + di2, iz[s] + dj2].add(per_star_source[s] * weights_dir * self.light_speed**(-2))

                            elif self.injection_geometry == "3D":
                                di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
                                weights3 = jnp.exp(-(di3**2 + dj3**2 + dk3**2) / (2 * sigma**2))
                                mask_x = (di3 >= 0).astype(weights3.dtype)
                                weights_dir = weights3 * mask_x
                                weights_dir = weights_dir / (weights_dir.sum() + 1e-30)
                                sol = sol.at[1, ix[s] + di3, iy[s] + dj3, iz[s] + dk3].add(per_star_source[s] * weights_dir * self.light_speed**(-2))

                        #old version non functionnal
                        # xi           = jnp.arange(43, 57)
                        # total_source = jnp.sum(per_star_source)
                        # sol          = sol.at[1:3, xi, 0, xi].add(
                        #     total_source / len(xi)  
                    
                        #     )#self.light_speed**2 *
        # ── Debug ─────────────────────────────────────────────────────────────
        if self.debug:
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
            jax.debug.print("X range: [{}, {}] (size: {})", x_min.astype(jnp.int32), x_max.astype(jnp.int32), x_size.astype(jnp.int32))
            jax.debug.print("Y range: [{}, {}] (size: {})", y_min.astype(jnp.int32), y_max.astype(jnp.int32), y_size.astype(jnp.int32))
            jax.debug.print("Count log(|value|) < threshold: {}", n_below_log)
            jax.debug.print("X where log < threshold: [{}, {}] (size: {})", x_below_min.astype(jnp.int32), x_below_max.astype(jnp.int32), x_below_size.astype(jnp.int32))
            jax.debug.print("Y where log < threshold: [{}, {}] (size: {})", y_below_min.astype(jnp.int32), y_below_max.astype(jnp.int32), y_below_size.astype(jnp.int32))
            xmid = self.mesh_shape[0] // 2
            ymid = self.mesh_shape[1] // 2
            zmid = self.mesh_shape[2] // 2
            Fx_line = sol[1, :, ymid, zmid]
            jax.debug.print("Fx line: {}", Fx_line)
            jax.debug.print("sum Fx(x>x0) = {}", jnp.sum(Fx_line[xmid+1:]))
            jax.debug.print("sum Fx(x<x0) = {}", jnp.sum(Fx_line[:xmid]))
            x0 = 0
            Fx_line = sol[1, :, ymid, zmid]
            jax.debug.print("Fx line at y=50,z=50: {}", Fx_line)
            jax.debug.print("Fx at injection point x=0: {}", Fx_line[x0])
            jax.debug.print("sum Fx for x>0: {}", jnp.sum(Fx_line[x0+1:]))

        params_out               = dict(params)
        params_out["star_ages"]  = star_ages_new
        return sol, params_out

    def get_N_gamma(self, star_masses, star_ages_old, star_ages_new, star_metallicities, sol):
        emission_old  = self.get_stellar_emission(star_ages_old, star_metallicities)
        emission_new  = self.get_stellar_emission(star_ages_new, star_metallicities)
        delta_emission = emission_new - emission_old
        cell_volume   = self.dx ** (sol.ndim - 1)
        return (star_masses * delta_emission) * self.escape_fraction / cell_volume

    def get_N_gamma_stromgen_sphere(self):
        return self.stromgren_rate