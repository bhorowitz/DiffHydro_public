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
    
    Implements: N_i^{n+1} = N_i^n + (f_esc/V) * sum_stars m_* [Pi_i(tau^{n+1}, Z) - Pi_i(tau^n, Z)]
    """
    
    def __init__(
        self,
        escape_fraction=0.1,
        stellar_spectrum_func=None,
        dx=1.0,
        injection_mode="physical",
        stromgren_rate=1e-7,
        gaussian_star = True,
        injection_geometry = "3D",
        injection_momentum = False,
        eq=None,
        debug = False,
    ):# changer petit a petit stronmgren 
        """
        Parameters
        ----------
        escape_fraction : float
            Fraction f_esc of photons that escape the source region
        stellar_spectrum_func : callable
            Function Pi_i(age, metallicity) -> photon emission rate
            If None, use a simple blackbody-based default
        dx : float
            Grid spacing (cell volume = dx^3 in 3D or dx^2 in 2D)
        eq : EquationManager, optional
            Equation manager; if provided, light_speed is taken from eq.light_speed
        """
        self.escape_fraction = escape_fraction
        self.stellar_spectrum_func = stellar_spectrum_func
        self.dx = dx
        self.injection_mode = injection_mode
        self.stromgren_rate = stromgren_rate
        self.gaussian_star = gaussian_star
        self.injection_geometry = injection_geometry
        self.injection_momentum = injection_momentum
        self.debug = debug
        self.light_speed = eq.light_speed if eq is not None else 1.0
    def get_stellar_emission(self, star_age, star_metallicity):
        """
        Compute a simple ionizing photon emission rate as a function of age and metallicity.
        
        This is a placeholder; replace with actual Starburst99, FSPS, or similar.
        """
        if self.stellar_spectrum_func is not None:
            return self.stellar_spectrum_func(star_age, star_metallicity)
        
        # Simple default: emission rate decays with age and scales with metallicity.
        # In practice, use look-up tables or pre-computed SED models
        age_factor = jnp.exp(-star_age / 10.0)  # Decay with time
        Z_factor = jnp.maximum(star_metallicity, 1e-4)  # Metallicity effect
        return age_factor * Z_factor
    
    def timestep(self, sol):
        """
        Conservative estimate: radiative sources don't impose CFL constraints
        unless coupled to gas energy. For now, use large dt.
        """
        print("timestepblast")
        return 1e30  # No strict timestep constraint in the time step of hydro_core


    def force(self, i, sol, params, dt):
        """
        Apply stellar radiation source.
        
        Assumes params contains:
        - 'star_masses': array of stellar particle masses
        - 'star_ages': array of stellar particle ages (at time t^n)
        - 'star_ages_new': array of stellar particle ages (at time t^{n+1})
        - 'star_metallicities': array of stellar particle metallicities
        - 'star_positions': array of stellar particle cell indices (optional, for spatial mapping)
        
        Parameters
        ----------
        i : int
            Timestep index (unused here)
        sol : Array
            State vector [E_gamma, F_gamma_x, F_gamma_y, F_gamma_z, ...]
        params : dict
            Physical parameters and stellar particle data
        dt : float
            Current timestep
            
        Returns
        -------
        sol : Array
            Updated state
        params : dict
            Unchanged
        """
        # Early exit if no stellar data
        if "star_masses" not in params or params["star_masses"] is None:
            return sol, params

        star_masses = jnp.asarray(params["star_masses"])
        star_ages_old = jnp.asarray(params["star_ages"])
        star_ages_new = star_ages_old + dt
        star_metallicities = jnp.asarray(params["star_metallicities"])

        if self.injection_mode == "stromgren":
            # Uniform per-star source rate (placeholder model), then aggregate by cell.
            per_star_source = self.get_N_gamma_stromgen_sphere() * dt
        else:
            # Source is computed per-star, then summed only in cells that contain stars.
            per_star_source = self.get_N_gamma(
                star_masses,
                star_ages_old,
                star_ages_new,
                star_metallicities,
                sol,
            ) * dt

        # Inject photons only at the first timestep (i == 0).
        # inject_now = jnp.equal(i, 0)
        # per_star_source = jnp.where(inject_now, per_star_source, 0.0)

        # Après le calcul de per_star_source...
        
        if "star_positions" not in params or params["star_positions"] is None:
            sol = sol.at[0, 50, 50, 50].add(jnp.sum(per_star_source))
            params_out = dict(params)
            params_out["star_ages"] = star_ages_new
            return sol, params_out
        
        star_positions = jnp.asarray(params["star_positions"], dtype=jnp.int32)
        if jnp.ndim(per_star_source) == 0:
            per_star_source = jnp.full((star_positions.shape[0],), per_star_source)
        
        ix = star_positions[:, 0]
        iy = star_positions[:, 1]
        iz = star_positions[:, 2]
        
        if not self.gaussian_star:
            sol = sol.at[0, ix, iy, iz].add(per_star_source)
        else:
            # attention sigma dois etre largement plus petit que la taille de la grille pour que ca fasse une gaussienne
            sigma = 1 #modifier cette ligne et celle d'apres pour faire une gaussienne plus ou moins large
            offsets = jnp.arange(-5, 6)
            di, dj, dk = jnp.meshgrid(offsets, offsets, offsets, indexing='ij')
            weights = jnp.exp(-(di**2 + dj**2 + dk**2) / (2 * sigma**2))
            weights = weights / weights.sum()

            if self.injection_geometry == "2D":
                def inject_star_2D_YZ(sol, args):
                    yi, zi, src = args
                    return sol.at[0, yi + di, zi + dj].add(src * weights)
                
                for s in range(star_positions.shape[0]):
                    sol = inject_star_2D_YZ(sol, (iy[s], iz[s], per_star_source[s]))
            
            elif self.injection_geometry == "3D":
                def inject_star_3D(sol, args):
                    xi, yi, zi, src = args
                    # jax.debug.print("src={}", src)
                    return sol.at[0, xi + di, yi + dj, zi + dk].add(src * weights)
                
                for s in range(star_positions.shape[0]):
                    sol = inject_star_3D(sol, (ix[s], iy[s], iz[s], per_star_source[s]))

        #Injection on the Moment
        if self.injection_momentum == True:
            def injection_moment_1D_X(sol): # attention ici c'est pas modulaire
                xi = jnp.arange(25, 75)
                total_source = jnp.sum(per_star_source)
                return sol.at[1, xi, :, :].add(self.light_speed **2 * total_source / len(xi)) # attention ici c'est pas modulaire du tout, a revoir pour faire une injection plus physique et plus modulaire
            sol = injection_moment_1D_X(sol)

        params_out = dict(params)
    
        params_out["star_ages"] = star_ages_new

        if self.debug == True:
            # JAX-safe debug prints (compatible with jit/pjit tracing)
            z_idx = 50
            z_slice = sol[0, :, :, z_idx]
            nonzero_count = jnp.count_nonzero(z_slice)
            # Static-size argwhere to remain JIT-safe. Invalid rows are padded with -1.
            coords_xy = jnp.argwhere(z_slice != 0, size=z_slice.size, fill_value=-1)
            x_idx = coords_xy[:, 0]
            y_idx = coords_xy[:, 1]
            valid = x_idx >= 0
            vals = jnp.where(valid, z_slice[x_idx, y_idx], 0.0)
        
            # Coordinates where log(|value|) is below a chosen threshold.
            log_threshold = -17
            log_abs_vals = jnp.where(valid, jnp.log(vals + 1e-300), jnp.inf)
            below_log_mask = valid & (log_abs_vals < log_threshold)
            n_below_log = jnp.count_nonzero(below_log_mask)
            x_below_min = jnp.min(jnp.where(below_log_mask, x_idx, 1000))
            x_below_max = jnp.max(jnp.where(below_log_mask, x_idx, -1))
            y_below_min = jnp.min(jnp.where(below_log_mask, y_idx, 1000))
            y_below_max = jnp.max(jnp.where(below_log_mask, y_idx, -1))
            x_below_size = jnp.maximum(x_below_max - x_below_min + 1, 0)
            y_below_size = jnp.maximum(y_below_max - y_below_min + 1, 0)
            x_below_min_i = x_below_min.astype(jnp.int32)
            x_below_max_i = x_below_max.astype(jnp.int32)
            y_below_min_i = y_below_min.astype(jnp.int32)
            y_below_max_i = y_below_max.astype(jnp.int32)
            x_below_size_i = x_below_size.astype(jnp.int32)
            y_below_size_i = y_below_size.astype(jnp.int32)
        
            # Compute bounding box sizes
            x_min = jnp.min(jnp.where(valid, x_idx, 1000))
            x_max = jnp.max(jnp.where(valid, x_idx, -1))
            y_min = jnp.min(jnp.where(valid, y_idx, 1000))
            y_max = jnp.max(jnp.where(valid, y_idx, -1))
            x_size = jnp.maximum(x_max - x_min + 1, 0)
            y_size = jnp.maximum(y_max - y_min + 1, 0)
            x_min_i = x_min.astype(jnp.int32)
            x_max_i = x_max.astype(jnp.int32)
            y_min_i = y_min.astype(jnp.int32)
            y_max_i = y_max.astype(jnp.int32)
            x_size_i = x_size.astype(jnp.int32)
            y_size_i = y_size.astype(jnp.int32)

            # Extract 1D arrays for x and y with z fixed at 50
            z_idx = 50

            # Non-zero values on 1D lines (JIT-safe static-size indices).
            line_x = sol[0, :, 50, z_idx]
            line_y = sol[0, 50, :, z_idx]
            jax.debug.print("Line x at y=50, z=50: {line}", line=line_x)
            jax.debug.print("Line y at x=50, z=50: {line}", line=line_y)
            non_zero_x = jnp.argwhere(line_x != 0, size=line_x.size, fill_value=-1)[:, 0]
            non_zero_y = jnp.argwhere(line_y != 0, size=line_y.size, fill_value=-1)[:, 0]

            # log10(|value|) below threshold on the same 1D lines (JIT-safe).
            log_threshold = -11
            log_line_x = jnp.log10(line_x + 1e-300)
            log_line_y = jnp.log10(line_y + 1e-300)
            log_x = jnp.argwhere(log_line_x < log_threshold, size=log_line_x.size, fill_value=-1)[:, 0]
            log_y = jnp.argwhere(log_line_y < log_threshold, size=log_line_y.size, fill_value=-1)[:, 0]

            # Debug outputs
            jax.debug.print("Non-zero x for y=50, z=50: {x}", x=non_zero_x)
            jax.debug.print("Non-zero y for x=50, z=50: {y}", y=non_zero_y)
            jax.debug.print("Log x for y=50, z=50: {x}", x=log_x)
            jax.debug.print("Log y for x=50, z=50: {y}", y=log_y)

            jax.debug.print("\n=== Timestep {} === Non-zero on [0,:,:,{}] ===", i, z_idx)
            jax.debug.print("Non-zero count: {}", nonzero_count)
            jax.debug.print("X range: [{}, {}] (size: {})", x_min_i, x_max_i, x_size_i)
            jax.debug.print("Y range: [{}, {}] (size: {})", y_min_i, y_max_i, y_size_i)
            jax.debug.print("Z plane: [{}] (size: 1)", z_idx)
            jax.debug.print("Coordinates x: {}", x_idx)
            jax.debug.print("Coordinates y: {}", y_idx)
            jax.debug.print("Values      : {}", vals)
            jax.debug.print("Log threshold: {}", log_threshold)
            jax.debug.print("Count log(|value|) < threshold: {}", n_below_log)
            jax.debug.print("X where log(|value|) < threshold: [{}, {}] (size: {})", x_below_min_i, x_below_max_i, x_below_size_i)
            jax.debug.print("Y where log(|value|) < threshold: [{}, {}] (size: {})", y_below_min_i, y_below_max_i, y_below_size_i)
            jax.debug.print("Format target: [0][x][y][{}]", z_idx)
        
        return sol, params_out
    
    def get_N_gamma(self, star_masses, star_ages_old, star_ages_new, star_metallicities, sol):
        """Compute per-star photon source from Delta Pi (dt handled in force)."""
        emission_old = self.get_stellar_emission(star_ages_old, star_metallicities)
        emission_new = self.get_stellar_emission(star_ages_new, star_metallicities)
        delta_emission = emission_new - emission_old
        cell_volume = self.dx ** (sol.ndim - 1)
        return (star_masses * delta_emission) * self.escape_fraction / cell_volume
    
    def get_N_gamma_stromgen_sphere(self):
        """Simple Stromgren-like photon rate placeholder."""
        return self.stromgren_rate