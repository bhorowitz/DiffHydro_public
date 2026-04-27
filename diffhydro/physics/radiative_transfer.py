"""
Radiative transfer module (stub).

Placeholder functions for future radiative transfer physics.
All methods currently return stub values.
"""

import jax.numpy as jnp


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

#version du chat
class StellarRadiationForce:
    """
    Radiative source term from stellar populations.
    Updates the E_gamma field based on stellar mass, age and metallicity.
    
    Implements: N_i^{n+1} = N_i^n + (f_esc/V) * sum_stars m_* [Pi_i(tau^{n+1}, Z) - Pi_i(tau^n, Z)]
    """
    
    def __init__(self, escape_fraction=0.1, stellar_spectrum_func=None, dx=1.0):
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
        """
        self.escape_fraction = escape_fraction
        self.stellar_spectrum_func = stellar_spectrum_func
        self.dx = dx
        
    def get_stellar_emission(self, star_age, star_metallicity):
        """
        Compute Pi_i(age, Z) = cumulative photon emission from a stellar population.
        
        This is a placeholder; replace with actual Starburst99, FSPS, or similar.
        """
        if self.stellar_spectrum_func is not None:
            return self.stellar_spectrum_func(star_age, star_metallicity)
        
        # Simple default: Assume emission scales with age and metallicity
        # In practice, use look-up tables or pre-computed SED models
        age_factor = jnp.exp(-star_age / 10.0)  # Decay with time
        Z_factor = jnp.maximum(star_metallicity, 1e-4)  # Metallicity effect
        return age_factor * Z_factor
    
    def timestep(self, sol):
        """
        Conservative estimate: radiative sources don't impose CFL constraints
        unless coupled to gas energy. For now, use large dt.
        """
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
        if 'star_masses' not in params or params['star_masses'] is None:
            return sol, params
        
        star_masses = jnp.asarray(params['star_masses'])
        star_ages_old = jnp.asarray(params['star_ages'])
        star_ages_new = star_ages_old + dt
        star_metallicities = jnp.asarray(params['star_metallicities'])
        
        # Compute emission at t^n and t^{n+1}
        emission_old = self.get_stellar_emission(star_ages_old, star_metallicities)
        emission_new = self.get_stellar_emission(star_ages_new, star_metallicities)
        
        # Delta emission (photons emitted during this step)
        delta_emission = emission_new - emission_old  # Array of shape (n_stars,)
        
        # Cell volume (adapt for 2D/3D)
        cell_volume = self.dx ** (sol.ndim - 1)
        
        # Source strength per cell
        # sum_stars m_* * delta_emission * f_esc / V
        total_source = jnp.sum(star_masses * delta_emission) * self.escape_fraction / cell_volume
        
        # Add to E_gamma (first component, index 0)
        # In practice, you might want to distribute this spatially based on star_positions
        sol = sol.at[0].add(total_source )

        params_out = dict(params)
        params_out['star_ages'] = star_ages_new
        return sol, params_out