from __future__ import annotations

from jax import Array
from dataclasses import dataclass, field
import jax.numpy as jnp
import jax


@dataclass
class EquationManager:
    """
    Multi-density Euler equation manager for radiative transfer.

    Multiple density fields share a single velocity and pressure field.
    Each density evolves via: drhoi/dt + div(rhoi * u) = 0
    Momentum and energy use rho_total = sum(rhoi).

    Storage layout (n_dens individual densities):
      Active (first 5, identical to standard Euler):
        Primitives:    [rho_total, vx, vy, vz, p]
        Conservatives: [rho_total, rho_total*vx, rho_total*vy, rho_total*vz, E_tot]

      Passive (slots 5 .. 5+n_dens):
        Primitives:    [Y_1, Y_2, ..., Y_N]   mass fractions Y_i = rho_i / rho_total
        Conservatives: [rho_1, rho_2, ..., rho_N]   individual densities

    n_cons = 5 + n_dens

    The active block is fed directly to the Riemann solver.
    Individual densities are advected as passive scalars.
    """
    gamma: float = 1.6
    eps: float = 1e-20
    isothermal: bool = False
    isothermal_sound_speed: float = 1.0

    density_names: tuple[str, ...] = ("rho",)

    # Derived (computed in __post_init__)
    n_dens: int = field(init=False) #new variable
    n_cons: int = field(init=False) #variable origine
    density_map: dict[str, int] = field(init=False, repr=False) #peut etre overkill pas sur a utiliser

    def __post_init__(self):
        self.n_dens = len(self.density_names)
        if self.n_dens < 1:
            raise ValueError(f"Must have at least 1 density. Got {self.n_dens}.")

        # Map density name -> passive slot index (0-based within passive block)
        self.density_map = {name: i for i, name in enumerate(self.density_names)}
        self.n_cons = 5 + self.n_dens

        # Active indices (same as standard Euler)
        self.mass_ids = 0
        self.vel_ids = (1, 2, 3)
        self.energy_ids = 4

        self.velocity_minor_axes = ((2, 3), (3, 1), (1, 2))
        self.equation_type = "SINGLE-PHASE"
        self.thermal_conductivity_model = "SUTHERLAND"
        self.sutherland_parameters = [0.1, 1.0, 1.0]
        self.cfl = 0.4
        self.mesh_shape = [100, 100, 100]
        self.R = 1.0
        self.cp = self.gamma / (self.gamma - 1.0) * self.R
        self.isothermal_sound_speed = float(self.isothermal_sound_speed)
        if self.isothermal_sound_speed <= 0.0:
            raise ValueError("isothermal_sound_speed must be > 0.")

    def get_conservatives_from_primitives(self, primitives):#changement d unite passer de rho vitesse et pression au unite conserve tho moment energie total 
        """
        Primitives  [rho_total, vx, vy, vz, p, Y_1, ..., Y_N]
        -> Conservatives [rho_total, rho_total*vx, rho_total*vy, rho_total*vz, E_tot, rho_1, ..., rho_N]
        """
        prim_a = primitives[self.active_slice]

        rho = prim_a[self.mass_ids]
        u, v, w = (prim_a[i] for i in self.vel_ids)
        p = prim_a[self.energy_ids]
        if self.isothermal:
            p = self.get_isothermal_pressure(rho)

        e = self.get_specific_energy(p, rho)
        kin = 0.5 * (u*u + v*v + w*w)
        Etot = rho * (kin + e)

        cons_a = jnp.stack([rho, rho*u, rho*v, rho*w, Etot], axis=0)

        if primitives.shape[0] > self.n_active:
            prim_p = primitives[self.passive_slice]
            cons_p = rho[jnp.newaxis, ...] * prim_p
            return jnp.concatenate([cons_a, cons_p], axis=0)

        return cons_a

    def get_primitives_from_conservatives(self, conservatives):
        """
        Conservatives [rho_total, rho_total*vx, rho_total*vy, rho_total*vz, E_tot, rho_1, ..., rho_N]
        -> Primitives  [rho_total, vx, vy, vz, p, Y_1, ..., Y_N]
        """
        cons_a = conservatives[self.active_slice]

        rho = cons_a[self.mass_ids]
        rho_safe = jnp.maximum(rho, self.eps)
        inv_rho = 1.0 / rho_safe

        u = cons_a[1] * inv_rho
        v = cons_a[2] * inv_rho
        w = cons_a[3] * inv_rho

        E = cons_a[4] * inv_rho
        kin = 0.5 * (u*u + v*v + w*w)
        e = jnp.maximum(E - kin, self.eps)

        if self.isothermal:
            p = self.get_isothermal_pressure(rho_safe)
        else:
            p = self.get_pressure(e, rho_safe)

        prim_a = jnp.stack([rho_safe, u, v, w, p], axis=0)

        if conservatives.shape[0] > self.n_active:
            cons_p = conservatives[self.passive_slice]
            prim_p = cons_p * inv_rho[jnp.newaxis, ...]
            return jnp.concatenate([prim_a, prim_p], axis=0)

        return prim_a
    
    
    @property
    def active_slice(self):
        return slice(0, 5)
    
    def get_pressure(self, e, rho):
        if self.isothermal:
            return self.get_isothermal_pressure(rho)
        return (self.gamma - 1.0) * jnp.maximum(e, self.eps) * jnp.maximum(rho, self.eps)
     # --- Slices for solver interface ---

    @property
    def n_active(self):
        """5 active vars for Riemann solver: rho_total, vx, vy, vz, p."""
        return 5

    @property
    def passive_slice(self):
        return slice(5, self.n_cons)
    
    def get_signal_speed(self, primitives, axis):
        p = primitives[self.energy_ids]
        rho = primitives[self.mass_ids]
        return self.get_speed_of_sound(p, rho)
    
    def get_speed_of_sound(self, p, rho):
        """Alias for get_sound_speed (required by Riemann solvers)."""
        if self.isothermal:
            rho_safe = jnp.maximum(rho, self.eps)
            return jnp.full_like(rho_safe, self.isothermal_sound_speed)
        return jnp.sqrt(self.gamma * jnp.maximum(p, self.eps) / jnp.maximum(rho, self.eps))
    
    def get_specific_energy(self, p, rho):
        if self.isothermal:
            p_safe = self.get_isothermal_pressure(rho)
        else:
            p_safe = jnp.maximum(p, self.eps)
        rho_safe = jnp.maximum(rho, self.eps)
        return p_safe / (rho_safe * (self.gamma - 1.0) + self.eps)
    
    # # --- Physical fluxes ---

    def get_fluxes_xi(self, primitives, conservatives, axis: int):
        """
        Physical flux in direction axis (0, 1, 2).

        Works with both active-only (5 vars) and full state (5 + n_dens).
        """
        prim_a = primitives[self.active_slice]
        cons_a = conservatives[self.active_slice]

        rho = prim_a[self.mass_ids]
        u, v, w = prim_a[1], prim_a[2], prim_a[3]
        p = prim_a[self.energy_ids]
        Etot = cons_a[-1]

        vel = (u, v, w)
        ui = vel[axis]
        rho_ui = rho * ui

        fx_rhou = rho_ui * u
        fx_rhov = rho_ui * v
        fx_rhow = rho_ui * w

        if axis == 0:   fx_rhou = fx_rhou + p
        elif axis == 1: fx_rhov = fx_rhov + p
        else:           fx_rhow = fx_rhow + p

        fx_E = ui * (Etot + p)

        flux_a = jnp.stack([rho_ui, fx_rhou, fx_rhov, fx_rhow, fx_E], axis=0)

        if conservatives.shape[0] > 5:
            # Passive: rho_i * u_axis
            cons_p = conservatives[self.passive_slice]
            flux_p = cons_p * ui
            return jnp.concatenate([flux_a, flux_p], axis=0)

        return flux_a
    
#     # @property
#     # def n_passive(self):
#     #     return self.n_dens

#     # @property
#     # def density_passive_slice(self):
#     #     """Slice for individual densities in cons/prim arrays (passive block)."""
#     #     return slice(5, 5 + self.n_dens)

#     # @property
#     # def momentum_slice(self):
#     #     return slice(1, 4)

#     # def get_rho_total(self, conserved):
#     #     """rho_total is the first conservative variable."""
#     #     return conserved[0]

#     # def get_individual_densities(self, conservatives):
#     #     """Individual density fields from conservative array."""
#     #     return conservatives[self.density_passive_slice]

#     # def get_mass_fractions(self, primitives):
#     #     """Mass fractions Y_i from primitive array."""
#     #     return primitives[self.density_passive_slice]

#     # # --- Thermodynamics ---

#     # def get_isothermal_pressure(self, rho):
#     #     rho_safe = jnp.maximum(rho, self.eps)
#     #     cs2 = self.isothermal_sound_speed ** 2
#     #     return cs2 * rho_safe





#     # def get_sound_speed(self, p, rho):
#     #     rho_safe = jnp.maximum(rho, self.eps)
#     #     if self.isothermal:
#     #         return jnp.full_like(rho_safe, self.isothermal_sound_speed)
#     #     return jnp.sqrt(self.gamma * jnp.maximum(p, self.eps) / rho_safe)



#     # # --- Primitive <-> Conservative ---





#     # # --- Wavespeeds ---

#     # def get_wavespeeds_xi(self, primitives, axis: int):
#     #     rho = primitives[self.mass_ids]
#     #     p = primitives[self.energy_ids]
#     #     ui = primitives[self.vel_ids[axis]]
#     #     a = self.get_sound_speed(p, rho)
#     #     return ui - a, ui + a

#     # # --- Utilities (interface parity with equationmanager.py) ---

#     # def get_specific_heat_capacity(self, T):
#     #     return self.cp

#     # def get_specific_heat_ratio(self, T):
#     #     return self.gamma

#     # def get_psi(self, p, rho):
#     #     return p / rho

#     # def get_grueneisen(self, rho, T=None):
#     #     return self.gamma - 1

#     # def get_temperature(self, p, rho):
#     #     p_safe = jnp.maximum(p, self.eps)
#     #     rho_safe = jnp.maximum(rho, self.eps)
#     #     T = p_safe / (rho_safe * self.R + self.eps)
#     #     return jnp.clip(T, 0.10, 10.0**10.5)

#     # def project_conservatives_to_eos(self, conservatives):
#     #     if not self.isothermal:
#     #         return conservatives
#     #     primitives = self.get_primitives_from_conservatives(conservatives)
#     #     return self.get_conservatives_from_primitives(primitives)

#     # def get_total_energy(self, p, rho, u, v, w):
#     #     return p / (self.gamma - 1) + 0.5 * rho * (u * u + v * v + w * w)

#     # def get_total_enthalpy(self, p, rho, u, v, w):
#     #     return (self.get_total_energy(p, rho, u, v, w) + p) / rho

#     # def get_stagnation_temperature(self, p, rho, u, v, w):
#     #     T = self.get_temperature(p, rho)
#     #     cp = self.get_specific_heat_capacity(T)
#     #     return T + 0.5 * (u * u + v * v + w * w) / cp



#     # def _set_transport_properties(self, func):
#     #     if self.thermal_conductivity_model == "CUSTOM":
#     #         self.thermal_conductivity_fun = func
#     #     elif self.thermal_conductivity_model == "SUTHERLAND":
#     #         self.kappa_ref = self.sutherland_parameters[0]
#     #         self.T_ref_kappa = self.sutherland_parameters[1]
#     #         self.C_kappa = self.sutherland_parameters[2]
#     #     else:
#     #         raise NotImplementedError

#     # def get_thermal_conductivity(self, temperature, primitives,
#     #                              density=None, partial_densities=None,
#     #                              volume_fractions=None):
#     #     T = temperature

#     #     if self.thermal_conductivity_model == "ELBADRY":
#     #         MHKB = 115.98518596699539
#     #         EPSILON = 3468.366826027353
#     #         rho_total = primitives[self.mass_ids]
#     #         p = primitives[self.energy_ids]
#     #         T_phys = 1.272727 * MHKB * p / (rho_total + self.eps)
#     #         temp7 = T_phys / 1.0e7
#     #         temp4 = T_phys / 1.0e4
#     #         ne2 = EPSILON * rho_total
#     #         kappa_hot = (1.7e11 * temp7**2.5) / (1.0 + 0.029 * jnp.log(temp7 / jnp.sqrt(ne2)))
#     #         kappa_cool = 2.5e5 * jnp.sqrt(temp4)
#     #         kappa = jnp.where(T_phys > 6.6e4, kappa_hot, kappa_cool)
#     #         kappa = kappa * 1.4 * MHKB / (rho_total + self.eps)
#     #         return kappa

#     #     elif self.thermal_conductivity_model == "SUTHERLAND":
#     #         t_1 = (self.T_ref_kappa + self.C_kappa) / (T + self.C_kappa)
#     #         t_2 = (T / self.T_ref_kappa) ** 1.5
#     #         return self.kappa_ref * t_1 * t_2

#     #     else:
#     #         raise NotImplementedError
