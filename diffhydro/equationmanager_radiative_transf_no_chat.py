from __future__ import annotations

from jax import Array
from dataclasses import dataclass, field
import jax.numpy as jnp
import jax


@dataclass
class EquationManager:
    """
    Generic compressible Euler equation manager with optional passive scalars.

        Active variables (always first, fixed order for compatibility):
            primitives active:  [E_gamma, F_gamma_x, F_gamma_y, F_gamma_z]
            conservatives active: [E_gamma, E_gamma*F_gamma_x, E_gamma*F_gamma_y, E_gamma*F_gamma_z]

    Passive variables (any extra slots beyond active):
      primitives passive:  s_k
      conservatives passive: E_gamma * s_k

    The code below avoids hard-coded indices in the numerics by:
      - defining active indices once in __post_init__
      - using active_slice/passive_slice everywhere else.
    """
    gamma: float = 1.6
    n_cons: int = 4 #avant c etait 5
    eps: float = 1e-20
    isothermal: bool = False
    light_speed: float = 1#3e8 #en m.s-1

    # Active variable names/order (single source of truth)
    active_names: tuple[str, ...] = ("E_gamma", "F_gamma_x", "F_gamma_y", "F_gamma_z")#,"p" 
    # passive_names: tuple[str, ...] = ("E_gamma",) #new

    # Derived index maps (filled in __post_init__)
    active_map: dict[str, int] = field(init=False, repr=False)
    mass_ids: int = field(init=False)
    vel_ids: tuple[int, int, int] = field(init=False)
    n_active: int = field(init=False)
    # energy_ids: int = field(init=False)
    def __post_init__(self):
        self.n_active = len(self.active_names)
        if self.n_active > self.n_cons:
            raise ValueError(
                f"Expected max {self.n_cons} total vars. Got {self.n_active} active."
            )

        self.active_map = {name: i for i, name in enumerate(self.active_names)}
        self.mass_ids = self.active_map["E_gamma"]
        self.vel_ids = (
            self.active_map["F_gamma_x"],
            self.active_map["F_gamma_y"],
            self.active_map["F_gamma_z"],
        )
        # self.energy_ids = self.active_map["p"]
        self.velocity_minor_axes = ((2, 3), (3, 1), (1, 2))
        self.equation_type = "SINGLE-PHASE"#equation_information.equation_type
        self.thermal_conductivity_model = "SUTHERLAND"
        self.sutherland_parameters = [0.1, 1.0, 1.0]
        self.cfl = 0.4
        self.mesh_shape = [100,100,100]
        self.R = 1.0
        self.cp = self.gamma / (self.gamma - 1.0) * self.R
        self.light_speed = float(self.light_speed)
        if self.light_speed <= 0.0:
            raise ValueError("light_speed must be > 0.")
        
    def get_conservatives_from_primitives(self, primitives):
        """
        primitives: (>= n_active, ...)
        returns conservatives: (>= n_active, ...)
        """
        prim_a = primitives[self.active_slice]

        E_gamma = prim_a[self.mass_ids]
        u, v, w = (prim_a[i] for i in self.vel_ids)
        # p = prim_a[self.energy_ids]
        if self.isothermal:
            p = self.get_isothermal_pressure(E_gamma)
            
        # e = self.get_specific_energy(E_gamma)
        # kin = 0.5 * (u*u + v*v + w*w)
        # Etot = E_gamma * (kin + e) #a modifier
        cons_a = jnp.stack([E_gamma, E_gamma*u, E_gamma*v, E_gamma*w], axis=0) #Etot

        if primitives.shape[0] > self.n_active:
            prim_p = primitives[self.passive_slice]                 # (n_passive,...)
            cons_p = E_gamma[jnp.newaxis, ...] * prim_p             # E_gamma*s_k
            return jnp.concatenate([cons_a, cons_p], axis=0)

        return cons_a

    def get_primitives_from_conservatives(self, conservatives):
        """
        conservatives: (>= n_active, ...)
        returns primitives: (>= n_active, ...)
        """
        cons_a = conservatives[self.active_slice]

        E_gamma = cons_a[self.mass_ids]
        E_gamma_safe = jnp.maximum(E_gamma, self.eps)
        inv_E_gamma = 1.0 / E_gamma_safe

        u = cons_a[1] * inv_E_gamma
        v = cons_a[2] * inv_E_gamma
        w = cons_a[3] * inv_E_gamma

        # E = cons_a[4] * inv_E_gamma
        # kin = 0.5 * (u*u + v*v + w*w)
        # e = jnp.maximum(E - kin, self.eps) #a modifier

        # if self.isothermal:
        #     p = self.get_isothermal_pressure(E_gamma_safe)
        # else:
        #     p = self.get_pressure(e, E_gamma_safe)
            
        prim_a = jnp.stack([E_gamma_safe, u, v, w], axis=0) #new ,p

        if conservatives.shape[0] > self.n_active:
            cons_p = conservatives[self.passive_slice]              # (n_passive,...)
            prim_p = cons_p * inv_E_gamma[jnp.newaxis, ...]             # s_k
            return jnp.concatenate([prim_a, prim_p], axis=0)

        return prim_a

   # ---------------------------
    # Convenience slices
    # ---------------------------
    @property
    def active_slice(self):
        return slice(0, self.n_active) #modifier ici pour choisir quoi utiliser probablement 4 
    
    @property
    def passive_slice(self):
        return slice(self.n_active, self.n_cons)

    @property
    def n_passive(self) -> int:
        return max(0, self.n_cons - self.n_active)

    def get_pressure(self, e, E_gamma):
        if self.isothermal:
            return self.get_isothermal_pressure(E_gamma)
        return (self.gamma - 1.0) * jnp.maximum(e, self.eps) * jnp.maximum(E_gamma, self.eps)
    
    def get_signal_speed(self, primitives, axis): #probablement a changer pour c of speed
        # p = primitives[self.energy_ids]
        E_gamma = primitives[self.mass_ids]
        return self.get_speed_of_sound(E_gamma)
    
    def get_speed_of_sound(self, E_gamma: Array) -> Array: # self, p: Array,probablement a changer pour retourne speed of light
        """See base class. """
        E_gamma_safe = jnp.maximum(E_gamma, self.eps)
        return jnp.full_like(E_gamma_safe, self.light_speed)

    # ---------------------------
    # def get_specific_energy(self, E_gamma: Array):
    #     E_gamma_safe = jnp.maximum(E_gamma, self.eps)
    #     # p_safe = jnp.maximum(p, self.eps)
    #     energy_thermique = 0.0
    #     if self.isothermal:
    #         # p_safe = self.get_isothermal_pressure(E_gamma_safe)
    #     return energy_thermique + #p_safe / ((self.gamma - 1.0) * E_gamma_safe + self.eps)

    def get_fluxes_xi(self, primitives, conservatives, axis: int):# peut etre ici qu on calcule le "p" pour la partie de transfert radiatif
        """
        Physical flux in direction axis=0,1,2.
        Returns flux array same leading dim as conservatives/primitives.
        """

        prim_a = primitives[self.active_slice]
        cons_a = conservatives[self.active_slice]

        E_gamma = prim_a[self.mass_ids]
        u, v, w = (prim_a[i] for i in self.vel_ids)
        # p = prim_a[self.energy_ids]
        # Etot = cons_a[-1]  # last active conservative slot

        vel = (u, v, w)
        ui = vel[axis]
        E_gamma_ui = E_gamma * ui

        # momentum flux components
        fx_E_gammau = E_gamma_ui * u
        fx_E_gammav = E_gamma_ui * v
        fx_E_gammaw = E_gamma_ui * w

        if axis == 0:
            fx_E_gammau = fx_E_gammau #+ p a modifier pour le p de transfert radiatif
        elif axis == 1:
            fx_E_gammav = fx_E_gammav # + p a modifier pour le p de transfert radiatif
        else:
            fx_E_gammaw = fx_E_gammaw  #+ p a modifier pour le p de transfert radiatif

        # No energy equation in this 4-variable RT manager: keep active flux size = 4.
        flux_a = jnp.stack([E_gamma_ui, fx_E_gammau, fx_E_gammav, fx_E_gammaw], axis=0)

        # if conservatives.shape[0] > self.n_active:
        #     cons_p = conservatives[self.passive_slice]              # (n_passive,...)
        #     flux_p = cons_p * ui                                    # (E_gamma*s_k)*ui
        #     return jnp.concatenate([flux_a, flux_p], axis=0)

        return flux_a



#     def get_isothermal_pressure(self, E_gamma):
#         E_gamma_safe = jnp.maximum(E_gamma, self.eps)
#         cs2 = self.light_speed * self.light_speed
#         return cs2 * E_gamma_safe

#     # ---------------------------
#     # Thermodynamics




#     def get_sound_speed(self, p, E_gamma):
#         E_gamma_safe = jnp.maximum(E_gamma, self.eps)
#         if self.isothermal:
#             return jnp.full_like(E_gamma_safe, self.light_speed)
#         p_safe = jnp.maximum(p, self.eps)
#         return jnp.sqrt(self.gamma * p_safe / E_gamma_safe)

#     # ---------------------------
#     # Primitive <-> Conservative
#     # ---------------------------
    
#     # ---------------------------
#     # Physical fluxes
#     # ---------------------------

#     # ---------------------------
#     # Wavespeeds (for CFL / solvers)
#     # ---------------------------
#     def get_wavespeeds_xi(self, primitives, axis: int):
#         prim_a = primitives[self.active_slice]
#         E_gamma = prim_a[self.mass_ids]
#         p = prim_a[self.energy_ids]
#         ui = prim_a[self.vel_ids[axis]]
#         a = self.get_sound_speed(p, E_gamma)
#         return ui - a, ui + a

#     def get_specific_heat_capacity(self, T: Array): #-> Union[float, Array]:
#         """Calculates the specific heat coefficient per unit mass.
#         [c_p] = J / kg / K

#         :param T: _description_
#         :type T: Array
#         :raises NotImplementedError: _description_
#         :return: _description_
#         :rtype: Array
#         """
#         return self.cp

#     def get_specific_heat_ratio(self, T: Array): #-> Union[float, Array]:
#         """Calculates the specific heat ratio.

#         :param T: _description_
#         :type T: Array
#         :raises NotImplementedError: _description_
#         :return: _description_
#         :rtype: Array
#         """
#         return self.gamma

#     def get_E_gamma(self, p: Array, E_gamma: Array) -> Array:
#         """See base class. """
#         return p / E_gamma

#     def get_grueneisen(self, E_gamma: Array, T: Array = None) -> Array:
#         """See base class. """
#         return self.gamma - 1


#     def get_pressure(self, e, E_gamma):
#         if self.isothermal:
#             return self.get_isothermal_pressure(E_gamma)
#         return (self.gamma - 1.0) * jnp.maximum(e, self.eps) * jnp.maximum(E_gamma, self.eps)

#     def get_temperature(self, p, E_gamma):
#         p_safe   = jnp.maximum(p,   self.eps)
#         E_gamma_safe = jnp.maximum(E_gamma, self.eps)
#         T = p_safe / (E_gamma_safe * self.R + self.eps)
#         # cap used temp to keep any downstream log/exp/table sane
#         return jnp.clip(T, 0.10, 10.0**10.5) 

#     def get_specific_energy(self, p, E_gamma):
#         if self.isothermal:
#             p_safe = self.get_isothermal_pressure(E_gamma)
#         else:
#             p_safe = jnp.maximum(p, self.eps)
#         E_gamma_safe = jnp.maximum(E_gamma, self.eps)
#         return p_safe / (E_gamma_safe * (self.gamma - 1.0) + self.eps)

#     def project_conservatives_to_eos(self, conservatives):
#         if not self.isothermal:
#             return conservatives
#         primitives = self.get_primitives_from_conservatives(conservatives)
#         return self.get_conservatives_from_primitives(primitives)

#     def get_total_energy(
#             self,
#             p:Array,
#             E_gamma:Array,
#             u:Array,
#             v:Array,
#             w:Array
#             ) -> Array:
#         """See base class. """
#         # Total energy per unit volume
#         # (sensible, i.e., without heat of formation)
#         return p / (self.gamma - 1) + 0.5 * E_gamma * ( (u * u + v * v + w * w) )

#     def get_total_enthalpy(
#             self,
#             p:Array,
#             E_gamma:Array,
#             u:Array,
#             v:Array,
#             w:Array
#             ) -> Array:
#         """See base class. """
#         # Total specific enthalpy
#         # (sensible, i.e., without heat of formation)
#         return (self.get_total_energy(p, E_gamma, u, v, w) + p) / E_gamma

#     def get_stagnation_temperature(
#             self,
#             p:Array,
#             E_gamma:Array,
#             u:Array,
#             v:Array,
#             w:Array
#         ) -> Array:
#         T = self.get_temperature(p, E_gamma)
#         cp = self.get_specific_heat_capacity(T)
#         return T + 0.5 * (u * u + v * v + w * w) / cp


    
#     def _set_transport_properties(self,func) -> None:

#         if self.thermal_conductivity_model is not None:
#             if self.thermal_conductivity_model == "CUSTOM":
#                 self.thermal_conductivity_fun = func

#             elif self.thermal_conductivity_model == "SUTHERLAND":
#                 sutherland_parameters = self.sutherland_parameters
#                 self.kappa_ref = sutherland_parameters[0]
#                 self.T_ref_kappa = sutherland_parameters[1]
#                 self.C_kappa = sutherland_parameters[2]
#             else:
#                 raise
#         else:
#             raise
    
#     def get_thermal_conductivity_old(
#             self,
#             temperature: Array,
#             primitives: Array,
#             density: Array = None,
#             partial_densities: Array = None,
#             volume_fractions: Array = None,
#         ) -> Array:
#         """Computes the thermal conductivity

#         :param T: _description_
#         :type T: Array
#         :raises NotImplementedError: _description_
#         :return: _description_
#         :rtype: Array
#         """
        
#         T = temperature
#  #       checkify.check(jax.numpy.all(T>0), "temperature must be non-negative, got {i}", i=T.min())

            
#         if self.thermal_conductivity_model == "CUSTOM":
#             thermal_conductivity = self.thermal_conductivity_fun(T)
    
#         elif self.thermal_conductivity_model == "SUTHERLAND":
#             t_1 = ((self.T_ref_kappa + self.C_kappa)/(T + self.C_kappa))
#             t_2 = (T/self.T_ref_kappa)**1.5
#             thermal_conductivity = \
#                 self.kappa_ref * t_1 * t_2
#         elif self.thermal_conductivity_model == "ELBADRY":
#             ALPHA= 813.142554365 
#             # 1e-20/(1.4mH)^2 erg*s^{-1}*cm^3 in simulation unit
#             # (cooling, Sutherland & Dopita 1993)
#             BETA =406571277.182611837
#             #1e-20/(1.4mH)^2 erg*s^{-1}*cm^3 in simulation unit
#             #heating, Kim & Ostriker 2015)
#             GAMMA= 406.571277183
#             # 1.4mH/cm^3 in simulation unit
#             #(hydrogen number density, Kim & Ostriker 2015)
#             DELTA=0.03459841649374997
#             # 1.2/(1.4mH)*1e2 cm^3 in simulation unit
#             #(electron number density, El-Badry 2019)
#             EPSILON=3468.366826027353
#             # erg*s^{-1}*cm^{-1} in simulation unit
#             #thermal conductivity, El-Badry 2019)
#             ZETA=5.111496271545331E-12
#             #hydrogen mass over Boltzmann constant in simulation unit
#             MHKB=115.98518596699539

#             E_gamma = primitives[self.mass_ids]
#             p   = primitives[self.energy_ids]
            
#             # Physical temperature like Athena (Kelvin)
#             # numeric factor mirrors their 1.272727*MHKB*p/E_gamma
#             T_phys = 1.272727 * MHKB * p / (E_gamma + self.eps)
            
#             # Hot vs. cool branches (Spitzer / Parker)
#             temp7 = T_phys / 1.0e7
#             ne2   = jnp.maximum(EPSILON * E_gamma, self.eps)  # guard
            
#             kappa_hot  = (1.7e11 * temp7**2.5) / (1.0 + 0.029 * jnp.log(temp7 / jnp.sqrt(ne2)))  # NATURAL log
#             kappa_cool = 2.5e5 * jnp.sqrt(jnp.maximum(T_phys / 1.0e4, 0.0))
            
#             kappa = jnp.where(T_phys > 6.6e4, kappa_hot, kappa_cool)
            
#             # Athena "adjust for units": multiply by 1.4*MHKB/E_gamma
#             kappa = kappa * 1.4 * MHKB / (E_gamma + self.eps)
            
#             # Apply El-Badry ceiling here (saturation comes later)
#             k_ceiling = 1.8e12/ DELTA * 1.4 * MHKB #1.8e12 / DELTA * 1.4 * MHKB
#             thermal_conductivity = jnp.minimum(kappa, k_ceiling)
#         else:
#             raise NotImplementedError

#         return thermal_conductivity
    
#     def get_thermal_conductivity(self, temperature, primitives, 
#                                            density=None, partial_densities=None, 
#                                            volume_fractions=None):
#         """
#         Corrected thermal conductivity matching Athena++ implementation.
#         This returns κ in "code units" that already includes unit adjustments.
#         """
#         T = temperature

#         if self.thermal_conductivity_model == "ELBADRY":
#             # Constants (same as Athena++)
#             MHKB = 115.98518596699539
#             EPSILON = 3468.366826027353

#             E_gamma = primitives[self.mass_ids]
#             p   = primitives[self.energy_ids]

#             # Temperature in Kelvin (matching Athena++ line 416)
#             T_phys = 1.272727 * MHKB * p / (E_gamma + self.eps)

#             # Hot vs. cool branches (matching Athena++ lines 418-427)
#             temp7 = T_phys / 1.0e7
#             temp4 = T_phys / 1.0e4
#             ne2   = EPSILON * E_gamma

#             # Spitzer conductivity for hot gas (T > 6.6e4 K)
#             kappa_hot = (1.7e11 * temp7**2.5) / (1.0 + 0.029 * jnp.log(temp7 / jnp.sqrt(ne2)))

#             # Parker conductivity for cool gas (T < 6.6e4 K)
#             kappa_cool = 2.5e5 * jnp.sqrt(temp4)

#             # Branch selection
#             kappa = jnp.where(T_phys > 6.6e4, kappa_hot, kappa_cool)

#             # Adjust for Athena++ units (line 430)
#             # This converts from physical units to code units
#             kappa = kappa * 1.4 * MHKB / (E_gamma + self.eps)

#             return kappa

#         elif self.thermal_conductivity_model == "SUTHERLAND":
#             t_1 = ((self.T_ref_kappa + self.C_kappa)/(T + self.C_kappa))
#             t_2 = (T/self.T_ref_kappa)**1.5
#             thermal_conductivity = self.kappa_ref * t_1 * t_2
#             return thermal_conductivity

#         else:
#             raise NotImplementedError