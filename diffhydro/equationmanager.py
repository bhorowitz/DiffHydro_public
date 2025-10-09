from jax import Array 
from functools import partial
from typing import List
import jax.numpy as jnp
from jax.experimental import checkify

import jax 
#EDITED FOR DIFFHYDRO

#need to figure out where cfl goes, probably into flux directly?

class EquationManager:
    """ The EquationManager stores information on the system of equations that is being solved.
    Besides providing indices for the different variables, the EquationManager provides the 
    equation-specific primitive/conservative conversions and the flux calculation.

    NOTE! For DiffHydro it also rolls in the materials properties as an (safe) Ideal Gas. Might go back to 
    materials class if lots of weird fluids emerge...
    """
    def __init__(
            self,
           # material_manager: MaterialManager,
          #  equation_information: EquationInformation
            ) -> None:

       # self.equation_information = equation_information

        self.mass_ids = 0#equation_information.mass_ids 
        self.vel_ids = (1,2,3)#equation_information.velocity_ids 
        self.energy_ids = -1
        self.velocity_minor_axes = ((2, 3), (3, 1), (1, 2))
        self.equation_type = "SINGLE-PHASE"#equation_information.equation_type
        self.gamma = 1.6
        self.thermal_conductivity_model = "SUTHERLAND"
        self.sutherland_parameters = [0.1, 1.0, 1.0]
        self.eps = 1E-20
        self.cfl = 0.3
        self.mesh_shape = [100,100,100]
        self.R = 1.0
        self.cp = self.gamma / (self.gamma - 1.0) * self.R
        
    def get_conservatives_from_primitives(self, primitives: Array) -> Array:
        """Converts primitive variables to conservative ones.
        Wrapper for 5 equation DIM and single-phase/level-set model.

        :param primitives: _description_
        :type primitives: Array
        :return: _description_
        :rtype: Array
        """
        if self.equation_type == "SINGLE-PHASE":
            rho = primitives[self.mass_ids] # = rho
            e = self.get_specific_energy(primitives[self.energy_ids], rho)
            rhou = rho * primitives[self.vel_ids[0]] # = rho * u
            rhov = rho * primitives[self.vel_ids[1]] # = rho * v
            rhow = rho * primitives[self.vel_ids[2]] # = rho * w
            E = rho * (0.5 * (
                primitives[self.vel_ids[0]] * primitives[self.vel_ids[0]] \
                + primitives[self.vel_ids[1]] * primitives[self.vel_ids[1]] \
                + primitives[self.vel_ids[2]] * primitives[self.vel_ids[2]]) + e)  # E = rho * (1/2 u^2 + e)
            conservatives = jnp.stack([rho, rhou, rhov, rhow, E], axis=0)
        
        else:
            raise NotImplementedError

        return conservatives

    def get_primitives_from_conservatives(self, conservatives: Array) -> Array:
        """Converts conservative variables to primitive variables.

        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :return: Buffer of primitive variables
        :rtype: Array
        """           

        if self.equation_type == "SINGLE-PHASE":
            rho = conservatives[self.mass_ids]  # rho = rho
            one_rho = 1.0 / rho
            u = conservatives[self.vel_ids[0]] * one_rho  # u = rho*u / rho
            v = conservatives[self.vel_ids[1]] * one_rho  # v = rho*v / rho
            w = conservatives[self.vel_ids[2]] * one_rho  # w = rho*w / rho
            e = conservatives[self.energy_ids] * one_rho - 0.5 * (u * u + v * v + w * w)
            p = self.get_pressure(e, rho) # p = (gamma-1) * ( E - 1/2 * (rho*u) * u)

            primitives = jnp.stack([rho, u, v, w, p], axis=0)

        else:
            raise NotImplementedError

        return primitives

    def get_fluxes_xi(
            self,
            primitives: Array,
            conservatives: Array,
            axis: int
            ) -> Array:
        """Computes the physical flux in a specified spatial direction.
        Cf. Eq. (3.65) in Toro.

        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :param axis: Spatial direction along which fluxes are calculated
        :type axis: int
        :return: Physical fluxes in axis direction
        :rtype: Array
        """

        if self.equation_type == "SINGLE-PHASE":
            rho_ui = conservatives[axis+1] # (rho u_i)
            rho_ui_u1 = conservatives[axis+1] * primitives[self.vel_ids[0]] # (rho u_i) * u_1
            rho_ui_u2 = conservatives[axis+1] * primitives[self.vel_ids[1]] # (rho u_i) * u_2
            rho_ui_u3 = conservatives[axis+1] * primitives[self.vel_ids[2]] # (rho u_i) * u_3
            ui_Ep = primitives[axis+1] * (conservatives[self.energy_ids] + primitives[self.energy_ids])
            if axis == 0:
                rho_ui_u1 += primitives[self.energy_ids]
            elif axis == 1:
                rho_ui_u2 += primitives[self.energy_ids]
            elif axis == 2:
                rho_ui_u3 += primitives[self.energy_ids]

            flux_xi = jnp.stack([
                rho_ui,
                rho_ui_u1,
                rho_ui_u2,
                rho_ui_u3,
                ui_Ep],
                axis=0)
        
        else:
            raise NotImplementedError

        return flux_xi

    def get_specific_heat_capacity(self, T: Array): #-> Union[float, Array]:
        """Calculates the specific heat coefficient per unit mass.
        [c_p] = J / kg / K

        :param T: _description_
        :type T: Array
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """
        return self.cp

    def get_specific_heat_ratio(self, T: Array): #-> Union[float, Array]:
        """Calculates the specific heat ratio.

        :param T: _description_
        :type T: Array
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """
        return self.gamma

    def get_psi(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return p / rho

    def get_grueneisen(self, rho: Array, T: Array = None) -> Array:
        """See base class. """
        return self.gamma - 1

    def get_speed_of_sound(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return jnp.sqrt( self.gamma * jnp.maximum( p, self.eps) / jnp.maximum( rho, self.eps ) )

    def get_pressure(self, e: Array, rho: Array) -> Array:
        """See base class. """
        return (self.gamma - 1.0) * e * rho

    def get_temperature(self, p: Array, rho: Array) -> Array:
        """See base class. """
        return p / (rho * self.R + self.eps)

    def get_specific_energy(self, p:Array, rho:Array) -> Array:
        """See base class. """
        # Specific internal energy
        return p / (rho * (self.gamma - 1.0))

    def get_total_energy(
            self,
            p:Array,
            rho:Array,
            u:Array,
            v:Array,
            w:Array
            ) -> Array:
        """See base class. """
        # Total energy per unit volume
        # (sensible, i.e., without heat of formation)
        return p / (self.gamma - 1) + 0.5 * rho * ( (u * u + v * v + w * w) )

    def get_total_enthalpy(
            self,
            p:Array,
            rho:Array,
            u:Array,
            v:Array,
            w:Array
            ) -> Array:
        """See base class. """
        # Total specific enthalpy
        # (sensible, i.e., without heat of formation)
        return (self.get_total_energy(p, rho, u, v, w) + p) / rho

    def get_stagnation_temperature(
            self,
            p:Array,
            rho:Array,
            u:Array,
            v:Array,
            w:Array
        ) -> Array:
        T = self.get_temperature(p, rho)
        cp = self.get_specific_heat_capacity(T)
        return T + 0.5 * (u * u + v * v + w * w) / cp


    
    def _set_transport_properties(self,func) -> None:

        if self.thermal_conductivity_model is not None:
            if self.thermal_conductivity_model == "CUSTOM":
                self.thermal_conductivity_fun = func

            elif self.thermal_conductivity_model == "SUTHERLAND":
                sutherland_parameters = self.sutherland_parameters
                self.kappa_ref = sutherland_parameters[0]
                self.T_ref_kappa = sutherland_parameters[1]
                self.C_kappa = sutherland_parameters[2]
            else:
                raise
        else:
            raise
    
    def get_thermal_conductivity(
            self,
            temperature: Array,
            primitives: Array,
            density: Array = None,
            partial_densities: Array = None,
            volume_fractions: Array = None,
        ) -> Array:
        """Computes the thermal conductivity

        :param T: _description_
        :type T: Array
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """
        
        T = temperature
 #       checkify.check(jax.numpy.all(T>0), "temperature must be non-negative, got {i}", i=T.min())

            
        if self.thermal_conductivity_model == "CUSTOM":
            thermal_conductivity = self.thermal_conductivity_fun(T)
    
        elif self.thermal_conductivity_model == "SUTHERLAND":
            t_1 = ((self.T_ref_kappa + self.C_kappa)/(T + self.C_kappa))
            t_2 = (T/self.T_ref_kappa)**1.5
            thermal_conductivity = \
                self.kappa_ref * t_1 * t_2
        elif self.thermal_conductivity_model == "ELBADRY":
            ALPHA= 813.142554365 
            # 1e-20/(1.4mH)^2 erg*s^{-1}*cm^3 in simulation unit
            # (cooling, Sutherland & Dopita 1993)
            BETA =406571277.182611837
            #1e-20/(1.4mH)^2 erg*s^{-1}*cm^3 in simulation unit
            #heating, Kim & Ostriker 2015)
            GAMMA= 406.571277183
            # 1.4mH/cm^3 in simulation unit
            #(hydrogen number density, Kim & Ostriker 2015)
            DELTA=0.03459841649374997
            # 1.2/(1.4mH)*1e2 cm^3 in simulation unit
            #(electron number density, El-Badry 2019)
            EPSILON=3468.366826027353
            # erg*s^{-1}*cm^{-1} in simulation unit
            #thermal conductivity, El-Badry 2019)
            ZETA=5.111496271545331E-12
            #hydrogen mass over Boltzmann constant in simulation unit
            MHKB=115.98518596699539

            temp = temperature/1E7
            n_e2 = EPSILON * primitives[self.mass_ids]
            kappa_hot = (1.7e11 * temp_7**2.5) / (1 + 0.029 * jnp.log10(temp_7 / jnp.sqrt(n_e2)))
                # thermal conductivity for neutral atomic collisions (Parker 1953)
            temp_4 = temperature/ 1.0E4
            kappa_cool = 2.5E5 * jnp.sqrt(temp_4)
                    #adjast for Athena++ units
            kappa = jnp.where(T>6.6E4, kappa_hot, kappa_cool)
            kappa = kappa * 1.4 * MHKB / primitives[self.mass_ids]
            kc = 1.8E12 / DELTA * 1.4 * MHKB

            kappa_max = 1E16
            
            thermal_conductivity = jnp.minimum(kc*jnp.ones(temp.shape), 1.0 / (1.0 / kappa + 1.0 / kappa_max))
            
        else:
            raise NotImplementedError

        return thermal_conductivity