from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from .signal_speeds import compute_sstar

#adapted from jaxfluids

class RiemannSolver(ABC):
    """Abstract base class for Riemann solvers.

    RiemannSolver has two fundamental attributes: a material manager and a signal speed.
    The solve_riemann_problem_xi method solves the one-dimensional Riemann problem.
    """

    def __init__(
            self,
#            material_manager: MaterialManager, 
            equation_manager,
            signal_speed: Callable,
            **kwargs
            ) -> None:

        self.eps = 1E-20#precision.get_eps()

#        self.material_manager = material_manager
        self.equation_manager = equation_manager
#        self.equation_information = equation_manager.equation_information
        self.signal_speed = signal_speed

        # MINOR AXIS DIRECTIONS 
        self.velocity_minor = self. equation_manager.velocity_minor_axes

        self.equation_type = self.equation_manager.equation_type
        self.is_surface_tension = False#self.equation_information.active_physics.is_surface_tension

        self.mass_ids = self.equation_manager.mass_ids
        self.velocity_ids = self.equation_manager.vel_ids #vel_id vs. velocity_id?
        self.energy_ids = self.equation_manager.energy_ids 
    
    def solve_riemann_problem_xi(
            self, 
            primitives_L: Array, 
            primitives_R: Array,
            conservatives_L: Array, 
            conservatives_R: Array, 
            axis: int,
            **kwargs
        ) -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        """Solves one-dimensional Riemann problem in the direction as specified 
        by the axis argument. Wrapper function which calls, depending on the equation type,
        one of

        1) _solve_riemann_problem_xi_single_phase
        2) _solve_riemann_problem_xi_diffuse_five_equation

        :param primitives_L: primtive variable buffer left of cell face
        :type primitives_L: Array
        :param primitives_R: primtive variable buffer right of cell face
        :type primitives_R: Array
        :param conservatives_L: conservative variable buffer left of cell face
        :type conservatives_L: Array
        :param conservatives_R: conservative variable buffer right of cell face
        :type conservatives_R: Array
        :param axis: Spatial direction along which Riemann problem is solved.
        :type axis: int
        :return: _description_
        :rtype: Tuple[Array, Union[Array, None], Union[Array, None]]
        """

        if self.equation_type in ("SINGLE-PHASE",):
            return self._solve_riemann_problem_xi_single_phase(
                primitives_L, primitives_R,
                conservatives_L, conservatives_R,
                axis, **kwargs)
    
        else:
            raise NotImplementedError

    @abstractmethod
    def _solve_riemann_problem_xi_single_phase(
            self,
            primitives_L: Array, 
            primitives_R: Array,
            conservatives_L: Array, 
            conservatives_R: Array, 
            axis: int,
            **kwargs
        ) -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        """Solves one-dimensional single-phase Riemann problem
        in the direction as specified by the axis argument.

        :param primitives_L: _description_
        :type primitives_L: Array
        :param primitives_R: _description_
        :type primitives_R: Array
        :param conservatives_L: _description_
        :type conservatives_L: Array
        :param conservatives_R: _description_
        :type conservatives_R: Array
        :param axis: _description_
        :type axis: int
        :return: _description_
        :rtype: Tuple[Array, Union[Array, None], Union[Array, None]]
        """
        pass


class LaxFriedrichs(RiemannSolver):

    def __init__(
            self,
            equation_manager,
            signal_speed: Callable,
            **kwargs
            ) -> None:
        super().__init__(equation_manager, signal_speed)

    def _solve_riemann_problem_xi_single_phase(
            self, 
            primitives_L: Array,
            primitives_R: Array, 
            conservatives_L: Array,
            conservatives_R: Array, 
            axis: int,
            **kwargs
            ) -> Tuple[Array, Array, Array]:

        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        speed_of_sound_L = self.equation_manager.get_speed_of_sound(primitives_L[self.equation_manager.energy_ids],primitives_L[self.equation_manager.mass_ids])
        speed_of_sound_R = self.equation_manager.get_speed_of_sound(primitives_R[self.equation_manager.energy_ids],primitives_R[self.equation_manager.mass_ids])

        alpha = jnp.maximum(
            jnp.max(jnp.abs(primitives_L[self.velocity_ids[axis]]) + speed_of_sound_L), 
            jnp.max(jnp.abs(primitives_R[self.velocity_ids[axis]]) + speed_of_sound_R))

        fluxes_xi = 0.5 * (fluxes_L + fluxes_R) - 0.5 * alpha * (conservatives_R - conservatives_L)
            
        return fluxes_xi, None, None
    

class HLLC(RiemannSolver):
    """HLLC Riemann Solver
    Toro et al. 1994

    Supports:
    1) Single-phase / Two-phase level-set
    3) 5-Equation Diffuse-interface model

    For single-phase or two-phase level-set equations, the standard Riemann
    solver proposed by Toro is used. For diffuse-interface method, the HLLC
    modification as proposed by Coralic & Colonius with the surface-tension
    extension by Garrick is used.
    """

    def __init__(
            self,
#            material_manager: MaterialManager, 
            equation_manager,
            signal_speed: Callable,
            **kwargs
            ) -> None:
        super().__init__(equation_manager, signal_speed)
        
        self.s_star = compute_sstar

    def _solve_riemann_problem_xi_single_phase(
            self,
            primitives_L: Array,
            primitives_R: Array,
            conservatives_L: Array,
            conservatives_R: Array,
            axis: int,
            **kwargs
            ) -> Tuple[Array, Array, Array]:
        
        speed_of_sound_L = self.equation_manager.get_speed_of_sound(primitives_L[self.equation_manager.energy_ids],primitives_L[self.equation_manager.mass_ids])
        speed_of_sound_R = self.equation_manager.get_speed_of_sound(primitives_R[self.equation_manager.energy_ids],primitives_R[self.equation_manager.mass_ids])

        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(
            primitives_L[self.velocity_ids[axis]],
            primitives_R[self.velocity_ids[axis]],
            speed_of_sound_L,
            speed_of_sound_R,
            rho_L=primitives_L[self.mass_ids],
            rho_R=primitives_R[self.mass_ids],
            p_L=primitives_L[self.energy_ids],
            p_R=primitives_R[self.energy_ids],
            gamma=self.equation_manager.gamma
        )
        wave_speed_contact = self.s_star(
            primitives_L[self.velocity_ids[axis]],
            primitives_R[self.velocity_ids[axis]],
            primitives_L[self.energy_ids],
            primitives_R[self.energy_ids],
            primitives_L[self.mass_ids],
            primitives_R[self.mass_ids],
            wave_speed_simple_L, 
            wave_speed_simple_R)

        wave_speed_L = jnp.minimum(wave_speed_simple_L, 0.0)
        wave_speed_R = jnp.maximum(wave_speed_simple_R, 0.0)

        # Toro 10.73
        pre_factor_L = (wave_speed_simple_L - primitives_L[self.velocity_ids[axis]]) / (wave_speed_simple_L - wave_speed_contact) * primitives_L[self.mass_ids]
        pre_factor_R = (wave_speed_simple_R - primitives_R[self.velocity_ids[axis]]) / (wave_speed_simple_R - wave_speed_contact) * primitives_R[self.mass_ids]

        # TODO check out performance with u_star_L = jnp.expand_dims(prefactor_L) / jnp.ones_like() 
        # to avoid list + jnp.stack
        u_star_L = [
            pre_factor_L,
            pre_factor_L,
            pre_factor_L,
            pre_factor_L,
            pre_factor_L * (conservatives_L[self.energy_ids] / conservatives_L[self.mass_ids] + (wave_speed_contact - primitives_L[self.velocity_ids[axis]]) * (wave_speed_contact + primitives_L[self.energy_ids] / primitives_L[self.mass_ids] / (wave_speed_simple_L - primitives_L[self.velocity_ids[axis]]) )) ]
        u_star_L[self.velocity_ids[axis]] *= wave_speed_contact
        u_star_L[self.velocity_minor[axis][0]] *= primitives_L[self.velocity_minor[axis][0]]
        u_star_L[self.velocity_minor[axis][1]] *= primitives_L[self.velocity_minor[axis][1]]
        u_star_L = jnp.stack(u_star_L)

        u_star_R = [
            pre_factor_R,
            pre_factor_R,
            pre_factor_R,
            pre_factor_R,
            pre_factor_R * (conservatives_R[self.energy_ids] / conservatives_R[self.mass_ids] + (wave_speed_contact - primitives_R[self.velocity_ids[axis]]) * (wave_speed_contact + primitives_R[self.energy_ids] / primitives_R[self.mass_ids] / (wave_speed_simple_R - primitives_R[self.velocity_ids[axis]]) )) ]
        u_star_R[self.velocity_ids[axis]] *= wave_speed_contact
        u_star_R[self.velocity_minor[axis][0]] *= primitives_R[self.velocity_minor[axis][0]]
        u_star_R[self.velocity_minor[axis][1]] *= primitives_R[self.velocity_minor[axis][1]]
        u_star_R = jnp.stack(u_star_R)

        # Phyiscal fluxes
        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        # Toro 10.72
        flux_star_L = fluxes_L + wave_speed_L * (u_star_L - conservatives_L)
        flux_star_R = fluxes_R + wave_speed_R * (u_star_R - conservatives_R)

        # Kind of Toro 10.71
        fluxes_xi = 0.5 * (1 + jnp.sign(wave_speed_contact)) * flux_star_L \
                  + 0.5 * (1 - jnp.sign(wave_speed_contact)) * flux_star_R
        return fluxes_xi, None, None