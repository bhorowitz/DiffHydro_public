from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from .signal_speeds import compute_sstar

#adapted from jaxfluids, added more magnetic-solvers, need to reorganize sometime...

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
        self.velocity_minor = self.equation_manager.velocity_minor_axes

        self.equation_type = self.equation_manager.equation_type

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

        if self.equation_type in ("SINGLE-PHASE","MHD"): #uses same riemann problem setup casework for both
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


class HLL_MHD(RiemannSolver):
    """
    Two-wave HLL Riemann solver for ideal MHD.
    Uses Davis-type bounds S_L, S_R from fast magnetosonic speeds.
    """
    def __init__(self, equation_manager, signal_speed, **kwargs):
        super().__init__(equation_manager, signal_speed)

    def _solve_riemann_problem_xi_single_phase(
            self,
            primitives_L, primitives_R,
            conservatives_L, conservatives_R,
            axis: int,
            **kwargs):
        # Physical fluxes from the equation manager (same pattern as other solvers)
        F_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        F_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        # Normal velocities
        uL = primitives_L[self.velocity_ids[axis]]
        uR = primitives_R[self.velocity_ids[axis]]

        # Fast magnetosonic speed bounds (requires EquationManagerMHD.get_fast_magnetosonic_speed)
        c_fL = self.equation_manager.get_fast_magnetosonic_speed(primitives_L, axis)
        c_fR = self.equation_manager.get_fast_magnetosonic_speed(primitives_R, axis)

        # Davis bounds
        S_L = jnp.minimum(uL - c_fL, uR - c_fR)
        S_R = jnp.maximum(uL + c_fL, uR + c_fR)

        # Degenerate case: fallback to local Lax–Friedrichs if S_R ≈ S_L
        denom = jnp.where(jnp.abs(S_R - S_L) < self.eps, 1.0, S_R - S_L)

        # HLL flux: (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)
        fluxes_xi = (S_R * F_L - S_L * F_R + S_L * S_R * (conservatives_R - conservatives_L)) / denom

        return fluxes_xi, None, None


class HLLD_MHD(RiemannSolver):
    """HLLD Riemann Solver for ideal MHD
    Miyoshi & Kusano 2005
    
    Four-wave solver with robust degeneracy handling and HLL fallback.
    """

    def __init__(
            self,
            equation_manager,
            signal_speed,
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
        
        # Get indices
        rho_i = self.mass_ids
        energy_i = self.energy_ids
        
        # Extract primitive variables
        rho_L = jnp.maximum(primitives_L[rho_i], self.eps)
        rho_R = jnp.maximum(primitives_R[rho_i], self.eps)
        p_L = jnp.maximum(primitives_L[energy_i], self.eps)
        p_R = jnp.maximum(primitives_R[energy_i], self.eps)
        
        # Magnetic field components
        B1_L = primitives_L[4]
        B2_L = primitives_L[5]
        B3_L = primitives_L[6]
        B1_R = primitives_R[4]
        B2_R = primitives_R[5]
        B3_R = primitives_R[6]
        
        # Normal and tangential components (axis-dependent)
        if axis == 0:
            Bn_L, Bt1_L, Bt2_L = B1_L, B2_L, B3_L
            Bn_R, Bt1_R, Bt2_R = B1_R, B2_R, B3_R
            vn_L = primitives_L[1]
            vt1_L = primitives_L[2]
            vt2_L = primitives_L[3]
            vn_R = primitives_R[1]
            vt1_R = primitives_R[2]
            vt2_R = primitives_R[3]
        elif axis == 1:
            Bn_L, Bt1_L, Bt2_L = B2_L, B3_L, B1_L
            Bn_R, Bt1_R, Bt2_R = B2_R, B3_R, B1_R
            vn_L = primitives_L[2]
            vt1_L = primitives_L[3]
            vt2_L = primitives_L[1]
            vn_R = primitives_R[2]
            vt1_R = primitives_R[3]
            vt2_R = primitives_R[1]
        else:  # axis == 2
            Bn_L, Bt1_L, Bt2_L = B3_L, B1_L, B2_L
            Bn_R, Bt1_R, Bt2_R = B3_R, B1_R, B2_R
            vn_L = primitives_L[3]
            vt1_L = primitives_L[1]
            vt2_L = primitives_L[2]
            vn_R = primitives_R[3]
            vt1_R = primitives_R[1]
            vt2_R = primitives_R[2]
        
        # Total magnetic pressure
        B2_L = B1_L*B1_L + B2_L*B2_L + B3_L*B3_L
        B2_R = B1_R*B1_R + B2_R*B2_R + B3_R*B3_R
        ptot_L = p_L + 0.5 * B2_L
        ptot_R = p_R + 0.5 * B2_R
        
        # Fast magnetosonic speeds
        cf_L = self.equation_manager.get_fast_magnetosonic_speed(primitives_L, axis)
        cf_R = self.equation_manager.get_fast_magnetosonic_speed(primitives_R, axis)
        
        # Outer wave speeds (Davis estimate)
        S_L = jnp.minimum(vn_L - cf_L, vn_R - cf_R)
        S_R = jnp.maximum(vn_L + cf_L, vn_R + cf_R)
        
        # Physical fluxes (needed for both HLLD and HLL fallback)
        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)
        
        # Early exit for supersonic flow
        fluxes_xi = jnp.where(S_L >= 0.0, fluxes_L,
                    jnp.where(S_R <= 0.0, fluxes_R, 
                              jnp.zeros_like(fluxes_L)))  # placeholder, will be overwritten
        
        # Check if we need to compute intermediate states
        compute_intermediates = (S_L < 0.0) & (S_R > 0.0)
        
        # Contact wave speed S_M (Miyoshi & Kusano eq. 38)
        denominator_M = (S_R - vn_R) * rho_R - (S_L - vn_L) * rho_L
        # Safe division
        denominator_M_safe = jnp.where(jnp.abs(denominator_M) < self.eps, 
                                       jnp.sign(denominator_M) * self.eps, 
                                       denominator_M)
        
        numerator_M = ((S_R - vn_R) * rho_R * vn_R - (S_L - vn_L) * rho_L * vn_L - 
                      ptot_R + ptot_L)
        S_M = numerator_M / denominator_M_safe
        
        # Normal magnetic field in star region (conserved across contact)
        Bn_star = ((S_R - vn_R) * rho_R * Bn_R - (S_L - vn_L) * rho_L * Bn_L) / \
                  denominator_M_safe
        
        # Star region densities (Miyoshi & Kusano eq. 43)
        rho_L_star = rho_L * (S_L - vn_L) / (S_L - S_M + self.eps)
        rho_R_star = rho_R * (S_R - vn_R) / (S_R - S_M + self.eps)
        rho_L_star = jnp.maximum(rho_L_star, self.eps)
        rho_R_star = jnp.maximum(rho_R_star, self.eps)
        
        # Total pressure in star region (Miyoshi & Kusano eq. 41)
        ptot_star = ((S_R - vn_R) * rho_R * ptot_L - (S_L - vn_L) * rho_L * ptot_R + 
                     rho_L * rho_R * (S_R - vn_R) * (S_L - vn_L) * (vn_R - vn_L)) / \
                    denominator_M_safe
        ptot_star = jnp.maximum(ptot_star, self.eps)
        
        # Check for weak field limit: if |Bn_star| is very small, fall back to HLL
        Bn_star_sq = Bn_star * Bn_star
        weak_field_threshold = self.eps * 100.0
        use_HLL = (Bn_star_sq < weak_field_threshold) | (jnp.abs(denominator_M) < self.eps)
        
        # HLL flux (fallback)
        denom_HLL = S_R - S_L + self.eps
        flux_HLL = (S_R * fluxes_L - S_L * fluxes_R + S_L * S_R * (conservatives_R - conservatives_L)) / denom_HLL
        
        # Alfvén wave speeds (Miyoshi & Kusano eq. 51)
        sqrt_rho_L_star = jnp.sqrt(rho_L_star)
        sqrt_rho_R_star = jnp.sqrt(rho_R_star)
        S_L_star = S_M - jnp.abs(Bn_star) / (sqrt_rho_L_star + self.eps)
        S_R_star = S_M + jnp.abs(Bn_star) / (sqrt_rho_R_star + self.eps)
        
        # Left star state (between S_L and S_L_star)
        # Denominator for tangential components (can be singular)
        denom_L = rho_L * (S_L - vn_L) * (S_L - S_M) - Bn_star_sq
        denom_L_safe = jnp.where(jnp.abs(denom_L) < self.eps,
                                 jnp.sign(denom_L) * self.eps,
                                 denom_L)
        
        factor_L = Bn_star * (S_M - vn_L) / denom_L_safe
        vt1_L_star = vt1_L - Bt1_L * factor_L
        vt2_L_star = vt2_L - Bt2_L * factor_L
        
        numer_Bt_L = rho_L * (S_L - vn_L) * (S_L - vn_L) - Bn_star_sq
        Bt1_L_star = Bt1_L * numer_Bt_L / denom_L_safe
        Bt2_L_star = Bt2_L * numer_Bt_L / denom_L_safe
        
        # Right star state (between S_R_star and S_R)
        denom_R = rho_R * (S_R - vn_R) * (S_R - S_M) - Bn_star_sq
        denom_R_safe = jnp.where(jnp.abs(denom_R) < self.eps,
                                 jnp.sign(denom_R) * self.eps,
                                 denom_R)
        
        factor_R = Bn_star * (S_M - vn_R) / denom_R_safe
        vt1_R_star = vt1_R - Bt1_R * factor_R
        vt2_R_star = vt2_R - Bt2_R * factor_R
        
        numer_Bt_R = rho_R * (S_R - vn_R) * (S_R - vn_R) - Bn_star_sq
        Bt1_R_star = Bt1_R * numer_Bt_R / denom_R_safe
        Bt2_R_star = Bt2_R * numer_Bt_R / denom_R_safe
        
        # Double star states (between S_L_star and S_R_star)
        sign_Bn = jnp.sign(Bn_star)
        sum_sqrt_rho = sqrt_rho_L_star + sqrt_rho_R_star + self.eps
        
        vt1_star_star = (sqrt_rho_L_star * vt1_L_star + sqrt_rho_R_star * vt1_R_star + 
                         sign_Bn * (Bt1_R_star - Bt1_L_star)) / sum_sqrt_rho
        vt2_star_star = (sqrt_rho_L_star * vt2_L_star + sqrt_rho_R_star * vt2_R_star + 
                         sign_Bn * (Bt2_R_star - Bt2_L_star)) / sum_sqrt_rho
        
        Bt1_star_star = (sqrt_rho_L_star * Bt1_R_star + sqrt_rho_R_star * Bt1_L_star + 
                         sign_Bn * sqrt_rho_L_star * sqrt_rho_R_star * (vt1_R_star - vt1_L_star)) / \
                        sum_sqrt_rho
        Bt2_star_star = (sqrt_rho_L_star * Bt2_R_star + sqrt_rho_R_star * Bt2_L_star + 
                         sign_Bn * sqrt_rho_L_star * sqrt_rho_R_star * (vt2_R_star - vt2_L_star)) / \
                        sum_sqrt_rho
        
        # Build conservative star states
        def build_star_state(rho_star, vn_star, vt1_star, vt2_star, 
                            Bn_star_val, Bt1_star, Bt2_star, ptot_star_val, axis):
            if axis == 0:
                u_star = vn_star
                v_star = vt1_star
                w_star = vt2_star
                B1_star = Bn_star_val
                B2_star = Bt1_star
                B3_star = Bt2_star
            elif axis == 1:
                v_star = vn_star
                w_star = vt1_star
                u_star = vt2_star
                B2_star = Bn_star_val
                B3_star = Bt1_star
                B1_star = Bt2_star
            else:
                w_star = vn_star
                u_star = vt1_star
                v_star = vt2_star
                B3_star = Bn_star_val
                B1_star = Bt1_star
                B2_star = Bt2_star
            
            ke = 0.5 * rho_star * (u_star*u_star + v_star*v_star + w_star*w_star)
            me = 0.5 * (B1_star*B1_star + B2_star*B2_star + B3_star*B3_star)
            p_star = jnp.maximum(ptot_star_val - me, self.eps)
            # FIXED: internal energy needs rho_star factor
            e_int = p_star / (self.equation_manager.gamma - 1.0)
            E_star = ke + me + e_int
            
            return jnp.stack([rho_star, rho_star*u_star, rho_star*v_star, rho_star*w_star,
                            B1_star, B2_star, B3_star, E_star])
        
        U_L_star = build_star_state(rho_L_star, S_M, vt1_L_star, vt2_L_star,
                                    Bn_star, Bt1_L_star, Bt2_L_star, ptot_star, axis)
        U_R_star = build_star_state(rho_R_star, S_M, vt1_R_star, vt2_R_star,
                                    Bn_star, Bt1_R_star, Bt2_R_star, ptot_star, axis)
        U_L_star_star = build_star_state(rho_L_star, S_M, vt1_star_star, vt2_star_star,
                                         Bn_star, Bt1_star_star, Bt2_star_star, ptot_star, axis)
        U_R_star_star = build_star_state(rho_R_star, S_M, vt1_star_star, vt2_star_star,
                                         Bn_star, Bt1_star_star, Bt2_star_star, ptot_star, axis)
        
        # Star fluxes
        flux_L_star = fluxes_L + S_L * (U_L_star - conservatives_L)
        flux_R_star = fluxes_R + S_R * (U_R_star - conservatives_R)
        flux_L_star_star = flux_L_star + S_L_star * (U_L_star_star - U_L_star)
        flux_R_star_star = flux_R_star + S_R_star * (U_R_star_star - U_R_star)
        
        # Select appropriate flux based on wave pattern (full HLLD)
        flux_HLLD = jnp.where(0.0 <= S_L, fluxes_L,
                    jnp.where(0.0 <= S_L_star, flux_L_star,
                    jnp.where(0.0 <= S_M, flux_L_star_star,
                    jnp.where(0.0 <= S_R_star, flux_R_star_star,
                    jnp.where(0.0 <= S_R, flux_R_star, fluxes_R)))))
        
        # Choose between HLLD and HLL based on field strength and degeneracy
        fluxes_xi = jnp.where(use_HLL, flux_HLL, flux_HLLD)
        
        # Final safety: ensure no NaNs or Infs
        fluxes_xi = jnp.nan_to_num(fluxes_xi, nan=0.0, posinf=0.0, neginf=0.0)
        
        return fluxes_xi, None, None
