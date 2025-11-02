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
    """HLLD Riemann Solver for ideal MHD (Miyoshi & Kusano, 2005)

    - Careful algebra for star and double-star states
    - Symmetric/robust total-pressure formula
    - Stable handling when Bn ~ 0 (HLL fallback)
    - Uses jnp.select (ordered conditions) to avoid jnp.where chaining pitfalls
    """

    def __init__(self, equation_manager, signal_speed=None, wft=1.0):
        self.equation_type = "SINGLE-PHASE"
        self.equation_manager = equation_manager
        self.signal_speed = signal_speed
        self.wft = wft
        self.eps = getattr(equation_manager, "eps", 1e-12)
        self.gamma = getattr(equation_manager, "gamma", 5.0/3.0)
        # Common indices
        self.mass_ids = getattr(equation_manager, "mass_ids", 0)
        self.energy_ids = getattr(equation_manager, "energy_ids", 4)  
    # ---------------------- helpers ---------------------- #
    @staticmethod
    def _axis_unpack(prims, axis, vel_ids, mag_ids):
        """Return (rho, p, vn, vt1, vt2, Bn, Bt1, Bt2, (B1,B2,B3)).
        Assumes prims layout consistent with equation_manager.  """
        B1_i, B2_i, B3_i = mag_ids
        u_i, v_i, w_i = vel_ids
        rho = prims[0]  # assuming mass density first in primitives
        p = prims[4]    # in your code, primitives[energy_i] stores gas pressure

        B1, B2, B3 = prims[B1_i], prims[B2_i], prims[B3_i]
        if axis == 0:
            vn, vt1, vt2 = prims[u_i], prims[v_i], prims[w_i]
            Bn, Bt1, Bt2 = B1, B2, B3
        elif axis == 1:
            vn, vt1, vt2 = prims[v_i], prims[w_i], prims[u_i]
            Bn, Bt1, Bt2 = B2, B3, B1
        else:
            vn, vt1, vt2 = prims[w_i], prims[u_i], prims[v_i]
            Bn, Bt1, Bt2 = B3, B1, B2
        return rho, p, vn, vt1, vt2, Bn, Bt1, Bt2, (B1, B2, B3)

    @staticmethod
    def _build_state(rho, vn, vt1, vt2, Bn, Bt1, Bt2, ptot, axis, gamma, eps):
        """Build conservative state U = [rho, rho*u, rho*v, rho*w, B1, B2, B3, E]."""
        if axis == 0:
            u, v, w = vn, vt1, vt2
            B1, B2, B3 = Bn, Bt1, Bt2
        elif axis == 1:
            v, w, u = vn, vt1, vt2
            B2, B3, B1 = Bn, Bt1, Bt2
        else:
            w, u, v = vn, vt1, vt2
            B3, B1, B2 = Bn, Bt1, Bt2

        ke = 0.5 * rho * (u*u + v*v + w*w)
        me = 0.5 * (B1*B1 + B2*B2 + B3*B3)
        p_gas = jnp.maximum(ptot - me, eps)
        e_int = p_gas / (gamma - 1.0)
        E = ke + me + e_int
        return jnp.stack([rho, rho*u, rho*v, rho*w, B1, B2, B3, E])

    # ---------------- main solver (single-face) ---------------- #
    def _solve_riemann_problem_xi_single_phase(self,
                                               primitives_L,
                                               primitives_R,
                                               conservatives_L,
                                               conservatives_R,
                                               axis,
                                               **kwargs) -> Tuple[jnp.ndarray, None, None]:
        # indices / params
        u_i, v_i, w_i = self.equation_manager.vel_ids
        b1_i, b2_i, b3_i = self.equation_manager.mag_ids
        rho_i = self.mass_ids
        p_i = self.energy_ids  # gas pressure index in primitives

        # Extract primitive components (per-axis oriented)
        rhoL = jnp.maximum(primitives_L[rho_i], self.eps)
        rhoR = jnp.maximum(primitives_R[rho_i], self.eps)
        pL   = jnp.maximum(primitives_L[p_i], self.eps)
        pR   = jnp.maximum(primitives_R[p_i], self.eps)

        (rhoL_, pL_, vnL, vt1L, vt2L, BnL, Bt1L, Bt2L, BvecL) = self._axis_unpack(
            primitives_L, axis, (u_i, v_i, w_i), (b1_i, b2_i, b3_i))
        (rhoR_, pR_, vnR, vt1R, vt2R, BnR, Bt1R, Bt2R, BvecR) = self._axis_unpack(
            primitives_R, axis, (u_i, v_i, w_i), (b1_i, b2_i, b3_i))
        # consistency guards
        rhoL = jnp.maximum(rhoL, self.eps)
        rhoR = jnp.maximum(rhoR, self.eps)
        pL = jnp.maximum(pL, self.eps)
        pR = jnp.maximum(pR, self.eps)

        # total (gas + magnetic) pressure
        B2L = BvecL[0]*BvecL[0] + BvecL[1]*BvecL[1] + BvecL[2]*BvecL[2]
        B2R = BvecR[0]*BvecR[0] + BvecR[1]*BvecR[1] + BvecR[2]*BvecR[2]
        pTL = pL + 0.5 * B2L
        pTR = pR + 0.5 * B2R

        # Fast magnetosonic speeds (per-axis)
        cfL = self.equation_manager.get_fast_magnetosonic_speed(primitives_L, axis)
        cfR = self.equation_manager.get_fast_magnetosonic_speed(primitives_R, axis)

        # Davis estimates for outer signal speeds
        SL = jnp.minimum(vnL - cfL, vnR - cfR)
        SR = jnp.maximum(vnL + cfL, vnR + cfR)

        # Physical fluxes on each side (needed for HLL fallback too)
        FL = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)[:8]
        FR = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)[:8]

        # Supersonic regions
        flux_upwind = jnp.where(SL >= 0.0, FL,
                         jnp.where(SR <= 0.0, FR, jnp.zeros_like(FL)))
        # NOTE: Avoid Python branching on traced booleans. We'll compute middle-state
        # fluxes and then select per-face using array masks.
        need_middle = (SL < 0.0) & (SR > 0.0)
        # (No early return here; keep everything JAX-traceable.)
        # --- continue to build HLLD/HLL below and select with masks ---


        # Contact speed S_M (Miyoshi & Kusano eq. 38)
        denomM = (SR - vnR) * rhoR - (SL - vnL) * rhoL
        denomM = jnp.where(jnp.abs(denomM) < self.eps, jnp.sign(denomM) * self.eps, denomM)
        numM = (
            (SR - vnR) * rhoR * vnR - (SL - vnL) * rhoL * vnL - (pTR - pTL)
        )
        SM = numM / denomM

        # Bn across the fan: for CT schemes Bn is continuous; use a stable interface value
        # Prefer arithmetic average to damp inconsistency when inputs slightly disagree.
        Bn_star = 0.5 * (BnL + BnR)

        # Star-region densities (eq. 43)
        rhoL_star = rhoL * (SL - vnL) / (SL - SM)
        rhoR_star = rhoR * (SR - vnR) / (SR - SM)
        rhoL_star = jnp.maximum(rhoL_star, self.eps)
        rhoR_star = jnp.maximum(rhoR_star, self.eps)

        # Total pressure in star region – robust symmetric form
        pT_star_L = pTL + rhoL * (SL - vnL) * (SM - vnL)
        pT_star_R = pTR + rhoR * (SR - vnR) * (SM - vnR)
        pT_star = 0.5 * (pT_star_L + pT_star_R)
        pT_star = jnp.maximum(pT_star, self.eps)

        # HLL fallback (pre-compute)
        denomHLL = SR - SL + self.eps
        Udiff = conservatives_R - conservatives_L
        F_HLL = (SR * FL - SL * FR + SL * SR * Udiff) / denomHLL
        # Degeneracy/weak-field detection (per-face array). We will *not* Python-return here;
        # instead we use it in a mask for flux selection.
        weak_Bn = (Bn_star * Bn_star) < jnp.maximum(1e-12, 1e-8 * jnp.maximum(B2L, B2R))


        # Alfvén wave speeds in star states
        aL = jnp.abs(Bn_star) / jnp.sqrt(rhoL_star)
        aR = jnp.abs(Bn_star) / jnp.sqrt(rhoR_star)
        SL_star = SM - aL
        SR_star = SM + aR

        # Tangential states in * (eq. 46–49)
        Bn2 = Bn_star * Bn_star
        # Left
        denomL = rhoL * (SL - vnL) * (SL - SM) - Bn2
        denomL = jnp.where(jnp.abs(denomL) < self.eps, jnp.sign(denomL) * self.eps, denomL)
        vt1L_star = vt1L - Bn_star * Bt1L * (SM - vnL) / denomL
        vt2L_star = vt2L - Bn_star * Bt2L * (SM - vnL) / denomL
        Bt1L_star = Bt1L * (rhoL * (SL - vnL) * (SL - vnL) - Bn2) / denomL
        Bt2L_star = Bt2L * (rhoL * (SL - vnL) * (SL - vnL) - Bn2) / denomL

        # Right
        denomR = rhoR * (SR - vnR) * (SR - SM) - Bn2
        denomR = jnp.where(jnp.abs(denomR) < self.eps, jnp.sign(denomR) * self.eps, denomR)
        vt1R_star = vt1R - Bn_star * Bt1R * (SM - vnR) / denomR
        vt2R_star = vt2R - Bn_star * Bt2R * (SM - vnR) / denomR
        Bt1R_star = Bt1R * (rhoR * (SR - vnR) * (SR - vnR) - Bn2) / denomR
        Bt2R_star = Bt2R * (rhoR * (SR - vnR) * (SR - vnR) - Bn2) / denomR

        # Double-star averages across Alfvén fan
        sgnBn = jnp.sign(Bn_star)
        srL = jnp.sqrt(rhoL_star)
        srR = jnp.sqrt(rhoR_star)
        denomSS = srL + srR + self.eps

        vt1_ss = (srL * vt1L_star + srR * vt1R_star + sgnBn * (Bt1R_star - Bt1L_star)) / denomSS
        vt2_ss = (srL * vt2L_star + srR * vt2R_star + sgnBn * (Bt2R_star - Bt2L_star)) / denomSS
        Bt1_ss = (srL * Bt1R_star + srR * Bt1L_star + sgnBn * srL * srR * (vt1R_star - vt1L_star)) / denomSS
        Bt2_ss = (srL * Bt2R_star + srR * Bt2L_star + sgnBn * srL * srR * (vt2R_star - vt2L_star)) / denomSS

        # Build conservative states
        UL_star = self._build_state(rhoL_star, SM, vt1L_star, vt2L_star, Bn_star, Bt1L_star, Bt2L_star,
                                    pT_star, axis, self.gamma, self.eps)
        UR_star = self._build_state(rhoR_star, SM, vt1R_star, vt2R_star, Bn_star, Bt1R_star, Bt2R_star,
                                    pT_star, axis, self.gamma, self.eps)
        UL_ss = self._build_state(rhoL_star, SM, vt1_ss, vt2_ss, Bn_star, Bt1_ss, Bt2_ss,
                                  pT_star, axis, self.gamma, self.eps)
        UR_ss = self._build_state(rhoR_star, SM, vt1_ss, vt2_ss, Bn_star, Bt1_ss, Bt2_ss,
                                  pT_star, axis, self.gamma, self.eps)

        # Star fluxes
        FL_star = FL + SL * (UL_star - conservatives_L)
        FR_star = FR + SR * (UR_star - conservatives_R)
        FL_ss = FL_star + SL_star * (UL_ss - UL_star)
        FR_ss = FR_star + SR_star * (UR_ss - UR_star)

        # Select flux according to wavefan ordering
        conds = [SL >= 0.0,
                 (SL < 0.0) & (SL_star >= 0.0),
                 (SL_star < 0.0) & (SM >= 0.0),
                 (SM < 0.0) & (SR_star >= 0.0),
                 (SR_star < 0.0) & (SR >= 0.0)]
        choices = [FL, FL_star, FL_ss, FR_ss, FR_star]
        F_HLLD = jnp.select(conds, choices, FR)
        # Final selection between HLLD and HLL (degeneracy safeguard), then between
        # transonic and supersonic regions. Make masks broadcastable to flux shape (8, Nx, Ny, 1).
        use_HLL = (jnp.abs(denomM) < 10.0*self.eps) | weak_Bn | (jnp.abs(denomL) < 10.0*self.eps) | (jnp.abs(denomR) < 10.0*self.eps)

        # --- mask shaping helpers ---
        def _mask_to_flux_shape(mask):
            # mask expected (Nx, Ny, 1, 1) or (Nx, Ny, 1)
            m = jnp.squeeze(mask, axis=-2) if mask.ndim == 4 else mask  # drop the penultimate singleton if present
            # Now m has shape (Nx, Ny, 1)
            m = m[None, ...]  # (1, Nx, Ny, 1)
            return m

        use_HLL_m = _mask_to_flux_shape(use_HLL)
        need_middle_m = _mask_to_flux_shape(need_middle)

        F_mid = jnp.where(use_HLL_m, F_HLL, F_HLLD)
        # Choose upwind if not transonic, otherwise HLL/HLLD mid-fan flux
        F = jnp.where(need_middle_m, F_mid, flux_upwind)
        return jnp.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), None, None

    
class HLLD_MHD_old(RiemannSolver):
    """HLLD Riemann Solver for ideal MHD
    Seems maybe more prone to errors, will discontinue eventually...
    Miyoshi & Kusano 2005
    
    Four-wave solver with (hopefully) robust degeneracy handling and HLL fallback.
    """

    def __init__(
            self,
            equation_manager,
            signal_speed =None, #not used, just for completeness/same format
            wft=1.0,
            **kwargs
            ) -> None:
        super().__init__(equation_manager, signal_speed)
        self.wft = wft
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
        
        # --- indices ---
        rho_i      = self.mass_ids
        energy_i   = self.energy_ids            # pressure index in primitives
        u_i, v_i, w_i = self.equation_manager.vel_ids
        b1_i, b2_i, b3_i = self.equation_manager.mag_ids
        
        # --- extract prims ---
        rho_L = jnp.maximum(primitives_L[rho_i], self.eps)
        rho_R = jnp.maximum(primitives_R[rho_i], self.eps)
        p_L   = jnp.maximum(primitives_L[energy_i], self.eps)
        p_R   = jnp.maximum(primitives_R[energy_i], self.eps)
        
        B1_L, B2_L, B3_L = primitives_L[b1_i], primitives_L[b2_i], primitives_L[b3_i]
        B1_R, B2_R, B3_R = primitives_R[b1_i], primitives_R[b2_i], primitives_R[b3_i]
        
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
        
        # Fast magnetosonic speeds, instead of normal hydro soundspeed
        cf_L = self.equation_manager.get_fast_magnetosonic_speed(primitives_L, axis)
        cf_R = self.equation_manager.get_fast_magnetosonic_speed(primitives_R, axis)
        
        # Outer wave speeds (Davis estimate)
        S_L = jnp.minimum(vn_L - cf_L, vn_R - cf_R)
        S_R = jnp.maximum(vn_L + cf_L, vn_R + cf_R)
        
        # Physical fluxes (needed for both HLLD and HLL fallback)
        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)[:8]
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)[:8]
        
        # Early exit for supersonic flow
        fluxes_xi = jnp.where(S_L >= 0.0, fluxes_L,
                    jnp.where(S_R <= 0.0, fluxes_R, 
                              jnp.zeros_like(fluxes_L)))  # placeholder, will be overwritten later on
        
        # Check if we need to compute intermediate states
        compute_intermediates = (S_L < 0.0) & (S_R > 0.0)
        
        # Contact wave speed S_M (Miyoshi & Kusano eq. 38)
        denominator_M = (S_R - vn_R) * rho_R - (S_L - vn_L) * rho_L
        # Safe division
      #  denominator_M_safe = jnp.where(jnp.abs(denominator_M) < self.eps, 
      #                                 jnp.sign(denominator_M) * self.eps, 
       #                                denominator_M)
        denominator_M_safe = jnp.where(jnp.abs(denominator_M) < self.eps, self.eps, denominator_M)

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
     #   ptot_star = ((S_R - vn_R) * rho_R * ptot_L - (S_L - vn_L) * rho_L * ptot_R + 
     #                rho_L * rho_R * (S_R - vn_R) * (S_L - vn_L) * (vn_R - vn_L)) / \
     #               denominator_M_safe
        ptot_star = (
                (S_R - vn_R) * rho_R * ptot_R - (S_L - vn_L) * rho_L * ptot_L
                + rho_L * rho_R * (S_R - vn_R) * (S_L - vn_L) * (vn_R - vn_L)
            ) / denominator_M_safe

        ptot_star = jnp.maximum(ptot_star, self.eps)
        
        # Check for weak field limit: if |Bn_star| is very small, fall back to HLL
        Bn_star_sq = Bn_star * Bn_star
        weak_field_threshold = self.eps * self.wft
        use_HLL = (Bn_star_sq < weak_field_threshold) | (jnp.abs(denominator_M) < self.eps)
        
        # HLL flux (fallback)
        denom_HLL = S_R - S_L + self.eps
        flux_HLL = (S_R * fluxes_L - S_L * fluxes_R + S_L * S_R * (conservatives_R - conservatives_L)) / denom_HLL
        
        # Alfvén wave speeds 
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
        
        # Final safety: ensure no NaNs or Infs, probably not best practice... might remove..
        fluxes_xi = jnp.nan_to_num(fluxes_xi, nan=0.0, posinf=0.0, neginf=0.0)
        
        return fluxes_xi, None, None