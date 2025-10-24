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

# diffhydro/solver/riemann_solver.py

class HLLC_MHD(RiemannSolver):
    """
    HLLC-like Riemann solver for ideal MHD (contact-resolving).
    Assumptions:
      - Ideal MHD, cell-centered B
      - W = [rho, u, v, w, B1, B2, B3, p]
      - U = [rho, rho*u, rho*v, rho*w, B1, B2, B3, E]
    Captures the contact wave (S_M). Does not construct full Alfvén star states.
    Falls back to HLL when the fan is degenerate.
    """

    def __init__(self, equation_manager, signal_speed=None, **kwargs):
        super().__init__(equation_manager, signal_speed)

    def _solve_riemann_problem_xi_single_phase(
        self,
        WL, WR, UL, UR,
        axis: int,
        **kwargs
    ):
        eq  = self.equation_manager
        eps = self.eps

        rho_i   = eq.mass_ids
        u_i, v_i, w_i = eq.vel_ids
        Bx_i, By_i, Bz_i = eq.mag_ids
        E_i     = eq.energy_ids

        # Convenience selectors for "normal" component indices in global coords
        momn_i = {0: u_i, 1: v_i, 2: w_i}[axis]
        Bn_i   = {0: Bx_i, 1: By_i, 2: Bz_i}[axis]

        # Normal velocities (u_n) and fast speeds on each side
        uL = WL[eq.vel_ids[axis]]
        uR = WR[eq.vel_ids[axis]]
        cL = eq.get_fast_magnetosonic_speed(WL, axis)
        cR = eq.get_fast_magnetosonic_speed(WR, axis)

        # Two outer signal bounds (Davis)
        SL = jnp.minimum(uL - cL, uR - cR)
        SR = jnp.maximum(uL + cL, uR + cR)

        # Physical fluxes on each side (ideal MHD)
        FL = eq.get_fluxes_xi(WL, UL, axis)
        FR = eq.get_fluxes_xi(WR, UR, axis)

        # HLL denominator and fallback flux/state
        denom = jnp.maximum(SR - SL, eps)
        UHLL  = (SR * UR - SL * UL + (FL - FR)) / denom
        FHLL  = (SR * FL - SL * FR + SL * SR * (UR - UL)) / denom

        # Single-valued normal magnetic field inside the fan (robust choice: upwind)
        BnL = WL[Bn_i]; BnR = WR[Bn_i]
        Bn_star = jnp.where(jnp.abs(SL) > jnp.abs(SR), BnL, BnR)

        # Contact speed S_M estimated from HLL state
        SM = (UHLL[momn_i] - Bn_star * UHLL[Bn_i]) / jnp.maximum(UHLL[rho_i], eps)
        SM = jnp.clip(SM, SL + 1e-12, SR - 1e-12)

        # Star-state densities (contact-resolving)
        rhoL_star = WL[rho_i] * (SL - uL) / jnp.maximum(SL - SM, eps)
        rhoR_star = WR[rho_i] * (SR - uR) / jnp.maximum(SR - SM, eps)

        # Build “minimal” star conservatives per side.
        # Strategy:
        #   - enforce normal velocity = S_M
        #   - scale tangential momenta with rho* (kept from side’s tangential v)
        #   - carry B unchanged except enforce shared Bn = Bn_star
        #   - recompute total energy from side pressure p and star (rho*, u*, B*)
        def star_conservatives(W, U, side, rho_star):
            # Side-local unpack
            rho  = jnp.maximum(rho_star, eps)
            un_w = W[eq.vel_ids[axis]]
            vt1  = W[eq.velocity_minor_axes[axis][0]]
            vt2  = W[eq.velocity_minor_axes[axis][1]]
            B1, B2, B3 = W[Bx_i], W[By_i], W[Bz_i]
            p    = W[E_i]

            # Map the shared Bn* back to (B1,B2,B3)
            if axis == 0:
                B1s, B2s, B3s = Bn_star, B2, B3
                us, vs, ws    = SM, vt1, vt2
            elif axis == 1:
                B1s, B2s, B3s = B1, Bn_star, B3
                us, vs, ws    = vt2, SM, vt1
            else:
                B1s, B2s, B3s = B1, B2, Bn_star
                us, vs, ws    = vt1, vt2, SM

            # Conservative star state (contact-only restoration)
            Ustar = jnp.zeros_like(U)
            Ustar = Ustar.at[rho_i].set(rho)
            Ustar = Ustar.at[u_i].set(rho * us)
            Ustar = Ustar.at[v_i].set(rho * vs)
            Ustar = Ustar.at[w_i].set(rho * ws)
            Ustar = Ustar.at[Bx_i].set(B1s)
            Ustar = Ustar.at[By_i].set(B2s)
            Ustar = Ustar.at[Bz_i].set(B3s)

            # Total energy from side pressure and star kinematics/fields
            ke = 0.5 * rho * (us*us + vs*vs + ws*ws)
            me = 0.5 * (B1s*B1s + B2s*B2s + B3s*B3s)
            e_int = eq.get_specific_energy(p, rho) * rho
            E_s = ke + me + jnp.maximum(e_int, eps)
            Ustar = Ustar.at[E_i].set(E_s)

            # Wave speed from the chosen side
            S = jnp.where(side == 0, SL, SR)
            Fside = eq.get_fluxes_xi(W, U, axis)
            Fstar = Fside + S * (Ustar - U)
            return Ustar, Fstar

        UL_star, FL_star = star_conservatives(WL, UL, side=0, rho_star=rhoL_star)
        UR_star, FR_star = star_conservatives(WR, UR, side=1, rho_star=rhoR_star)

        # Region selection (standard HLLC fan logic)
        flux = jnp.where(0.0 <= SL, FL,
                 jnp.where(0.0 <= SM, FL_star,
                   jnp.where(0.0 <  SR, FR_star, FR)))

        # Degenerate fan → fallback to HLL
        degenerate = (jnp.abs(SR - SL) < 100*eps) | jnp.isnan(SM) | jnp.isinf(SM) \
                     | (rhoL_star <= eps) | (rhoR_star <= eps)
        flux = jnp.where(degenerate, FHLL, flux)
        flux = jnp.nan_to_num(flux, 0.0, 0.0, 0.0)
        return flux, None, None


class HLLD_MHD(RiemannSolver):
    """
    HLLD Riemann solver for ideal MHD (Miyoshi & Kusano, 2005).
    Assumes:
      - Primitives W = [rho, u, v, w, B1, B2, B3, p]
      - Conservatives U = [rho, rho*u, rho*v, rho*w, B1, B2, B3, E]
    Uses safe divisions and HLL fallback for degenerate wave fans.
    """

    def __init__(self, equation_manager, signal_speed=None, **kwargs):
        super().__init__(equation_manager, signal_speed)

    # ---------- helpers ----------
    def _safe_div(self, num, den, eps):
        den_safe = jnp.where(jnp.abs(den) < eps,
                             jnp.where(den >= 0.0, eps, -eps),
                             den)
        return num / den_safe

    def _clamp_pos(self, x, eps):
        return jnp.where(x > eps, x, eps)

    def _pick_axis(self, W, axis, eq):
        """Rotate to 'axis' as the normal direction.
        Returns: un, vt1, vt2, Bn, Bt1, Bt2, and a function to un-rotate (u,v,w,B1,B2,B3)."""
        u_i, v_i, w_i = eq.vel_ids
        Bx_i, By_i, Bz_i = eq.mag_ids

        if axis == 0:
            un, vt1, vt2 = W[u_i], W[v_i], W[w_i]
            Bn, Bt1, Bt2 = W[Bx_i], W[By_i], W[Bz_i]
            def unrotate(un_, vt1_, vt2_, Bn_, Bt1_, Bt2_):
                return (un_, vt1_, vt2_, Bn_, Bt1_, Bt2_)
        elif axis == 1:
            un, vt1, vt2 = W[v_i], W[w_i], W[u_i]
            Bn, Bt1, Bt2 = W[By_i], W[Bz_i], W[Bx_i]
            def unrotate(un_, vt1_, vt2_, Bn_, Bt1_, Bt2_):
                # map back: normal along y
                return (vt2_, un_, vt1_, Bt2_, Bn_, Bt1_)
        else:
            un, vt1, vt2 = W[w_i], W[u_i], W[v_i]
            Bn, Bt1, Bt2 = W[Bz_i], W[Bx_i], W[By_i]
            def unrotate(un_, vt1_, vt2_, Bn_, Bt1_, Bt2_):
                # map back: normal along z
                return (vt1_, vt2_, un_, Bt1_, Bt2_, Bn_)
        return un, vt1, vt2, Bn, Bt1, Bt2, unrotate

    # ---------- core ----------
    def _solve_riemann_problem_xi_single_phase(
        self,
        WL, WR, UL, UR,
        axis: int,
        **kwargs
    ):
        eq = self.equation_manager
        eps = self.eps

        rho_i = eq.mass_ids
        u_i, v_i, w_i = eq.vel_ids
        Bx_i, By_i, Bz_i = eq.mag_ids
        E_i = eq.energy_ids  # expected -1

        # rotate to axis-normal frame
        uL, vt1L, vt2L, BnL, Bt1L, Bt2L, unrot = self._pick_axis(WL, axis, eq)
        uR, vt1R, vt2R, BnR, Bt1R, Bt2R, _     = self._pick_axis(WR, axis, eq)
        rhoL, rhoR = WL[rho_i], WR[rho_i]
        pL,   pR   = WL[E_i],  WR[E_i]

        # total pressure
        B2L = WL[Bx_i]*WL[Bx_i] + WL[By_i]*WL[By_i] + WL[Bz_i]*WL[Bz_i]
        B2R = WR[Bx_i]*WR[Bx_i] + WR[By_i]*WR[By_i] + WR[Bz_i]*WR[Bz_i]
        ptL = pL + 0.5 * B2L
        ptR = pR + 0.5 * B2R

        # fast magnetosonic speeds (normal)
        cfL = eq.get_fast_magnetosonic_speed(WL, axis)
        cfR = eq.get_fast_magnetosonic_speed(WR, axis)

        # outer wave bounds (Davis)
        SL = jnp.minimum(uL - cfL, uR - cfR)
        SR = jnp.maximum(uL + cfL, uR + cfR)
        SR = jnp.maximum(SR, SL)  # enforce ordering
        denom = self._clamp_pos(SR - SL, eps)

        # physical fluxes
        FL = eq.get_fluxes_xi(WL, UL, axis)
        FR = eq.get_fluxes_xi(WR, UR, axis)

        # HLL state and flux (used for fallback and some intermediates)
        UHLL = (SR * UR - SL * UL + (FL - FR)) / denom
        FHLL = (SR * FL - SL * FR + SL * SR * (UR - UL)) / denom

        # choose single Bn* across fan (upwind choice is robust)
        Bn_i = {0: Bx_i, 1: By_i, 2: Bz_i}[axis]
        Bn_star = jnp.where(jnp.abs(SL) > jnp.abs(SR), BnL, BnR)

        # contact speed SM from HLL state
        momn_i = {0: u_i, 1: v_i, 2: w_i}[axis]
        SM = self._safe_div(UHLL[momn_i] - Bn_star * UHLL[Bn_i],
                            UHLL[rho_i], eps)

        # star densities
        rhoL_star = self._clamp_pos(rhoL * self._safe_div(SL - uL, SL - SM, eps), eps)
        rhoR_star = self._clamp_pos(rhoR * self._safe_div(SR - uR, SR - SM, eps), eps)

        # tangential star states (Alfvén jumps)
        def tangential_star(vt1, vt2, Bt1, Bt2, rho, un, S):
            denom_side = self._clamp_pos(rho * (S - SM), eps)
            vt1s = vt1 - Bn_star * Bt1 / denom_side
            vt2s = vt2 - Bn_star * Bt2 / denom_side
            Bt1s = Bt1 * self._safe_div(S - un, S - SM, eps)
            Bt2s = Bt2 * self._safe_div(S - un, S - SM, eps)
            return vt1s, vt2s, Bt1s, Bt2s

        vt1L_s, vt2L_s, Bt1L_s, Bt2L_s = tangential_star(vt1L, vt2L, Bt1L, Bt2L, rhoL, uL, SL)
        vt1R_s, vt2R_s, Bt1R_s, Bt2R_s = tangential_star(vt1R, vt2R, Bt1R, Bt2R, rhoR, uR, SR)

        # conservative star states and star fluxes
        def cons_star(W, U, rho_s, vt1s, vt2s, Bn_s, Bt1s, Bt2s, S):
            # build axis-frame primitives
            un_s = SM
            u_s, v_s, w_s, B1_s, B2_s, B3_s = unrot(un_s, vt1s, vt2s, Bn_s, Bt1s, Bt2s)

            # energy from primitives and fields
            rho_s = self._clamp_pos(rho_s, eps)
            ke = 0.5 * rho_s * (u_s*u_s + v_s*v_s + w_s*w_s)
            me = 0.5 * (B1_s*B1_s + B2_s*B2_s + B3_s*B3_s)
            p  = W[E_i]
            e_int = eq.get_specific_energy(p, rho_s) * rho_s
            E_s = ke + me + e_int

            U_s = jnp.zeros_like(U)
            U_s = U_s.at[rho_i].set(rho_s)
            U_s = U_s.at[u_i].set(rho_s * u_s)
            U_s = U_s.at[v_i].set(rho_s * v_s)
            U_s = U_s.at[w_i].set(rho_s * w_s)
            U_s = U_s.at[Bx_i].set(B1_s)
            U_s = U_s.at[By_i].set(B2_s)
            U_s = U_s.at[Bz_i].set(B3_s)
            U_s = U_s.at[E_i].set(E_s)

            # star flux from side: F* = F + S (U* - U)
            F_side = eq.get_fluxes_xi(W, U, axis)
            F_star = F_side + S * (U_s - U)
            return U_s, F_star

        UL_star, FL_star = cons_star(WL, UL, rhoL_star, vt1L_s, vt2L_s, Bn_star, Bt1L_s, Bt2L_s, SL)
        UR_star, FR_star = cons_star(WR, UR, rhoR_star, vt1R_s, vt2R_s, Bn_star, Bt1R_s, Bt2R_s, SR)

        # select flux by region
        # regions: 0<SL => FL ; SL<=0<SM => FL* ; SM<=0<SR => FR* ; 0>=SR => FR
        flux_sel = jnp.where(0.0 <= SL, FL,
                     jnp.where(0.0 <= SM, FL_star,
                       jnp.where(0.0 < SR, FR_star, FR)))

        # degeneracy detection → fallback to HLL
        degenerate = (jnp.abs(SR - SL) < 100*eps) \
                     | jnp.isnan(SM) | jnp.isinf(SM) \
                     | (rhoL_star <= eps) | (rhoR_star <= eps)
        flux = jnp.where(degenerate, FHLL, flux_sel)
 #       flux = jnp.nan_to_num(flux, 0.0, 0.0, 0.0)

        return flux, None, None
