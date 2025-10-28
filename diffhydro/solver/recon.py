import jax
from jax import Array 
from functools import partial
from typing import List
from .limiter import LIMITER_DICT
import jax.numpy as jnp
#Routines adapted from JAX-FLUIDS, possibly in the future will integrate there entire package...

class TENO5_alt:
    """
    Fu et al. (2016): Targeted ENO, 5th-order.
    - Correct L/R face conventions and stencil alignment.
    - Sharp cutoff (delta_k) with robust fallback if all stencils are rejected.
    - Safer eps default for FP32.
    API: reconstruct_xi(buffer, axis, j, dx=None, **kwargs)
         j=0 -> LEFT state at i+1/2 (from cell i)
         j=1 -> RIGHT state at i+1/2 (from cell i+1)
    """

    def __init__(
        self,
        linear_weights=(0.05, 0.55, 0.40),  # optimized spectral 
        C=1.0,
        q=6,
        CT=1e-5,
        eps=1e-12
    ):
        # Linear (optimal) weights; set to (0.1, 0.6, 0.3) for classical 5th-order
        self.dr_ = linear_weights
        self.C = C
        self.q = q
        self.CT = CT
        self.eps = eps

        # 3 quadratic candidate polynomials (WENO5/TENO5 standard)
        self.cr_ = (
            ( 1/3, -7/6, 11/6),
            (-1/6,  5/6,  1/3),
            ( 1/3,  5/6, -1/6),
        )

    def _smoothness(self, s0, s1, s2, s3, s4):
        # Jiang–Shu betas (WENO5)
        beta_0 = (13.0/12.0)*(s0 - 2*s1 + s2)**2 + 0.25*(s0 - 4*s1 + 3*s2)**2
        beta_1 = (13.0/12.0)*(s1 - 2*s2 + s3)**2 + 0.25*(s1 - s3)**2
        beta_2 = (13.0/12.0)*(s2 - 2*s3 + s4)**2 + 0.25*(3*s2 - 4*s3 + s4)**2
        tau_5 = jnp.abs(beta_0 - beta_2)
        return beta_0, beta_1, beta_2, tau_5

    def _candidates(self, s0, s1, s2, s3, s4):
        # Three 3-point polynomials at x_{i+1/2}
        p0 = self.cr_[0][0]*s0 + self.cr_[0][1]*s1 + self.cr_[0][2]*s2
        p1 = self.cr_[1][0]*s1 + self.cr_[1][1]*s2 + self.cr_[1][2]*s3
        p2 = self.cr_[2][0]*s2 + self.cr_[2][1]*s3 + self.cr_[2][2]*s4
        return p0, p1, p2

    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx: float = None, **kwargs) -> Array:
        # --- 5-point windows per face ---
        # j=0: LEFT state at i+1/2, from cell i  -> [i-2, i-1, i, i+1, i+2]
        # j=1: RIGHT state at i+1/2, from cell i+1 -> [i-1, i, i+1, i+2, i+3]
        if j == 0:
            s0 = jnp.roll(buffer, -2, axis=axis)
            s1 = jnp.roll(buffer, -1, axis=axis)
            s2 = buffer
            s3 = jnp.roll(buffer, +1, axis=axis)
            s4 = jnp.roll(buffer, +2, axis=axis)
        elif j == 1:
            s0 = jnp.roll(buffer, -1, axis=axis)
            s1 = buffer
            s2 = jnp.roll(buffer, +1, axis=axis)
            s3 = jnp.roll(buffer, +2, axis=axis)
            s4 = jnp.roll(buffer, +3, axis=axis)
        else:
            raise ValueError("TENO5.reconstruct_xi expects j in {0,1}")

        # --- smoothness, indicators, and sharp cutoff ---
        beta0, beta1, beta2, tau5 = self._smoothness(s0, s1, s2, s3, s4)

        gamma0 = (self.C + tau5 / (beta0 + self.eps))**self.q
        gamma1 = (self.C + tau5 / (beta1 + self.eps))**self.q
        gamma2 = (self.C + tau5 / (beta2 + self.eps))**self.q

        gsum = gamma0 + gamma1 + gamma2
        # probabilities
        pi0 = gamma0 / (gsum + self.eps)
        pi1 = gamma1 / (gsum + self.eps)
        pi2 = gamma2 / (gsum + self.eps)

        # sharp cutoff δ_k
        d0 = jnp.where(pi0 < self.CT, 0.0, 1.0)
        d1 = jnp.where(pi1 < self.CT, 0.0, 1.0)
        d2 = jnp.where(pi2 < self.CT, 0.0, 1.0)

        # targeted linear weights
        w0 = d0 * self.dr_[0]
        w1 = d1 * self.dr_[1]
        w2 = d2 * self.dr_[2]
        wsum = w0 + w1 + w2

        # robust fallback: if all cut, revert to linear weights
        all_cut = (wsum <= 0.0)
        w0 = jnp.where(all_cut, self.dr_[0], w0)
        w1 = jnp.where(all_cut, self.dr_[1], w1)
        w2 = jnp.where(all_cut, self.dr_[2], w2)
        wsum = jnp.where(all_cut, 1.0, wsum)

        om0 = w0 / wsum
        om1 = w1 / wsum
        om2 = w2 / wsum

        # --- three candidates and final blend ---
        p0, p1, p2 = self._candidates(s0, s1, s2, s3, s4)
        face_state = om0 * p0 + om1 * p1 + om2 * p2
        return face_state

class TENO5():
    ''' Fu et al. - 2016 -  A family of high-order targeted ENO schemes for compressible-fluid simulations'''    
    
    def __init__(self):
        #NOT WORKING!
        
        # Coefficients for 5-th order convergence
        # self.dr_ = [1/10, 6/10, 3/10]
        # Coefficients for optimized spectral properties
        
        self.dr_ = [0.05, 0.55, 0.40]

        self.eps = 1E-20
        
        self.cr_ = [
            [1/3, -7/6, 11/6], 
            [-1/6, 5/6, 1/3], 
            [1/3, 5/6, -1/6]
        ]

        self.C = 1.0
        self.q = 6
        self.CT = 1e-5

    #    self._stencil_size = 6
    #    self.array_slices([range(-3, 2, 1), range(2, -3, -1)])
     #   self.stencil_slices([range(0, 5, 1), range(5, 0, -1)])

    def reconstruct_xi(self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs) -> Array:
      #  s1_ = self.s_[j][axis]
        adj = 0
       # if j == 0 or j== 1:
       #     s_0 = jnp.roll(buffer,-2+adj, axis=axis)
       #     s_1 = jnp.roll(buffer,-1+adj, axis=axis)
       #     s_2 = jnp.roll(buffer,adj, axis=axis)
       #     s_3 = jnp.roll(buffer,1+adj, axis=axis)
       #     s_4 = jnp.roll(buffer,2+adj, axis=axis)

        #need to test...
        if j == 0:  # LEFT state at i+1/2, from cell i
            # [i-2, i-1, i, i+1, i+2]
            s_0 = jnp.roll(buffer, -2, axis=axis)
            s_1 = jnp.roll(buffer, -1, axis=axis)
            s_2 = buffer
            s_3 = jnp.roll(buffer, +1, axis=axis)
            s_4 = jnp.roll(buffer, +2, axis=axis)
        
        elif j == 1:  # RIGHT state at i+1/2, from cell i+1
            # [i-1, i, i+1, i+2, i+3]
            s_0 = jnp.roll(buffer, -1, axis=axis)
            s_1 = buffer
            s_2 = jnp.roll(buffer, +1, axis=axis)
            s_3 = jnp.roll(buffer, +2, axis=axis)
            s_4 = jnp.roll(buffer, +3, axis=axis)
        else:
            raise ValueError("TENO5.reconstruct_xi expects j in {0,1}")        
            #could maybe vectorize this nicer...
        beta_0 = 13.0 / 12.0 * (s_0 - 2 * s_1 + s_2) \
            * (s_0 - 2 * s_1 + s_2) \
            + 1.0 / 4.0 * (s_0 - 4 * s_1 + 3 * s_2) \
            * (s_0 - 4 * s_1 + 3 * s_2)
        beta_1 = 13.0 / 12.0 * (s_1 - 2 * s_2 + s_3) \
            * (s_1 - 2 * s_2 + s_3) \
            + 1.0 / 4.0 * (s_1 - s_3) * (s_1 - s_3)
        beta_2 = 13.0 / 12.0 * (s_2 - 2 * s_3 + s_4) \
            * (s_2 - 2 * s_3 + s_4) \
            + 1.0 / 4.0 * (3 * s_2 - 4 * s_3 + s_4) \
            * (3 * s_2 - 4 * s_3 + s_4)

        tau_5 = jnp.abs(beta_0 - beta_2)

        # SMOOTHNESS MEASURE
        gamma_0 = (self.C + tau_5 / (beta_0 + self.eps))**self.q
        gamma_1 = (self.C + tau_5 / (beta_1 + self.eps))**self.q
        gamma_2 = (self.C + tau_5 / (beta_2 + self.eps))**self.q


        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2)

        # SHARP CUTOFF FUNCTION
        delta_0 = jnp.where(gamma_0 * one_gamma_sum < self.CT, 0, 1)
        delta_1 = jnp.where(gamma_1 * one_gamma_sum < self.CT, 0, 1)
        delta_2 = jnp.where(gamma_2 * one_gamma_sum < self.CT, 0, 1)

        w0 = delta_0 * self.dr_[0]
        w1 = delta_1 * self.dr_[1]
        w2 = delta_2 * self.dr_[2]

        # TODO eps should not be necessary
        one_dk = 1.0 / (w0 + w1 + w2 + self.eps)

        omega_0 = w0 * one_dk 
        omega_1 = w1 * one_dk 
        omega_2 = w2 * one_dk 

        p_0 = self.cr_[0][0] * s_0 + self.cr_[0][1] * s_1 + self.cr_[0][2] * s_2
        p_1 = self.cr_[1][0] * s_1 + self.cr_[1][1] * s_2 + self.cr_[1][2] * s_3
        p_2 = self.cr_[2][0] * s_2 + self.cr_[2][1] * s_3 + self.cr_[2][2] * s_4

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2
        return cell_state_xi_j


class WENO1:
    def __init__(self):
        pass
    
    def reconstruct_xi(self,buffer: Array, axis: int, j: int, dx = None, **kwargs) -> Array:
        cell_state_xi_j = jnp.roll(buffer,-j,axis=axis)    
        return cell_state_xi_j


class MUSCL3:
    def __init__(self,limiter: str):
        self.limiter = LIMITER_DICT[limiter]
        
    def f_sten(self,idd,j):
        #clunky, can probably refactor out
        return idd-j-1
        
    def reconstruct_xi(self,
                buffer: Array,
                axis: int,
                j: int = None,
            ) -> Array:
        
        """MUSCL-type reconstruction with different limiters.
    
        psi_{i+1/2}^L = psi_i     + 0.5 * phi(r_L) * (psi_{i} - psi_{i-1})
        psi_{i+1/2}^R = psi_{i+1} - 0.5 * phi(r_R) * (psi_{i+2} - psi_{i+1})
    
        r_L = (phi_{i+1} - phi_{i}) / (phi_{i} - phi_{i-1})
        r_R = (phi_{i+1} - phi_{i}) / (phi_{i+2} - phi_{i+1})
    
        """
        eps_ad = 1e-6
        if j == 0: #left
                delta_central = jnp.roll(buffer,self.f_sten(2,j), axis=axis) - jnp.roll(buffer,self.f_sten(1,j), axis=axis)
                delta_upwind =  jnp.roll(buffer,self.f_sten(1,j), axis=axis) - jnp.roll(buffer,self.f_sten(0,j), axis=axis)
                r = jnp.where(
                    delta_upwind >= eps_ad,
                    delta_central / (delta_upwind + eps_ad), 
                    (delta_central + eps_ad) / (delta_upwind + eps_ad))
                limiter = self.limiter(r)
                cell_state_xi_j = jnp.roll(buffer,0,axis=axis) + 0.5 * limiter * (jnp.roll(buffer,-1,axis=axis) - jnp.roll(buffer,0,axis=axis)) 
        if j == 1: #right
                delta_central = -1* jnp.roll(buffer,self.f_sten(2,j), axis=axis) + jnp.roll(buffer,self.f_sten(1,j), axis=axis)
                delta_upwind =  -1* jnp.roll(buffer,self.f_sten(1,j), axis=axis) + jnp.roll(buffer,self.f_sten(0,j), axis=axis)
    
                r = jnp.where(
                    delta_upwind >= eps_ad**2, 
                    delta_central / (delta_upwind + eps_ad), 
                    (delta_central + eps_ad) / (delta_upwind + eps_ad**2))
                limiter = self.limiter(r)
                cell_state_xi_j = jnp.roll(buffer,-1,axis=axis) - 0.5 * limiter * (jnp.roll(buffer,-2,axis=axis) - jnp.roll(buffer,-1,axis=axis))
        return cell_state_xi_j


class PPM:
    """
    Piecewise Parabolic Method (Colella–Woodward style, simplified) with MUSCL-style
    slope limiting via LIMITER_DICT, optional contact steepening, and final
    monotonic face clipping. Mirrors MUSCL3.reconstruct_xi(buffer, axis, j).
    """

    def __init__(self,
                 limiter: str = "MC",      # any key in LIMITER_DICT (e.g. VANLEER, SUPERBEE, MC, ...)
                 use_clip: bool = True,
                 steepen: bool = True,
                 rho_idx: int | None = None,
                 p_idx: int | None = None,
                 vel_ids: tuple[int, int, int] | None = None,
                 chi0: float = 0.5,
                 chi1: float = 1.5,
                 beta: float = 0.25,
                 eps: float = 1e-12):
        self.limiter_fun = LIMITER_DICT[limiter]  # same pattern as MUSCL3 :contentReference[oaicite:3]{index=3}
        self.use_clip = use_clip
        self.steepen  = steepen
        self.rho_idx  = rho_idx
        self.p_idx    = p_idx
        self.vel_ids  = vel_ids
        self.chi0 = chi0
        self.chi1 = chi1
        self.beta = beta
        self.eps  = eps

    def reconstruct_xi(self, buffer: Array, axis: int, j: int = None) -> Array:
        """
        buffer: primitives (all comps), shape [nvar, ...]
        axis:   sweep axis (same convention as MUSCL3)
        j:      0 => left state at i+1/2 (from cell i)
                1 => right state at i+1/2 (from cell i+1)
        """
        # rolling stencil
        qim2 = jnp.roll(buffer, +2, axis=axis)
        qim1 = jnp.roll(buffer, +1, axis=axis)
        qi   = buffer
        qip1 = jnp.roll(buffer, -1, axis=axis)
        qip2 = jnp.roll(buffer, -2, axis=axis)

        # --- MUSCL-style limited slopes using phi(r) ---
        eps_ad = 1e-6

        # Ratios for the "i" and "i+1" cells (centered around each face):
        # r_i   = (q_{i+1}-q_i)   / (q_i - q_{i-1})
        # r_ip1 = (q_{i+2}-q_{i+1}) / (q_{i+1} - q_i)
        num_i   = (qip1 - qi)
        den_i   = (qi - qim1)
        r_i = jnp.where(den_i >= eps_ad, num_i / (den_i + eps_ad),
                        (num_i + eps_ad) / (den_i + eps_ad))

        num_ip1 = (qip2 - qip1)
        den_ip1 = (qip1 - qi)
        r_ip1 = jnp.where(den_ip1 >= eps_ad, num_ip1 / (den_ip1 + eps_ad),
                          (num_ip1 + eps_ad) / (den_ip1 + eps_ad))

        phi_i   = self.limiter_fun(r_i)
        phi_ip1 = self.limiter_fun(r_ip1)

        # Limited one-sided slopes (MUSCL-like)
        # delta_i   ~ phi(r_i)   * (q_i   - q_{i-1})
        # delta_ip1 ~ phi(r_ip1) * (q_{i+2} - q_{i+1})  (note: right-biased per MUSCL3) :contentReference[oaicite:5]{index=5}
        delta_i   = phi_i   * (qi   - qim1)
        delta_ip1 = phi_ip1 * (qip2 - qip1)

        # --- PPM 4th-order face prediction with curvature correction ---
        curv_i   = (qip1 - qi)   - (qi   - qim1)
        curv_ip1 = (qip2 - qip1) - (qip1 - qi)

        # Left state at i+1/2 from cell i, Right state at i+1/2 from cell i+1
        qL_iphalf = qi   + 0.5*delta_i   - (1.0/6.0)*curv_i
        qR_iphalf = qip1 - 0.5*delta_ip1 + (1.0/6.0)*curv_ip1

        # --- monotonic bracketing at the face (TVD-ish safety) ---
        if self.use_clip:
            qmin = jnp.minimum(qi, qip1)
            qmax = jnp.maximum(qi, qip1)
            qL_iphalf = jnp.clip(qL_iphalf, qmin, qmax)
            qR_iphalf = jnp.clip(qR_iphalf, qmin, qmax)

        # --- optional contact steepening (uses rho/p/u along sweep axis) ---
        if self.steepen and (self.rho_idx is not None) and (self.p_idx is not None) and (self.vel_ids is not None):
            ax0 = axis - 1  # choose velocity component aligned with this sweep
            rho_im1 = jnp.roll(buffer[self.rho_idx], +1, axis=axis)
            rho_i   = buffer[self.rho_idx]
            rho_ip1 = jnp.roll(buffer[self.rho_idx], -1, axis=axis)

            p_im1 = jnp.roll(buffer[self.p_idx], +1, axis=axis)
            p_i   = buffer[self.p_idx]
            p_ip1 = jnp.roll(buffer[self.p_idx], -1, axis=axis)

            u_im1 = jnp.roll(buffer[self.vel_ids[ax0]], +1, axis=axis)
            u_ip1 = jnp.roll(buffer[self.vel_ids[ax0]], -1, axis=axis)

            chi = jnp.abs(rho_ip1 - rho_im1) / (jnp.minimum(jnp.minimum(rho_im1, rho_i), rho_ip1) + self.eps)
            pmax = jnp.maximum(jnp.maximum(p_im1, p_i), p_ip1)
            dpp  = (p_ip1 - p_im1)
            phi_p = jnp.exp(- (dpp*dpp) / (self.beta * (pmax + self.eps) * (pmax + self.eps)))
            gu = ((u_ip1 - u_im1) < 0.0).astype(buffer.dtype)

            s = (chi - self.chi0) / (self.chi1 - self.chi0)
            s = jnp.clip(s, 0.0, 1.0) * phi_p * gu

            qL_iphalf = (1.0 - s) * qL_iphalf + s * qi
            qR_iphalf = (1.0 - s) * qR_iphalf + s * qip1

            if self.use_clip:
                qmin = jnp.minimum(qi, qip1)
                qmax = jnp.maximum(qi, qip1)
                qL_iphalf = jnp.clip(qL_iphalf, qmin, qmax)
                qR_iphalf = jnp.clip(qR_iphalf, qmin, qmax)

        if j == 0:
            return qL_iphalf
        elif j == 1:
            return qR_iphalf
        else:
            raise ValueError("PPM.reconstruct_xi expects j in {0,1}")
