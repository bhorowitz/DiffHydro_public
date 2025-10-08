import jax
from jax import Array 
from functools import partial
from typing import List
from .limiter import LIMITER_DICT
import jax.numpy as jnp
#Routines adapted from JAX-FLUIDS, possibly in the future will integrate there entire package...

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

        #SOMETHING IS WRONG HERE!
        if j== 0: #right
            s_0 = jnp.roll(buffer,-2, axis=axis)
            s_1 = jnp.roll(buffer,-1, axis=axis)
            s_2 = jnp.roll(buffer,0, axis=axis)
            s_3 = jnp.roll(buffer,1, axis=axis)
            s_4 = jnp.roll(buffer,2, axis=axis)
        if j== 1:  #left
            add = 0
            s_0 = jnp.roll(buffer,-3+add, axis=axis)
            s_1 = jnp.roll(buffer,-2+add, axis=axis)
            s_2 = jnp.roll(buffer,-1+add, axis=axis)
            s_3 = jnp.roll(buffer,0+add, axis=axis)
            s_4 = jnp.roll(buffer,1+add, axis=axis)
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
        eps_ad = 1e-10
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
