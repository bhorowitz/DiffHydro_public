from jax import Array 
from functools import partial
from typing import List
from .solver import recon
import jax.numpy as jnp

class ConductiveFlux:
    def __init__(self,
                EquationManager,
                 Solver,
                 Recon,
                 positivity = False
                ):
        self.eq_manage = EquationManager
        self.solver = Solver
        self.recon = Recon
        
        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        self.dx_o = 1
        
        try: #3d
            self.flux_shapes = (5,EquationManager.mesh_shape[0],EquationManager.mesh_shape[1],EquationManager.mesh_shape[2])
        except: #2d, z velocity axis usually just constant...
            self.flux_shapes = (5,EquationManager.mesh_shape[0],EquationManager.mesh_shape[1])

        
    def flux(self,sol,ax,params):
        primitives = self.eq_manage.get_primitives_from_conservatives(sol)
        
        primitives_xi_L = self.recon.reconstruct_xi(
            primitives,
            axis=ax,
            j=0)
        
        primitives_xi_R = self.recon.reconstruct_xi(
            primitives,
            axis=ax,
            j=1)


        conservative_xi_L = self.eq_manage.get_conservatives_from_primitives(primitives_xi_L)
        conservative_xi_R = self.eq_manage.get_conservatives_from_primitives(primitives_xi_R)

        if self.positivity:
            conservative_xi_L, primitives_xi_L, count_L = self.compute_positivity_preserving_interpolation(
                primitives=primitives,
                primitives_xi_j=primitives_xi_L,
                j=0,
                axis=ax)
            conservative_xi_R, primitives_xi_R, count_R = self.compute_positivity_preserving_interpolation(
                primitives=primitives,
                primitives_xi_j=primitives_xi_R,
                j=1,
                axis=ax)
        #final axis is from vel-id, so one off... should standardize at some point!
        flux,_,_ = self.solver.solve_riemann_problem_xi(primitives_xi_L,primitives_xi_R,conservative_xi_L,conservative_xi_R,ax-1)
        return flux
    

    def timestep(self,sol):
        print(sol)
        v = jnp.abs(sol[1:-1]/sol[0])
    
        temp_quant = (self.eq_manage.gamma-1)*(sol[-1]-sol[0]*jnp.sum(v**2.0,axis=0)/2.0)
        P = jnp.maximum(jnp.where(temp_quant>0,temp_quant,0),0.0)
    
        cs = jnp.sqrt(self.eq_manage.gamma*P/sol[0])
    
        #cmax = jnp.nanmax(jnp.nanmax(v)+cs)
        cmax = jnp.max(jnp.max(v)+cs)
        dt = self.eq_manage.cfl*self.dx_o/cmax
        return dt
    
    def compute_positivity_preserving_interpolation(self,
            primitives: Array,
            primitives_xi_j: Array,
            j: int,
            axis: int):
        
        cell_state_xi_safe_j = self.positivity_stencil.reconstruct_xi(
            primitives, axis, j)

        rho_j = primitives_xi_j[self.eq_manage.mass_ids]
        
        mask = jnp.where(rho_j < self.eq_manage.eps, 0, 1)
        counter = jnp.sum(1 - mask)    # TODO check in parallel

        primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

        # CHECK ENERGY / PRESSURE
        p_j = primitives_xi_j[self.eq_manage.energy_ids]
        
        # OPTION 1 - CHECK PRESSURE DIRECTLY
        mask = jnp.where(p_j < self.eq_manage.eps, 0, 1)

        # OPTION 2 - CHECK VIA INTERNAL ENERGY
        # rhoe_j = rho_j * self.material_manager.get_specific_energy(p_j, rho=rho_j)
        # mask_j = jnp.where(rhoe_j - pb < self.eps.pressure, 0, 1)

        counter += jnp.sum(1 - mask)    # TODO check in parallel
        primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

        conservative_xi_j = self.eq_manage.get_conservatives_from_primitives(primitives_xi_j)

        return conservative_xi_j,primitives_xi_j,counter