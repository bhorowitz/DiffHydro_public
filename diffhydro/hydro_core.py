from jax import Array 
from functools import partial
from typing import List
import jax.numpy as np
from diffhydro import NoBoundary, NoForcing
import jax
from jax import jit
import jax.numpy as jnp

@jax.tree_util.register_pytree_node_class
class hydro:
    #TO DO, pretty up this area...
    def __init__(self,
                 n_super_step = 600,
                 max_dt = 0.5, 
                 boundary = NoBoundary,
                 snapshots = False,
                splitting_schemes=[[3,1,2,2,1,3],[1,2,3,3,2,1],[2,3,1,1,3,2]], #cyclic permutations
                fluxes = None, #convection, conduction
                forces = [NoForcing], #gravity, etc.
                maxjit=False):
        #parameters that are held constant per run (i.e. probably don't want to take derivatives with respect to...)
   #     self.init_dt = init_dt # tiny starting timestep to smooth out anything too sharp
        self.splitting_schemes = splitting_schemes #strang splitting for x,y,z sweeps
        self.max_dt = max_dt
        self.boundary = boundary
        #supersteps, each superstep has len(splitting_schemes) time steps
        self.n_super_step = n_super_step
        self.snapshots = snapshots #poorly names/
        self.outputs = []
        self.fluxes = fluxes
        self.forces = forces
        self.maxjit = maxjit
        self.dx_o = 1.0
        self.timescale = jnp.zeros(self.n_super_step)

    
    def timestep(self,fields):
        dt = []
        for flux in self.fluxes:
            dt.append(flux.timestep(fields))
        for force in self.forces:
            dt.append(force.timestep(fields))
        print(dt)
        return jnp.min(jnp.array(dt))
    
    def flux(self,sol,ax,params):
        total_flux = jnp.zeros(sol.shape)
        for flux in self.fluxes:
            total_flux += flux.flux(sol,ax,params)
        return total_flux
    
    def forcing(self,i,sol,params): #all axis independant? 
        total_force = jnp.zeros(sol.shape)
        for force in self.forces:
            total_force += force.force(i,sol,params)
        return total_force
    
    def solve_step(self,sol,dt,ax,params):
        ##RK2 method
        
        fu1 = self.flux(sol,ax,params) 
        #first order upwind
        rhs_cons = (fu1 - jnp.roll(fu1, 1, axis=ax)) 
        
        u1 = sol - rhs_cons * dt / (2.0 * self.dx_o)
        #second order step

        fu = self.flux(sol,ax,params)  
            
        rhs_cons = (fu - jnp.roll(fu, 1, axis=ax))  #fu or fu1?
        
        sol = sol - (rhs_cons) * dt / self.dx_o
        return sol
    
    @jax.checkpoint
    def sweep_stack(self,state,dt,i):
        sol,params = state
        for scheme in self.splitting_schemes:
            for nn,ax in enumerate(scheme):
                sol = self.boundary.impose(sol,ax)
                sol = self.solve_step(sol,dt/len(scheme)*2,int(ax),params)                 
                # experimental
                sol = sol.at[0].set(jnp.abs(sol[0])) #experimental...
                sol = sol.at[-1].set(jnp.abs(sol[-1])) #experimental...
    
        return sol
    
   # @jax.jit
    def evolve(self,input_fields,params):
        self.outputs=[]
        #main loop
        state = (input_fields,params)

        #need to rework the UI to get out snapshots from jitted function, hack for now...
        if self.maxjit:
            state  = jax.lax.fori_loop(0, self.n_super_step, self.hydrostep_adapt, state)
        else:
            for i in range(0,self.n_super_step):
                state = self.hydrostep_adapt(i,state)
                if self.snapshots:
                    if i%self.snapshots==0: #comment out most times...
                        self.outputs.append(state)
        return state
        
    @partial(jit, static_argnums=0)
    def hydrostep_adapt(self,i,state):
        fields,params = state
        print("f",fields)
        ttt = self.timestep(fields)
        ttt = jnp.minimum(self.max_dt,ttt)
        dt = (ttt)
        return self._hydrostep(i,state,dt)
    
    @jax.jit
    def _hydrostep(self,i,state,dt):
        fields,params = state

        #save actual timescale used, mostly important if you are using hydro_adapt
#        self.timescale[i].set(dt)
        
        hydro_output = self.sweep_stack(state,dt,i)
        
        fields = hydro_output

        fields = self.forcing(i,fields,params)
            
        return (fields,params)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def tree_flatten(self):
        #this method is needed for JAX control flow
        children = ()  # arrays / dynamic values
        aux_data = {
                    "boundary":self.boundary,
                    "snapshots":self.snapshots,
                   "splitting_schemes":self.splitting_schemes,
                    "fluxes":self.fluxes,"forces":self.forces,"maxjit":self.maxjit}  # static values
        return (children, aux_data)