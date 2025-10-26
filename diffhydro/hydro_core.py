from jax import Array 
from functools import partial
from typing import List
import jax.numpy as np
from diffhydro import NoBoundary, NoForcing
import jax
from jax import jit
import jax.numpy as jnp
from .solver.integrator import INTEGRATOR_DICT



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
                forces = [NoForcing()], #gravity, etc.
                maxjit=False,
                use_mol=False,
                 use_ctu=False,
                integrator="RK2"):
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
        self.use_mol = use_mol
        self.integrator = INTEGRATOR_DICT[integrator]  # callable
        self.use_ctu = use_ctu
        print("using CTU?",use_ctu)
    
    def timestep(self,fields):
        dt = []
        for flux in self.fluxes:
            dt.append(flux.timestep(fields))
        for force in self.forces:
            dt.append(force.timestep(fields))
        print("dt",dt)
        return jnp.min(jnp.array(dt))
    
    def flux(self,sol,ax,params):
        total_flux = jnp.zeros(sol.shape)
        for flux in self.fluxes: 
            #note it is ordered, to allow a flux_correction depending on calculated fluxes
            #make sure your order is correct for that though!
            total_flux += flux.flux(sol,ax,params,total_flux)
        return total_flux
    
    def forcing(self,i,sol,params,dt): #all axis independant? 
        for force in self.forces:
            sol = force.force(i, sol, params, dt)  # each returns UPDATED fields
        return sol
        
    def split_solve_step(self,sol,dt,ax,params):
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
                sol = self.split_solve_step(sol,dt/len(scheme)*2,int(ax),params)                 
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

        fields = self.forcing(i,fields,params,dt)
            
        return (fields,params)

    @jax.jit
    def _hydrostep(self, i, state, dt):
        
        #split forcing outside of core hydro loop
        
        fields, params = state
        
        fields = self.forcing(i, fields, params, dt/2)          

        if self.use_mol:
            fields = self.mol_solve_step(fields, dt, params)  # <<< unsplit RK2
        else:
            fields = self.sweep_stack(state, dt, i)           #
            
        fields = self.forcing(i, fields, params, dt/2)          
        return (fields, params)
    def rhs_unsplit(self, sol, params):
        rhs = jnp.zeros_like(sol)
        # assume first axis is variable index; spatial axes start at 1
        for ax in range(1, sol.ndim):
            sol_b = self.boundary.impose(sol, ax)            
            fu = self.flux(sol_b, ax, params)                
            rhs = rhs - (fu - jnp.roll(fu, 1, axis=ax)) / self.dx_o
        return rhs

    def mol_solve_step(self, sol, dt, params):
        if getattr(self, "use_ctu", False):
            # pass dt into rhs via a closure
            return self.integrator(lambda u, p: self.rhs_ctu(u, p, dt), sol, dt, params)
        else:
            return self.integrator(self.rhs_unsplit, sol, dt, params)  # existing path  :contentReference[oaicite:1]{index=1}

    def rhs_ctu(self, sol, params, dt):
        print("CTU")
        #THE MONSTER! Need to figure out a better org for it...
        """
        3D CTU(+CT) RHS for MHD (and GLM-MHD).
        Reuses:
          - recon.reconstruct_xi()
          - solver.solve_riemann_problem_xi()
          - eq.get_fluxes_xi()
        """
        eq     = self.fluxes[0].eq_manage
        recon  = self.fluxes[0].recon
        solver = self.fluxes[0].solver
        dx     = self.dx_o
        dt_dx  = dt / dx
    
        Uc = sol
        for ax in range(1, Uc.ndim):
            Uc = self.boundary.impose(Uc, ax)
        Wc = eq.get_primitives_from_conservatives(Uc)
    
        def phys_flux(W, U, axis0):
            return eq.get_fluxes_xi(W, U, axis0)
    
        def riemann(WL, WR, UL, UR, axis0):
            F, _, _ = solver.solve_riemann_problem_xi(WL, WR, UL, UR, axis0)
            return F
    
        def reconstruct_faces(W, axis, j):
            return recon.reconstruct_xi(W, axis=axis, j=j)
    
        glm_on = getattr(eq, "use_glm", False)
        def slice8(U): return U[:8] if (glm_on and U.shape[0] == 9) else U
    
        # Reconstruct along each axis (x=1,y=2,z=3)
        WLx, WRx = reconstruct_faces(Wc, 1, 0), reconstruct_faces(Wc, 1, 1)
        WLy, WRy = reconstruct_faces(Wc, 2, 0), reconstruct_faces(Wc, 2, 1)
        WLz, WRz = reconstruct_faces(Wc, 3, 0), reconstruct_faces(Wc, 3, 1)
    
        ULx, URx = eq.get_conservatives_from_primitives(WLx), eq.get_conservatives_from_primitives(WRx)
        ULy, URy = eq.get_conservatives_from_primitives(WLy), eq.get_conservatives_from_primitives(WRy)
        ULz, URz = eq.get_conservatives_from_primitives(WLz), eq.get_conservatives_from_primitives(WRz)
    
        # MUSCL–Hancock half-time prediction (normal only)
        def half_time_predict(WL, WR, UL, UR, axis0):
            F_L = phys_flux(WL, UL, axis0)
            F_R = phys_flux(WR, UR, axis0)
            W_Lh = WL - 0.5 * dt_dx * (F_R - F_L)
            W_Rh = WR - 0.5 * dt_dx * (F_R - F_L)
            return W_Lh, W_Rh
    
        WLx_h, WRx_h = half_time_predict(WLx, WRx, ULx, URx, 0)
        WLy_h, WRy_h = half_time_predict(WLy, WRy, ULy, URy, 1)
        WLz_h, WRz_h = half_time_predict(WLz, WRz, ULz, URz, 2)
    
        # First-pass Riemann on half-time face states
        ULx_h, URx_h = eq.get_conservatives_from_primitives(WLx_h), eq.get_conservatives_from_primitives(WRx_h)
        ULy_h, URy_h = eq.get_conservatives_from_primitives(WLy_h), eq.get_conservatives_from_primitives(WRy_h)
        ULz_h, URz_h = eq.get_conservatives_from_primitives(WLz_h), eq.get_conservatives_from_primitives(WRz_h)
    
        Fx_h = riemann(WLx_h, WRx_h, slice8(ULx_h), slice8(URx_h), 0)
        Fy_h = riemann(WLy_h, WRy_h, slice8(ULy_h), slice8(URy_h), 1)
        Fz_h = riemann(WLz_h, WRz_h, slice8(ULz_h), slice8(URz_h), 2)
    
        # CTU transverse corrections (each face corrected by fluxes from other 2 dirs)
        def ctu_correct(WL_h, WR_h, F_a, F_b, ax_a, ax_b):
            dFa_da = (jnp.roll(F_a, -1, axis=ax_a+1) - jnp.roll(F_a, 1, axis=ax_a+1)) * (0.5/dx)
            dFb_db = (jnp.roll(F_b, -1, axis=ax_b+1) - jnp.roll(F_b, 1, axis=ax_b+1)) * (0.5/dx)
            WL_ctu = WL_h - 0.5 * dt * (dFa_da + dFb_db)
            WR_ctu = WR_h - 0.5 * dt * (dFa_da + dFb_db)
            return WL_ctu, WR_ctu
            
        WLx_ctu, WRx_ctu = ctu_correct(WLx_h, WRx_h, Fy_h, Fz_h, 1, 2)
        WLy_ctu, WRy_ctu = ctu_correct(WLy_h, WRy_h, Fz_h, Fx_h, 2, 0)
        WLz_ctu, WRz_ctu = ctu_correct(WLz_h, WRz_h, Fx_h, Fy_h, 0, 1)
    
        # Final Riemann on CTU-corrected states
        ULx_ctu, URx_ctu = eq.get_conservatives_from_primitives(WLx_ctu), eq.get_conservatives_from_primitives(WRx_ctu)
        ULy_ctu, URy_ctu = eq.get_conservatives_from_primitives(WLy_ctu), eq.get_conservatives_from_primitives(WRy_ctu)
        ULz_ctu, URz_ctu = eq.get_conservatives_from_primitives(WLz_ctu), eq.get_conservatives_from_primitives(WRz_ctu)
    
        Fx = riemann(WLx_ctu, WRx_ctu, slice8(ULx_ctu), slice8(URx_ctu), 0)
        Fy = riemann(WLy_ctu, WRy_ctu, slice8(ULy_ctu), slice8(URy_ctu), 1)
        Fz = riemann(WLz_ctu, WRz_ctu, slice8(ULz_ctu), slice8(URz_ctu), 2)
    
        # Optional GLM coupling (same pattern for each axis)
        if glm_on:
            def glm_modify(WL, WR, F, axis0):
                Bn_L = (WL[4], WL[5], WL[6])[axis0]
                Bn_R = (WR[4], WR[5], WR[6])[axis0]
                psi_L, psi_R = WL[-1], WR[-1]
                c_h = getattr(eq, "glm_ch", 1.0)
                F_Bn  = 0.5*(psi_L+psi_R) - 0.5*c_h*(Bn_R - Bn_L)
                F_psi = 0.5*(c_h**2)*(Bn_L+Bn_R) - 0.5*c_h*(psi_R - psi_L)
                F = F.at[(4,5,6)[axis0]].set(F_Bn)
                return jnp.vstack([F, F_psi])
            Fx = glm_modify(WLx_ctu, WRx_ctu, Fx, 0)
            Fy = glm_modify(WLy_ctu, WRy_ctu, Fy, 1)
            Fz = glm_modify(WLz_ctu, WRz_ctu, Fz, 2)
        
        # Flux divergence
        rhs = jnp.zeros_like(sol)
        rhs = rhs - (Fx - jnp.roll(Fx, 1, axis=1)) / dx
        rhs = rhs - (Fy - jnp.roll(Fy, 1, axis=2)) / dx
        rhs = rhs - (Fz - jnp.roll(Fz, 1, axis=3)) / dx
    
        # --- Constrained Transport update ---
       # --- Constrained Transport update (upwinded EMFs, Gardiner–Stone style) ---
        if False:
            #bugged...
            # All fluxes Fx,Fy,Fz have shape (n_cons, nx, ny, nz)
            # Each contains magnetic flux components:
            #   Fx[5]=F(By)= -Ez,  Fx[6]=F(Bz)= +Ey
            #   Fy[4]=F(Bx)= +Ez,  Fy[6]=F(Bz)= -Ex
            #   Fz[4]=F(Bx)= -Ey,  Fz[5]=F(By)= +Ex
            
            # 1. Derive face-centered electric fields from flux components
            Ex_face = 0.5 * ((-Fy[6]) + Fz[5])      # E_x ≈ ½(−Fy(Bz) + Fz(By))
            Ey_face = 0.5 * ((Fx[6])  - Fz[4])      # E_y ≈ ½(+Fx(Bz) − Fz(Bx))
            Ez_face = 0.5 * ((-Fx[5]) + Fy[4])      # E_z ≈ ½(−Fx(By) + Fy(Bx))
            
            # 2. Corner-averaged, upwinded EMFs (following GS05 eq. 59)
            # We assume your data layout (nx, ny, nz) with periodic BCs.
            
            def avg4(A):
                # Average over four neighboring faces in the xy-plane (used for Ez)
                return 0.25 * (A + jnp.roll(A, -1, axis=1)
                                 + jnp.roll(A, -1, axis=2)
                                 + jnp.roll(jnp.roll(A, -1, axis=1), -1, axis=2))
            
            # Example for E_z corners; others are cyclic permutations
            Ez_corner = avg4(Ez_face)
            Ex_corner = avg4(Ex_face)
            Ey_corner = avg4(Ey_face)
            
            # Optional upwind bias using local velocities (normal component sign)
            # Requires face-centered velocities from CTU states; omit for now
            # Ez_corner = Ez_corner - 0.25 * dt/dx * (dFy_dx - dFx_dy)
            
            # 3. Compute curl(E) on the uniform grid
            dBx_dt = -(jnp.roll(Ez_corner, -1, axis=1) - Ez_corner) / dx \
                     + (jnp.roll(Ey_corner, -1, axis=2) - Ey_corner) / dx
            dBy_dt = -(jnp.roll(Ex_corner, -1, axis=2) - Ex_corner) / dx \
                     + (jnp.roll(Ez_corner, -1, axis=0) - Ez_corner) / dx
            dBz_dt = -(jnp.roll(Ey_corner, -1, axis=0) - Ey_corner) / dx \
                     + (jnp.roll(Ex_corner, -1, axis=1) - Ex_corner) / dx
            
            # 4. Overwrite magnetic rows in RHS
            rhs = rhs.at[4].set(dBx_dt)
            rhs = rhs.at[5].set(dBy_dt)
            rhs = rhs.at[6].set(dBz_dt)
        if True:
            #debug without CT, use PCCT or something like that instead for the CT
            rhsBx, rhsBy, rhsBz = rhs[4:7]
            rhs_numerical = -(Fx[4:7] - jnp.roll(Fx[4:7], 1, axis=1))/dx \
                            -(Fy[4:7] - jnp.roll(Fy[4:7], 1, axis=2))/dx \
                            -(Fz[4:7] - jnp.roll(Fz[4:7], 1, axis=3))/dx
            rhs = rhs.at[4:7].set(rhs_numerical)
        
        return rhs

        
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def tree_flatten(self):
        #this method is needed for JAX control flow, probably some easier way to do it though...
        children = ()  # arrays / dynamic values
        aux_data = {
                    "boundary":self.boundary,
                    "snapshots":self.snapshots,
                   "splitting_schemes":self.splitting_schemes,
                    "fluxes":self.fluxes,"forces":self.forces,"maxjit":self.maxjit,
                "use_mol":self.use_mol,"use_ctu":self.use_ctu}  # static values
        return (children, aux_data)