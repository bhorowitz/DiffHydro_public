from jax import Array 
from functools import partial
from typing import List
import jax.numpy as np
from diffhydro import NoBoundary, NoForcing
import jax
from jax import jit
import jax.numpy as jnp
import os

#reorg into halo_helper sometime
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.pjit import pjit
from .solver.integrator import INTEGRATOR_DICT
from .utils.parallel import halo_helper
from jax.experimental import mesh_utils, multihost_utils
#from jax.experimental import maps as maps
from jax.sharding import PartitionSpec as P  # keep P from jax.sharding
from jax.experimental.shard_map import shard_map

import jax.lax as lax

import numpy as onp
from jax.experimental import io_callback  # side-effect callback inside jit/pjit



def save_snapshot_np(path, arr_host):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onp.save(path, onp.asarray(arr_host))


def roll_with_halo(self, array, shift, axis):
    """
    Halo-aware roll for distributed arrays using shard_map.
    """
    # Single device or not distributed on this axis - use regular roll
    axis_idx = axis - 1  # axis 0 is variables, spatial axes start at 1
    if self.pmesh_shape[axis_idx] == 1:
        return jnp.roll(array, shift, axis=axis)

    # For multi-step shifts, do them one at a time
    # This is less efficient but handles arbitrary shifts
    if abs(shift) > 1:
        result = array
        step = 1 if shift > 0 else -1
        for _ in range(abs(shift)):
            result = self.roll_with_halo(result, step, axis)
        return result

    axis_name = ('x', 'y', 'z')[axis_idx]
    n_devices = self.pmesh_shape[axis_idx]

    # Define the operation that will run on each shard
    def _exchange_halos(local_array):
        # Build permutation for communication
        if shift == 1:
            perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]
            # Get last slice to send forward
            boundary = jax.lax.slice_in_dim(local_array, -1, None, axis=axis)
            received = jax.lax.ppermute(boundary, axis_name, perm)
            # Prepend received data
            interior = jax.lax.slice_in_dim(local_array, 0, -1, axis=axis)
            return jnp.concatenate([received, interior], axis=axis)
        elif shift == -1:
            perm = [(i, (i - 1) % n_devices) for i in range(n_devices)]
            # Get first slice to send backward
            boundary = jax.lax.slice_in_dim(local_array, 0, 1, axis=axis)
            received = jax.lax.ppermute(boundary, axis_name, perm)
            # Append received data
            interior = jax.lax.slice_in_dim(local_array, 1, None, axis=axis)
            return jnp.concatenate([interior, received], axis=axis)
        else:
            raise ValueError(f"Only shift=±1 supported in base case, got {shift}")

    # Use shard_map to bind the axis names for ppermute
    return shard_map(
        _exchange_halos,
        mesh=self.mesh,
        in_specs=self.FIELD_XYZ,
        out_specs=self.FIELD_XYZ,
        check_rep=False
    )(array)


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
                use_mol=False,
                use_ctu=False,
                pmesh_shape= (1,1,1) ,
                integrator="RK2",
                snapshot_every: int | None = None,
                snapshot_dir: str = "snapshots",
                track_time: bool = True):
        #parameters that are held constant per run (i.e. probably don't want to take derivatives with respect to...)
   #     self.init_dt = init_dt # tiny starting timestep to smooth out anything too sharp
        self.splitting_schemes = splitting_schemes #strang splitting for x,y,z sweeps
        self.max_dt = max_dt
        self.boundary = None
        #supersteps, each superstep has len(splitting_schemes) time steps
        self.n_super_step = n_super_step
        self.snapshots = snapshots #poorly names/
        self.outputs = []
        self.fluxes = fluxes
        self.forces = forces
        self.dx_o = 1.0
        self.timescale = jnp.zeros(self.n_super_step)
        self.use_mol = use_mol
        self.integrator = INTEGRATOR_DICT[integrator]  # callable
        self.use_ctu = use_ctu
        self.pmesh_shape = pmesh_shape #parallelism
        
        devices = mesh_utils.create_device_mesh(self.pmesh_shape)
        self.mesh =  Mesh(devices, ('x', 'y','z'))
        self.FIELD_XYZ = P(None, 'x', 'y','z')
        
        # --- NEW runtime state ---
        self.sim_time: float = 0.0
        self.track_time: bool = track_time
        self.snapshot_every: int | None = snapshot_every
        self.snapshot_dir: str = snapshot_dir

        # Make snapshot dir on host 0 (safe if it already exists)
        if self.snapshot_every is not None and jax.process_index() == 0:
            os.makedirs(self.snapshot_dir, exist_ok=True)

        # Initialize boundary class with mesh info
        if boundary is None:
            # Default to periodic with multi-GPU support
            from .boundary import PeriodicBoundarySimple
            self.boundary = PeriodicBoundarySimple(
                mesh=self.mesh,
                pmesh_shape=self.pmesh_shape,
                field_spec=self.FIELD_XYZ,
                roll_fn=self.roll_with_halo  # Pass our halo exchange function
            )
        elif isinstance(boundary, type):
            # boundary is a class, instantiate it
            self.boundary = boundary(
                mesh=self.mesh,
                pmesh_shape=self.pmesh_shape,
                field_spec=self.FIELD_XYZ
            )
        else:
            # boundary is already an instance
            self.boundary = boundary
            # Inject mesh info if not already present
            if hasattr(self.boundary, 'mesh'):
                self.boundary.mesh = self.mesh
                self.boundary.pmesh_shape = self.pmesh_shape
                self.boundary.field_spec = self.FIELD_XYZ
    
                
    def evolve_with_callbacks(self, input_fields, params):
        sh_arr = NamedSharding(self.mesh, self.FIELD_XYZ)
        fields0 = jax.device_put(input_fields, sh_arr)
        t0 = jnp.array(0.0, dtype=fields0.dtype)

        snapshot_every = (self.snapshot_every if getattr(self, "snapshot_every", None) is not None
                          else (int(getattr(self, "snapshots", 0)) if getattr(self, "snapshots", 0) else 0))
        snapshot_every = int(snapshot_every) if snapshot_every else 0

        snapshot_dir = self.snapshot_dir
        mesh_shape = self.mesh.shape

        # Save shard with device index
        def _save_shard_np_cb(step_i, x_idx, y_idx, z_idx, arr_host):
            import os, numpy as onp
            # Compute linear device index on host
            linear_idx = int(x_idx) * (mesh_shape['y'] * mesh_shape['z']) + \
                         int(y_idx) * mesh_shape['z'] + int(z_idx)
            os.makedirs(snapshot_dir or ".", exist_ok=True)
            path = os.path.join(snapshot_dir, f"fields_step_{int(step_i):06d}_device_{linear_idx}.npy")
            onp.save(path, onp.asarray(arr_host))

        def _one_step(fields, params, i, t_scalar):
            (fields_out, params_out), dt = self.hydrostep_adapt(i, (fields, params), t_scalar)
            return fields_out, params_out, dt

        def run_loop(fields, params, t):
            def body(i, carry):
                fields, params, t = carry
                fields, params, dt = _one_step(fields, params, i, t)
                t = t + dt

                if snapshot_every > 0:
                    def _do_snapshot(_):
                        # Create a shard_map just to access axis indices
                        def save_local_shard(local_fields):
                            x_idx = lax.axis_index('x')
                            y_idx = lax.axis_index('y')
                            z_idx = lax.axis_index('z')
                            # Save with mesh coordinates
                            io_callback(_save_shard_np_cb, None, 
                                       i, x_idx, y_idx, z_idx, local_fields)
                            return ()

                        shard_map(
                            save_local_shard,
                            mesh=self.mesh,
                            in_specs=self.FIELD_XYZ,
                            out_specs=P(),  # Returns ()
                            check_rep=False
                        )(fields)
                        return ()

                    lax.cond((i % snapshot_every) == 0, _do_snapshot, lambda _: (), operand=None)

                return (fields, params, t)

            return lax.fori_loop(0, self.n_super_step, body, (fields, params, t))

        evolve_pjit = pjit(
            run_loop,
            in_shardings=(sh_arr, None, None),
            out_shardings=(sh_arr, None, None),
        )

        with self.mesh:
            fields_f, params_f, t_f = evolve_pjit(fields0, params, t0)

        self.sim_time = float(t_f)
        return fields_f, params_f
                
    def roll_with_halo(self, array, shift, axis):
        """
        Halo-aware roll for distributed arrays using shard_map.
        """
        # Single device or not distributed on this axis - use regular roll
        axis_idx = axis - 1  # axis 0 is variables, spatial axes start at 1
        if self.pmesh_shape[axis_idx] == 1:
            return jnp.roll(array, shift, axis=axis)

        axis_name = ('x', 'y', 'z')[axis_idx]
        n_devices = self.pmesh_shape[axis_idx]

        # Define the operation that will run on each shard
        def _exchange_halos(local_array):
            # Build permutation for communication
            if shift == 1:
                perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]
                # Get last slice to send forward
                boundary = jax.lax.slice_in_dim(local_array, -1, None, axis=axis)
                received = jax.lax.ppermute(boundary, axis_name, perm)
                # Prepend received data
                interior = jax.lax.slice_in_dim(local_array, 0, -1, axis=axis)
                return jnp.concatenate([received, interior], axis=axis)
            elif shift == -1:
                perm = [(i, (i - 1) % n_devices) for i in range(n_devices)]
                # Get first slice to send backward
                boundary = jax.lax.slice_in_dim(local_array, 0, 1, axis=axis)
                received = jax.lax.ppermute(boundary, axis_name, perm)
                # Append received data
                interior = jax.lax.slice_in_dim(local_array, 1, None, axis=axis)
                return jnp.concatenate([interior, received], axis=axis)
            else:
                raise ValueError(f"Only shift=±1 supported, got {shift}")

        # Use shard_map to bind the axis names for ppermute
        return shard_map(
            _exchange_halos,
            mesh=self.mesh,
            in_specs=self.FIELD_XYZ,
            out_specs=self.FIELD_XYZ,
            check_rep=False
        )(array)
    
    @jax.jit
    def timestep(self,fields):
        dt = []
        for flux in self.fluxes:
            dt.append(flux.timestep(fields))
        for force in self.forces:
            dt.append(force.timestep(fields))
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

    def split_solve_step(self, sol, dt, ax, params):
        """RK2 method, need to put in integrator choice at some point..."""

        # First stage
        fu1 = self.flux(sol, ax, params) 
        rhs_cons = (fu1 - self.roll_with_halo(fu1, 1, ax))  # WITH HALO EXCHANGE

        u1 = sol - rhs_cons * dt / (2.0 * self.dx_o)

        # Second stage
        fu = self.flux(u1, ax, params)  # Note: should this be u1 instead of sol?
        rhs_cons = (fu - self.roll_with_halo(fu, 1, ax))    # WITH HALO EXCHANGE

        sol = sol - (rhs_cons) * dt / self.dx_o
        return sol

    @jax.checkpoint
    def sweep_stack(self,state,dt,i):
        sol,params = state
        for scheme in self.splitting_schemes:
            for nn,ax in enumerate(scheme):
                sol = self.boundary.impose(sol,ax)
                sol = self.split_solve_step(sol,dt/(len(scheme)),int(ax),params)                 
                # experimental
                sol = sol.at[0].set(jnp.abs(sol[0])) #experimental...
                sol = sol.at[-1].set(jnp.abs(sol[-1])) #experimental...
    
        return sol
    
   # @jax.jit
    def evolve_old(self,input_fields,params):
        self.outputs=[]
        #main loop
      #  state = (input_fields,params)
        sharding = NamedSharding(self.mesh, self.FIELD_XYZ)
        state = (jax.device_put(input_fields, sharding), params)
        #need to rework the UI to get out snapshots from jitted function, hack for now...
        if self.maxjit:
            state  = jax.lax.fori_loop(0, self.n_super_step, self.hydrostep_adapt, state)
        else:
            for i in range(0,self.n_super_step):
                state,_t = self.hydrostep_adapt(i,state,0)
                if self.snapshots:
                    if i%self.snapshots==0: #comment out most times...
                        self.outputs.append(state)
        return state
        
 #   @partial(jit, static_argnums=0)

    def hydrostep_adapt(self, i, state, current_time):
        fields, params = state
        ttt = self.timestep(fields)
        ttt = jnp.minimum(self.max_dt, ttt)
        dt = (ttt)
        fields, params = self._hydrostep(i, (fields, params), dt)
        # return both the new state and the dt so host can accumulate time
        return (fields, params), dt
    

   # @jax.jit
    def _hydrostep(self, i, state, dt):

        #split forcing outside of core hydro loop
        
        fields, params = state

        fields = self.forcing(i, fields, params, dt/2)          

        if self.use_mol:
            fields = self.mol_solve_step(fields, dt, params)  # <<< unsplit
        else:
            fields = self.sweep_stack(state, dt, i)           #
            
        fields = self.forcing(i, fields, params, dt/2) 

        return (fields, params)
    
    
    # Wrap the *step* in pjit so named axes exist when rhs_unsplit runs.
    def build_pjit_step(self):
        def _step(fields, params, i):
            # hydrostep_adapt expects (i, state) where state=(fields, params)
            (fields_out, params_out),_t = self.hydrostep_adapt(i, (fields, params),0)
            return fields_out, params_out

        return pjit(
            _step,
            in_shardings=(self.FIELD_XYZ, None, None),  # (fields, params, i)
            out_shardings=(self.FIELD_XYZ, None),       # (fields_out, params_out)
        )

    

    def evolve(self, input_fields, params):
        # 1) Create sharding spec for fields
        sh_arr = NamedSharding(self.mesh, self.FIELD_XYZ)
        fields = jax.device_put(input_fields, sh_arr)

        # 2) Define and wrap the step function with pjit
        def _one_step(fields, params, i):
            (fields_out, params_out),_t = self.hydrostep_adapt(i, (fields, params),0)
            return fields_out, params_out

        pjit_step = pjit(
            _one_step,
            in_shardings=(sh_arr, None, None),
            out_shardings=(sh_arr, None),
        )

        # 3) Run the loop (no mesh context needed!)
        def body(i, carry):
            fields, params = carry
            # i is a JAX scalar here; fine to pass into pjit_step as long as it
            # doesn't change shapes / trigger recompiles.
            fields, params = pjit_step(fields, params, i)
            return (fields, params)
        
        fields, params = lax.fori_loop(
            0, self.n_super_step, body, (fields, params)
        )
       # self.outputs = []
       # for i in range(self.n_super_step):
       #     fields, params = pjit_step(fields, params, i)
       ##     if self.snapshots and i % self.snapshots == 0:
        #        self.outputs.append((fields, params))

        return fields, params
    
    def rhs_unsplit(self, sol, params):
        """
        Unsplit RHS computation with proper halo exchanges via boundary class.
        """
        rhs = jnp.zeros_like(sol)

        # Loop over spatial axes
        for ax in range(1, sol.ndim):
            if sol.shape[ax] <= 1:
                continue
            # STEP 1: Apply boundary conditions (includes halo exchange)
            sol_b = self.boundary.impose(sol, ax)

            # STEP 2: For wide stencils (TENO5, PPM), sync halos multiple times
            # Each impose() call exchanges one layer of halos
            # For TENO5 (needs ±2 cells), call 2-3 times to be safe
            sol_b = self.boundary.impose(sol, ax, width=3)

            # STEP 3: Compute flux (now reconstruction can safely use jnp.roll)
            fu = self.flux(sol_b, ax, params)

            # STEP 4: Compute divergence with proper halo exchange
            rhs = rhs - (fu - self.roll_with_halo(fu, 1, ax)) / self.dx_o

        return rhs

    def mol_solve_step(self, sol, dt, params):
        if getattr(self, "use_ctu", False):
            # pass dt into rhs via a closure
            return self.integrator(lambda u, p: self.rhs_ctu(u, p, dt), sol, dt, params)
        else:
            return self.integrator(self.rhs_unsplit, sol, dt, params)  # existing path

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
        def ctu_correct(self, WL_h, WR_h, F_a, F_b, ax_a, ax_b):
            # Need to map axis indices correctly
            # ax_a, ax_b are 0,1,2 but roll_with_halo expects 1,2,3
            dFa_da = (self.roll_with_halo(F_a, -1, ax_a+1) - 
                      self.roll_with_halo(F_a, 1, ax_a+1)) * (0.5/dx)
            dFb_db = (self.roll_with_halo(F_b, -1, ax_b+1) - 
                      self.roll_with_halo(F_b, 1, ax_b+1)) * (0.5/dx)
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
        rhs = rhs - (Fx - self.roll_with_halo(Fx, 1, 1)) / dx  # ✓ axis=1 for x
        rhs = rhs - (Fy - self.roll_with_halo(Fy, 1, 2)) / dx  # ✓ axis=2 for y
        rhs = rhs - (Fz - self.roll_with_halo(Fz, 1, 3)) / dx  # ✓ axis=3 for z
    
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
            
        def avg4(A, ax1, ax2):
        # Average over four neighboring faces
            return 0.25 * (A + self.roll_with_halo(A, -1, ax1)
                         + self.roll_with_halo(A, -1, ax2)
                         + self.roll_with_halo(self.roll_with_halo(A, -1, ax1), -1, ax2))

            # Example for E_z corners; others are cyclic permutations
            Ez_corner = avg4(Ez_face)
            Ex_corner = avg4(Ex_face)
            Ey_corner = avg4(Ey_face)
            
            # Optional upwind bias using local velocities (normal component sign)
            # Requires face-centered velocities from CTU states; omit for now
            # Ez_corner = Ez_corner - 0.25 * dt/dx * (dFy_dx - dFx_dy)
            
            # 3. Compute curl(E) on the uniform grid
            dBx_dt = -(self.roll_with_halo(Ez_corner, -1, 2) - Ez_corner) / dx \
                 + (self.roll_with_halo(Ey_corner, -1, 3) - Ey_corner) / dx

            
            dBy_dt = -(self.roll_with_halo(Ez_corner, -1, 2) - Ex_corner) / dx \
                 + (self.roll_with_halo(Ey_corner, -1, 0) - Ez_corner) / dx
            dBz_dt = -(self.roll_with_halo(Ez_corner, -1, 0) - Ey_corner) / dx \
                 + (self.roll_with_halo(Ey_corner, -1, 1) - Ex_corner) / dx
            
            
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
                "use_mol":self.use_mol,"use_ctu":self.use_ctu, "pmesh_shape":self.pmesh_shape}  # static values
        return (children, aux_data)