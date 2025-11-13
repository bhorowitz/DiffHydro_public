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

# ---- Remat/checkpoint compatibility shim ----
import jax

try:
    # Newer JAX: checkpoint + (optional) policies module
    _remat = jax.checkpoint
    try:
        from jax.experimental import checkpoint_policies as _ckp  # may not exist on older JAX
        REMAT_POLICY = _ckp.checkpoint_dots  # good default when available
    except Exception:
        REMAT_POLICY = None
except AttributeError:
    # Older JAX: fall back to remat
    _remat = jax.remat
    REMAT_POLICY = None

def remat(fn):
    # Use a policy if available in your JAX; otherwise plain remat/checkpoint
    return _remat(fn) if REMAT_POLICY is None else _remat(fn, policy=REMAT_POLICY)
# ---------------------------------------------


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
                use_mol=True,
                use_ct=False,
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
        self._integrator_name = integrator
        self.use_ct = use_ct
        self.iBx, self.iBy, self.iBz = 4, 5, 6  # if Euler run, these rows may not exist

        self.pmesh_shape = pmesh_shape #parallelism
        
        devices = mesh_utils.create_device_mesh(self.pmesh_shape)
        self.mesh =  Mesh(devices, ('x', 'y','z'))
        self.FIELD_XYZ = P(None, 'x', 'y','z')
        
        # --- NEW runtime state ---
        self.sim_time: float = 0.0
        self.track_time: bool = track_time
        self.snapshot_every: int | None = snapshot_every
        self.snapshot_dir: str = snapshot_dir
        
        self.compute_dtype = jnp.float32
        self.state_dtype = jnp.float32
        
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
        dt_hist0 = jnp.zeros((self.n_super_step,), dtype=fields0.dtype)
        
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

        def run_loop(fields, params, t, dt_hist):
            def body(i, carry):
                fields, params, t, dt_hist = carry
                fields, params, dt = _one_step(fields, params, i, t)
                t = t + dt
                dt_hist = dt_hist.at[i].set(dt)  # <- record per-step dt

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

                return (fields, params, t, dt_hist)

            return lax.fori_loop(0, self.n_super_step, body, (fields, params, t, dt_hist0))

        evolve_pjit = pjit(
            run_loop,
            in_shardings=(sh_arr, None, None, None),
            out_shardings=(sh_arr, None, None, None),
            donate_argnums=(0,)
        )

        with self.mesh:
            fields_f, params_f, t_f, dt_hist = evolve_pjit(fields0, params, t0, dt_hist0)

        self.sim_time = float(t_f)
        return fields_f, params_f, dt_hist
                
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

    @partial(remat)
    def sweep_stack(self,state,dt,i):
        sol,params = state
        for scheme in self.splitting_schemes:
            for nn,ax in enumerate(scheme):
                sol = self.boundary.impose(sol,ax)
                sol = self.split_solve_step(sol,dt/(len(self.splitting_schemes)),int(ax),params)                 
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
        if True:
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
        # split forcing outside of core hydro loop
        fields, params = state
        fields = self.forcing(i, fields, params, dt/2)

        if self.use_mol and self.use_ct:
            jax.debug.print("use ct")
            fields = self.mol_solve_step_ct(fields, dt, params)  # <<< unsplit (MOL + CT-on-state)
        elif self.use_mol:
            fields = self.mol_solve_step(fields, dt, params)  # <<< unsplit (MOL + CT-on-state)

        else:
            fields = self.sweep_stack(state, dt, i)

        fields = self.forcing(i, fields, params, dt/2)
        return (fields, params)

    
    def evolve_with_dt_schedule(self, input_fields, params, dt_array):
        """
        Evolve using a pre-specified array of dt's, in order.

        Parameters
        ----------
        input_fields : Array
            Initial field state on host.
        params : Any
            Initial params.
        dt_array : Array-like, shape (n_steps,)
            Sequence of time steps to apply. n_steps determines how many
            hydro steps we take.

        Returns
        -------
        fields_f : Array
            Final field state on the device mesh.
        params_f : Any
            Final params.
        t_f : Array
            Final simulation time (sum of dt_array).
        """
        # 1) Shard fields
        sh_arr = NamedSharding(self.mesh, self.FIELD_XYZ)
        fields = jax.device_put(input_fields.astype(self.state_dtype), sh_arr)

        # 2) Put dt_array on device, make sure dtype matches
        dt_array = jnp.asarray(dt_array, dtype=fields.dtype)
        n_steps = dt_array.shape[0]

        # 3) Single hydro step that uses dt_array[i]
        def _one_step(fields, params, i, dt_array):
            dt = dt_array[i]
            fields, params = self._hydrostep(i, (fields, params), dt)
            return fields, params

        checkpointed_step = remat(_one_step)

        # We shard fields, params unsharded, dt_array replicated
        pjit_step = pjit(
            checkpointed_step,
            in_shardings=(sh_arr, None, None, None),
            out_shardings=(sh_arr, None),
            donate_argnums=(0,),
        )

        # 4) fori_loop over step index
        def body(i, carry):
            fields, params = carry
            fields, params = pjit_step(fields, params, i, dt_array)
            return (fields, params)

        fields_f, params_f = lax.fori_loop(
            0, n_steps, body, (fields, params)
        )

        # Final time is just sum of the dt's
        t_f = jnp.sum(dt_array)
     #   self.sim_time = float(t_f)

        return fields_f, params_f, t_f
    def evolve_till_time(
        self,
        input_fields,
        params,
        t_target: float,
        max_steps: int | None = None,
    ):
        """
        Evolve the system until the simulation time reaches `t_target`
        (or until `max_steps` steps are taken), using a JAX while_loop.

        Returns
        -------
        fields_f : Array
            Final field state on the device mesh.
        params_f : Any
            Final params.
        t_f : Array
            Final simulation time (scalar, same dtype as fields).
        dt_hist : Array
            Per-step dt history of length `self.n_super_step`.
            Only the first `n_steps` entries are filled; the rest remain 0.
        n_steps : Array
            Number of steps actually taken (int32 scalar).
        """
        # Sharding for the field array
        sh_arr = NamedSharding(self.mesh, self.FIELD_XYZ)
        fields0 = jax.device_put(input_fields, sh_arr)

        # Initial time and step counter
        t0 = jnp.array(0.0, dtype=fields0.dtype)
        step0 = jnp.array(0, dtype=jnp.int32)

        # dt history with a fixed, static length for JIT/pjit
        # We re-use n_super_step as an upper bound for safety.
        max_hist_len = self.n_super_step
        dt_hist0 = jnp.zeros((max_hist_len,), dtype=fields0.dtype)

        # Target time and step cap as JAX scalars
        t_target = jnp.asarray(t_target, dtype=fields0.dtype)
        max_steps = (
            jnp.asarray(max_steps, dtype=jnp.int32)
            if max_steps is not None
            else jnp.asarray(self.n_super_step, dtype=jnp.int32)
        )

        def _one_step(fields, params, i, t_scalar):
            # Same stepping logic as in evolve_with_callbacks
            (fields_out, params_out), dt = self.hydrostep_adapt(i, (fields, params), t_scalar)
            return fields_out, params_out, dt

        def run_loop(fields, params, t, dt_hist, step, t_target, max_steps):
            # while (t < t_target) and (step < max_steps)
            def cond_fn(carry):
                fields, params, t, dt_hist, step = carry
                return jnp.logical_and(t < t_target, step < max_steps)

            def body_fn(carry):
                fields, params, t, dt_hist, step = carry

                # Use current step as the loop index for hydrostep_adapt and dt_hist
                fields_new, params_new, dt = _one_step(fields, params, step, t)
                t_new = t + dt

                # Record dt for this step (if within allocated history length)
                dt_hist_new = jax.lax.cond(
                    step < max_hist_len,
                    lambda _dt_hist: _dt_hist.at[step].set(dt),
                    lambda _dt_hist: _dt_hist,
                    dt_hist,
                )

                step_new = step + jnp.array(1, dtype=step.dtype)
                return (fields_new, params_new, t_new, dt_hist_new, step_new)

            fields_f, params_f, t_f, dt_hist_f, step_f = lax.while_loop(
                cond_fn,
                body_fn,
                (fields, params, t, dt_hist, step),
            )
            return fields_f, params_f, t_f, dt_hist_f, step_f

        evolve_pjit = pjit(
            run_loop,
            in_shardings=(sh_arr, None, None, None, None, None, None),
            out_shardings=(sh_arr, None, None, None, None),
            donate_argnums=(0,),  # donate fields
        )

        with self.mesh:
            fields_f, params_f, t_f, dt_hist, n_steps = evolve_pjit(
                fields0, params, t0, dt_hist0, step0, t_target, max_steps
            )

        self.sim_time = float(t_f)
        return fields_f, params_f, t_f, dt_hist, n_steps

    def evolve(self, input_fields, params):
        # 1) Create sharding spec for fields
        sh_arr = NamedSharding(self.mesh, self.FIELD_XYZ)
        fields = jax.device_put(input_fields.astype(self.state_dtype), sh_arr)

        # 2) Define and wrap the step function with pjit
        def _one_step(fields, params, i):
            (fields_out, params_out),_t = self.hydrostep_adapt(i, (fields, params),0)
            return fields_out.astype(input_fields.dtype), params_out

        
        checkpointed_step = remat(_one_step)

        
        pjit_step = pjit(
            checkpointed_step,
            in_shardings=(sh_arr, None, None),
            out_shardings=(sh_arr, None),
            donate_argnums=(0,)
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
    
    @partial(remat)
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
            sol_b = self.boundary.impose(sol_b, ax, width=3)

            # STEP 3: Compute flux (now reconstruction can safely use jnp.roll)
            fu = self.flux(sol_b, ax, params)

            # STEP 4: Compute divergence with proper halo exchange
            rhs = rhs - (fu - self.roll_with_halo(fu, 1, ax)) / self.dx_o

        #magnetic stuff, ignored if no magnetic fields
        if getattr(self, "ct", False):
            if sol.shape[0] > self.iBx:
                rhs = rhs.at[self.iBx].set(0.0)
            if sol.shape[0] > self.iBy:
                rhs = rhs.at[self.iBy].set(0.0)
            if sol.shape[0] > self.iBz:
                rhs = rhs.at[self.iBz].set(0.0)
        return rhs
    
    def mol_solve_step(self, sol, dt, params):
        #normal solve without MHD CT
        return self.integrator(self.rhs_unsplit, sol, dt, params)  
    
    
    ###
    
    def evolve_memory_efficient(self, input_fields, params, checkpoint_every=10):
        """
        Memory-efficient evolution with configurable checkpointing.

        Parameters
        ----------
        input_fields : Array
            Initial fields
        params : Any
            Parameters
        checkpoint_every : int
            Number of steps between checkpoints. Higher = less memory, more recomputation.
            Typical values: 5-20 depending on your memory budget.
        """
        sh_arr = NamedSharding(self.mesh, self.FIELD_XYZ)
        fields = jax.device_put(input_fields.astype(self.state_dtype), sh_arr)

        def _single_step(fields, params, i):
            """Single hydro step - not checkpointed"""
            (fields_out, params_out), _t = self.hydrostep_adapt(i, (fields, params), 0)
            return fields_out.astype(input_fields.dtype), params_out

        def _block_of_steps(fields, params, block_idx):
            """
            Run checkpoint_every steps. Only this function is checkpointed,
            so intermediate states within the block are recomputed during backprop.
            """
            start_i = block_idx * checkpoint_every

            def substep(j, carry):
                fields, params = carry
                i = start_i + j
                fields, params = _single_step(fields, params, i)
                return (fields, params)

            return lax.fori_loop(0, checkpoint_every, substep, (fields, params))

        # Checkpoint only the blocks
        checkpointed_block = remat(_block_of_steps)

        pjit_block = pjit(
            checkpointed_block,
            in_shardings=(sh_arr, None, None),
            out_shardings=(sh_arr, None),
            donate_argnums=(0,)
        )

        # Main loop over checkpointed blocks
        n_blocks = self.n_super_step // checkpoint_every

        def body(block_idx, carry):
            fields, params = carry
            fields, params = pjit_block(fields, params, block_idx)
            return (fields, params)

        fields, params = lax.fori_loop(0, n_blocks, body, (fields, params))

        # Handle remaining steps (if n_super_step not divisible by checkpoint_every)
        remainder = self.n_super_step % checkpoint_every
        if remainder > 0:
            start_i = n_blocks * checkpoint_every

            def final_substep(j, carry):
                fields, params = carry
                i = start_i + j
                fields, params = _single_step(fields, params, i)
                return (fields, params)

            pjit_final = pjit(
                lambda f, p: lax.fori_loop(0, remainder, final_substep, (f, p)),
                in_shardings=(sh_arr, None),
                out_shardings=(sh_arr, None),
                donate_argnums=(0,)
            )
            fields, params = pjit_final(fields, params)

        return fields, params


    # Add this as a method to your hydro class:
    def add_memory_efficient_evolve_method(hydro_class):
        """
        Monkey-patch to add the memory-efficient evolve method.
        Usage:
            hydro.evolve = hydro.evolve_memory_efficient.__get__(hydro, type(hydro))
        """
        hydro_class.evolve_memory_efficient = evolve_memory_efficient
        return hydro_class

    
        # ---------------- MOL + CT-on-updated-state ----------------

    def mol_solve_step_ct(self, sol, dt, params):
        """
        MOL with CT applied on the UPDATED state, using this step's fluxes.
        For SSPRK3 / RK2 integrators we inline the stages to insert CT after each stage.
        For other integrators we apply CT once after the full step (fallback).
        
        Hopefully I figure out a nicer way to do this, but easy to code up...
        """
        name = self._integrator_name.upper()

        if name in ("SSPRK3", "RK3", "SSP3"):
            # --- SSPRK(3,3) ---
            # stage 1
            k1 = self.rhs_unsplit(sol, params); u1 = sol + dt * k1
            u1 = self._apply_ct_on_state(u1, params, dt)

            # stage 2
            k2 = self.rhs_unsplit(u1, params); u2 = 0.75 * sol + 0.25 * (u1 + dt * k2)
            # Effective substep on convex combo -> 0.25*dt contributes to new part; use 0.25*dt for CT
            u2 = self._apply_ct_on_state(u2, params, 0.25 * dt)

            # stage 3
            k3 = self.rhs_unsplit(u2, params); u3 = (1.0/3.0) * sol + (2.0/3.0) * (u2 + dt * k3)
            # Effective increment is (2/3)*dt on the last convex part; apply CT with that weight
            u3 = self._apply_ct_on_state(u3, params, (2.0/3.0) * dt)
            return u3

        elif name in ("RK2", "HEUN", "MIDPOINT"):
            # --- RK2 (Heun) ---
            k1 = self.rhs_unsplit(sol, params); u1 = sol + dt * k1
            u1 = self._apply_ct_on_state(u1, params, dt)

            k2 = self.rhs_unsplit(u1, params)
            u2_pred = sol + 0.5 * dt * (k1 + k2)
            # Apply CT for the second half contribution (0.5*dt)
            u2 = self._apply_ct_on_state(u2_pred, params, 0.5 * dt)
            return u2

        elif name in ("RK4",):
            # Fallback: apply CT once after a classic RK4 step using provided integrator
            u = self.integrator(self.rhs_unsplit, sol, dt, params)
            u = self._apply_ct_on_state(u, params, dt)
            return u

        else:
            # Unknown integrator: apply CT once
            u = self.integrator(self.rhs_unsplit, sol, dt, params)
            u = self._apply_ct_on_state(u, params, dt)
            return u

    # ---------------- CT on updated state ----------------

    def _apply_ct_on_state(self, sol, params, dt):
        """
        Constrained Transport applied to the *updated* state (MOL path).
        - Build edge-centered EMFs from face fluxes on the updated state.
        - Take curl(-E) to get face-centered dB/dt.
        - Average faces to cell centers and add to B components in `sol`.
        Works in 2D and 3D (if the state has 3 spatial dims).
        
        not properlly parallelized for multi-gpu, probably will work in forward at least
        """
        # If no magnetic rows_present, nothing to do
        if sol.shape[0] <= self.iBy:
            return sol

        # 1) Per-axis fluxes on the UPDATED state
        Fx = self.flux(sol, 1, params)  # (vars, x, y[, z])
        Fy = self.flux(sol, 2, params)
        Fz = self.flux(sol, 3, params) if sol.ndim >= 4 else None

        # 2) EMF mapping from magnetic flux rows (face-centered)
        #   Fx[By] = -E_z,  Fx[Bz] = +E_y
        #   Fy[Bx] = +E_z,  Fy[Bz] = -E_x
        #   Fz[Bx] = -E_y,  Fz[By] = +E_x
        Ez_face = 0.5 * (-Fx[self.iBy] + Fy[self.iBx])

        if sol.ndim == 3:
            # ---------- 2D (vars, x, y) ----------
            # corners (i+1/2, j+1/2) from faces: average over x(0) and y(1)
            Ez_corner = 0.25 * (
                Ez_face
                + jnp.roll(Ez_face, -1, axis=0)
                + jnp.roll(Ez_face, -1, axis=1)
                + jnp.roll(jnp.roll(Ez_face, -1, axis=0), -1, axis=1)
            )

            # curl(-E_z k̂):
            # dBx/dt on x-faces =  ∂(-Ez)/∂y  ;  dBy/dt on y-faces = -∂(-Ez)/∂x
            dbx_face = ( -Ez_corner + jnp.roll(-Ez_corner, 1, axis=1) ) / self.dx_o   # derivative in y
            dby_face = (  Ez_corner - jnp.roll( Ez_corner, 1, axis=0) ) / self.dx_o   # derivative in x

            # face → cell-center averages along the normal axis
            dBx = 0.5 * (dbx_face + jnp.roll(dbx_face, 1, axis=0))  # average along x
            dBy = 0.5 * (dby_face + jnp.roll(dby_face, 1, axis=1))  # average along y

            sol = sol.at[self.iBx].add(dt * dBx)
            sol = sol.at[self.iBy].add(dt * dBy)
            return sol

        # ---------- 3D (vars, x, y, z) ----------
        # Additional EMFs from other flux rows
        Ex_face = 0.5 * ((-Fy[self.iBz]) + Fz[self.iBy])
        Ey_face = 0.5 * (( Fx[self.iBz]) - Fz[self.iBx])

        def avg4(A, ax_a, ax_b):
            """Average A with neighbors shifted by -1 along (ax_a, ax_b) in A's own axes."""
            A1 = jnp.roll(A, -1, axis=ax_a)
            A2 = jnp.roll(A, -1, axis=ax_b)
            A3 = jnp.roll(A1, -1, axis=ax_b)
            return 0.25 * (A + A1 + A2 + A3)

        # After slicing var, EMFs are (x,y,z) with axes (0,1,2)
        Ez_corner = avg4(Ez_face, 0, 1)  # x–y corners
        Ex_corner = avg4(Ex_face, 1, 2)  # y–z corners
        Ey_corner = avg4(Ey_face, 0, 2)  # x–z corners

        # dB/dt = -curl(E) using corner EMFs
        dBx_face = (-(Ez_corner - jnp.roll(Ez_corner, 1, axis=1)) / self.dx_o   # -∂Ez/∂y
                    + ( Ey_corner - jnp.roll( Ey_corner, 1, axis=2)) / self.dx_o)  # +∂Ey/∂z
        dBy_face = (-( Ex_corner - jnp.roll( Ex_corner, 1, axis=2)) / self.dx_o   # -∂Ex/∂z
                    + ( Ez_corner - jnp.roll(Ez_corner, 1, axis=0)) / self.dx_o)  # +∂Ez/∂x
        dBz_face = (-( Ey_corner - jnp.roll( Ey_corner, 1, axis=0)) / self.dx_o   # -∂Ey/∂x
                    + ( Ex_corner - jnp.roll( Ex_corner, 1, axis=1)) / self.dx_o)  # +∂Ex/∂y

        # face → cell-center averages along normal axes
        dBx = 0.5 * (dBx_face + jnp.roll(dBx_face, 1, axis=0))  # along x
        dBy = 0.5 * (dBy_face + jnp.roll(dBy_face, 1, axis=1))  # along y
        dBz = 0.5 * (dBz_face + jnp.roll(dBz_face, 1, axis=2))  # along z

        sol = sol.at[self.iBx].add(dt * dBx)
        sol = sol.at[self.iBy].add(dt * dBy)
        sol = sol.at[self.iBz].add(dt * dBz)
        return sol

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
                    "fluxes":self.fluxes,"forces":self.forces,
                "use_mol":self.use_mol,"use_ct":self.use_ct, "pmesh_shape":self.pmesh_shape}  # static values
        return (children, aux_data)