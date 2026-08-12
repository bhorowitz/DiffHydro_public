# Type JAX de base pour annoter les tenseurs.
from jax import Array 
# partial permet de preconfigurer des decorators/fonctions.
from functools import partial
# Typage standard Python.
from typing import List
# Alias NumPy JAX (utilise parfois sous np, parfois jnp).
# import numpy as np
# Valeurs par defaut du package (conditions aux limites / forcing).
from diffhydro import NoBoundary, NoForcing
# Namespace principal JAX.
import jax
# jit importe directement (meme si jax.jit est aussi utilise).
from jax import jit
# Alias principal utilise dans tout le fichier.
import jax.numpy as jnp
# I/O systeme pour snapshots et chemins.
import os

# TODO: eventuellement regrouper la logique halo dans halo_helper.
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.pjit import pjit
from .solver.integrator import INTEGRATOR_DICT
from .utils.parallel import halo_helper
from jax.experimental import mesh_utils, multihost_utils
# Variante historique laissee en commentaire.
#from jax.experimental import maps as maps
from jax.sharding import PartitionSpec as P  # keep P from jax.sharding
from jax.experimental.shard_map import shard_map

import jax.lax as lax

import numpy as onp
from jax.experimental import io_callback  # side-effect callback inside jit/pjit
from .utils.debug_checks import _check_finite, _check_float_status

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
    # Use a policy if available in your JAX version; otherwise plain remat/checkpoint
    return _remat(fn) if REMAT_POLICY is None else _remat(fn, policy=REMAT_POLICY)
# ---------------------------------------------


def save_snapshot_np(path, arr_host):
    # Creates the parent folder if needed before writing the .npy file.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convertit explicitement vers NumPy host puis sauvegarde.
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

    # Nom logique de l'axe dans la mesh JAX.
    axis_name = ('x', 'y', 'z')[axis_idx]
    # Nombre de partitions sur cet axe.
    n_devices = self.pmesh_shape[axis_idx]

    # Define the operation that will run on each shard
    def _exchange_halos(local_array):
        # Build permutation for communication
        if shift == 1:
            # Mapping circulaire vers le voisin +1.
            perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]
            # Get last slice to send forward
            boundary = jax.lax.slice_in_dim(local_array, -1, None, axis=axis)
            received = jax.lax.ppermute(boundary, axis_name, perm)
            # Prepend received data
            interior = jax.lax.slice_in_dim(local_array, 0, -1, axis=axis)
            return jnp.concatenate([received, interior], axis=axis)
        elif shift == -1:
            # Mapping circulaire vers le voisin -1.
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
                forces = [], #gravity, etc.
                use_mol=True,
                use_ct=False,
                pmesh_shape= (1,1,1) ,
                integrator="RK2",
                snapshot_every: int | None = None,
                snapshot_dir: str = "snapshots",
                snapshot_prefix: str = "fields",
                track_time: bool = True,
                debug_fixed_dt: float | None = None,
                periodic_flux_divergence: bool = True,
                dx: float = 1.0
                ):
        # Fixed simulation parameters (rather static with respect to optimization).
   #     self.init_dt = init_dt # tiny starting timestep to smooth out anything too sharp
        self.splitting_schemes = splitting_schemes #strang splitting for x,y,z sweeps
        self.max_dt = max_dt
        self.debug_fixed_dt = debug_fixed_dt
        self.boundary = None
        # Number of global iterations (each super-step applies a complete sweep scheme).
        self.n_super_step = n_super_step
        # Liste d'objets responsables des flux numeriques.
        self.fluxes = fluxes
        # Liste d'objets responsables des termes sources / forces.
        self.forces = forces
        # Reference spatial step (assumed uniform here).
        self.dx_o = dx
        self.use_mol = use_mol
        # Selects the time integration function by name.
        self.integrator = INTEGRATOR_DICT[integrator]  # callable
        self._integrator_name = integrator
        self.use_ct = use_ct
        # Index of magnetic components in the state tensor.
        self.iBx, self.iBy, self.iBz = 4, 5, 6  # if Euler run, these rows may not exist

        self.pmesh_shape = pmesh_shape #parallelism
        
        # Builds the logical mesh of devices for spatial parallelization.
        devices = mesh_utils.create_device_mesh(self.pmesh_shape)
        self.mesh =  Mesh(devices, ('x', 'y','z'))
        # The field is sharded only on spatial axes (not on the variables axis).
        self.FIELD_XYZ = P(None, 'x', 'y','z')
        
        # --- NEW runtime state ---
        # Temps physique cumule de la simulation.
        self.sim_time: float = 0.0
        self.track_time: bool = track_time
        self.periodic_flux_divergence = periodic_flux_divergence
        self.snapshot_every: int | None = snapshot_every
        self.snapshot_dir: str = snapshot_dir
        self.snapshot_prefix: str = snapshot_prefix
        
        # Dtypes explicites pour stabiliser la compilation et les conversions.
        self.compute_dtype = jnp.float64
        self.state_dtype = jnp.float64
        
        # Make snapshot dir on host 0 (safe if it already exists)
        # Avoids concurrent folder creations on all hosts.
        if self.snapshot_every is not None and jax.process_index() == 0:
            os.makedirs(self.snapshot_dir, exist_ok=True)

        # Initialise la gestion des conditions aux limites avec les infos de mesh.
        if boundary is None:
            # Default to periodic with multi-GPU support
            from .boundary import PeriodicBoundarySimple
            self.boundary = PeriodicBoundarySimple(
                mesh=self.mesh,
                pmesh_shape=self.pmesh_shape,
                field_spec=self.FIELD_XYZ,
                # Provides the halo-aware shift primitive for inter-device exchanges.
                roll_fn=self.roll_with_halo
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
            # If the instance exposes these attributes, synchronize them with the hydro object.
            if hasattr(self.boundary, 'mesh'):
                self.boundary.mesh = self.mesh
                self.boundary.pmesh_shape = self.pmesh_shape
                self.boundary.field_spec = self.FIELD_XYZ
    
                
    def evolve_with_callbacks(self, input_fields, params):
        # Describes how the field tensor is distributed over the mesh.
        sh_arr = NamedSharding(self.mesh, self.FIELD_XYZ)
        # Sends the initial state to devices with the defined sharding.
        fields0 = jax.device_put(input_fields, sh_arr)
        # Temps initial scalaire device-side.
        t0 = jnp.array(0.0, dtype=fields0.dtype)
        # History of dt allocated to fixed size (static shape for JAX).
        dt_hist0 = jnp.zeros((self.n_super_step,), dtype=fields0.dtype)
        
        # Compatibility with old "snapshots" field if snapshot_every is absent.
        snapshot_every = (self.snapshot_every if getattr(self, "snapshot_every", None) is not None
                          else (int(getattr(self, "snapshots", 0)) if getattr(self, "snapshots", 0) else 0))
        # Normalizes to Python integer (0 = disabled).
        snapshot_every = int(snapshot_every) if snapshot_every else 0

        # Local captures for host callback (simpler and more explicit).
        snapshot_dir = self.snapshot_dir
        mesh_shape = self.mesh.shape

        # Save shard with device index
        def _save_shard_np_cb(step_i, x_idx, y_idx, z_idx, arr_host):
            import os, numpy as onp
            # Recomposes the linear device index from its mesh coordinates.
            linear_idx = int(x_idx) * (mesh_shape['y'] * mesh_shape['z']) + \
                         int(y_idx) * mesh_shape['z'] + int(z_idx)
            # Creates the snapshots folder then writes the local shard.
            os.makedirs(snapshot_dir or ".", exist_ok=True)
            path = os.path.join(snapshot_dir, f"{self.snapshot_prefix}_step_{int(step_i):06d}_device_{linear_idx}.npy")
            onp.save(path, onp.asarray(arr_host))

        def _one_step(fields, params, i, t_scalar):
            # Advances by an adaptive step and also retrieves the dt used.
            (fields_out, params_out), dt = self.hydrostep_adapt(i, (fields, params), t_scalar)
            return fields_out, params_out, dt

        def run_loop(fields, params, t, dt_hist):
            def body(i, carry):
                fields, params, t, dt_hist = carry
                # Executes a hydro step then accumulates the simulated time.
                fields, params, dt = _one_step(fields, params, i, t)
                t = t + dt
                # Saves the dt of step i for diagnostics / post-processing.
                dt_hist = dt_hist.at[i].set(dt)  # <- record per-step dt

                if snapshot_every > 0:
                    def _do_snapshot(_):
                        # Launches a local function per shard to retrieve mesh indices.
                        def save_local_shard(local_fields):
                            x_idx = lax.axis_index('x')
                            y_idx = lax.axis_index('y')
                            z_idx = lax.axis_index('z')
                            # Host callback: serializes the shard with device coordinates.
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

                    # Triggers snapshot only on multiples of snapshot_every.
                    lax.cond((i % snapshot_every) == 0, _do_snapshot, lambda _: (), operand=None)

                return (fields, params, t, dt_hist)

            # Static XLA loop over n_super_step iterations.
            return lax.fori_loop(0, self.n_super_step, body, (fields, params, t, dt_hist0))

        # Compiles the complete loop in pjit for efficient distributed execution.
        evolve_pjit = pjit(
            run_loop,
            in_shardings=(sh_arr, None, None, None),
            out_shardings=(sh_arr, None, None, None),
            donate_argnums=(0,)
        )

        # Contexte mesh obligatoire pour executer collectives et shardings nommes.
        with self.mesh:
            fields_f, params_f, t_f, dt_hist = evolve_pjit(fields0, params, t0, dt_hist0)

        # Brings back the final time as Python float for the state object.
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

        # Selection of the logical axis and the number of concerned devices.
        axis_name = ('x', 'y', 'z')[axis_idx]
        n_devices = self.pmesh_shape[axis_idx]

        # Define the operation that will run on each shard
        def _exchange_halos(local_array):
            # Build permutation for communication
            if shift == 1:
                # Forward cyclic permutation (send to neighbor +1).
                perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]
                # Get last slice to send forward
                boundary = jax.lax.slice_in_dim(local_array, -1, None, axis=axis)
                received = jax.lax.ppermute(boundary, axis_name, perm)
                # Prepend received data
                interior = jax.lax.slice_in_dim(local_array, 0, -1, axis=axis)
                return jnp.concatenate([received, interior], axis=axis)
            elif shift == -1:
                # Backward cyclic permutation (send to neighbor -1).
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
        # CFL constraints / physical constraints from fluxes (advection, diffusion, ...).
        for flux in self.fluxes:
            dt.append(flux.timestep(fields))
        # Additional constraints from source terms (forces).
        for force in self.forces:
            dt.append(force.timestep(fields))
        # We select the most restrictive global dt.
        return jnp.min(jnp.array(dt))
    
    def flux(self,sol,ax,params):
        # Total flux accumulator (same shape as the state).
        total_flux = jnp.zeros(sol.shape)
        for flux in self.fluxes: 
            #note it is ordered, to allow a flux_correction depending on calculated fluxes
            #make sure your order is correct for that though!
            # Each flux can depend on the cumulative value already calculated (successive corrections).
            total_flux += flux.flux(sol,ax,params,total_flux)
        return total_flux
    
    def forcing(self,i,sol,params,dt): #all axis independant?
        # Applies each force sequentially by propagating updated fields + parameters.
        for force in self.forces:
            sol,params = force.force(i, sol, params, dt)  # each returns UPDATED fields
        return sol,params

    def split_solve_step(self, sol, dt, ax, params):
        """RK2 method, need to put in integrator choice at some point..."""

        # Stage 1 RK2: calcul du flux et divergence conservative.
        fu1 = self.flux(sol, ax, params) 
        rhs_cons = (fu1 - self.roll_with_halo(fu1, 1, ax))  # WITH HALO EXCHANGE

        # Intermediate state at half-step.
        u1 = sol - rhs_cons * dt / (2.0 * self.dx_o)

        # Stage 2 RK2: recalculates fluxes on the intermediate state.
        fu = self.flux(u1, ax, params)  # Note: should this be u1 instead of sol?
        rhs_cons = (fu - self.roll_with_halo(fu, 1, ax))    # WITH HALO EXCHANGE

        # Final conservative update over one complete step.
        sol = sol - (rhs_cons) * dt / self.dx_o
        return sol

    @partial(remat)
    def sweep_stack(self,state,dt,i):
        sol,params = state
        # Traversal of different sweep orders (Strang/cyclic).
        for scheme in self.splitting_schemes:
            # Traversal of axes in the current order.
            for nn,ax in enumerate(scheme):
                # Apply boundaries before computing fluxes on the axis.
                sol = self.boundary.impose(sol,ax)
                # Integrates a fraction of the total dt associated with the active scheme.
                sol = self.split_solve_step(sol,dt/(len(self.splitting_schemes)),int(ax),params)                 
                # experimental
                # Simple projection to avoid negative values on certain variables.
                sol = sol.at[0].set(jnp.abs(sol[0]))
                sol = sol.at[-1].set(jnp.abs(sol[-1]))
    
        return sol
        
 #   @partial(jit, static_argnums=0)

    def hydrostep_adapt(self, i, state, current_time):
        fields, params = state
        fields = fields.at[0].set(jnp.maximum(fields[0], 0.0))
        # Calculates an admissible local dt then bounds it by max_dt.
        ttt = self.timestep(fields)
        ttt = jnp.minimum(self.max_dt, ttt)
        if self.debug_fixed_dt is not None: #debug option by the chat
            fixed_dt = jnp.array(self.debug_fixed_dt, dtype=ttt.dtype)
            dt = jnp.minimum(ttt, fixed_dt)
            jax.debug.print("hydro_core: debug_fixed_dt active, dt = {}", dt)
        else:
            dt = ttt
        # Applies a hydro step with the selected dt.
        fields, params = self._hydrostep(i, (fields, params), dt)
        # return both the new state and the dt so host can accumulate time
        return (fields, params), dt
    

   # @jax.jit
    def _hydrostep(self, i, state, dt):
        # split forcing outside of core hydro loop
        fields, params = state
        # Strang splitting of forces: half-step before hydrodynamics.
        fields, params = self.forcing(i, fields, params, dt/2)
        # Main hydrodynamic update part.
        if self.use_mol and self.use_ct:
            #jax.debug.print("use ct")
            # Method of lines + constrained transport.
            fields = self.mol_solve_step_ct(fields, dt, params)  # <<< unsplit (MOL + CT-on-state)
        elif self.use_mol:
            # Method of lines without CT.
            fields = self.mol_solve_step(fields, dt, params)  # <<< unsplit (MOL + CT-on-state)

        else:
            # Fallback split-directionnel classique.
            fields = self.sweep_stack(state, dt, i)

        # Quick sanity check after the hydro update (before final forcing)
        _check_finite("sol after hydro", fields)

        # Final half-step of forces (closes the symmetric splitting).
        fields, params = self.forcing(i, fields, params, dt/2)
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
        # 1) Prepare le sharding des champs et place l'etat initial sur device.
        sh_arr = NamedSharding(self.mesh, self.FIELD_XYZ)
        fields = jax.device_put(input_fields.astype(self.state_dtype), sh_arr)

        # 2) Copie dt_array cote device avec un dtype coherent.
        dt_array = jnp.asarray(dt_array, dtype=fields.dtype)
        # Nombre de pas impose par la longueur du tableau dt.
        n_steps = dt_array.shape[0]

        # 3) Defini un pas hydro qui lit son dt dans le planning.
        def _one_step(fields, params, i, dt_array):
            dt = dt_array[i]
            fields, params = self._hydrostep(i, (fields, params), dt)
            return fields, params

        # Checkpointing: reduit la memoire backward au prix de recomputation.
        checkpointed_step = remat(_one_step)

        # Champs shardes, params non-shardes, dt_array replique sur la mesh.
        pjit_step = pjit(
            checkpointed_step,
            in_shardings=(sh_arr, None, None, None),
            out_shardings=(sh_arr, None),
            donate_argnums=(0,),
        )

        # 4) Boucle sur tous les pas du planning.
        def body(i, carry):
            fields, params = carry
            fields, params = pjit_step(fields, params, i, dt_array)
            return (fields, params)

        fields_f, params_f = lax.fori_loop(
            0, n_steps, body, (fields, params)
        )

        # Temps final = somme des dt imposes.
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
        # Prepare le sharding des champs.
        sh_arr = NamedSharding(self.mesh, self.FIELD_XYZ)
        # Place l'etat initial sur devices avec ce sharding.
        fields0 = jax.device_put(input_fields, sh_arr)
        # print("caca evolve")
        # Etat initial de la boucle: temps = 0, compteur de pas = 0.
        t0 = jnp.array(0.0, dtype=fields0.dtype)
        step0 = jnp.array(0, dtype=jnp.int32)

        # Historique dt de taille fixe (contraintes shape statiques XLA).
        # We re-use n_super_step as an upper bound for safety.
        max_hist_len = self.n_super_step
        dt_hist0 = jnp.zeros((max_hist_len,), dtype=fields0.dtype)

        # Cibles converties en scalaires JAX pour rester dans le graphe compile.
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
            # Condition de sortie: temps cible atteint ou nombre max de pas atteint.
            def cond_fn(carry):
                fields, params, t, dt_hist, step = carry
                return jnp.logical_and(t < t_target, step < max_steps)

            def body_fn(carry):
                fields, params, t, dt_hist, step = carry

                # Utilise l'index courant pour pilotage et enregistrement.
                fields_new, params_new, dt = _one_step(fields, params, step, t)
                t_new = t + dt

                # Enregistre dt seulement si l'index reste dans le buffer alloue.
                dt_hist_new = jax.lax.cond(
                    step < max_hist_len,
                    lambda _dt_hist: _dt_hist.at[step].set(dt),
                    lambda _dt_hist: _dt_hist,
                    dt_hist,
                )

                # Increments the step counter.
                step_new = step + jnp.array(1, dtype=step.dtype)
                return (fields_new, params_new, t_new, dt_hist_new, step_new)

            fields_f, params_f, t_f, dt_hist_f, step_f = lax.while_loop(
                cond_fn,
                body_fn,
                (fields, params, t, dt_hist, step),
            )
            return fields_f, params_f, t_f, dt_hist_f, step_f

        # Compiles the while loop in distributed version.
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

        # Updates the final time on the Python object side.
        self.sim_time = float(t_f)
        return fields_f, params_f, t_f, dt_hist, n_steps

    def evolve(self, input_fields, params):
        # 1) Describes the spatial distribution of the tensor on the mesh.
        sh_arr = NamedSharding(self.mesh, self.FIELD_XYZ)
        # 2) Converts/copies the initial state to sharded devices.
        fields = jax.device_put(input_fields.astype(self.state_dtype), sh_arr)

        # 3) Declares a unit time step (called in the JAX loop).
        def _one_step(fields, params, i):
            (fields_out, params_out),_t = self.hydrostep_adapt(i, (fields, params),0)
            return fields_out.astype(input_fields.dtype), params_out

        
        checkpointed_step = remat(_one_step)

        
        # 4) Compiles this step in distributed pjit version.
        pjit_step = pjit(
            checkpointed_step,
            in_shardings=(sh_arr, None, None),
            out_shardings=(sh_arr, None),
            donate_argnums=(0,)
        )

        # 5) Boucle principale des super-steps.
        def body(i, carry):
            fields, params = carry
            # i is a JAX scalar here; fine to pass into pjit_step as long as it
            # doesn't change shapes / trigger recompiles.
            fields, params = pjit_step(fields, params, i)
            return (fields, params)
        
        # 6) Execute n_super_step iterations on the sharded state.
        fields, params = lax.fori_loop(
            0, self.n_super_step, body, (fields, params)
        )

        return fields, params
    
    @partial(remat)#rhs right hand side
    def rhs_unsplit(self, sol, params): #cf article equation 11
        """
        Unsplit RHS computation with proper halo exchanges via boundary class.
        """
        # Conservative source term dU/dt initialized to zero.
        rhs = jnp.zeros_like(sol)

        # Loop over spatial axes
        for ax in range(1, sol.ndim):
            if sol.shape[ax] <= 1:
                continue
            # STEP 1: imposes boundaries and synchronizes necessary halos.
            sol_b = self.boundary.impose(sol, ax)

            # STEP 2: pour stencil large, assure une largeur de halo suffisante.
            # Each impose() call exchanges one layer of halos
            # For TENO5 (needs ±2 cells), call 2-3 times to be safe
            sol_b = self.boundary.impose(sol_b, ax, width=3)

            # STEP 3: calculates numerical fluxes on this axis.
            fu = self.flux(sol_b, ax, params)

            # STEP 4: divergence des flux (forme conservative) avec roll halo-aware.
            
            #add by GPT no use to fix the momentum injection. the injection function without.
            # if self.periodic_flux_divergence:
            #     flux_left = self.roll_with_halo(fu, 1, ax)
            # else:
            #     flux_left = jnp.roll(fu, 1, axis=ax)
            #     zero_face = jnp.zeros_like(jax.lax.slice_in_dim(fu, 0, 1, axis=ax))
            #     flux_left = jax.lax.dynamic_update_slice_in_dim(
            #         flux_left, zero_face, 0, axis=ax
            #     )
            # rhs = rhs - (fu - flux_left) / self.dx_o #eq 39 qrticle
            rhs = rhs - (fu - self.roll_with_halo(fu, 1, ax)) / self.dx_o #eq 39 qrticle
        # Neutralizes dB/dt here when CT explicitly manages the magnetic field.
        if getattr(self, "ct", False):
            if sol.shape[0] > self.iBx:
                rhs = rhs.at[self.iBx].set(0.0)
            if sol.shape[0] > self.iBy:
                rhs = rhs.at[self.iBy].set(0.0)
            if sol.shape[0] > self.iBz:
                rhs = rhs.at[self.iBz].set(0.0)
        return rhs
    
    def mol_solve_step(self, sol, dt, params):
        # Standard time integration of unsplit RHS (without explicit CT).
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
        # Prepare sharding + copy of the initial state.
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

        # Checkpoint seulement au niveau des blocs complets.
        checkpointed_block = remat(_block_of_steps)

        pjit_block = pjit(
            checkpointed_block,
            in_shardings=(sh_arr, None, None),
            out_shardings=(sh_arr, None),
            donate_argnums=(0,)
        )

        # Nombre de blocs pleins de taille checkpoint_every.
        n_blocks = self.n_super_step // checkpoint_every

        def body(block_idx, carry):
            fields, params = carry
            fields, params = pjit_block(fields, params, block_idx)
            return (fields, params)

        fields, params = lax.fori_loop(0, n_blocks, body, (fields, params))

        # Manages the end of loop if the number of steps is not an exact multiple.
        remainder = self.n_super_step % checkpoint_every
        if remainder > 0:
            start_i = n_blocks * checkpoint_every

            def final_substep(j, carry):
                fields, params = carry
                i = start_i + j
                fields, params = _single_step(fields, params, i)
                return (fields, params)

            # Compiles a mini-loop for the remaining steps.
            pjit_final = pjit(
                lambda f, p: lax.fori_loop(0, remainder, final_substep, (f, p)),
                in_shardings=(sh_arr, None),
                out_shardings=(sh_arr, None),
                donate_argnums=(0,)
            )
            fields, params = pjit_final(fields, params)

        return fields, params


    def add_memory_efficient_evolve_method(hydro_class):
        """
        Monkey-patch to add the memory-efficient evolve method.
        Usage:
            hydro.evolve = hydro.evolve_memory_efficient.__get__(hydro, type(hydro))
        """
        # Dynamically attaches the method on the target class.
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
        # Normalizes the integrator name for branch comparisons.
        name = self._integrator_name.upper()

        if name in ("SSPRK3", "RK3", "SSP3"):
            # --- SSPRK(3,3) ---
            # stage 1
            # Stage 1: derivative then explicit prediction.
            k1 = self.rhs_unsplit(sol, params); u1 = sol + dt * k1
            u1 = self._apply_ct_on_state(u1, params, dt)

            # stage 2
            # Stage 2: combinaison convexe SSP + correction CT ponderee.
            k2 = self.rhs_unsplit(u1, params); u2 = 0.75 * sol + 0.25 * (u1 + dt * k2)
            # Effective substep on convex combo -> 0.25*dt contributes to new part; use 0.25*dt for CT
            u2 = self._apply_ct_on_state(u2, params, 0.25 * dt)

            # stage 3
            # Stage 3: closure SSPRK3 then CT on final contribution.
            k3 = self.rhs_unsplit(u2, params); u3 = (1.0/3.0) * sol + (2.0/3.0) * (u2 + dt * k3)
            # Effective increment is (2/3)*dt on the last convex part; apply CT with that weight
            u3 = self._apply_ct_on_state(u3, params, (2.0/3.0) * dt)
            return u3

        elif name in ("RK2", "HEUN", "MIDPOINT"):
            # --- RK2 (Heun) ---
            # RK2 stage 1.
            # print("caca rk2")
            # jax.debug.print("E min dans Rk2 stage 1: {}", jnp.min(sol[0]))
            k1 = self.rhs_unsplit(sol, params); u1 = sol + dt * k1
            u1 = self._apply_ct_on_state(u1, params, dt)

            # RK2 stage 2 + moyenne de Heun.
            k2 = self.rhs_unsplit(u1, params)
            u2_pred = sol + 0.5 * dt * (k1 + k2)
            # Apply CT for the second half contribution (0.5*dt)
            u2 = self._apply_ct_on_state(u2_pred, params, 0.5 * dt)
            return u2

        elif name in ("RK4",):
            # Fallback: apply CT once after a classic RK4 step using provided integrator
            # Fallback: on reapplique l'integrateur generique puis CT global.
            u = self.integrator(self.rhs_unsplit, sol, dt, params)
            u = self._apply_ct_on_state(u, params, dt)
            return u

        else:
            # Unknown integrator: apply CT once
            # Case unknown integrator: same robust fallback strategy.
            u = self.integrator(self.rhs_unsplit, sol, dt, params)
            u = self._apply_ct_on_state(u, params, dt)
            return u

    # ---------------- CT on updated state ----------------


    
    def _apply_ct_on_state(self, sol, params, dt):
        """
        BORIS-style 2D CT:
          - build corner Ez directly from face fluxes (Fx[By], Fy[Bx])
          - update face-centered B via curl(-Ez)
          - average face->center and add to sol's cell-centered B
        """
        if sol.shape[0] <= self.iBy:
            return sol
    
        # Directional fluxes necessary for the reconstruction of EMF Ez.
        Fx = self.flux(sol, 1, params)  # axis=1 means x
        Fy = self.flux(sol, 2, params)  # axis=2 means y
    
        # BORIS corner EMF (Ez) at (i+1/2, j+1/2):
        # Ez = 0.25 * ( -Fx(By)_i,j - Fx(By)_i,j+1 + Fy(Bx)_i,j + Fy(Bx)_i+1,j )
        # Using rolls:
        FxBy = Fx[self.iBy]              # shape (x,y)
        FyBx = Fy[self.iBx]              # shape (x,y)
    
        Ez_corner = 0.25 * (
            -FxBy
            -jnp.roll(FxBy, -1, axis=1)          # j+1
            +FyBx
            +jnp.roll(FyBx, -1, axis=0)          # i+1
        )
    
        # BORIS then calls get_curl(-Ez, dx):
        # bx =  (Az - roll(Az,1,y))/dx
        # by = -(Az - roll(Az,1,x))/dx
        # with Az = -Ez (EMF). So:
        # dbx_face = ( (-Ez) - roll((-Ez),1,y) )/dx = -(Ez - roll(Ez,1,y))/dx
        # dby_face = -( (-Ez) - roll((-Ez),1,x) )/dx = +(Ez - roll(Ez,1,x))/dx
        dbx_face = -(Ez_corner - jnp.roll(Ez_corner, 1, axis=1)) / self.dx_o  # ∂(-Ez)/∂y
        dby_face = +(Ez_corner - jnp.roll(Ez_corner, 1, axis=0)) / self.dx_o  # -∂(-Ez)/∂x
    
        # If you're storing cell-centered B in sol, you need face->center averaging.
        # Match BORIS: they evolve face bx/by then average to centers for the Riemann solve.
        dBx = 0.5 * (dbx_face + jnp.roll(dbx_face, 1, axis=0))  # average x-faces -> centers
        dBy = 0.5 * (dby_face + jnp.roll(dby_face, 1, axis=1))  # average y-faces -> centers
    
        # Injects the CT correction into cell-centered magnetic components.
        sol = sol.at[self.iBx].add(dt * dBx)
        sol = sol.at[self.iBy].add(dt * dBy)
        return sol
        
    def _apply_ct_on_state_3D(self, sol, params, dt):
        """
        Constrained Transport applied to the *updated* state (MOL path).
        - Build edge-centered EMFs from face fluxes on the updated state.
        - Take curl(-E) to get face-centered dB/dt.
        - Average faces to cell centers and add to B components in `sol`.
        Works in 2D and 3D (if the state has 3 spatial dims).
        
        not properlly parallelized for multi-gpu, probably will work in forward at least
        """
        # If no magnetic rows present, nothing to do
        # If the magnetic components do not exist, exit without effect.
        if sol.shape[0] <= self.iBy:
            return sol

        # 1) Per-axis fluxes on the UPDATED state
        # Flux per direction on the already updated state.
        Fx = self.flux(sol, 1, params)  # (vars, x, y[, z])
        Fy = self.flux(sol, 2, params)
        Fz = self.flux(sol, 3, params) if sol.ndim >= 4 else None

        # 2) EMF mapping from magnetic flux rows (face-centered)
        #   Fx[By] = -E_z,  Fx[Bz] = +E_y
        #   Fy[Bx] = +E_z,  Fy[Bz] = -E_x
        #   Fz[Bx] = -E_y,  Fz[By] = +E_x
        # EMF Ez reconstruite depuis les composantes de flux magnetique.
        Ez_face = 0.5 * (-Fx[self.iBy] + Fy[self.iBx])

        if sol.ndim == 3:
            # ---------- 2D (vars, x, y) ----------
            # corners (i+1/2, j+1/2) from faces: average over x(0) and y(1)
            # Interpole Ez des faces vers coins (moyenne 4 points).
            Ez_corner = 0.25 * (
                Ez_face
                + jnp.roll(Ez_face, -1, axis=0)
                + jnp.roll(Ez_face, -1, axis=1)
                + jnp.roll(jnp.roll(Ez_face, -1, axis=0), -1, axis=1)
            )

            # curl(-E_z k̂):
            # dBx/dt on x-faces =  ∂(-Ez)/∂y  ;  dBy/dt on y-faces = -∂(-Ez)/∂x
            # Derivees discretes de curl(-E) sur les faces.
            dbx_face = ( -Ez_corner + jnp.roll(-Ez_corner, 1, axis=1) ) / self.dx_o   # derivative in y
            dby_face = (  Ez_corner - jnp.roll( Ez_corner, 1, axis=0) ) / self.dx_o   # derivative in x

            # face → cell-center averages along the normal axis
            # Projection face->centre pour coherer avec un stockage centre-cellule.
            dBx = 0.5 * (dbx_face + jnp.roll(dbx_face, 1, axis=0))  # average along x
            dBy = 0.5 * (dby_face + jnp.roll(dby_face, 1, axis=1))  # average along y

            sol = sol.at[self.iBx].add(dt * dBx)
            sol = sol.at[self.iBy].add(dt * dBy)
            return sol

        # ---------- 3D (vars, x, y, z) ----------
        # Additional EMFs from other flux rows
        # EMFs complementaires en 3D.
        Ex_face = 0.5 * ((-Fy[self.iBz]) + Fz[self.iBy])
        Ey_face = 0.5 * (( Fx[self.iBz]) - Fz[self.iBx])

        def avg4(A, ax_a, ax_b):
            """Average A with neighbors shifted by -1 along (ax_a, ax_b) in A's own axes."""
            # Voisins sur le premier axe de coin.
            A1 = jnp.roll(A, -1, axis=ax_a)
            # Voisins sur le second axe de coin.
            A2 = jnp.roll(A, -1, axis=ax_b)
            # Voisin diagonal combine.
            A3 = jnp.roll(A1, -1, axis=ax_b)
            # Moyenne bilineaire locale.
            return 0.25 * (A + A1 + A2 + A3)

        # After slicing var, EMFs are (x,y,z) with axes (0,1,2)
        Ez_corner = avg4(Ez_face, 0, 1)  # x–y corners
        Ex_corner = avg4(Ex_face, 1, 2)  # y–z corners
        Ey_corner = avg4(Ey_face, 0, 2)  # x–z corners

        # dB/dt = -curl(E) using corner EMFs
        # Composition discrete de -curl(E) composante par composante.
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

        # Injection finale des increments magnetiques au pas dt.
        sol = sol.at[self.iBx].add(dt * dBx)
        sol = sol.at[self.iBy].add(dt * dBy)
        sol = sol.at[self.iBz].add(dt * dBz)
        return sol

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruit l'objet depuis les parties statiques + dynamiques.
        return cls(*children, **aux_data)

    def tree_flatten(self):
        #this method is needed for JAX control flow, probably some easier way to do it though...
        # Pas d'enfants dynamiques exposes ici (etat porte hors de l'objet).
        children = ()  # arrays / dynamic values
        # Meta-donnees statiques necessaires pour recreer l'instance.
        aux_data = {
                    "boundary":self.boundary,
                   "splitting_schemes":self.splitting_schemes,
                    "fluxes":self.fluxes,"forces":self.forces,
                "use_mol":self.use_mol,"use_ct":self.use_ct, "pmesh_shape":self.pmesh_shape,
                # dx doit survivre au round-trip pytree, sinon self.dx_o revient a 1.0
                # dans toute methode ou l'objet est passe en argument d'un jit.
                "dx":self.dx_o}  # static values
        return (children, aux_data)
