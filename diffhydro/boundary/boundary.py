# boundary.py
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


class PeriodicBoundaryExplicit:
    """
    Alternative implementation that explicitly pads arrays with ghost cells.
    More expensive but easier to understand and debug.
    """
    
    def __init__(self, mesh=None, pmesh_shape=(1,1,1), field_spec=None):
        self.mesh = mesh
        self.pmesh_shape = pmesh_shape
        self.field_spec = field_spec if field_spec is not None else P(None, 'x', 'y', 'z')
    
    def impose(self, sol, axis,width=1):
        """
        Apply periodic BCs with explicit ghost cell padding.
        """
        axis_idx = axis - 1
        is_sharded = self.pmesh_shape[axis_idx] > 1
        
        if not is_sharded:
            # For non-sharded axes, periodic BC is automatic with jnp.roll
            return sol
        else:
            # Add ghost cells by exchanging with neighbors
            return self._add_ghost_cells(sol, axis, width=width)
    
    def _add_ghost_cells(self, sol, axis, width=1):
        """
        Explicitly pad array with ghost cells from neighboring GPUs.
        
        Args:
            sol: Solution array
            axis: Axis to pad (1, 2, or 3)
            width: Number of ghost cell layers to add
        
        Returns:
            Array with shape increased by 2*width along the given axis
        """
        axis_idx = axis - 1
        axis_name = ('x', 'y', 'z')[axis_idx]
        n_devices = self.pmesh_shape[axis_idx]
        
        if n_devices <= 1:
            return sol
        
        def _pad_with_halos(local_sol):
            """
            Pad local array with ghost cells from neighbors.
            """
            # Get slices to send to neighbors
            # Left ghost: receive from previous GPU's right edge
            # Right ghost: receive from next GPU's left edge
            
            left_slice = jax.lax.slice_in_dim(local_sol, 0, width, axis=axis)
            right_slice = jax.lax.slice_in_dim(local_sol, -width, None, axis=axis)
            
            # Exchange with neighbors (periodic wrapping)
            perm_forward = [(i, (i + 1) % n_devices) for i in range(n_devices)]
            perm_backward = [(i, (i - 1) % n_devices) for i in range(n_devices)]
            
            left_ghost = jax.lax.ppermute(right_slice, axis_name, perm_backward)
            right_ghost = jax.lax.ppermute(left_slice, axis_name, perm_forward)
            
            # Concatenate: [left_ghost | local_sol | right_ghost]
            padded = jnp.concatenate([left_ghost, local_sol, right_ghost], axis=axis)
            
            return padded
        
        return shard_map(
            _pad_with_halos,
            mesh=self.mesh,
            in_specs=self.field_spec,
            out_specs=self.field_spec,
            check_rep=False
        )(sol)


class PeriodicBoundarySimple:
    """
    Simplest implementation: just ensure data is visible across GPU boundaries.
    Uses roll_with_halo-style approach.
    """
    
    def __init__(self, mesh=None, pmesh_shape=(1,1,1), field_spec=None, roll_fn=None):
        self.mesh = mesh
        self.pmesh_shape = pmesh_shape
        self.field_spec = field_spec if field_spec is not None else P(None, 'x', 'y', 'z')
        self.roll_fn = roll_fn  # Pass in hydro.roll_with_halo
    
    def impose(self, sol, axis):
        """
        For periodic BCs with multi-GPU, just ensure halos are synced.
        The actual periodic wrapping is handled by roll operations in the flux computation.
        """
        axis_idx = axis - 1
        is_sharded = self.pmesh_shape[axis_idx] > 1
        
        if not is_sharded or self.roll_fn is None:
            return sol
        
        # "Touch" the boundaries by doing a roll and roll-back
        # This forces JAX to synchronize halos
        # Roll forward
        temp = self.roll_fn(sol, 1, axis)
        # Roll backward
        temp = self.roll_fn(temp, -1, axis)
        
        # At this point, halos should be synced
        # Return original array (the temp was just to trigger sync)
        return sol


class NoBoundary:
    """
    No-op boundary for testing or when boundaries are handled elsewhere.
    """
    def __init__(self, **kwargs):
        pass
    
    def impose(self, sol, axis, width=1):
        """No boundary conditions applied."""
        return sol
    
class OutflowBoundary:
    """
    Probably broken right now...

    Outflow (zero-gradient) boundary conditions with multi-GPU support.
    
    At physical boundaries: du/dn = 0 (extrapolate from interior)
    At GPU boundaries: exchange halos normally
    """
    
    def __init__(self, mesh=None, pmesh_shape=(1,1,1), field_spec=None):
        self.mesh = mesh
        self.pmesh_shape = pmesh_shape
        self.field_spec = field_spec if field_spec is not None else P(None, 'x', 'y', 'z')
    
    def impose(self, sol, axis):
        """
        Apply outflow BCs with proper multi-GPU handling.
        """
        # Handle 2D case: if z-dimension is 1, skip z-axis
        if axis >= sol.ndim:
            return sol
        
        # If the axis dimension is 1, no BC needed (degenerate dimension)
        if sol.shape[axis] <= 1:
            return sol
        
        axis_idx = axis - 1
        is_sharded = self.pmesh_shape[axis_idx] > 1
        
        if not is_sharded:
            # Single GPU or not sharded on this axis
            # Apply zero-gradient at physical boundaries
            return self._apply_zero_gradient_local(sol, axis)
        else:
            # Multi-GPU: need to handle GPU boundaries AND physical boundaries
            return self._apply_zero_gradient_distributed(sol, axis)
    
    def _apply_zero_gradient_local(self, sol, axis):
        """
        Apply zero-gradient BC on a single GPU.
        """
        # Skip if axis is out of bounds or degenerate
        if axis >= sol.ndim or sol.shape[axis] <= 1:
            return sol
        
        # Left boundary: set first cell equal to second cell
        interior_left = jax.lax.slice_in_dim(sol, 1, 2, axis=axis)
        sol = jax.lax.dynamic_update_slice_in_dim(sol, interior_left, 0, axis=axis)
        
        # Right boundary: set last cell equal to second-to-last cell
        interior_right = jax.lax.slice_in_dim(sol, -2, -1, axis=axis)
        sol = jax.lax.dynamic_update_slice_in_dim(sol, interior_right, 
                                                   sol.shape[axis]-1, axis=axis)
        
        return sol
    
    def _apply_zero_gradient_distributed(self, sol, axis):
        """
        Apply zero-gradient BC with multi-GPU.
        """
        # Skip if axis is out of bounds or degenerate
        if axis >= sol.ndim or sol.shape[axis] <= 1:
            return sol
        
        axis_idx = axis - 1
        axis_name = ('x', 'y', 'z')[axis_idx]
        n_devices = self.pmesh_shape[axis_idx]
        
        def _process_boundaries(local_sol):
            """
            Handle both GPU boundaries and physical boundaries.
            """
            my_rank = jax.lax.axis_index(axis_name)
            
            # Step 1: Exchange halos with neighboring GPUs
            left_boundary = jax.lax.slice_in_dim(local_sol, 0, 1, axis=axis)
            right_boundary = jax.lax.slice_in_dim(local_sol, -1, None, axis=axis)
            
            # Exchange with neighbors
            if n_devices > 1:
                perm_forward = [(i, (i + 1) % n_devices) for i in range(n_devices)]
                left_halo = jax.lax.ppermute(right_boundary, axis_name, perm_forward)
                
                perm_backward = [(i, (i - 1) % n_devices) for i in range(n_devices)]
                right_halo = jax.lax.ppermute(left_boundary, axis_name, perm_backward)
                
                # Step 2: Override halos at physical boundaries with extrapolation
                interior_left = jax.lax.slice_in_dim(local_sol, 1, 2, axis=axis)
                left_halo = jnp.where(my_rank == 0, interior_left, left_halo)
                
                interior_right = jax.lax.slice_in_dim(local_sol, -2, -1, axis=axis)
                right_halo = jnp.where(my_rank == n_devices - 1, interior_right, right_halo)
                
                # Step 3: Update boundary cells
                local_sol = jax.lax.dynamic_update_slice_in_dim(
                    local_sol, left_halo, 0, axis=axis)
                local_sol = jax.lax.dynamic_update_slice_in_dim(
                    local_sol, right_halo, local_sol.shape[axis] - 1, axis=axis)
            
            return local_sol
        
        if self.mesh is None:
            return _process_boundaries(sol)
        else:
            return shard_map(
                _process_boundaries,
                mesh=self.mesh,
                in_specs=self.field_spec,
                out_specs=self.field_spec,
                check_rep=False
            )(sol)


class PeriodicBoundary:
    """
    Periodic boundary conditions - the simplest case.
    Just exchange halos; the periodic topology is automatic.
    """
    
    def __init__(self, mesh=None, pmesh_shape=(1,1,1), field_spec=None):
        self.mesh = mesh
        self.pmesh_shape = pmesh_shape
        self.field_spec = field_spec if field_spec is not None else P(None, 'x', 'y', 'z')
    
    def impose(self, sol, axis):
        """
        For periodic BCs, just ensure halos are synced.
        No modification of boundary values needed - jnp.roll handles wrapping.
        """
        # Handle 2D case: if axis doesn't exist or is degenerate, skip
        if axis >= sol.ndim or sol.shape[axis] <= 1:
            return sol
        
        axis_idx = axis - 1
        is_sharded = self.pmesh_shape[axis_idx] > 1
        
        if not is_sharded:
            # Not sharded - jnp.roll automatically wraps (periodic)
            return sol
        else:
            # Sharded - exchange halos with periodic wrapping
            return self._exchange_halos(sol, axis)
    
    def _exchange_halos(self, sol, axis):
        """
        Exchange one layer of halos with periodic wrapping.
        """
        axis_idx = axis - 1
        axis_name = ('x', 'y', 'z')[axis_idx]
        n_devices = self.pmesh_shape[axis_idx]
        
        def _exchange(local_sol):
            # Send boundaries to neighbors
            left_boundary = jax.lax.slice_in_dim(local_sol, 0, 1, axis=axis)
            right_boundary = jax.lax.slice_in_dim(local_sol, -1, None, axis=axis)
            
            # Periodic exchange
            perm_forward = [(i, (i + 1) % n_devices) for i in range(n_devices)]
            perm_backward = [(i, (i - 1) % n_devices) for i in range(n_devices)]
            
            _ = jax.lax.ppermute(right_boundary, axis_name, perm_forward)
            _ = jax.lax.ppermute(left_boundary, axis_name, perm_backward)
            
            return local_sol
        
        if self.mesh is None:
            return sol
        else:
            return shard_map(
                _exchange,
                mesh=self.mesh,
                in_specs=self.field_spec,
                out_specs=self.field_spec,
                check_rep=False
            )(sol)


class ReflectiveBoundary:
    """
    Broken right now...
    Reflective (wall) boundary conditions with multi-GPU support.
    """
    
    def __init__(self, mesh=None, pmesh_shape=(1,1,1), field_spec=None, 
                 vel_ids=(1, 2, 3)):
        self.mesh = mesh
        self.pmesh_shape = pmesh_shape
        self.field_spec = field_spec if field_spec is not None else P(None, 'x', 'y', 'z')
        self.vel_ids = vel_ids
    
    def impose(self, sol, axis):
        """
        Apply reflective BCs with proper multi-GPU handling.
        """
        # Handle 2D case
        if axis >= sol.ndim or sol.shape[axis] <= 1:
            return sol
        
        axis_idx = axis - 1
        is_sharded = self.pmesh_shape[axis_idx] > 1
        
        if not is_sharded:
            return self._apply_reflective_local(sol, axis)
        else:
            return self._apply_reflective_distributed(sol, axis)
    
    def _apply_reflective_local(self, sol, axis):
        """
        Apply reflective BC on a single GPU.
        """
        if axis >= sol.ndim or sol.shape[axis] <= 1:
            return sol
        
        # Determine which velocity component is normal
        normal_vel_idx = self.vel_ids[axis - 1]
        
        # Left boundary
        interior_left = jax.lax.slice_in_dim(sol, 1, 2, axis=axis)
        boundary_left = interior_left.at[normal_vel_idx].set(-interior_left[normal_vel_idx])
        sol = jax.lax.dynamic_update_slice_in_dim(sol, boundary_left, 0, axis=axis)
        
        # Right boundary
        interior_right = jax.lax.slice_in_dim(sol, -2, -1, axis=axis)
        boundary_right = interior_right.at[normal_vel_idx].set(-interior_right[normal_vel_idx])
        sol = jax.lax.dynamic_update_slice_in_dim(sol, boundary_right, 
                                                   sol.shape[axis] - 1, axis=axis)
        
        return sol
    
    def _apply_reflective_distributed(self, sol, axis):
        """
        Apply reflective BC with multi-GPU.
        """
        if axis >= sol.ndim or sol.shape[axis] <= 1:
            return sol
        
        axis_idx = axis - 1
        axis_name = ('x', 'y', 'z')[axis_idx]
        n_devices = self.pmesh_shape[axis_idx]
        normal_vel_idx = self.vel_ids[axis - 1]
        
        def _process_reflective(local_sol):
            my_rank = jax.lax.axis_index(axis_name)
            
            left_boundary = jax.lax.slice_in_dim(local_sol, 0, 1, axis=axis)
            right_boundary = jax.lax.slice_in_dim(local_sol, -1, None, axis=axis)
            
            if n_devices > 1:
                perm_forward = [(i, (i + 1) % n_devices) for i in range(n_devices)]
                left_halo = jax.lax.ppermute(right_boundary, axis_name, perm_forward)
                
                perm_backward = [(i, (i - 1) % n_devices) for i in range(n_devices)]
                right_halo = jax.lax.ppermute(left_boundary, axis_name, perm_backward)
                
                # Apply reflection at physical boundaries
                interior_left = jax.lax.slice_in_dim(local_sol, 1, 2, axis=axis)
                wall_left = interior_left.at[normal_vel_idx].set(-interior_left[normal_vel_idx])
                left_halo = jnp.where(my_rank == 0, wall_left, left_halo)
                
                interior_right = jax.lax.slice_in_dim(local_sol, -2, -1, axis=axis)
                wall_right = interior_right.at[normal_vel_idx].set(-interior_right[normal_vel_idx])
                right_halo = jnp.where(my_rank == n_devices - 1, wall_right, right_halo)
                
                local_sol = jax.lax.dynamic_update_slice_in_dim(
                    local_sol, left_halo, 0, axis=axis)
                local_sol = jax.lax.dynamic_update_slice_in_dim(
                    local_sol, right_halo, local_sol.shape[axis] - 1, axis=axis)
            
            return local_sol
        
        if self.mesh is None:
            return _process_reflective(sol)
        else:
            return shard_map(
                _process_reflective,
                mesh=self.mesh,
                in_specs=self.field_spec,
                out_specs=self.field_spec,
                check_rep=False
            )(sol)