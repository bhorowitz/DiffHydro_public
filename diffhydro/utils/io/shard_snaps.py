import os
import numpy as np


def load_snapshot_cpu(
    step_i: int,
    snapshot_dir: str,
    pmesh_shape=(1, 1, 1),
    partition_spec=(None, "x", "y", "z"),
):
    """
    Load and recombine a full snapshot from individual device shards,
    without requiring a hydrosim object or any GPUs.

    Parameters
    ----------
    step_i : int
        Snapshot step index (the number in fields_step_XXXXXX_device_YYY.npy).
    snapshot_dir : str
        Directory where snapshot shards were written.
    pmesh_shape : tuple(int, int, int)
        The parallel mesh shape (nx, ny, nz) used when writing snapshots.
        Must match the pmesh_shape you used in the hydro run.
    partition_spec : tuple
        PartitionSpec-like tuple indicating which field axes are sharded.
        For the current hydro implementation this is always:
            (None, "x", "y", "z")
        meaning:
            axis 0 = variables (not sharded)
            axis 1 = x, axis 2 = y, axis 3 = z

    Returns
    -------
    full_field : np.ndarray
        Reconstructed full snapshot array on CPU.
    """
    nx, ny, nz = pmesh_shape

    # Load all shards into a dict indexed by (x, y, z)
    shards = {}
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                linear_idx = x * (ny * nz) + y * nz + z
                path = os.path.join(
                    snapshot_dir,
                    f"fields_step_{step_i:06d}_device_{linear_idx}.npy",
                )
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing shard file: {path}")
                shards[(x, y, z)] = np.load(path)

    # Inspect one shard to get per-shard field shape
    field_shape = shards[(0, 0, 0)].shape

    # Build mapping from mesh axis names -> field dimension index
    # e.g. (None, "x", "y", "z") -> {"x": 1, "y": 2, "z": 3}
    axis_map = {}
    for dim, name in enumerate(partition_spec):
        if name is not None:
            axis_map[name] = dim

    # Now reconstruct by concatenating along each partitioned axis
    # in the order z -> y -> x (to match how linear_idx is built).

    # First: handle z
    result = shards
    if nz > 1 and "z" in axis_map:
        z_axis = axis_map["z"]
        new_result = {}
        for x in range(nx):
            for y in range(ny):
                parts = [result[(x, y, z)] for z in range(nz)]
                new_result[(x, y)] = np.concatenate(parts, axis=z_axis)
        result = new_result
    elif nz == 1:
        # No z partitioning, drop z index
        result = {(x, y): result[(x, y, 0)] for x in range(nx) for y in range(ny)}

    # Next: handle y
    if ny > 1 and "y" in axis_map:
        y_axis = axis_map["y"]
        new_result = {}
        for x in range(nx):
            parts = [result[(x, y)] for y in range(ny)]
            new_result[x] = np.concatenate(parts, axis=y_axis)
        result = new_result
    elif ny == 1:
        result = {x: result[(x, 0)] for x in range(nx)}

    # Finally: handle x
    if nx > 1 and "x" in axis_map:
        x_axis = axis_map["x"]
        parts = [result[x] for x in range(nx)]
        full_field = np.concatenate(parts, axis=x_axis)
    else:
        full_field = result[0]

    return full_field


def load_snapshot(hydrosim, step_i, snapshot_dir=None):
    """Load and recombine a full snapshot from individual device shards."""
    import numpy as np
    import os
    
    if snapshot_dir is None:
        snapshot_dir = hydrosim.snapshot_dir
    
    mesh_shape = hydrosim.mesh.shape
    nx, ny, nz = mesh_shape['x'], mesh_shape['y'], mesh_shape['z']
    
    # Load all shards into a dictionary indexed by (x, y, z)
    shards = {}
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                linear_idx = x * (ny * nz) + y * nz + z
                path = os.path.join(snapshot_dir, f"fields_step_{step_i:06d}_device_{linear_idx}.npy")
                shards[(x, y, z)] = np.load(path)
    
    # Figure out which field axes correspond to which mesh axes
    # by inspecting FIELD_XYZ
    field_shape = shards[(0, 0, 0)].shape
    partition_spec = hydrosim.FIELD_XYZ
    
    # Map mesh axis names to field dimension indices
    axis_map = {}  # e.g., {'x': 0, 'y': 2} means mesh-x maps to field dim 0, mesh-y to field dim 2
    for field_dim, spec_name in enumerate(partition_spec):
        if spec_name is not None:
            axis_map[spec_name] = field_dim
    
    print(f"Shard shape: {field_shape}")
    print(f"Partition spec: {partition_spec}")
    print(f"Axis mapping: {axis_map}")
    
    # Concatenate along each partitioned axis in order: z, y, x
    result = shards
    
    # Concatenate along z if partitioned
    if nz > 1 and 'z' in axis_map:
        z_axis = axis_map['z']
        new_result = {}
        for x in range(nx):
            for y in range(ny):
                z_parts = [result[(x, y, z)] for z in range(nz)]
                new_result[(x, y)] = np.concatenate(z_parts, axis=z_axis)
        result = new_result
    elif nz == 1:
        # No z partitioning, just drop z index
        result = {(x, y): result[(x, y, 0)] for x in range(nx) for y in range(ny)}
    
    # Concatenate along y if partitioned
    if ny > 1 and 'y' in axis_map:
        y_axis = axis_map['y']
        new_result = {}
        for x in range(nx):
            y_parts = [result[(x, y)] for y in range(ny)]
            new_result[x] = np.concatenate(y_parts, axis=y_axis)
        result = new_result
    elif ny == 1:
        result = {x: result[(x, 0)] for x in range(nx)}
    
    # Concatenate along x if partitioned
    if nx > 1 and 'x' in axis_map:
        x_axis = axis_map['x']
        x_parts = [result[x] for x in range(nx)]
        full_field = np.concatenate(x_parts, axis=x_axis)
    else:
        full_field = result[0]
    
    return full_field