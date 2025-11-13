import os
import numpy as np

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