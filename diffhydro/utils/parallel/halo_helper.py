import jax
import jax.lax as lax

def _axis_len(a, ax): return a.shape[ax]

def _neighbor_pairs_1d(mesh, axis_name):
    # Build (src->dst) pairs for a 1D ring along axis_name
    idxs = list(range(mesh.devices.shape[mesh.axis_names.index(axis_name)]))
    pairs = []
    for i in idxs:
        left  = (i - 1) % len(idxs)
        right = (i + 1) % len(idxs)
    # We’ll return functions to target left/right via collective_permute
    return idxs

def _send_recv_boundary(slice_to_send, src_dst_pairs):
    # src_dst_pairs: list[(src, dst)]
    return lax.collective_permute(slice_to_send, src_dst_pairs)

def halo_pad_1d(x, *, pad: int, axis: int, axis_name: str, periodic: bool = True):
    """
    Add a 1D halo of width `pad` along `axis` by exchanging boundary slabs
    with neighbors along the named mesh axis `axis_name`.

    Works with pjit/GSPMD when called under `with mesh:` where `axis_name`
    is one of the mesh's named axes (e.g., 'x', 'y', 'z').

    Args:
      x:           Array sharded along `axis` by the mesh axis `axis_name`.
      pad:         Halo width (in cells) to add on each side along `axis`.
      axis:        Integer array axis to halo-extend (0-based over x.ndim).
      axis_name:   Name of the mesh axis that shards `axis` (e.g., 'x').
      periodic:    If False, use zeros outside the global domain.

    Returns:
      x_halo: Array with shape equal to `x` except `shape[axis]` increased by 2*pad.
    """
    # Grab boundary slabs to send
    n = x.shape[axis]
    left_send  = lax.slice_in_dim(x, 0,   pad, axis=axis)       # [ ... pad ]
    right_send = lax.slice_in_dim(x, n-pad, n,   axis=axis)     # [ ... pad ]

    # Gather every shard's boundary slabs along the mesh axis.
    # Shapes become: [mesh_size, ... pad ...]
    left_all  = lax.all_gather(left_send,  axis_name)
    right_all = lax.all_gather(right_send, axis_name)

    # Get my coordinate along the mesh axis and the axis size (on-device scalars).
    my_idx  = lax.axis_index(axis_name)        # 0..mesh_size-1
    ax_size = lax.psum(1, axis_name)           # mesh_size as a scalar value

    # Compute neighbor indices (periodic ring by default)
    left_nbr  = (my_idx - 1) % ax_size
    right_nbr = (my_idx + 1) % ax_size

    # Pull the appropriate neighbor slabs from the gathered stacks.
    # left_recv = right_send from my LEFT neighbor
    # right_recv = left_send from my RIGHT neighbor
    left_recv  = jnp.take(right_all, left_nbr,  axis=0, mode='wrap')
    right_recv = jnp.take(left_all,  right_nbr, axis=0, mode='wrap')

    if not periodic:
        # For non-periodic, zero out the outer slabs where neighbors don't exist.
        # (leftmost shard has no left neighbor; rightmost has no right neighbor)
        is_leftmost  = (my_idx == 0)
        is_rightmost = (my_idx == ax_size - 1)
        # Broadcast bools to the slab shapes
        while is_leftmost.ndim < left_recv.ndim:
            is_leftmost = is_leftmost[..., None]
        while is_rightmost.ndim < right_recv.ndim:
            is_rightmost = is_rightmost[..., None]
        left_recv  = jnp.where(is_leftmost,  jnp.zeros_like(left_recv),  left_recv)
        right_recv = jnp.where(is_rightmost, jnp.zeros_like(right_recv), right_recv)

    # Concatenate halos around x along the target axis
    x_halo = jnp.concatenate([left_recv, x, right_recv], axis=axis)
    return x_halo