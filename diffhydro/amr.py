# diffhydro/amr.py
import jax
import jax.numpy as jnp

@jax.tree_util.register_pytree_node_class
class AMRBlock:
    def __init__(self, U, mask, origin, dx):
        self.U = U            # [C, H, W] dense tile incl. halos if you prefer
        self.mask = mask      # [1, H, W] float (1=active)
        self.origin = origin  # (i,j) in coarse-level cell indices
        self.dx = dx

    def tree_flatten(self):
        return ((self.U, self.mask), {"origin": self.origin, "dx": self.dx})

    @classmethod
    def tree_unflatten(cls, aux, children):
        U, mask = children
        return cls(U, mask, aux["origin"], aux["dx"])
    
@jax.tree_util.register_pytree_node_class
class AMRLevel:
    def __init__(self, ratio, blocks):
        self.ratio = ratio      # e.g., 2
        self.blocks = tuple(blocks)
    def tree_flatten(self):
        # ratio is static metadata; blocks is the dynamic tree
        return ((self.blocks,), {"ratio": self.ratio})
    @classmethod
    def tree_unflatten(cls, aux, children):
        (blocks,) = children
        return cls(aux["ratio"], blocks)
        
@jax.tree_util.register_pytree_node_class
class AMRHierarchy:
    def __init__(self, levels):
        self.levels = tuple(levels)
        
    def tree_flatten(self):
        return ((self.levels,), {})
    @classmethod
    def tree_unflatten(cls, aux, children):
        (levels,) = children
        return cls(levels)