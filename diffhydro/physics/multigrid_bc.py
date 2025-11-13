# multigrid_gs_np.py
# Non-periodic multigrid Poisson solver for JAX (Dirichlet / Neumann BCs).
# - Discretization: 2nd-order 6-point Laplacian, NO wrap
# - Restriction: separable full weighting (no wrap; BC-aware)
# - Prolongation: separable linear upsampling by 2 (no wrap; BC-aware)
# - Smoother: weighted Jacobi (omega=2/3), JIT-tiled with fori_loop
# - Cycle driver: Python recursion (V- or W-cycle via mu), robust under JIT callers

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray

# -----------------------------------------------------------------------------
# Boundary-Condition utilities
# -----------------------------------------------------------------------------

@dataclass
class BC:
    """Boundary condition description."""
    bc_type: str = "dirichlet"   # "dirichlet" or "neumann"
    value: float = 0.0           # Dirichlet value at faces (ignored for neumann)

    def is_dirichlet(self) -> bool:
        return self.bc_type.lower().startswith("d")

    def is_neumann(self) -> bool:
        return self.bc_type.lower().startswith("n")


def make_interior_mask(shape: Tuple[int, ...]) -> Array:
    """1 on interior cells, 0 on boundary faces (any ndim)."""
    mask = jnp.ones(shape, dtype=jnp.float32)
    nd = len(shape)
    for ax in range(nd):
        idx0 = [slice(None)] * nd; idx0[ax] = 0
        idx1 = [slice(None)] * nd; idx1[ax] = -1
        mask = mask.at[tuple(idx0)].set(0.0)
        mask = mask.at[tuple(idx1)].set(0.0)
    return mask


def enforce_bc(phi: Array, bc: BC) -> Array:
    """Clamp boundary cells to BC (Dirichlet) or mirror (Neumann)."""
    if bc.is_dirichlet():
        mask = make_interior_mask(phi.shape)
        val = jnp.asarray(bc.value, phi.dtype)
        return mask * phi + (1.0 - mask) * val
    else:
        # Neumann: zero-normal derivative by mirroring boundary-adjacent cells.
        nd = phi.ndim
        out = phi
        for ax in range(nd):
            # low face: copy index 1 -> 0
            idx_b = [slice(None)] * nd; idx_b[ax] = 0
            idx_i = [slice(None)] * nd; idx_i[ax] = 1
            out = out.at[tuple(idx_b)].set(out[tuple(idx_i)])
            # high face: copy index -2 -> -1
            idx_b[ax] = -1; idx_i[ax] = -2
            out = out.at[tuple(idx_b)].set(out[tuple(idx_i)])
        return out


# Neighbor helpers without wrap, honoring BCs.
def _neighbor_left(a: Array, axis: int, bc: BC) -> Array:
    nd = a.ndim
    out = jnp.zeros_like(a)
    sl_tgt = [slice(None)] * nd; sl_src = [slice(None)] * nd
    sl_tgt[axis] = slice(1, None)
    sl_src[axis] = slice(0, -1)
    out = out.at[tuple(sl_tgt)].set(a[tuple(sl_src)])
    # boundary fill at i==0
    if bc.is_dirichlet():
        val = jnp.asarray(bc.value, a.dtype)
        out = out.at[tuple([slice(None)] * axis + [0] + [slice(None)] * (nd - axis - 1))].set(val)
    else:
        # Neumann: mirror boundary cell itself
        face = a.take(indices=0, axis=axis)
        out = out.at[tuple([slice(None)] * axis + [0] + [slice(None)] * (nd - axis - 1))].set(face)
    return out


def _neighbor_right(a: Array, axis: int, bc: BC) -> Array:
    nd = a.ndim
    out = jnp.zeros_like(a)
    sl_tgt = [slice(None)] * nd; sl_src = [slice(None)] * nd
    sl_tgt[axis] = slice(0, -1)
    sl_src[axis] = slice(1, None)
    out = out.at[tuple(sl_tgt)].set(a[tuple(sl_src)])
    # boundary fill at i==-1
    if bc.is_dirichlet():
        val = jnp.asarray(bc.value, a.dtype)
        out = out.at[tuple([slice(None)] * axis + [-1] + [slice(None)] * (nd - axis - 1))].set(val)
    else:
        face = a.take(indices=-1, axis=axis)
        out = out.at[tuple([slice(None)] * axis + [-1] + [slice(None)] * (nd - axis - 1))].set(face)
    return out


# -----------------------------------------------------------------------------
# Discrete operator (no wrap) and residual
# -----------------------------------------------------------------------------

def apply_poisson_np(u: Array, h: float, bc: BC) -> Array:
    """
    A u = (sum 6 neighbors - 2d*u) / h^2 with non-periodic BC handling.
    Dirichlet: neighbors outside domain use bc.value.
    Neumann: neighbors outside domain mirror the boundary cell.
    """
    nd = u.ndim
    sum_n = jnp.zeros_like(u)
    for ax in range(nd):
        sum_n = sum_n + _neighbor_left(u, ax, bc) + _neighbor_right(u, ax, bc)
    return (sum_n - 2.0 * nd * u) / (h * h)


def residual_np(F: Array, u: Array, h: float, bc: BC) -> Array:
    return F - apply_poisson_np(u, h, bc)


# -----------------------------------------------------------------------------
# Restriction / Prolongation (no wrap; BC-aware)
# -----------------------------------------------------------------------------

def restrict_full_weighting_np(fine: Array, bc: BC) -> Array:
    """
    Separable full-weighting by factor 2, without wrap. For each axis:
    smooth with 1D kernel [1/4, 1/2, 1/4] using BC-aware neighbors, then ::2.
    """
    x = fine
    nd = x.ndim
    for ax in range(nd):
        left  = _neighbor_left(x, ax, bc)
        right = _neighbor_right(x, ax, bc)
        x = 0.25 * left + 0.5 * x + 0.25 * right
    sl = tuple(slice(None, None, 2) for _ in range(nd))
    return x[sl]


def _upsample1d_linear_np(a: Array, axis: int, bc: BC) -> Array:
    """
    Linear upsampling by factor 2 along one axis, no wrap.
    Even positions copy a; odd are average with right neighbor (BC fill at edge).
    """
    n = a.shape[axis]
    new_n = 2 * n
    new_shape = tuple(a.shape[i] if i != axis else new_n for i in range(a.ndim))
    out = jnp.zeros(new_shape, dtype=a.dtype)

    # even
    idx = [slice(None)] * a.ndim; idx[axis] = slice(0, new_n, 2)
    out = out.at[tuple(idx)].set(a)

    # right neighbor (BC-aware)
    r = _neighbor_right(a, axis, bc)
    odd_vals = 0.5 * (a + r)
    idx[axis] = slice(1, new_n, 2)
    out = out.at[tuple(idx)].set(odd_vals)
    return out


def prolong_trilinear_np(coarse: Array, fine_shape: Tuple[int, ...], bc: BC) -> Array:
    x = coarse
    for ax in range(coarse.ndim):
        x = _upsample1d_linear_np(x, ax, bc)
    if x.shape != fine_shape:
        raise ValueError(f"prolong_trilinear_np: shape {x.shape} != {fine_shape}")
    return x


# -----------------------------------------------------------------------------
# Smoother: Weighted Jacobi (BC-aware, JIT fused)
# -----------------------------------------------------------------------------

@jax.jit
def smooth_weighted_jacobi_np(u0: Array,
                              F: Array,
                              h: float,
                              iters: int,
                              omega: float = 2.0/3.0,
                              bc: Optional[BC] = None) -> Array:
    """
    Weighted Jacobi with BC enforcement every sweep.
    For Dirichlet: clamp faces to bc.value.
    For Neumann: mirror faces to enforce zero normal derivative.
    """
    if bc is None:
        bc = BC("dirichlet", 0.0)

    def body_fun(_, u):
        nd = u.ndim
        sum_n = jnp.zeros_like(u)
        for ax in range(nd):
            sum_n = sum_n + _neighbor_left(u, ax, bc) + _neighbor_right(u, ax, bc)
        denom = 2.0 * nd
        u_star = (sum_n - (h * h) * F) / denom
        u_new = (1.0 - omega) * u + omega * u_star
        # Enforce BC after each sweep
        return enforce_bc(u_new, bc)

    return jax.lax.fori_loop(0, jnp.asarray(iters, jnp.int32), body_fun, enforce_bc(u0, bc))


# -----------------------------------------------------------------------------
# One V-/W-cycle (no wrap), recursion in Python for robustness
# -----------------------------------------------------------------------------

@dataclass
class PoissonCycleNP:
    F: Array
    v1: int = 2
    v2: int = 2
    mu: int = 1            # 1 => V-cycle, 2 => W-cycle
    l: int = 1             # recursion depth below this level
    eps: float = 1e-6
    h: float = 1.0
    bc: BC = BC("dirichlet", 0.0)

    def norm(self, U: Array) -> Array:
        r = residual_np(self.F, U, self.h, self.bc)
        return jnp.sqrt(jnp.mean(r * r))

    def __call__(self, U: Array) -> Array:
        return self.do_cycle(self.F, U, self.l, self.h, self.bc)

    def do_cycle(self, F: Array, U: Array, level: int, h: float, bc: BC) -> Array:
        # Coarsest grid: extra smoothing acts as "exact" solve
        if level <= 0 or min(U.shape) <= 2:
            Uc = smooth_weighted_jacobi_np(U, F, h, iters=32, omega=2.0/3.0, bc=bc)
            return enforce_bc(Uc, bc)

        # Pre-smooth
        U = smooth_weighted_jacobi_np(U, F, h, iters=self.v1, omega=2.0/3.0, bc=bc)

        # Residual and restriction
        r = residual_np(F, U, h, bc)
        Rc = restrict_full_weighting_np(r, bc)
        hc = 2.0 * h

        # Coarse-grid correction solve A e = Rc
        Ec = jnp.zeros_like(Rc, dtype=U.dtype)
        for _ in range(int(self.mu)):  # V/W-cycle
            Ec = self.do_cycle(Rc, Ec, level - 1, hc, bc)

        # Prolongate and correct
        Ef = prolong_trilinear_np(Ec, U.shape, bc)
        U = U + Ef
        U = enforce_bc(U, bc)

        # Post-smooth
        U = smooth_weighted_jacobi_np(U, F, h, iters=self.v2, omega=2.0/3.0, bc=bc)
        return enforce_bc(U, bc)


# -----------------------------------------------------------------------------
# Public driver
# -----------------------------------------------------------------------------

def multigrid_np(cycle: PoissonCycleNP, U: Array, eps: float, iter_cycle: int) -> Array:
    """
    Run a fixed number of cycles (kept in Python for JAX robustness).
    If you want tolerance-based stopping, add a host-side check around this loop.
    """
    U_out = U
    for _ in range(int(iter_cycle)):
        U_out = cycle(U_out)
        # Optional host-side early-stop:
        # if float(cycle.norm(U_out)) <= eps: break
    return U_out


def poisson_multigrid_np(F: Array,
                         U: Array,
                         l: int,
                         v1: int,
                         v2: int,
                         mu: int,
                         iter_cycle: int,
                         eps: float = 1e-6,
                         h: Optional[float] = None,
                         bc_type: str = "dirichlet",
                         bc_value: float = 0.0) -> Array:
    """
    Solve A phi = F with non-periodic BCs.
      - bc_type: "dirichlet" (phi = bc_value on faces) or "neumann" (zero-normal derivative)
      - bc_value: only used for Dirichlet
    """
    if h is None:
        h = 1.0
    bc = BC(bc_type, bc_value)
    cycle = PoissonCycleNP(F=F, v1=v1, v2=v2, mu=mu, l=l, eps=eps, h=h, bc=bc)
    return multigrid_np(cycle, U, eps, iter_cycle)
