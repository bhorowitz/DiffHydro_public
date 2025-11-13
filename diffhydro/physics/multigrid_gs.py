# multigrid_gs.py
# Periodic multigrid Poisson solver for JAX hydro, tracer-safe version.
# - Discretization: 2nd-order 6-point Laplacian (periodic)
# - Restriction: separable full weighting
# - Prolongation: separable trilinear interpolation (factor 2)
# - Smoother: weighted Jacobi (omega=2/3), fully vectorized
# - Outer MG driver: fixed # of V-cycles via Python for-loop (robust with JIT callers)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp


Array = jnp.ndarray


# -----------------------------------------------------------------------------
# Core discrete operators (periodic, vectorized)
# -----------------------------------------------------------------------------

def apply_poisson(u: Array, h: float) -> Array:
    """A u = -6u + sum of 6 neighbors, divided by h^2 (periodic 3D/2D/1D)."""
    nd = u.ndim
    lap = -2.0 * nd * u
    for ax in range(nd):
        lap = lap + jnp.roll(u, +1, axis=ax) + jnp.roll(u, -1, axis=ax)
    return lap / (h * h)


def residual(F: Array, u: Array, h: float) -> Array:
    """r = F - A u."""
    return F - apply_poisson(u, h)


# -----------------------------------------------------------------------------
# Restriction / Prolongation (periodic, separable)
# -----------------------------------------------------------------------------

def restrict_full_weighting(fine: Array) -> Array:
    """
    Separable full-weighting restriction by factor 2 per axis.
    For 3D this matches 1/8 sum over the 8 fine cells around the coarse point.
    Implemented as 1D [1/4, 1/2, 1/4] smoothing along each axis, then take ::2.
    """
    x = fine
    nd = x.ndim

    # 1D full-weighting kernel is [0.25, 0.5, 0.25] applied separably.
    for ax in range(nd):
        x = 0.25 * jnp.roll(x, +1, ax) + 0.5 * x + 0.25 * jnp.roll(x, -1, ax)

    # Subsample by 2 along each axis
    sl = tuple(slice(None, None, 2) for _ in range(nd))
    coarse = x[sl]
    return coarse


def _upsample1d_linear(a: Array, axis: int) -> Array:
    """
    Periodic linear upsampling by factor 2 along a single axis.
    Even positions copy a; odd positions are average with next cell.
    """
    n = a.shape[axis]
    new_n = 2 * n
    new_shape = tuple(a.shape[i] if i != axis else new_n for i in range(a.ndim))
    out = jnp.zeros(new_shape, dtype=a.dtype)

    # even: out[..., 0::2, ...] = a
    idx_even = [slice(None)] * a.ndim
    idx_even[axis] = slice(0, new_n, 2)
    out = out.at[tuple(idx_even)].set(a)

    # odd: out[..., 1::2, ...] = 0.5*(a + roll(a, -1))
    a_avg = 0.5 * (a + jnp.roll(a, -1, axis))
    idx_odd = [slice(None)] * a.ndim
    idx_odd[axis] = slice(1, new_n, 2)
    out = out.at[tuple(idx_odd)].set(a_avg)

    return out


def prolong_trilinear(coarse: Array, fine_shape: Tuple[int, ...]) -> Array:
    """
    Separable trilinear (bilinear/linear) interpolation to double each axis size.
    Does three 1D linear upsampling passes. Periodic wrap.
    """
    x = coarse
    for ax in range(coarse.ndim):
        x = _upsample1d_linear(x, axis=ax)
    # Safety: ensure the requested fine_shape (should match 2x on each axis)
    if x.shape != fine_shape:
        raise ValueError(f"prolong_trilinear: shape mismatch {x.shape} != {fine_shape}")
    return x


# -----------------------------------------------------------------------------
# Smoother: Weighted Jacobi (vectorized, periodic)
# -----------------------------------------------------------------------------

def jacobi_sweep(u: Array, F: Array, h: float, omega: float = 2.0 / 3.0) -> Array:
    """
    One weighted-Jacobi sweep for A u = F with A = (sum 6 nbrs - 2d*u)/h^2.
    Update formula (3D): u_new = (1-ω)*u + ω*( (sum_neighbors - h^2 F) / (2d) ).
    """
    nd = u.ndim
    # Sum of 6 neighbors in nd dims:
    sum_nbrs = jnp.zeros_like(u)
    for ax in range(nd):
        sum_nbrs = sum_nbrs + jnp.roll(u, +1, axis=ax) + jnp.roll(u, -1, axis=ax)

    denom = 2.0 * nd  # 2 per axis in the stencil
    u_star = (sum_nbrs - (h * h) * F) / denom
    return (1.0 - omega) * u + omega * u_star



@partial(jax.jit, static_argnums=(3,))
def smooth_weighted_jacobi(u0, F, h, iters, omega=2.0/3.0):
    """
    Weighted Jacobi smoother for A u = F with 6-point Laplacian.
    `iters` MUST be a Python int (or at least static) for reverse-mode to work.
    """
    iters = int(iters)  # force static loop bound

    def body(k, u):
        nd = u.ndim
        sum_n = jnp.zeros_like(u)
        for ax in range(nd):
            sum_n = sum_n + jnp.roll(u, +1, axis=ax) + jnp.roll(u, -1, axis=ax)
        denom = 2.0 * nd
        u_star = (sum_n - (h*h) * F) / denom
        return (1.0 - omega) * u + omega * u_star

    return jax.lax.fori_loop(0, iters, body, u0)


def smooth_weighted_jacobi_old(u0: Array, F: Array, h: float, iters: int, omega: float = 2.0 / 3.0) -> Array:
    """
    Run iters sweeps of weighted Jacobi (small fixed iters; plain Python loop is ok).
    """
    u = u0
    for _ in range(int(iters)):
        u = jacobi_sweep(u, F, h, omega=omega)
    return u


# -----------------------------------------------------------------------------
# One multigrid V-cycle (periodic), no JIT on the recursive structure
# -----------------------------------------------------------------------------

@dataclass
class PoissonCycle:
    F: Array
    v1: int = 2
    v2: int = 2
    mu: int = 1           # 1 => V-cycle, 2 => W-cycle
    l: int = 1           # levels below this (0 = coarsest)
    eps: float = 1e-6
    h: float = 1.0
    laplace: Optional[Callable[[Array, float], Array]] = None  # kept for API symmetry

    # --- API compatibility helpers used by callers (optional) ---
    def norm(self, U: Array) -> Array:
        """RMS residual norm (scalar) — useful if you want to monitor convergence."""
        r = residual(self.F, U, self.h)
        return jnp.sqrt(jnp.mean(r * r))

    def __call__(self, U: Array) -> Array:
        """One V-/W-cycle application."""
        return self.do_cycle(self.F, U, self.l, self.h)

    # --- Core cycle ---
    def do_cycle(self, F: Array, U: Array, level: int, h: float) -> Array:
        """
        Recursive V-cycle (mu=1) / W-cycle (mu=2).
        On the coarsest grid (level==0) we do a few extra Jacobi sweeps.
        """
        # Coarsest grid: "exact" solve via more smoothing sweeps
        if level <= 0 or min(U.shape) <= 2:
            Uc = smooth_weighted_jacobi(U, F, h, iters=16, omega=2.0 / 3.0)
            return Uc

        # Pre-smooth
        U = smooth_weighted_jacobi(U, F, h, iters=self.v1, omega=2.0 / 3.0)

        # Residual and restrict to coarse
        r = residual(F, U, h)
        Rc = restrict_full_weighting(r)

        # Coarse-grid operator uses doubled spacing
        hc = 2.0 * h

        # Coarse-grid correction: solve A e = Rc
        Ec = jnp.zeros_like(Rc, dtype=U.dtype)

        # mu times down-and-up (V:1, W:2)
        for _ in range(int(self.mu)):
            Ec = self.do_cycle(Rc, Ec, level - 1, hc)

        # Prolongate and correct
        Ef = prolong_trilinear(Ec, U.shape)
        U = U + Ef

        # Post-smooth
        U = smooth_weighted_jacobi(U, F, h, iters=self.v2, omega=2.0 / 3.0)
        return U


# -----------------------------------------------------------------------------
# Public driver
# -----------------------------------------------------------------------------

def multigrid(cycle: PoissonCycle, U: Array, eps: float, iter_cycle: int) -> Array:
    """
    Run a fixed number of V-/W-cycles. Using a plain Python loop keeps this
    robust under JIT/scan/pjit callers (no data-dependent conds inside jitted code).
    """
    U_out = U
    for _ in range(int(iter_cycle)):
        U_out = cycle(U_out)
        # Optional: if you *really* want a tolerance stop, do it on host:
        # if float(cycle.norm(U_out)) <= eps:
        #     break
    return U_out


def poisson_multigrid(F: Array,
                      U: Array,
                      l: int,
                      v1: int,
                      v2: int,
                      mu: int,
                      iter_cycle: int,
                      eps: float = 1e-6,
                      h: Optional[float] = None,
                      laplace: Optional[Callable[[Array, float], Array]] = apply_poisson) -> Array:
    """
    Solve A phi = F with periodic BCs using multigrid V-/W-cycles.
    Arguments mirror your previous API so it can be dropped in.
    """
    if h is None:
        # Default to unit spacing if not provided
        h = 1.0

    # Build the cycle object (Python). We keep `laplace` in the signature for API
    # compatibility, but the code above uses `apply_poisson` internally.
    cycle = PoissonCycle(F=F, v1=v1, v2=v2, mu=mu, l=l, eps=eps, h=h, laplace=laplace)

    # Run fixed number of cycles
    return multigrid(cycle, U, eps, iter_cycle)


# -----------------------------------------------------------------------------
# Custom-VJP wrapper: treat multigrid as an implicit linear solve
# -----------------------------------------------------------------------------

def make_poisson_mg_solver(l: int,
                           v1: int,
                           v2: int,
                           mu: int,
                           iter_cycle: int,
                           eps: float = 1e-6,
                           h: float = 1.0):
    """
    Build a custom-VJP multigrid Poisson solver with *fixed* MG parameters.

    Usage:
        solve = make_poisson_mg_solver(l=4, v1=2, v2=2, mu=2,
                                       iter_cycle=3, eps=1e-3, h=dx)
        phi = solve(F, U0)  # U0 is initial guess, same shape as F

    Forward:  phi = approx(A^{-1} F) via poisson_multigrid.
    Backward: g_F = approx(A^{-1} g_phi) using the same MG config.
    """

    @jax.custom_vjp
    def solve(F: Array, U0: Array) -> Array:
        # Forward wrapper that closes over MG params.
        return poisson_multigrid(
            F=F,
            U=U0,
            l=l,
            v1=v1,
            v2=v2,
            mu=mu,
            iter_cycle=iter_cycle,
            eps=eps,
            h=h,
            laplace=apply_poisson,
        )

    # ---- Forward pass for custom VJP ----
    def solve_fwd(F: Array, U0: Array):
        phi = poisson_multigrid(
            F=F,
            U=U0,
            l=l,
            v1=v1,
            v2=v2,
            mu=mu,
            iter_cycle=iter_cycle,
            eps=eps,
            h=h,
            laplace=apply_poisson,
        )
        # We don't actually need to save anything for backward because the
        # operator is linear and we reuse the same solver; residuals = ()
        residuals = ()
        return phi, residuals

    # ---- Backward pass (VJP) ----
    def solve_bwd(residuals, g_phi: Array):
        # residuals == (), unused
        del residuals

        # VJP wrt F: for symmetric A, (A^{-1})^T g_phi = A^{-1} g_phi.
        # So we just solve another Poisson problem:
        g_F = poisson_multigrid(
            F=g_phi,
            U=jnp.zeros_like(g_phi),
            l=l,
            v1=v1,
            v2=v2,
            mu=mu,
            iter_cycle=iter_cycle,
            eps=eps,
            h=h,
            laplace=apply_poisson,
        )

        # We choose *not* to propagate gradients through the initial guess U0;
        # if you wanted that for some reason, you'd need another linear solve.
        g_U0 = jnp.zeros_like(g_phi)

        return (g_F, g_U0)

    solve.defvjp(solve_fwd, solve_bwd)

    # Optional: JIT the custom-VJP'd function for speed.
    # Inputs are just (F, U0), MG params are closed over and thus static.
    return jax.jit(solve)