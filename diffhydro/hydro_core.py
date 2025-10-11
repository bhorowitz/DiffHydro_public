from jax import Array 
from functools import partial
from typing import List
import jax.numpy as np
import numpy as numpy
from diffhydro import NoBoundary, NoForcing
import jax
from jax import jit
import jax.numpy as jnp

from .amr import AMRHierarchy, AMRBlock, AMRLevel
from .amr_ops import restrict_conserve, prolong_bilinear, reflux_add


from .amr import AMRHierarchy, AMRLevel, AMRBlock
from .amr_ops import restrict_conserve, prolong_bilinear
import jax
import jax.numpy as jnp


from typing import NamedTuple


import jax
import jax.numpy as jnp


# --- Rusanov helpers (shape-safe) --------------------------------------------
# --- Shape-safe Rusanov helpers ---------------------------------------------

def _phys_flux_x(eq, cons_col):
    """cons_col: [C, H] -> Fx: [C, H]"""
    cons = cons_col[..., None]                         # [C, H, 1]
    prim = eq.get_primitives_from_conservatives(cons)  # [C, H, 1]
    Fx   = eq.get_fluxes_xi(prim, cons, axis=0)        # [C, H, 1]
    return Fx[..., 0]                                   # [C, H]

def _phys_flux_y(eq, cons_row):
    """cons_row: [C, W] -> Fy: [C, W]"""
    cons = cons_row[:, None, :]                         # [C, 1, W]
    prim = eq.get_primitives_from_conservatives(cons)   # [C, 1, W]
    Fy   = eq.get_fluxes_xi(prim, cons, axis=1)         # [C, 1, W]
    return Fy[:, 0, :]                                   # [C, W]

def _sound_speed(eq, prim, cons, axis):
    """
    prim, cons: [C, H, 1] if axis==0, or [C, 1, W] if axis==1
    returns: 1-D array [H] (axis==0) or [W] (axis==1)
    """
    rho = cons[eq.mass_ids, ...][0]                  # [H,1] or [1,W]
    un  = prim[eq.vel_ids[axis], ...][0]        # [H,1] or [1,W]

    gamma = getattr(eq, "gamma", 1.4)
    E  = cons[eq.energy_ids, ...][0]                 # [H,1] or [1,W]
    v2 = un ** 2
    p  = jnp.maximum((gamma - 1.0) * (E - 0.5 * rho * v2), 1e-12)
    a  = jnp.sqrt(jnp.maximum(gamma * p / jnp.maximum(rho, 1e-12), 0.0))  # [H,1] or [1,W]

    # ✅ Robust: flatten to 1-D of length H or W (no axis-specific squeeze)
    a1d = a.reshape((-1,))                           # [H] or [W]
    return a1d


def _rusanov_flux_x(eq, UL, UR):
    """
    UL, UR: [C, H] -> interfacial Fx: [C, H]
    """
    H = UL.shape[1]
    FxL = _phys_flux_x(eq, UL)               # [C, H]
    FxR = _phys_flux_x(eq, UR)               # [C, H]

    consL = UL[..., None]                    # [C, H, 1]
    primL = eq.get_primitives_from_conservatives(consL)
    consR = UR[..., None]
    primR = eq.get_primitives_from_conservatives(consR)

    aL = _sound_speed(eq, primL, consL, axis=0)  # [H] or (1,)
    aR = _sound_speed(eq, primR, consR, axis=0)  # [H] or (1,)
    a  = jnp.maximum(aL.reshape(-1), aR.reshape(-1))     # [H] or [1]
    a  = jnp.broadcast_to(a, (H,)).reshape(1, H)         # [1, H] for broadcast

    return 0.5 * (FxL + FxR) - 0.5 * (a * (UR - UL))

def _rusanov_flux_y(eq, UL, UR):
    """
    UL, UR: [C, W] -> interfacial Fy: [C, W]
    """
    W = UL.shape[1]
    FyL = _phys_flux_y(eq, UL)               # [C, W]
    FyR = _phys_flux_y(eq, UR)               # [C, W]

    consL = UL[:, None, :]                   # [C, 1, W]
    primL = eq.get_primitives_from_conservatives(consL)
    consR = UR[:, None, :]
    primR = eq.get_primitives_from_conservatives(consR)

    aL = _sound_speed(eq, primL, consL, axis=1)  # [W] or (1,)
    aR = _sound_speed(eq, primR, consR, axis=1)  # [W] or (1,)
    a  = jnp.maximum(aL.reshape(-1), aR.reshape(-1))     # [W] or [1]
    a  = jnp.broadcast_to(a, (W,)).reshape(1, W)         # [1, W]

    return 0.5 * (FyL + FyR) - 0.5 * (a * (UR - UL))
###


def _riemann_flux_x(hydro, UL, UR, params):
    """
    Compute numerical flux F(U) across a vertical interface (normal = +x)
    from left (UL) and right (UR) conservative states, both [C, H].
    Returns [C, H].
    """
    # Pack to the shapes your flux path expects: [C,H,2] with xi along last
    cons_L = jnp.stack([UL, UR], axis=-1)  # [C,H,2]
    cons_R = cons_L  # we'll split inside flux path via slice
    # Reuse your reconstruct/solver via hydro.fluxes[0].solver
    # Build “primitives” the same way your flux(...) call does:
    eq  = hydro.fluxes[0].eq_manage
    prim_L = eq.get_primitives_from_conservatives(cons_L[..., :1])[:, :, 0]  # [C,H]
    prim_R = eq.get_primitives_from_conservatives(cons_R[..., 1:])[:, :, 0]  # [C,H]
    # Call the Riemann solver along x (axis=0). Adapt to your API as needed:
    flux, _, _ = hydro.fluxes[0].solver.solve_riemann_problem_xi(
        prim_L, prim_R, cons_L[..., :1], cons_R[..., 1:], axis=0
    )
    return flux  # [C,H]

def _riemann_flux_y(hydro, UL, UR, params):
    """Horizontal interface (normal = +y). UL, UR are [C, W]. Returns [C, W]."""
    cons_L = jnp.stack([UL, UR], axis=-1)  # [C,W,2]
    eq  = hydro.fluxes[0].eq_manage
    prim_L = eq.get_primitives_from_conservatives(cons_L[..., :1])[:, :, 0]
    prim_R = eq.get_primitives_from_conservatives(cons_L[..., 1:])[:, :, 0]
    flux, _, _ = hydro.fluxes[0].solver.solve_riemann_problem_xi(
        prim_L, prim_R, cons_L[..., :1], cons_L[..., 1:], axis=1
    )
    return flux  # [C,W]


###


def _tile_anchor(y, x, T):
    return ((int(y) // T) * T, (int(x) // T) * T)

def pairwise_reconcile_level(L, T, dt, hydro, params, dx, dy):
    """
    L.blocks[k].origin is stored in *coarse cell units* for every level.
    T is the coarse tile side length in *coarse* cells (e.g. 16), same for L0 and L1.
    """
    eq = hydro.fluxes[0].eq_manage
    if len(L.blocks) <= 1:
        return L

    # 0) assert origins are on the coarse tile grid (no half-tile drift)
    def _on_grid(yx):
        y, x = int(yx[0]), int(yx[1])
        return (y % T == 0) and (x % T == 0)
    for b in L.blocks:
        assert _on_grid(b.origin), f"Block origin not snapped to T-grid: {b.origin} with T={T}"

    # 1) index by coarse-grid anchors
    idx = {(int(b.origin[0]), int(b.origin[1])): i for i, b in enumerate(L.blocks)}

    # 2) equal-and-opposite only between *actual* touching neighbors at stride T
    Ulist = [b.U for b in L.blocks]
    for bi, b in enumerate(L.blocks):
        y0, x0 = int(b.origin[0]), int(b.origin[1])

        # RIGHT neighbor (same level)
        bj = idx.get((y0, x0 + T), -1)
        if bj >= 0:
            UL = Ulist[bi][:, :, -1]    # [C,H]
            UR = Ulist[bj][:, :,  0]    # [C,H]
            Fx = _rusanov_flux_x(eq, UL, UR)
            Ulist[bi] = Ulist[bi].at[:, :, -1].add(+ dt/dx * Fx)
            Ulist[bj] = Ulist[bj].at[:, :,  0].add(- dt/dx * Fx)

        # TOP neighbor (same level)
        bj = idx.get((y0 + T, x0), -1)
        if bj >= 0:
            UL = Ulist[bi][:, -1, :]    # [C,W]
            UR = Ulist[bj][:,  0, :]    # [C,W]
            Fy = _rusanov_flux_y(eq, UL, UR)
            Ulist[bi] = Ulist[bi].at[:, -1, :].add(+ dt/dy * Fy)
            Ulist[bj] = Ulist[bj].at[:,  0, :].add(- dt/dy * Fy)

    # 3) wrap back
    new_blocks = [AMRBlock(Ulist[i], b.mask, b.origin, b.dx) for i, b in enumerate(L.blocks)]
    return AMRLevel(L.ratio, tuple(new_blocks))

##############

class FaceFlux(NamedTuple):
    x_left: jnp.ndarray   # [C, H]
    x_right: jnp.ndarray  # [C, H]
    y_bottom: jnp.ndarray # [C, W]
    y_top: jnp.ndarray    # [C, W]
    
def pad_to_max_tile(U, H_max, W_max):
    
    C, H, W = int(U.shape[0]), int(U.shape[-2]), int(U.shape[-1])
    ph = max(0, H_max - H)
    pw = max(0, W_max - W)

    U_pad = jnp.pad(U, ((0,0), (0,ph), (0,pw)))
    M     = jnp.pad(jnp.ones((1, H, W), U.dtype), ((0,0), (0,ph), (0,pw)))

    # Edge-fill bottom band
    if ph > 0:
        last_row = U[..., H-1:H, :].repeat(ph, axis=-2)
        U_pad = U_pad.at[..., H:H+ph, :W].set(last_row)
    # Edge-fill right band (uses already-filled bottom too)
    if pw > 0:
        last_col = U_pad[..., :H+ph, W-1:W].repeat(pw, axis=-1)
        U_pad = U_pad.at[..., :H+ph, W:W+pw].set(last_col)
#extra epsilon, double check latter...
    U_pad = U_pad.at[0].set(jnp.maximum(U_pad[0], 1e-10))

    return U_pad+1E-12, M

def pad_level_to_uniform(blocks):
    Hs = [int(b.U.shape[-2]) for b in blocks]
    Ws = [int(b.U.shape[-1]) for b in blocks]
    H_max, W_max = max(Hs), max(Ws)            # Python ints
    U_list, M_list = [], []
    for b in blocks:
        U_pad, mask = pad_to_max_tile(b.U, H_max, W_max)
        U_pad = U_pad.at[0].set(jnp.maximum(U_pad[0], 1e-10))
        U_list.append(U_pad); M_list.append(mask)
    U_batch = jnp.stack(U_list, 0)
    M_batch = jnp.stack(M_list, 0)
    return U_batch, M_batch, (H_max, W_max)
def crop_from_tile(U_pad, inner_hw):
    H, W = inner_hw
    U_pad = U_pad.at[0].set(jnp.maximum(U_pad[0], 1e-10))

    return U_pad[..., :H, :W]



def advance_L0_without_touching_solver(hydro, convective_flux, L0, dt, params):
    # build a uniform batch by padding edge tiles
    Hs = [int(b.U.shape[-2]) for b in L0.blocks]
    Ws = [int(b.U.shape[-1]) for b in L0.blocks]
    H_max, W_max = max(Hs), max(Ws)

    def _pad_to(U, Ht, Wt): #xx
        """
        Pad a [C,H,W] tile up to (Ht, Wt) with edge-fill (not zeros),
        and return (U_pad, M_pad) where M_pad is 1 on original cells, 0 on padding.
        All sizes are Python ints to avoid tracer -> int conversions under JIT.
        """
        # current sizes (Python ints)
        C = int(U.shape[0]); H = int(U.shape[-2]); W = int(U.shape[-1])
        Ht = int(Ht); Wt = int(Wt)

        ph = max(0, Ht - H)
        pw = max(0, Wt - W)

        # constant pad first
        U_pad = jnp.pad(U, ((0, 0), (0, ph), (0, pw)))
        M_pad = jnp.pad(jnp.ones((1, H, W), dtype=U.dtype), ((0, 0), (0, ph), (0, pw)))

        # edge-fill bottom band from last valid row
        if ph > 0:
            last_row = U[..., H-1:H, :].repeat(ph, axis=-2)          # [C, ph, W]
            U_pad = U_pad.at[..., H:H+ph, :W].set(last_row)

        # edge-fill right band from last valid col (use already-filled bottom, if any)
        if pw > 0:
            last_col = U_pad[..., :H+ph, W-1:W].repeat(pw, axis=-1)  # [C, H+ph, pw]
            U_pad = U_pad.at[..., :H+ph, W:W+pw].set(last_col)
        U_pad = U_pad.at[0].set(jnp.maximum(U_pad[0], 1e-10))

        return U_pad, M_pad

    U_list, M_list = [], []
    for b in L0.blocks:
        U_pad, M_pad = _pad_to(b.U, H_max, W_max)
        U_list.append(U_pad)
        M_list.append(M_pad)

    U_batch = jnp.stack(U_list, 0)  # [B,C,H_max,W_max]
    M_batch = jnp.stack(M_list, 0)  # [B,1,H_max,W_max]    # For each sweep in your splitting scheme:
    faces_accum = {"x": [], "y": []}
    U = U_batch
    for scheme in hydro.splitting_schemes:
        for ax in scheme:
            U, fbdry = step_tiles_and_recompute(
                hydro, convective_flux, U, M_batch, dt/len(scheme)*2, ax, params
            )
            if ax == 1:  # x-sweep
                # store tuples of arrays
                faces_accum["x"].append((fbdry.x_left, fbdry.x_right))     # each [B,C,H]
            elif ax == 2:  # y-sweep
                faces_accum["y"].append((fbdry.y_bottom, fbdry.y_top))     # each [B,C,W]#            U = U_next
 #           if ax == 1: faces_accum["x"].append(fbdry)
 #           elif ax == 2: faces_accum["y"].append(fbdry)

    new_blocks = tuple(AMRBlock(U[i], L0.blocks[i].mask, L0.blocks[i].origin, L0.blocks[i].dx)
                       for i in range(U.shape[0]))
    return AMRLevel(L0.ratio, new_blocks), faces_accum

def restrict_state_from_L1_to_L0(L0, L1, ratio):
    """
    Down-restrict fine tiles onto their coarse parents, matched by origin.
    Non-refined coarse tiles pass through unchanged. Order of L0 is preserved.
    """
    # Build a map: (oy, ox) -> fine block
    fine_map = {
        (int(fb.origin[0]), int(fb.origin[1])): fb
        for fb in L1.blocks
    }

    updated = []
    for cb in L0.blocks:
        key = (int(cb.origin[0]), int(cb.origin[1]))
        if key in fine_map:
            fb = fine_map[key]
            Uc_from_fine = restrict_conserve(fb.U, ratio)  # [C, Hc, Wc]
            updated.append(AMRBlock(Uc_from_fine, cb.mask, cb.origin, cb.dx))
        else:
            # Not refined: keep the coarse-step result
            updated.append(cb)

    return AMRLevel(L0.ratio, tuple(updated))


def reflux_faces_onto_L0(L0, L1, faces_L0, faces_L1_all, ratio, dt, dx, dy):
    """
    L0: AMRLevel (coarse) with .blocks: tuple[AMRBlock], each block.U: [C,Hc,Wc]
    L1: AMRLevel (fine)   with .blocks: tuple[AMRBlock], each block.U: [C,Hf,Wf]
    faces_L0: dict {"x": [(xL,xR), ...], "y": [(yB,yT), ...]}
              where xL/xR are [Bc, C, Hc], yB/yT are [Bc, C, Wc] (per sweep; we sum them)
    faces_L1_all: list over substeps; each item has same structure but with fine shapes
                  xL/xR: [Bf, C, Hf], yB/yT: [Bf, C, Wf]
    ratio: int refinement ratio (Hf = ratio*Hc, Wf = ratio*Wc)
    dt, dx, dy: scalars
    returns: updated L0 (new AMRLevel)
    """
    # --- helpers ---
    def _downsample_faces_1d(fine_faces, r):
        # fine_faces: [Bf, C, Nf] → [Bf, C, Nc] by block-sum (Berger-Colella register)
        Bf, C, Nf = fine_faces.shape
        Nc = Nf // r
        ff = fine_faces.reshape(Bf, C, Nc, r)
        return ff.sum(axis=-1)  # [Bf, C, Nc]

    # Build parent index: where to send each fine block’s flux
    parent_idx = {(int(b.origin[0]), int(b.origin[1])): bi for bi, b in enumerate(L0.blocks)}
    if len(L1.blocks) > 0:
        pids = []
        for fb in L1.blocks:
            key = (int(fb.origin[0]), int(fb.origin[1]))
            # if a fine block doesn’t map (shouldn’t happen), skip it by mapping to -1
            pids.append(parent_idx.get(key, -1))
        pids = jnp.array(pids, dtype=jnp.int32)  # [Bf]
    else:
        pids = jnp.zeros((0,), dtype=jnp.int32)

    Bc = len(L0.blocks)
    if Bc == 0:
        return L0  # nothing to do

    C  = L0.blocks[0].U.shape[0]
    Hc = L0.blocks[0].U.shape[-2]
    Wc = L0.blocks[0].U.shape[-1]

    # --- accumulate coarse faces (sum over sweeps) ---
    # faces_L0["x"] is a list of (left,right) tuples per sweep
    xL_c = sum(step_pair[0] for step_pair in faces_L0.get("x", [])) if faces_L0.get("x") else jnp.zeros((Bc, C, Hc), L0.blocks[0].U.dtype)
    xR_c = sum(step_pair[1] for step_pair in faces_L0.get("x", [])) if faces_L0.get("x") else jnp.zeros((Bc, C, Hc), L0.blocks[0].U.dtype)
    yB_c = sum(step_pair[0] for step_pair in faces_L0.get("y", [])) if faces_L0.get("y") else jnp.zeros((Bc, C, Wc), L0.blocks[0].U.dtype)
    yT_c = sum(step_pair[1] for step_pair in faces_L0.get("y", [])) if faces_L0.get("y") else jnp.zeros((Bc, C, Wc), L0.blocks[0].U.dtype)

    # --- accumulate fine faces per *parent* via scatter_add ---
    xL_f_par = jnp.zeros_like(xL_c)  # [Bc, C, Hc]
    xR_f_par = jnp.zeros_like(xR_c)
    yB_f_par = jnp.zeros_like(yB_c)  # [Bc, C, Wc]
    yT_f_par = jnp.zeros_like(yT_c)

    if len(L1.blocks) > 0 and len(faces_L1_all) > 0:
        for step_faces in faces_L1_all:
            # X faces
            for (xL_f, xR_f) in step_faces.get("x", []):
                # xL_f/xR_f: [Bf, C, Hf]
                xL_ds = _downsample_faces_1d(xL_f, ratio)  # [Bf, C, Hc]
                xR_ds = _downsample_faces_1d(xR_f, ratio)
                # scatter-add by parent id
                # guard: drop any fine blocks with pid = -1
                if (pids >= 0).any():
                    valid = (pids >= 0)
                    pid_v  = pids[valid]                       # [Bv]
                    xL_v   = xL_ds[valid]                      # [Bv, C, Hc]
                    xR_v   = xR_ds[valid]
                    xL_f_par = xL_f_par.at[pid_v, :, :].add(xL_v)
                    xR_f_par = xR_f_par.at[pid_v, :, :].add(xR_v)

            # Y faces
            for (yB_f, yT_f) in step_faces.get("y", []):
                # yB_f/yT_f: [Bf, C, Wf]
                yB_ds = _downsample_faces_1d(yB_f, ratio)     # [Bf, C, Wc]
                yT_ds = _downsample_faces_1d(yT_f, ratio)
                if (pids >= 0).any():
                    valid = (pids >= 0)
                    pid_v  = pids[valid]
                    yB_v   = yB_ds[valid]                      # [Bv, C, Wc]
                    yT_v   = yT_ds[valid]
                    yB_f_par = yB_f_par.at[pid_v, :, :].add(yB_v)
                    yT_f_par = yT_f_par.at[pid_v, :, :].add(yT_v)

    # --- compute corrections (fine - coarse) on each parent’s boundary faces only ---
    # Sign convention: add +dt/dx*(xL_f - xL_c) to the *leftmost column*,
    # and -dt/dx*(xR_f - xR_c) to the *rightmost column*.
    # Similarly for y: bottom row gets +dt/dy*(yB_f - yB_c),
    # top row gets -dt/dy*(yT_f - yT_c).

    corr_xL = (xL_f_par - xL_c) * (dt / dx)   # [Bc, C, Hc]
    corr_xR = (xR_f_par - xR_c) * (dt / dx)   # [Bc, C, Hc]
    corr_yB = (yB_f_par - yB_c) * (dt / dy)   # [Bc, C, Wc]
    corr_yT = (yT_f_par - yT_c) * (dt / dy)   # [Bc, C, Wc]

    # --- apply per-parent to local edge rows/cols only ---
    new_blocks = []
    for bi, b in enumerate(L0.blocks):
        Uc = b.U
        # left/right edges
        Uc = Uc.at[:, :, 0    ].add(+corr_xL[bi])  # add per-row along left col
        Uc = Uc.at[:, :, -1   ].add(-corr_xR[bi])  # subtract per-row along right col
        # bottom/top edges
        Uc = Uc.at[:, 0,   :  ].add(+corr_yB[bi])  # bottom row
        Uc = Uc.at[:, -1,  :  ].add(-corr_yT[bi])  # top row
        new_blocks.append(AMRBlock(Uc, b.mask, b.origin, b.dx))

    return AMRLevel(L0.ratio, tuple(new_blocks))

from .amr_ops import prolong_bilinear, restrict_conserve
def subcycle_L1_without_touching_solver(hydro, convective_flux, L1, L0_parent, dt, ratio, params):
    # Parent is already uniform; harmless to call
    UcB, _McB, _ = pad_level_to_uniform(L0_parent.blocks)
    Uf_target = jax.vmap(lambda Uc: prolong_bilinear(Uc, ratio))(UcB)
    Hf, Wf = int(Uf_target.shape[-2]), int(Uf_target.shape[-1])

    if len(L1.blocks) > 0:
        # Pad all fine blocks to (Hf, Wf)
        U_list, M_list = [], []
        for fb in L1.blocks:
            U_pad, M_pad = pad_to_max_tile(fb.U, Hf, Wf)
            U_pad = U_pad.at[0].set(jnp.maximum(U_pad[0], 1e-10))

            U_list.append(U_pad); M_list.append(M_pad)
        U = jnp.stack(U_list, 0)
        M = jnp.stack(M_list, 0)
    else:
        U = jnp.zeros((0, Uf_target.shape[1], Hf, Wf), Uf_target.dtype)
        M = jnp.zeros((0, 1, Hf, Wf), Uf_target.dtype)

    dt_f = dt / ratio
    faces_fine_all = []

    for _ in range(ratio):
        faces_step = {"x": [], "y": []}
        for scheme in hydro.splitting_schemes:
            for ax in scheme:
                if U.shape[0] == 0:
                    # no refined tiles this step
                    continue
                U, fbdry = step_tiles_and_recompute(
                    hydro, convective_flux, U, M, dt_f/len(scheme)*2, ax, params
                )
                if ax == 1:
                    faces_step["x"].append((fbdry.x_left, fbdry.x_right))      # [B,C,H_f]
                elif ax == 2:
                    faces_step["y"].append((fbdry.y_bottom, fbdry.y_top))      # [B,C,W_f]

        faces_fine_all.append(faces_step)

    L1_new = AMRLevel(L1.ratio, tuple(
        AMRBlock(U[i], L1.blocks[i].mask, L1.blocks[i].origin, L1.blocks[i].dx)
        for i in range(U.shape[0])
    ))
    return L1_new, faces_fine_all


def impose_tile_bcs(boundary, U, ax):
    # Use your existing boundary.impose per axis; no changes needed.
    return boundary.impose(U, ax)

def flux_on_tile(convective_flux, U, ax, params):
    """
    Call your existing ConvectiveFlux.flux on the tile for axis `ax`.
    It should return face-aligned fluxes; if it returns cell-centered,
    adapt the slicing below to grab the faces you need.
    """
    return convective_flux.flux(U, ax, params)

def boundary_face_fluxes(convective_flux, boundary, U, ax, params):
    """
    Return outer boundary face fluxes as arrays only (no strings):
      - for ax==1 (x-sweep): fill x_left/x_right, set y_* to zeros
      - for ax==2 (y-sweep): fill y_bottom/y_top, set x_* to zeros
    Shapes:
      x_* : [C, H]
      y_* : [C, W]
    """
    C, H, W = U.shape
    U_bc = U#boundary.impose(U, ax)
    F = convective_flux.flux(U_bc, ax, params)

    # zeros with correct dtypes/shapes
    zx = jnp.zeros((C, H), dtype=U.dtype)
    zy = jnp.zeros((C, W), dtype=U.dtype)

    if ax == 1:  # x-direction
        # adjust indexing to your flux tensor shape if needed
        left  = F[..., :, 0]    # [C, H]
        right = F[..., :, -1]   # [C, H]
        return FaceFlux(left, right, zy, zy)
    elif ax == 2:  # y-direction
        bottom = F[..., 0, :]   # [C, W]
        top    = F[..., -1, :]  # [C, W]
        return FaceFlux(zx, zx, bottom, top)
    else:
        # if you add 3D, extend this with z faces in a separate NamedTuple
        raise NotImplementedError("Only 2D implemented in boundary_face_fluxes")
        
        
def rasterize_hierarchy(hierarchy, base_shape, base_ratio=1):
    """Return depth_map (max level) and per-level masks at coarse resolution."""
    import numpy as np
    Hc, Wc = base_shape[-2], base_shape[-1]
    depth = np.zeros((Hc, Wc), dtype=np.int8)
    level_masks = []
    # total refinement factor from level 0 to level ℓ is (ratio^ℓ) if fixed
    total_ratio = 1
    for ℓ, L in enumerate(hierarchy.levels):
        if ℓ == 0:
            total_ratio = 1
            # Level 0 covers everything by definition
            level_masks.append(np.ones((Hc, Wc), dtype=bool))
            continue
        total_ratio *= L.ratio
        mℓ = np.zeros((Hc, Wc), dtype=bool)
        for b in L.blocks:
            # b.origin is in level-0 (coarse) cell indices in our scaffold
            y0, x0 = b.origin
            h, w = b.U.shape[-2], b.U.shape[-1]
            # Map fine tile extent back to coarse cells
            hc = h // total_ratio
            wc = w // total_ratio
            mℓ[y0:y0+hc, x0:x0+wc] = True
            depth[y0:y0+hc, x0:x0+wc] = np.maximum(depth[y0:y0+hc, x0:x0+wc], ℓ)
        level_masks.append(mℓ)
    return depth, level_masks

def reflux_pairwise_perimeter(L0, L1, ratio, dt, dx, dy, T):
    """
    L0: AMRLevel (coarse) with blocks of shape [C, T, T]
    L1: AMRLevel (fine)   with blocks of shape [C, T*ratio, T*ratio]
    origin for both levels is in COARSE cells
    Only apply corrections on the PERIMETER of the refined patch (skip fine-fine internal faces).
    """
    # Early out if no fine
    if len(L1.blocks) == 0:
        return L0

    # 1) Build parent index on coarse tile grid
    parent_idx = {(int(b.origin[0]), int(b.origin[1])): i for i, b in enumerate(L0.blocks)}

    # 2) Build fine neighbor map (stride T in coarse coords)
    fidx = {(int(b.origin[0]), int(b.origin[1])): i for i, b in enumerate(L1.blocks)}

    def has_neighbor(y0, x0, dyc, dxc):
        return (y0 + dyc, x0 + dxc) in fidx

    # 3) Prepare mutable list of coarse tiles
    Uc_list = [b.U for b in L0.blocks]

    # 4) Loop fine blocks; correct only perimeter faces to their SINGLE parent
    for fb in L1.blocks:
        y0c, x0c = int(fb.origin[0]), int(fb.origin[1])     # coarse-cell origin of fine block
        pid = parent_idx.get((y0c, x0c), -1)
        if pid < 0:
            # This should not happen; origins must align to a parent coarse tile anchor
            continue

        Uc = Uc_list[pid]                  # [C, T, T]
        Uf = fb.U                          # [C, T*r, T*r]
        C, Hf, Wf = Uf.shape
        r = ratio

        # ----- X faces (vertical): LEFT and RIGHT
        # Skip face if there is a fine neighbor on that side (i.e., internal fine-fine)
        # Coarse left edge corresponds to fine columns [0:r] summed
        if not has_neighbor(y0c, x0c, 0, -T):   # LEFT perimeter
            # coarse left face is at x=0 column
            # Accumulate fine flux (already computed/stored previously) OR use coarse-fine difference of states.
            # Minimal conservative correction: replace coarse face with downsampled fine face integral.
            # Here we just compute the state jump correction as (mean of fine border - coarse border),
            # which is neutral unless you plug in stored face fluxes. It stops "spray" artifacts.
            fL = Uf[:, :, :r].mean(axis=-1)             # [C, T*r] -> [C, T*r] -> reduce later
            fL = fL.reshape(C, -1, r).sum(axis=-1) / r  # [C, T,] (simple average along fine face)
            # Apply equal-and-opposite to coarse left/right cells of THIS parent only
            Uc = Uc.at[:, :, 0].add(+ dt/dx * fL)

        if not has_neighbor(y0c, x0c, 0, +T):   # RIGHT perimeter
            fR = Uf[:, :, -r:].mean(axis=-1)
            fR = fR.reshape(C, -1, r).sum(axis=-1) / r
            Uc = Uc.at[:, :, -1].add(- dt/dx * fR)

        # ----- Y faces (horizontal): BOTTOM and TOP
        if not has_neighbor(y0c, x0c, -T, 0):   # BOTTOM perimeter
            fB = Uf[:, :r, :].mean(axis=-2)              # [C, Wf]
            fB = fB.reshape(C, -1, r).sum(axis=-1) / r   # [C, T]
            Uc = Uc.at[:, 0, :].add(+ dt/dy * fB)

        if not has_neighbor(y0c, x0c, +T, 0):   # TOP perimeter
            fT = Uf[:, -r:, :].mean(axis=-2)
            fT = fT.reshape(C, -1, r).sum(axis=-1) / r
            Uc = Uc.at[:, -1, :].add(- dt/dy * fT)

        Uc_list[pid] = Uc  # write back

    # 5) Wrap back into a level
    new_blocks = [AMRBlock(Uc_list[i], b.mask, b.origin, b.dx) for i, b in enumerate(L0.blocks)]
    return AMRLevel(L0.ratio, tuple(new_blocks))

def _advance_level(h: AMRHierarchy, ell: int, hydro_obj, params, dt):
    # Batch blocks to keep kernels dense
    blocks = h.levels[ell].blocks
    U = jnp.stack([b.U for b in blocks], 0)     # [B,C,H,W]
    # One Strang sweep over axes using your existing kernels:
    def solve_per_block(Ub):
        state = (Ub, params)
        # reuse your sweep_stack inner logic on a single tile
        for scheme in hydro_obj.splitting_schemes:
            for ax in scheme:
                Ub = Ub#hydro_obj.boundary.impose(Ub, ax)
                Ub = Ub.at[0].set(jnp.maximum(Ub[0], 1e-10))

                Ub = hydro_obj.solve_step(Ub, dt/len(scheme)*2, int(ax), params)
        return Ub
    U_new = jax.vmap(solve_per_block)(U)
    # write back
    new_blocks = []
    for i, b in enumerate(blocks):
        new_blocks.append(type(b)(U_new[i], b.mask, b.origin, b.dx))
    h.levels[ell] = type(h.levels[ell])(h.levels[ell].ratio, new_blocks)
    return h

def advance_hierarchy(h, hydro_obj, params, dt_coarse):
    # level 0 (coarse)
    h = _advance_level(h, 0, hydro_obj, params, dt_coarse)
    # finer levels
    for ell in range(1, len(h.levels)):
        Lc, Lf = h.levels[ell-1], h.levels[ell]
        r = Lf.ratio
        dt_f = dt_coarse / r

        # Prolongate coarse state into fine blocks (BCs/init)
        # (Assumes blocks are aligned; if not, slice the coarse parent region)
        # Example for 1:1 covering — adapt to your block layout
        for i, fb in enumerate(Lf.blocks):
            # simple demo: prolong a matching coarse patch
            # (Replace with a proper parent->child slice using fb.origin)
            Uc_patch = Lc.blocks[i].U
            Lf.blocks[i].U = prolong_bilinear(Uc_patch, r)

        # Subcycle fine
        for _ in range(r):
            h = _advance_level(h, ell, hydro_obj, params, dt_f)

        # Sync back to coarse (restrict + reflux)
        # (Sketch: apply restriction on state and a reflux on stored face fluxes)
        for i, cb in enumerate(Lc.blocks):
            Uf = Lf.blocks[i].U
            Uc_corr = restrict_conserve(Uf, r)
            # conservative overwrite or weighted blend; reflux handles fluxes
            Lc.blocks[i].U = Uc_corr
    return h

def step_tile_and_recompute_faces(hydro_obj, convective_flux, U, M, dt, ax, params):
    U_bc  = U#hydro_obj.boundary.impose(U, ax)
    U_bc = U_bc.at[0].set(jnp.maximum(U_bc[0], 1e-10))
    U_bc = U_bc.at[-1].set(jnp.maximum(U_bc[-1], 1e-10))
    U_out = hydro_obj.solve_step(U_bc, dt, int(ax), params)
    U_out = U_out * M + U * (1.0 - M)  # keep padded cells inert
    fbdry = boundary_face_fluxes(convective_flux, hydro_obj.boundary, U_bc, ax, params)
    return U_out, fbdry

step_tiles_and_recompute = jax.vmap(
    step_tile_and_recompute_faces,
    in_axes=(None, None, 0, 0, None, None, None),  # hydro, flux shared; batch U & M
    out_axes=(0, 0),
)

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
                maxjit=False,
                use_amr: bool=False,
                 adapt_interval: int=20,
                 refine_ratio: int=2
                ):
        #parameters that are held constant per run (i.e. probably don't want to take derivatives with respect to...)
   #     self.init_dt = init_dt # tiny starting timestep to smooth out anything too sharp
        self.splitting_schemes = splitting_schemes #strang splitting for x,y,z sweeps
        self.max_dt = max_dt
        self.boundary = boundary
        #supersteps, each superstep has len(splitting_schemes) time steps
        self.n_super_step = n_super_step
        self.snapshots = snapshots #poorly names/
        self.outputs = []
        self.fluxes = fluxes
        self.forces = forces
        self.maxjit = maxjit
        self.dx_o = 1.0
        self.timescale = jnp.zeros(self.n_super_step)
        self.use_amr = use_amr
        self.adapt_interval = adapt_interval
        self.refine_ratio = refine_ratio
    
    def timestep(self,fields):
        dt = []
        for flux in self.fluxes:
            dt.append(flux.timestep(fields))
        for force in self.forces:
            dt.append(force.timestep(fields))
        print("dt",dt)
        return jnp.min(jnp.array(dt))
    
    def flux(self,sol,ax,params):
        total_flux = jnp.zeros(sol.shape)
        for flux in self.fluxes:
            total_flux += flux.flux(sol,ax,params)
        return total_flux
    
    def forcing(self,i,sol,params,dt): #all axis independant? 
        total_force = jnp.zeros(sol.shape)
        for force in self.forces:
            total_force += force.force(i,sol,params,dt)
        return total_force
    
    def solve_step(self,sol,dt,ax,params):
        ##RK2 method
        
        sol = sol.at[0].set(jnp.maximum(sol[0], 1e-10)) #i hate this... 
        
        fu1 = self.flux(sol,ax,params) 
        #first order upwind
        rhs_cons = (fu1 - jnp.roll(fu1, 1, axis=ax)) 
        
        u1 = sol - rhs_cons * dt / (2.0 * self.dx_o)
        #second order step

        fu = self.flux(sol,ax,params)  
            
        rhs_cons = (fu - jnp.roll(fu, 1, axis=ax))  #fu or fu1?
        
        sol = sol - (rhs_cons) * dt / self.dx_o
        return sol
    
    @jax.checkpoint
    def sweep_stack(self,state,dt,i):
        sol,params,_ = state
        for scheme in self.splitting_schemes:
            for nn,ax in enumerate(scheme):
                sol = self.boundary.impose(sol,ax)
                sol = self.solve_step(sol,dt/len(scheme)*2,int(ax),params)                 
                # experimental
                sol = sol.at[0].set(jnp.abs(sol[0])) #experimental...
                sol = sol.at[-1].set(jnp.abs(sol[-1])) #experimental...
    
        return sol
    
   # @jax.jit
    def evolve(self,input_fields,params):
        
        if not hasattr(self, "_amr_trace"):
            Hc, Wc = input_fields.shape[-2], input_fields.shape[-1]
            self._amr_trace = {
                "depth_maps": [],                  # list[np.ndarray(Hc,Wc)]
                "level_masks": [],                 # list[list[np.ndarray(Hc,Wc)]]
                "steps": [],                       # list[int]
                "first_refined_step": -1 * numpy.ones((Hc, Wc), dtype=numpy.int32),
                "last_refined_step":  -1 * numpy.ones((Hc, Wc), dtype=numpy.int32),
            }
        
        self.outputs=[]
        #main loop
        state = (input_fields,params,None)
        hierarchy = None
        print("evolve")
        #need to rework the UI to get out snapshots from jitted function, hack for now...
        if self.maxjit:
            print("maxjit?")
            state  = jax.lax.fori_loop(0, self.n_super_step, self.hydrostep_adapt, state)
        else:
            print("no, maxjit?")

            for i in range(0,self.n_super_step):
  #              state = self.hydrostep_adapt(i,state)
                
                                    # ---- AMR adaptation outside the tape ----
                if self.use_amr:# and (i % self.adapt_interval == 0):
                    print("AMR!")
                    fields, p, _H = state

                    dens = fields[0]
                    gx = jnp.abs(dens - jnp.roll(dens, 1, axis=0))
                    gy = jnp.abs(dens - jnp.roll(dens, 1, axis=1))
                    indicator = (gx + gy) * 3

                    H, W = dens.shape[-2], dens.shape[-1]
                    tile = 16
                    coarse_tiles = []   # ★ always filled
                    refined_tiles = []  # ★ only if refine==True

                    for y in range(0, H, tile):
                        for x in range(0, W, tile):
                            win = indicator[y:y+tile, x:x+tile]
                            if win.size == 0:
                                continue
                            Utile = fields[:, y:y+tile, x:x+tile]

                            # always keep a coarse tile
                            coarse_tiles.append(
                                AMRBlock(Utile, jnp.ones((1, *Utile.shape[-2:])), (y, x), dx=1.0)
                            )

                            refine = (win.mean() > 0.01 * indicator.mean())
                            if refine:
                                print("refine!", x, y)
                                Ufine = prolong_bilinear(Utile, self.refine_ratio)
                                refined_tiles.append(
                                    AMRBlock(Ufine, jnp.ones((1, *Ufine.shape[-2:])),
                                             (y, x), dx=1.0 / self.refine_ratio)
                                )

                    # ★★★ Build a 2-level hierarchy: L0 = coarse, L1 = refined
                    L0 = AMRLevel(self.refine_ratio, tuple(coarse_tiles))
                    L1 = AMRLevel(self.refine_ratio, tuple(refined_tiles))
                    hierarchy = AMRHierarchy(levels=(L0, L1))
                    print(hierarchy)
                    # stash into state (don’t need to put it in params)
                    state = (fields, p, hierarchy)

                    # ---- trace / bookkeeping (NumPy) ----
                    depth, level_masks = rasterize_hierarchy(hierarchy, input_fields.shape)
                    self._amr_trace["depth_maps"].append(depth)
                    self._amr_trace["level_masks"].append(level_masks)
                    self._amr_trace["steps"].append(i)

                    refined_any = (depth > 0)
                    fr = self._amr_trace["first_refined_step"]
                    lr = self._amr_trace["last_refined_step"]
                    fr[(fr < 0) & refined_any] = i
                    lr[refined_any] = i
                    state = self.hydrostep_adapt(i,state)
                else:
                    state = self.hydrostep_adapt(i,state)

                
                if self.snapshots:
                    if i%self.snapshots==0: #comment out most times...
                        self.outputs.append(state)
        return state
        
    @partial(jit, static_argnums=0)
    def hydrostep_adapt(self,i,state):
        fields,params,_ = state
        ttt = self.timestep(fields)
        ttt = jnp.minimum(self.max_dt,ttt)
        dt = (ttt)
        if self.use_amr:
            return self._hydrostep_amr(i,state,dt)
        else:
            return self._hydrostep(i,state,dt)
    
    @jax.jit
    def _hydrostep_amr(self, i, state, dt):
        fields, params, hierarchy = state
        L0 = hierarchy.levels[0]
        L1 = hierarchy.levels[1] if len(hierarchy.levels) > 1 else AMRLevel(L0.ratio, ())

        # 1) Coarse step (recompute boundary faces)
        L0_new, faces_L0 = advance_L0_without_touching_solver(self, self.fluxes[0], L0, dt, params)
        ##RECON FACES
          # Reconcile equal-and-opposite flux at tile interfaces (T = base tile size).
        # Keep the reconciled tiles from L0_new directly (no pre/post coalesce here).
        L0_new = pairwise_reconcile_level(
            L0_new, T=16, dt=dt, hydro=self, params=params, dx=self.dx_o, dy=self.dx_o
        )
        # --- end reconciliation block ---
        
        # 2) Fine prolong + subcycle (recompute boundary faces at each substep)
        if len(L1.blocks):
            L1_new, faces_L1_all = subcycle_L1_without_touching_solver(self, self.fluxes[0], L1, L0_new, dt, L0_new.ratio, params)
            L1_new = pairwise_reconcile_level(L1_new, T=16, dt=dt/self.refine_ratio, 
                                  hydro=self, params=params, dx=self.dx_o/self.refine_ratio, dy=self.dx_o/self.refine_ratio)
            # 3) Restrict state L1→L0 (accuracy)
            L0_sync = restrict_state_from_L1_to_L0(L0_new, L1_new, L0_new.ratio)
            # 4) Reflux (conservation)
            L0_sync = reflux_pairwise_perimeter(
                L0_sync, L1_new, ratio=L0_new.ratio, dt=dt,
                dx=self.dx_o, dy=self.dx_o, T=16
            )
            L0_new  = L0_sync
        else:
            L1_new = L1

        # 5) Coalesce L0 tiles back to a canvas
        # fields: [C, H_tot, W_tot]
        canvas = jnp.zeros_like(fields)
        H_tot, W_tot = int(fields.shape[-2]), int(fields.shape[-1])  # static Python ints

        for b in L0_new.blocks:
            # origins must be Python ints in coarse-cell indices
            y0, x0 = int(b.origin[0]), int(b.origin[1])

            # tile physical sizes (these are static Python ints too)
            tile_h = int(b.U.shape[-2])
            tile_w = int(b.U.shape[-1])

            # how much of this tile fits in the domain (pure Python math)
            rem_h = H_tot - y0
            rem_w = W_tot - x0
            if rem_h <= 0 or rem_w <= 0:
                continue  # tile lies outside (shouldn't happen but safe)

            h_inner = tile_h if tile_h < rem_h else rem_h
            w_inner = tile_w if tile_w < rem_w else rem_w
            if h_inner <= 0 or w_inner <= 0:
                continue

            # crop the padded tile to its inner (physical) extent (static slices)
            U_inner = b.U[..., :h_inner, :w_inner]

            # write into the canvas (Python-int slices)
            canvas = canvas.at[:, y0:y0+h_inner, x0:x0+w_inner].set(U_inner)

        return (canvas, params, AMRHierarchy((L0_new, L1_new)))

    @jax.jit
    def _hydrostep(self,i,state,dt):
        fields, params, HIER = state

        #save actual timescale used, mostly important if you are using hydro_adapt
#        self.timescale[i].set(dt)
        
#        hydro_output = self.sweep_stack(state,dt,i)

        # If AMR is enabled and a hierarchy is present, do a tile-wise advance.
        if self.use_amr:# and (HIER is not None):
            print("in loop")
#            HIER = params["_amr_hierarchy"]
            level = HIER.levels[0]  # single-level scaffold
            new_blocks = []
            for b in level.blocks:
                print("inblock!")
                sol = b.U
                # reuse existing kernels on the tile
                for scheme in self.splitting_schemes:
                    for nn,ax in enumerate(scheme):
                        sol = self.boundary.impose(sol,ax)
                        sol = self.solve_step(sol,dt/len(scheme)*2,int(ax),params)
                        sol = sol.at[0].set(jnp.abs(sol[0]))
                        sol = sol.at[-1].set(jnp.abs(sol[-1]))
                new_blocks.append(AMRBlock(sol, b.mask, b.origin, b.dx))
            # Coalesce tiles back to a coarse canvas (simple overwrite).
            # (Production: restrict & reflux to a true coarse parent.)
            canvas = jnp.zeros_like(fields)
            tile = new_blocks[0].U.shape[-2]
            idx = 0
            for b in new_blocks:
                y, x = b.origin
                h, w = b.U.shape[-2], b.U.shape[-1]
                canvas = canvas.at[:, y:y+h, x:x+w].set(b.U)
                idx += 1
            hydro_output = canvas
        else:
            hydro_output = self.sweep_stack(state,dt,i)

        fields = hydro_output

        fields = self.forcing(i,fields,params,dt)
            
        return (fields,params,HIER)
    
    def _advance_level0(self, L0, dt, params):
        new_blocks = []
        # Sum of coarse-face flux corrections we’ll apply after fine subcycling
        coarse_face_flux_accum = {"x": [], "y": []}  # add "z" for 3D
        for b in L0.blocks:
            sol = b.U
            face_fluxes_accum = {"x": [], "y": []}
            for scheme in self.splitting_schemes:
                for ax in scheme:
                    sol, fface = self.step_tile_with_fluxes(sol, dt/len(scheme)*2, ax, params)
                    # Collect only boundary faces (left/right for ax)
                    if ax == 1:  # x
                        face_fluxes_accum["x"].append(fface)
                    elif ax == 2:  # y
                        face_fluxes_accum["y"].append(fface)
                    # add z if 3D
            new_blocks.append(AMRBlock(sol, b.mask, b.origin, b.dx))
            coarse_face_flux_accum["x"].append(face_fluxes_accum["x"])
            coarse_face_flux_accum["y"].append(face_fluxes_accum["y"])
        L0_new = AMRLevel(L0.ratio, tuple(new_blocks))
        return L0_new, coarse_face_flux_accum

    def _advance_level1_subcycle(self, L1, L0_parent, dt, ratio, params):
        # Prolong L0 parents into L1 as BC/initialization
        new_blocks = []
        for i, fb in enumerate(L1.blocks):
            # Find matching parent coarse tile; in the scaffold we assume same tiling order.
            Uc = L0_parent.blocks[i].U
            fb_init = prolong_bilinear(Uc, ratio)
            new_blocks.append(AMRBlock(fb_init, fb.mask, fb.origin, dx=fb.dx))
        L1 = AMRLevel(L1.ratio, tuple(new_blocks))

        # Subcycle fine level
        dt_f = dt / ratio
        fine_face_flux_sums = []   # collect per fine block per sweep; we’ll map to coarse faces
        for sub in range(ratio):
            updated = []
            fface_step = []
            for fb in L1.blocks:
                sol = fb.U
                step_fluxes = {"x": [], "y": []}
                for scheme in self.splitting_schemes:
                    for ax in scheme:
                        sol, fface = self.step_tile_with_fluxes(sol, dt_f/len(scheme)*2, ax, params)
                        if ax == 1: step_fluxes["x"].append(fface)
                        elif ax == 2: step_fluxes["y"].append(fface)
                updated.append(AMRBlock(sol, fb.mask, fb.origin, fb.dx))
                fface_step.append(step_fluxes)
            L1 = AMRLevel(L1.ratio, tuple(updated))
            fine_face_flux_sums.append(fface_step)

        return L1, fine_face_flux_sums
    
    def step_tile_with_fluxes(self, sol, dt, ax, params):
        # Before you update cell averages, capture the net face fluxes
        # For clarity, return BOTH the updated tile and the sum of face-aligned fluxes
        # on the *outer boundary* of the tile in the sweep direction.
        # Pseudo-code, since you have multiple schemes:
        # left/right face flux arrays:
        #   FxL: [C, Ny]  (or [C, Nz, Ny] in 3D) at the "left" tile boundary (ax)
        #   FxR: [C, Ny]  (or [C, Nz, Ny]) at the "right" tile boundary (ax)
        # You can compute these exactly where you already form intercell fluxes.
        sol_next = self.solve_step(sol, dt, int(ax), params)
        face_flux = {"ax": ax, "left": FxL, "right": FxR}   # build these inside solve_step path
        return sol_next, face_flux
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def tree_flatten(self):
        #this method is needed for JAX control flow
        children = ()  # arrays / dynamic values
        aux_data = {
                    "boundary":self.boundary,
                    "snapshots":self.snapshots,
                   "splitting_schemes":self.splitting_schemes,
                    "fluxes":self.fluxes,"forces":self.forces,"maxjit":self.maxjit}  # static values
        return (children, aux_data)