# hydro_core.py
# AMR-by-lattice rework inspired by amr.py:
# - global lattice per level (tiles laid out on a regular Ny x Nx grid)
# - halos built by pure indexing using roll (periodic) and aligned stitching
# - refinement from a grid indicator + dilation -> block mask
# - prolong (kron) and restrict (avg pool)
# - local solves freeze ghosts; only interior updated (prevents wrap artifacts)
# - same-level interface reconciliation (Rusanov-like)
# - optional reflux L1->L0 (coarse/fine face flux replacement)
#
# This is intentionally self-contained; wire to your existing flux/equation modules.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp

Array = jnp.ndarray





# ------------------------ Level config & helpers ------------------------



@dataclass
class LevelConfig:
    Ny: int                # number of tiles vertically
    Nx: int                # number of tiles horizontally
    T:  int                # tile size (square, T x T)
    H:  int                # total domain height (cells)
    W:  int                # total domain width  (cells)
    r:  int                # refinement ratio to next level

    @property
    def shape_tiles(self) -> Tuple[int, int]:
        return (self.Ny, self.Nx)

    @property
    def shape_canvas(self) -> Tuple[int, int]:
        return (self.H, self.W)

def gather_halo_from_canvas(canvas: jnp.ndarray, y0: int, x0: int, T: int, h: int) -> jnp.ndarray:
    # canvas: [C,H,W], returns [C, T+2h, T+2h]
    C, H, W = canvas.shape
    ys = (jnp.arange(y0 - h, y0 + T + h) % H)
    xs = (jnp.arange(x0 - h, x0 + T + h) % W)
    return canvas[:, ys[:, None], xs[None, :]]

def step_tile_with_halo(hydro, Uhalo: jnp.ndarray, dt: float, ax: int, params: dict) -> jnp.ndarray:
    # Uhalo: [C, T+2h, T+2h]
    # replicate hydro.solve_step() but DO NOT call boundary.impose()
    fu1 = hydro.flux(Uhalo, ax, params)
    rhs = fu1 - jnp.roll(fu1, 1, axis=ax)
    U1  = Uhalo - dt * rhs

    fu2 = hydro.flux(U1, ax, params)
    rhs2 = fu2 - jnp.roll(fu2, 1, axis=ax)
    U2  = 0.5 * (Uhalo + U1 - dt * rhs2)
    return U2
    

def build_level0_config(H: int, W: int, T: int, r: int) -> LevelConfig:
    # require divisibility (simplifies; you can pad if you prefer)
    assert H % T == 0 and W % T == 0, "H and W must be divisible by base tile size T"
    Ny, Nx = H // T, W // T
    return LevelConfig(Ny=Ny, Nx=Nx, T=T, H=H, W=W, r=r)


def extract_tiles(U: Array, cfg: LevelConfig) -> Array:
    """Canvas [C, H, W] -> tiles [Ny, Nx, C, T, T] (row-major)."""
    C, H, W = U.shape
    Ny, Nx, T = cfg.Ny, cfg.Nx, cfg.T
    U4 = U.reshape(C, Ny, T, Nx, T)
    U4 = U4.transpose(1, 3, 0, 2, 4)   # [Ny, Nx, C, T, T]
    return U4


def assemble_from_tiles(tiles: Array, cfg: LevelConfig) -> Array:
    """Tiles [Ny, Nx, C, T, T] -> canvas [C, H, W]."""
    Ny, Nx, C, T, _ = tiles.shape
    U = tiles.transpose(2, 0, 3, 1, 4)    # [C, Ny, T, Nx, T]
    U = U.reshape(C, Ny*T, Nx*T)
    assert (Ny == cfg.Ny) and (Nx == cfg.Nx) and (T == cfg.T)
    return U


# ------------------------ Refinement mask from indicator ------------------------

def _sobel_like_indicator(rho: Array) -> Array:
    # simple grad magnitude
    gx = jnp.abs(rho - jnp.roll(rho, 1, axis=-1))
    gy = jnp.abs(rho - jnp.roll(rho, 1, axis=-2))
    return gx + gy


def _dilate_mask(mask: Array, iters: int = 1) -> Array:
    # binary dilation with 3x3 ones kernel, 'periodic' via roll max
    if iters <= 0:
        return mask
    for _ in range(iters):
        nb = [
            jnp.roll(mask,  1, axis=-1), jnp.roll(mask, -1, axis=-1),
            jnp.roll(mask,  1, axis=-2), jnp.roll(mask, -1, axis=-2),
            jnp.roll(jnp.roll(mask,  1, axis=-2),  1, axis=-1),
            jnp.roll(jnp.roll(mask,  1, axis=-2), -1, axis=-1),
            jnp.roll(jnp.roll(mask, -1, axis=-2),  1, axis=-1),
            jnp.roll(jnp.roll(mask, -1, axis=-2), -1, axis=-1),
        ]
        st = mask
        for n in nb:
            st = jnp.maximum(st, n)
        mask = st
    return mask


def refine_mask_from_indicator(U0: Array, cfg: LevelConfig, tau: float = 0.02, dilate: int = 1) -> Array:
    """
    U0: [C,H,W] coarse canvas
    returns L1 block mask on the L0 lattice: [Ny, Nx] bool
    """
    rho = U0[0]
    ind = _sobel_like_indicator(rho)
    thr = tau * (jnp.mean(ind) + 1e-12)
    refine_cells = ind > thr
    refine_cells = _dilate_mask(refine_cells, iters=dilate)

    # reduce to tiles by average>0
    Cmask = refine_cells.reshape(cfg.Ny, cfg.T, cfg.Nx, cfg.T).transpose(0, 2, 1, 3)  # [Ny,Nx,T,T]
    Bmask = (Cmask.sum(axis=(2, 3)) > 0)                                             # [Ny,Nx]
    return Bmask


# ------------------------ Prolongation / Restriction ------------------------

def prolong_kron(Uc: Array, r: int) -> Array:
    """Kronecker upsample: [C,Hc,Wc] -> [C,Hc*r,Wc*r]."""
    eye = jnp.ones((r, r), dtype=Uc.dtype)
    return jnp.kron(Uc, eye)


def restrict_avg(Uf: Array, r: int) -> Array:
    """Average pooling: [C,Hf,Wf] -> [C,Hc,Wc]."""
    C, Hf, Wf = Uf.shape
    assert Hf % r == 0 and Wf % r == 0
    Hc, Wc = Hf // r, Wf // r
    A = Uf.reshape(C, Hc, r, Wc, r)
    return A.mean(axis=(2, 4))


# ------------------------ Local step with frozen ghosts ------------------------

def _update_interior_old(u: Array, rhs: Array, ax: int, h: int, scale: float) -> Array:
    if ax == 1:   # x-sweep (width dim)
        return u.at[:, :, h:-h].add(-scale * rhs[:, :, h:-h])
    elif ax == 2: # y-sweep (height dim)
        return u.at[:, h:-h, :].add(-scale * rhs[:, h:-h, :])
    else:
        raise ValueError(f"bad axis {ax}")
        
def _update_interior(U_bc, rhs, ax, halo_w, scale):
    # U_bc, rhs: [C, H, W]; update ONLY the interior h:-h, h:-h
    h = int(halo_w)
    if h == 0:
        return U_bc - scale * rhs
    return U_bc.at[:, h:-h, h:-h].add(-scale * rhs[:, h:-h, h:-h])

def _roll_axis_from_sweep(ax: int) -> int:
    # ax: 1 = x (width, W axis); 2 = y (height, H axis)
    return 2 if ax == 1 else 1

def _solve_step_freeze_ghosts(hydro, U_bc, dt, ax, params, halo_w):
    ra   = _roll_axis_from_sweep(ax)
    fu1  = hydro.flux(U_bc, ax, params)
    rhs1 = fu1 - jnp.roll(fu1, 1, axis=ra)
    U1   = _update_interior(U_bc, rhs1, ax, halo_w, dt/(2.0*hydro.dx_o))
    fu2  = hydro.flux(U1, ax, params)
    rhs2 = fu2 - jnp.roll(fu2, 1, axis=ra)
    U2   = _update_interior(U1, rhs2, ax, halo_w, dt/hydro.dx_o)


    U2 = U2.at[0].set(jnp.maximum(U2[0], 1e-12))  # gentle safety floor
    return U2


# ------------------------ Halos by pure indexing on the tile lattice ------------------------

def _make_halos_x(tiles: Array, h: int) -> Tuple[Array, Array]:
    """
    tiles: [Ny,Nx,C,T,T]
    return (L, R) halos as [Ny,Nx,C,T,h] pulled from neighbor tiles (periodic).
    """
    # rightmost h columns from left neighbor:
    left_nb  = jnp.roll(tiles,  1, axis=1)   # shift Nx
    L = left_nb[..., -h:]                   # [..., T, h]
    # leftmost h columns from right neighbor:
    right_nb = jnp.roll(tiles, -1, axis=1)
    R = right_nb[..., :h]
    # transpose to [Ny,Nx,C,T,h]
    L = L  # already [Ny,Nx,C,T,h]
    R = R
    return L, R


def _make_halos_y(tiles, h):
    # tiles: [Ny, Nx, C, T, T]
    down_nb = jnp.roll(tiles,  1, axis=0)   # neighbor below (periodic)
    B = down_nb[..., -h:, :]                # take the last h rows
    up_nb   = jnp.roll(tiles, -1, axis=0)   # neighbor above (periodic)
    Tt = up_nb[..., :h,  :]                 # take the first h rows
    return B, Tt

def _make_halos_y_old(tiles, h):
    below_nb = jnp.roll(tiles, -1, axis=0)  # neighbor below (i+1)
    B = below_nb[..., :h, :]                # TOP h rows of the below neighbor

    above_nb = jnp.roll(tiles,  1, axis=0)  # neighbor above (i-1)
    Tt = above_nb[..., -h:, :]              # BOTTOM h rows of the above neighbor
    return B, Tt

def step_tiles_with_halo(hydro, tiles: Array, dt: float, ax: int, params: Dict, halo_w: int = 2) -> Array:
    """
    tiles: [Ny,Nx,C,T,T]
    returns stepped tiles (same shape), using halos from periodic neighbors.
    """
    Ny, Nx, C, T, _ = tiles.shape
    h = halo_w

    if int(ax) == 1:
        Lh, Rh = _make_halos_x(tiles, h)                     # [Ny,Nx,C,T,h]
        Ubc = jnp.concatenate([Lh, tiles, Rh], axis=-1)      # [Ny,Nx,C,T,T+2h]
        # vmaps over (Ny,Nx)
        step = jax.vmap(jax.vmap(lambda U: _solve_step_freeze_ghosts(hydro, U, dt, 1, params, h), in_axes=0), in_axes=0)
        Uout = step(Ubc)                                     # [Ny,Nx,C,T,T+2h]
        Uret = Uout[..., h:-h]                               # crop interior
    elif int(ax) == 2:
        Bh, Th = _make_halos_y(tiles, h)                     # [Ny,Nx,C,h,T]
        Ubc = jnp.concatenate([Bh, tiles, Th], axis=-2)      # [Ny,Nx,C,T+2h,T]
        step = jax.vmap(jax.vmap(lambda U: _solve_step_freeze_ghosts(hydro, U, dt, 2, params, h), in_axes=0), in_axes=0)
        Uout = step(Ubc)                                     # [Ny,Nx,C,T+2h,T]
        Uret = Uout[..., h:-h, :]                            # crop interior
    else:
        raise ValueError(f"bad axis {ax}")

    return Uret


# ------------------------ Same-level interface reconcile (Rusanov) ------------------------

def _sound_speed(eq, prim, cons, axis: int) -> jnp.ndarray:
    """
    Compute ideal-gas sound speed a = sqrt(gamma * p / rho).
    Accepts 2D slices prim, cons with shape [C, N] (N = H for x, W for y).
    Robust to accidental [1, N] rows via _row().
    Uses the full |u|^2 = u^2 + v^2 (+ w^2) from eq.vel_ids.
    """
    gamma = eq.gamma

    def _row(a):
        # ensure [N] no matter if a is [N] or [1, N]
        return a if a.ndim == 1 else a[0]

    rho = _row(cons[eq.mass_ids])     # [N]
    E   = _row(cons[eq.energy_ids])   # [N]

    # sum all velocity components (handles 2D/3D)
    v2 = 0.0
    for vid in eq.vel_ids:
        v = _row(prim[vid])           # [N]
        v2 = v2 + v * v

    p = (gamma - 1.0) * (E - 0.5 * rho * v2)              # [N]
    a = jnp.sqrt(jnp.maximum(gamma * p / jnp.maximum(rho, 1e-12), 0.0))  # [N]
    return a



def _phys_flux_x(eq, cons_col: Array) -> Array:
    # cons_col: [C,H,2] or [C,H] — we’ll just compute flux of each side separately outside
    return eq.get_fluxes_xi(eq.get_primitives_from_conservatives(cons_col), cons_col, axis=0)  # <-- API


def _phys_flux_y(eq, cons_row: Array) -> Array:
    return eq.get_fluxes_xi(eq.get_primitives_from_conservatives(cons_row), cons_row, axis=1)  # <-- API
def reconcile_interfaces(hydro, canvas, cfg, dt, params):
    eq = hydro.fluxes[0].eq_manage
    T, Ny, Nx = int(cfg.T), int(cfg.Ny), int(cfg.Nx)
    C, H, W   = canvas.shape
    dx = getattr(hydro, "dx_o", 1.0)
    dy = getattr(hydro, "dx_o", 1.0)

    U0 = canvas                     # freeze
    dU = jnp.zeros_like(U0)         # accumulate

    # --- vertical interfaces (x)
    for j in range(Nx):
        xL = ((j + 1) * T - 1) % W
        xR = ((j + 1) * T) % W
        UL, UR = U0[:, :, xL], U0[:, :, xR]

        primL = eq.get_primitives_from_conservatives(UL)
        primR = eq.get_primitives_from_conservatives(UR)
        FxL   = eq.get_fluxes_xi(primL, UL, axis=0)
        FxR   = eq.get_fluxes_xi(primR, UR, axis=0)
        a     = jnp.maximum(
                    _sound_speed(eq, primL, UL, axis=0),
                    _sound_speed(eq, primR, UR, axis=0)
                )[None, :]
        Fx = 0.5 * (FxL + FxR) - 0.5 * (a * (UR - UL))

        dU = dU.at[:, :, xL].add(- dt / dx * Fx)  # left loses
        dU = dU.at[:, :, xR].add(+ dt / dx * Fx)  # right gains

    # --- horizontal interfaces (y)
    for i in range(Ny):
        yB = ((i + 1) * T - 1) % H
        yT = ((i + 1) * T) % H
        UL, UR = U0[:, yB, :], U0[:, yT, :]

        primL = eq.get_primitives_from_conservatives(UL)
        primR = eq.get_primitives_from_conservatives(UR)
        FyL   = eq.get_fluxes_xi(primL, UL, axis=1)
        FyR   = eq.get_fluxes_xi(primR, UR, axis=1)
        a     = jnp.maximum(
                    _sound_speed(eq, primL, UL, axis=1),
                    _sound_speed(eq, primR, UR, axis=1)
                )[None, :]
        Fy = 0.5 * (FyL + FyR) - 0.5 * (a * (UR - UL))

        dU = dU.at[:, yB, :].add(- dt / dy * Fy)  # bottom loses
        dU = dU.at[:, yT, :].add(+ dt / dy * Fy)  # top gains

    return U0 + dU



# ------------------------ Reflux (coarse-fine) ------------------------

def reflux_L1_onto_L0(hydro, U0: Array, tiles1: Array, mask1: Array, cfg0: LevelConfig, dt: float) -> Array:
    """
    Very lightweight reflux: replace each refined parent’s boundary flux
    with the sum of fine fluxes across that boundary (downsampled).
    Here we approximate it by simply restricting the fine canvas to the
    parent region and replacing the parent *interior* (you can extend this
    with explicit face-flux bookkeeping if desired).
    """
    if tiles1 is None:
        return U0

    # assemble L1 canvas at fine resolution covering entire domain (unrefined areas 0)
    Ny, Nx, C, T, _ = tiles1.shape
    r = hydro.refine_ratio
    T0 = cfg0.T
    Tf = T0 * r

    # lay fine tiles into a fine canvas
    Uf = jnp.zeros((C, cfg0.H*r, cfg0.W*r), dtype=U0.dtype)
    for i in range(cfg0.Ny):
        for j in range(cfg0.Nx):
            if mask1[i, j]:
                y0 = i * Tf
                x0 = j * Tf
                Uf = Uf.at[:, y0:y0+Tf, x0:x0+Tf].set(tiles1[i, j])

    # restrict back to coarse and "blend" only over refined parents
    Uback = restrict_avg(Uf, r)  # [C,H,W]
    # conservative replacement on the union of refined parent tiles
    for i in range(cfg0.Ny):
        for j in range(cfg0.Nx):
            if mask1[i, j]:
                y0 = i * T0
                x0 = j * T0
                U0 = U0.at[:, y0:y0+T0, x0:x0+T0].set(Uback[:, y0:y0+T0, x0:x0+T0])
    return U0


# ------------------------ Hydro class ------------------------

class hydro:
    def __init__(
        self,
        fluxes,
        forces=(),
        boundary=None,
        recon=None,
        splitting_schemes=((1,2,2,1),(2,1,1,2)),
        use_amr: bool = True,
        adapt_interval: int = 1,
        refine_ratio: int = 2,
        base_tile: int = 16,
        max_dt: float = 1.0,
        dx: float = 1.0,
        maxjit: bool = False,
        snapshots: Optional[int] = None,
    ):
        self.fluxes = list(fluxes)
        self.forces = list(forces)
        self.boundary = boundary
        self.recon = recon
        self.splitting_schemes = tuple(tuple(s) for s in splitting_schemes)
        self.use_amr = use_amr
        self.adapt_interval = int(adapt_interval)
        self.refine_ratio = int(refine_ratio)
        self.base_tile = int(base_tile)
        self.max_dt = float(max_dt)
        self.dx_o = float(dx)
        self.maxjit = bool(maxjit)
        self.snapshots = snapshots

        self._amr_trace = {
            "depth_maps": [],
            "level_masks": [],
            "steps": [],
        }

    # ------------- PDE wrappers -------------

    def flux(self, sol: Array, ax: int, params: Dict) -> Array:
        total = jnp.zeros_like(sol)
        for f in self.fluxes:
            total = total + f.flux(sol, ax, params)  # uses your repo's API
        return total

    def timestep(self, fields: Array) -> float:
        dts = []
        for f in self.fluxes:
            dts.append(f.timestep(fields))  # your repo's API
        for g in self.forces:
            dts.append(g.timestep(fields))
        dt = jnp.min(jnp.stack(dts))
        return dt

    # ------------- time steppers -------------

    def _hydrostep_uniform(self, U, params, dt):
        for scheme in self.splitting_schemes:
            for ax in scheme:
                ra   = _roll_axis_from_sweep(ax)
                fu1  = self.flux(U, ax, params)
                rhs1 = fu1 - jnp.roll(fu1, 1, axis=ra)
                U    = U - (dt/(2.0*self.dx_o)) * rhs1

                fu2  = self.flux(U, ax, params)
                rhs2 = fu2 - jnp.roll(fu2, 1, axis=ra)
                U    = U - (dt/self.dx_o) * rhs2
                U    = U.at[0].set(jnp.maximum(U[0], 1e-12))
        return U
    
    def _hydrostep_amr(self, U0_in: Array, params: Dict, dt: float, step_idx: int) -> Array:
        """
        One AMR step:
          1) L0: step tiles with halos (periodic)
          2) reconcile same-level tile interfaces
          3) optional build L1, prolong, subcycle, restrict/reflux
        """
        C, H, W = U0_in.shape
        cfg0 = build_level0_config(H, W, self.base_tile, self.refine_ratio)

        # --- L0 tiles
        tiles0 = extract_tiles(U0_in, cfg0)

        # per-scheme splitting
        U0_canvas = assemble_from_tiles(tiles0, cfg0)
        for scheme in self.splitting_schemes:
            dt_ax = dt / len(scheme)
            for ax in scheme:
                tiles0     = step_tiles_with_halo(self, tiles0, dt_ax, ax, params, halo_w=2)
        U0_canvas  = assemble_from_tiles(tiles0, cfg0)

        # Optional: adapt every interval (build L1 mask)
        tiles1 = None
        mask1  = None
        if self.use_amr and (step_idx % self.adapt_interval == 0):
            Bmask = refine_mask_from_indicator(U0_canvas, cfg0, tau=0.02, dilate=1)  # [Ny,Nx] bool
            self._amr_trace["depth_maps"].append(Bmask.astype(jnp.int32))
            self._amr_trace["level_masks"].append([Bmask])
            self._amr_trace["steps"].append(int(step_idx))

            if jnp.any(Bmask):
                # build fine tiles by prolonging parent tiles on refined blocks
                r  = cfg0.r
                T0 = cfg0.T
                Tf = T0 * r

                # assemble parent tiles for refined blocks and prolong
                # build fine tiles by prolonging *all* parents (refined and unrefined)
                tiles1_list = []
                for i in range(cfg0.Ny):
                    row = []
                    for j in range(cfg0.Nx):
                        Uparent = tiles0[i, j]            # [C,T0,T0]
                        Uf = prolong_kron(Uparent, r)     # [C,Tf,Tf]
                        row.append(Uf)
                    tiles1_list.append(jnp.stack(row, axis=0))
                tiles1 = jnp.stack(tiles1_list, axis=0)   # [Ny,Nx,C,Tf,Tf]
                
                ## seam?
                
                cfg1 = LevelConfig(Ny=cfg0.Ny, Nx=cfg0.Nx, T=Tf, H=cfg0.H * r, W=cfg0.W * r, r=1)
                nsub = r
                for scheme in self.splitting_schemes:
                    for ax in scheme:
                        for _ in range(nsub):
                            
                            dt_sub = dt / len(scheme) / nsub
                            tiles1 = step_tiles_with_halo(self, tiles1, dt_sub * r, ax, params, halo_w=2)
                           # dt_sub   = dt / len(scheme) / nsub
                            #tiles1   = step_tiles_with_halo(self, tiles1, dt_sub, ax, params, halo_w=2)
                
                mask1  = Bmask
                # subcycle fine level
             #   nsub = r
             #   for scheme in self.splitting_schemes:
             #       for ax in scheme:
             #           for _ in range(nsub):
              #              tiles1 = step_tiles_with_halo(self, tiles1, dt / len(scheme) / nsub, ax, params, halo_w=2)

                # restrict/reflux onto parents
                U0_canvas = reflux_L1_onto_L0(self, U0_canvas, tiles1, mask1, cfg0, dt)

        return U0_canvas

    # ------------- public API -------------

    def evolve(self, input_fields: Array, params: Dict):
        U = input_fields
        self._amr_trace = {"depth_maps": [], "level_masks": [], "steps": []}

        outputs = []
        n_steps = getattr(self, "n_super_step", 10)  # keep compatibility with your driver
        for i in range(n_steps):
            # CFL / dt selection
            dt = jnp.minimum(self.max_dt, self.timestep(U))

            if self.use_amr:
                U = self._hydrostep_amr(U, params, float(dt), i)
            else:
                U = self._hydrostep_uniform(U, params, float(dt))

            if self.snapshots and (i % self.snapshots == 0):
                outputs.append(U)

        return U, self._amr_trace
