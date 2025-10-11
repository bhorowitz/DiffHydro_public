# hydro_core.py — AMR v3 (positivity + full fine halos)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp

Array = jax.Array

#RK2
def step_tiles_with_halo(hydro, tiles, dt, ax, halo_w: int, dx_o: float, params):
    """
    RK2 on tiles with a halo exchange between stages (periodic).
    tiles: [Ny, Nx, C, T, T]   ->   same shape
    ax: sweep axis (1=x, 2=y)
    """
    h = int(halo_w)
    Ny, Nx, C, T, _ = tiles.shape
    ra = 2 if int(ax) == 1 else 1  # array axis used in the finite-difference

    def _rhs(U_pad):
        # U_pad shape: [C, Hpad, Wpad] where pad = T (+ 2*h in the swept direction)
        fu = hydro.flux(U_pad, ax, params)
        return fu - jnp.roll(fu, 1, axis=ra)

    # ---- helpers to pad/crop along the swept direction ----
    if int(ax) == 1:
        def pad_with_halos(tiles_):
            left  = jnp.roll(tiles_,  1, axis=1)[..., -h:]  # [Ny,Nx,C,T,h]
            right = jnp.roll(tiles_, -1, axis=1)[..., :h]   # [Ny,Nx,C,T,h]
            return jnp.concatenate([left, tiles_, right], axis=-1)  # [Ny,Nx,C,T,T+2h]

        def crop_interior(arr):
            return arr[..., h:-h]  # crop x

    elif int(ax) == 2:
        def pad_with_halos(tiles_):
            down = jnp.roll(tiles_,  1, axis=0)[..., -h:, :]  # [Ny,Nx,C,h,T]
            up   = jnp.roll(tiles_, -1, axis=0)[..., :h, :]   # [Ny,Nx,C,h,T]
            return jnp.concatenate([down, tiles_, up], axis=-2)  # [Ny,Nx,C,T+2h,T]

        def crop_interior(arr):
            return arr[..., h:-h, :]  # crop y
    else:
        raise ValueError(f"bad axis {ax}")

    # ------------------------- Stage 1 (half step) -------------------------
    U0_pad   = pad_with_halos(tiles)                          # [Ny,Nx,C,*,*]
    rhs0_all = jax.vmap(jax.vmap(_rhs, in_axes=0), in_axes=0)(U0_pad)
    rhs0     = crop_interior(rhs0_all)                         # interior stencil
    tiles_1  = tiles - (dt / (2.0 * dx_o)) * rhs0             # half-step on interiors only

    # -------------------- Refresh halos from Stage-1 -----------------------
    U1_pad   = pad_with_halos(tiles_1)
    rhs1_all = jax.vmap(jax.vmap(_rhs, in_axes=0), in_axes=0)(U1_pad)
    rhs1     = crop_interior(rhs1_all)

    # ------------------------- Stage 2 (full step) -------------------------
    tiles_2  = tiles_1 - (dt / dx_o) * rhs1

    # Optional: mimic your gentle density floor on channel 0
    tiles_2  = tiles_2.at[..., 0, :, :].set(jnp.maximum(tiles_2[..., 0, :, :], 1e-12))
    return tiles_2

# ------------------------ Level config & helpers ------------------------
@dataclass
class LevelConfig:
    Ny: int; Nx: int; T: int; H: int; W: int; r: int
    @property
    def shape_tiles(self) -> Tuple[int, int]: return (self.Ny, self.Nx)
    @property
    def shape_canvas(self) -> Tuple[int, int]: return (self.H, self.W)

def build_level0_config(H: int, W: int, T: int, r: int) -> LevelConfig:
    assert H % T == 0 and W % T == 0, "H and W must be divisible by base tile size T"
    Ny, Nx = H // T, W // T
    return LevelConfig(Ny=Ny, Nx=Nx, T=T, H=H, W=W, r=r)

def extract_tiles(U: Array, cfg: LevelConfig) -> Array:
    C, H, W = U.shape; Ny, Nx, T = cfg.Ny, cfg.Nx, cfg.T
    U4 = U.reshape(C, Ny, T, Nx, T).transpose(1, 3, 0, 2, 4)   # [Ny, Nx, C, T, T]
    return U4

def assemble_from_tiles(tiles: Array, cfg: LevelConfig) -> Array:
    Ny, Nx, C, T, _ = tiles.shape
    U = tiles.transpose(2, 0, 3, 1, 4).reshape(C, Ny*T, Nx*T)
    return U

# ------------------------ Positivity helpers ------------------------
def _as_idx(x):
    # handle int or [int] forms
    if isinstance(x, (list, tuple)): return x[0]
    return x

def positivity_fix_with_eq(eq, U: Array, p_floor: float = 1e-12, rho_floor: float = 1e-12) -> Array:
    """Ensure rho>=rho_floor and p>=p_floor using eq_manage (Euler-like)."""
    rho_i = _as_idx(eq.mass_ids); E_i = _as_idx(eq.energy_ids)
    prim  = eq.get_primitives_from_conservatives(U)  # velocities etc.
    gamma = getattr(eq, 'gamma', 1.4)
    rho   = U[rho_i]
    # v^2 from primitive velocities
    v2 = 0.0
    for vid in getattr(eq, 'vel_ids', []):
        v = prim[vid]
        v2 = v2 + v*v
    E = U[E_i]
    KE = 0.5 * rho * v2
    Emin = KE + p_floor/(gamma - 1.0)
    U = U.at[rho_i].set(jnp.maximum(rho, rho_floor))
    U = U.at[E_i].set(jnp.maximum(E, Emin))
    return U

def positivity_fix_simple(U: Array, rho_floor: float = 1e-12, e_floor: float = 1e-12) -> Array:
    U = U.at[0].set(jnp.maximum(U[0], rho_floor))
    if U.shape[0] >= 4:
        U = U.at[3].set(jnp.maximum(U[3], e_floor))
    return U

def positivity_fix(hydro, U: Array) -> Array:
    eq = None
    if getattr(hydro, 'fluxes', None):
        eq = getattr(hydro.fluxes[0], 'eq_manage', None)
    if eq is not None:
        return positivity_fix_with_eq(eq, U)
    return positivity_fix_simple(U)

# ------------------------ Refinement mask with hysteresis ------------------------
def _sobel_like_indicator(rho: Array) -> Array:
    gx = jnp.abs(rho - jnp.roll(rho, 1, axis=-1))
    gy = jnp.abs(rho - jnp.roll(rho, 1, axis=-2))
    return gx + gy

def _dilate_mask(mask: Array, iters: int = 1) -> Array:
    if iters <= 0: return mask
    for _ in range(iters):
        nb = [jnp.roll(mask, s, axis=a) for a in (-1, -2) for s in (-1, 1)]
        st = mask
        for n in nb: st = jnp.maximum(st, n)
        mask = st
    return mask

def refine_mask_from_indicator_hyst(U0: Array, cfg: LevelConfig, prev_mask: Optional[Array],
                                    tau_low: float = 0.015, tau_high: float = 0.03, dilate: int = 2) -> Array:
    rho = U0[0]; ind = _sobel_like_indicator(rho); mean_ind = jnp.mean(ind) + 1e-12
    refine_hi = _dilate_mask(ind > (tau_high*mean_ind), iters=dilate)
    refine_lo = _dilate_mask(ind > (tau_low *mean_ind), iters=dilate)
    Cmask_hi = refine_hi.reshape(cfg.Ny, cfg.T, cfg.Nx, cfg.T).transpose(0,2,1,3)
    Cmask_lo = refine_lo.reshape(cfg.Ny, cfg.T, cfg.Nx, cfg.T).transpose(0,2,1,3)
    tile_hi = (Cmask_hi.sum(axis=(2,3)) > 0)
    tile_lo = (Cmask_lo.sum(axis=(2,3)) > 0)
    if prev_mask is None: return tile_hi
    return jnp.where(prev_mask, tile_lo, tile_hi)

# ------------------------ Prolongation / Restriction ------------------------
def _minmod(a, b): s = 0.5*(jnp.sign(a)+jnp.sign(b)); return s*jnp.minimum(jnp.abs(a), jnp.abs(b))

def prolong_PLM_minmod(Uc: Array, r: int) -> Array:
    C, Hc, Wc = Uc.shape
    Ux = _minmod(Uc - jnp.roll(Uc,1,axis=-1), jnp.roll(Uc,-1,axis=-1)-Uc)
    Uy = _minmod(Uc - jnp.roll(Uc,1,axis=-2), jnp.roll(Uc,-1,axis=-2)-Uc)
    Hf, Wf = Hc*r, Wc*r; Uf = jnp.zeros((C,Hf,Wf), dtype=Uc.dtype)
    k = jnp.arange(r) + 0.5; xi = (k/r) - 0.5; eta = (k/r) - 0.5
    Xi  = xi[None,:,None]; Eta = eta[:,None]
    for iy in range(Hc):
        for ix in range(Wc):
            u  = Uc[:,iy,ix][:,None,None]
            sx = Ux[:,iy,ix][:,None,None]
            sy = Uy[:,iy,ix][:,None,None]
            patch = u + Xi*sx + Eta*sy
            y0, x0 = iy*r, ix*r
            Uf = Uf.at[:, y0:y0+r, x0:x0+r].set(patch)
    return Uf

def restrict_avg(Uf: Array, r: int) -> Array:
    C, Hf, Wf = Uf.shape; Hc, Wc = Hf//r, Wf//r
    return Uf.reshape(C,Hc,r,Wc,r).mean(axis=(2,4))

# ------------------------ Local step with frozen ghosts ------------------------
def _roll_axis_from_sweep(ax: int) -> int: return 2 if ax == 1 else 1

def _update_interior(U_bc: Array, rhs: Array, halo_w: int, scale: float) -> Array:
    h = int(halo_w)
    return U_bc.at[:, h:-h, h:-h].add(-scale * rhs[:, h:-h, h:-h]) if h>0 else (U_bc - scale*rhs)

def _solve_step_freeze_ghosts(hydro, U_bc: Array, dt: float, ax: int, halo_w: int, dx_o: float, params: Dict) -> Array:
    ra = _roll_axis_from_sweep(ax)
    fu1  = hydro.flux(U_bc, ax, params); rhs1 = fu1 - jnp.roll(fu1, 1, axis=ra)
    U1   = _update_interior(U_bc, rhs1, halo_w, dt/(2.0*dx_o)); U1 = positivity_fix(hydro, U1)
    fu2  = hydro.flux(U1, ax, params);  rhs2 = fu2 - jnp.roll(fu2, 1, axis=ra)
    U2   = _update_interior(U1, rhs2, halo_w, dt/dx_o);         U2 = positivity_fix(hydro, U2)
    return U2

# ------------------------ Halos on tile lattice ------------------------
def _make_halos_x(tiles: Array, h: int):
    left_nb  = jnp.roll(tiles,  1, axis=1); right_nb = jnp.roll(tiles, -1, axis=1)
    return left_nb[..., -h:], right_nb[..., :h]

def _make_halos_y(tiles: Array, h: int):
    down_nb  = jnp.roll(tiles,  1, axis=0); up_nb    = jnp.roll(tiles, -1, axis=0)
    return down_nb[..., -h:, :], up_nb[..., :h, :]

def step_tiles_with_halo_old(hydro, tiles: Array, dt: float, ax: int, halo_w: int, dx_o: float, params: Dict) -> Array:
    Ny, Nx, C, T, _ = tiles.shape; h = int(halo_w)
    if int(ax) == 1:
        Lh, Rh = _make_halos_x(tiles, h)
        Ubc = jnp.concatenate([Lh, tiles, Rh], axis=-1)
        step = jax.vmap(jax.vmap(lambda U: _solve_step_freeze_ghosts(hydro, U, dt, 1, h, dx_o, params), in_axes=0), in_axes=0)
        Uout = step(Ubc); Uret = Uout[..., h:-h]
    elif int(ax) == 2:
        Bh, Th = _make_halos_y(tiles, h)
        Ubc = jnp.concatenate([Bh, tiles, Th], axis=-2)
        step = jax.vmap(jax.vmap(lambda U: _solve_step_freeze_ghosts(hydro, U, dt, 2, h, dx_o, params), in_axes=0), in_axes=0)
        Uout = step(Ubc); Uret = Uout[..., h:-h, :]
    else:
        raise ValueError(f'bad axis {ax}')
    return Uret

# ------------------------ Hydro class ------------------------
class hydro:
    def __init__(self, fluxes, forces=(), boundary=None, recon=None,
                 splitting_schemes=((1,2,2,1),(2,1,1,2)),
                 use_amr=True, adapt_interval=1, refine_ratio=2, base_tile=16,
                 max_dt=1.0, dx=1.0, maxjit=False, snapshots=None, n_super_step=5,
                 halo_width=3, tau_low=0.015, tau_high=0.03, dilate=2,
                 seam_reconcile=False):
        self.fluxes=list(fluxes); self.forces=list(forces); self.boundary=boundary; self.recon=recon
        self.splitting_schemes=tuple(tuple(s) for s in splitting_schemes)
        self.use_amr=bool(use_amr); self.adapt_interval=int(adapt_interval); self.refine_ratio=int(refine_ratio)
        self.base_tile=int(base_tile); self.max_dt=float(max_dt); self.dx_o=float(dx)
        self.maxjit=bool(maxjit); self.snapshots=snapshots; self.n_super_step=int(n_super_step)
        self.halo_width=int(halo_width); self.tau_low=float(tau_low); self.tau_high=float(tau_high)
        self.dilate=int(dilate); self.seam_reconcile=bool(seam_reconcile)
        self._amr_trace={'depth_maps':[], 'level_masks':[], 'steps':[]}; self._prev_mask=None

    # PDE wrappers
    def flux(self, sol: Array, ax: int, params: Dict) -> Array:
        total = jnp.zeros_like(sol)
        for f in self.fluxes: total = total + f.flux(sol, ax, params)
        return total

    def timestep(self, fields: Array) -> float:
        dts = [f.timestep(fields) for f in self.fluxes]
        for g in self.forces: dts.append(g.timestep(fields))
        return jnp.min(jnp.stack(dts)) if dts else jnp.asarray(self.max_dt)

    # Uniform step (no AMR)
    def _hydrostep_uniform(self, U: Array, params: Dict, dt: float) -> Array:
        for scheme in self.splitting_schemes:
            for ax in scheme:
                ra = 2 if ax==1 else 1
                fu1=self.flux(U,ax,params); rhs1=fu1 - jnp.roll(fu1,1,axis=ra)
                U=positivity_fix(self, U - (dt/(2.0*self.dx_o))*rhs1)
                fu2=self.flux(U,ax,params); rhs2=fu2 - jnp.roll(fu2,1,axis=ra)
                U=positivity_fix(self, U - (dt/self.dx_o)*rhs2)
        return U

    # AMR step
    def _hydrostep_amr(self, U0_in: Array, params: Dict, dt: float, step_idx: int) -> Array:
        C,H,W = U0_in.shape; cfg0=build_level0_config(H,W,self.base_tile,self.refine_ratio)
        tiles0 = extract_tiles(U0_in, cfg0)
        # coarse sweep
        for scheme in self.splitting_schemes:
            dt_ax = dt/len(scheme)
            for ax in scheme:
                tiles0 = step_tiles_with_halo(self, tiles0, dt_ax, ax, self.halo_width, self.dx_o, params)
        U0 = positivity_fix(self, assemble_from_tiles(tiles0, cfg0))

        # adapt + L1
        if self.use_amr and (step_idx % self.adapt_interval == 0):
            Bmask = refine_mask_from_indicator_hyst(U0, cfg0, self._prev_mask,
                                                    tau_low=self.tau_low, tau_high=self.tau_high, dilate=self.dilate)
            
            ###diagnostic print
            
            # --- after you compute Bmask (shape [Ny, Nx]) in _hydrostep_amr ---
            n_tiles = int(Bmask.sum())           # number of refined parent tiles at L1
            T0 = int(cfg0.T)                     # coarse tile size (cells per side)
            r  = int(cfg0.r)                     # refinement ratio

            # Coarse-area coverage (fraction of base grid covered by refined tiles)
            coarse_cells_refined = n_tiles * (T0 * T0)
            coverage_frac = float(coarse_cells_refined) / float(cfg0.H * cfg0.W)
            # Active fine cells used inside refined patches (useful for cost tracking)
            active_fine_cells = n_tiles * (T0 * r) * (T0 * r)

            # Store in your existing trace (so you can plot later)
            self._amr_trace.setdefault("n_refined_tiles", []).append(n_tiles)
            self._amr_trace.setdefault("refined_parent_cells", []).append(coarse_cells_refined)
            self._amr_trace.setdefault("active_fine_cells", []).append(active_fine_cells)
            self._amr_trace.setdefault("coverage_frac", []).append(coverage_frac)

            # Optional quick print
            if getattr(self, "verbose_amr", False):
                print(f"[step {step_idx:4d}] tiles={n_tiles:4d} | cov={100*coverage_frac:5.2f}% | fine_cells={active_fine_cells}")
            
            self._prev_mask=Bmask
            self._amr_trace['depth_maps'].append(Bmask.astype(jnp.int32))
            self._amr_trace['level_masks'].append([Bmask]); self._amr_trace['steps'].append(int(step_idx))
            if jnp.any(Bmask):
                r=cfg0.r; Tf=cfg0.T*r
                # FULL prolongation for *all* tiles (so halos are realistic)
                U1_full = prolong_PLM_minmod(U0, r); U1_full = positivity_fix(self, U1_full)
                tiles1  = extract_tiles(U1_full, LevelConfig(cfg0.Ny,cfg0.Nx,Tf,H*r,W*r,r))

                # subcycle fine everywhere (simple, robust); we only restrict from refined parents
                for scheme in self.splitting_schemes:
                    for ax in scheme:
                        for _ in range(r):
                            dt_sub = dt/len(scheme)/r; dx_f = self.dx_o / r
                            tiles1 = step_tiles_with_halo(self, tiles1, dt_sub, ax, self.halo_width, dx_f, params)
                U1_full = positivity_fix(self, assemble_from_tiles(tiles1, LevelConfig(cfg0.Ny,cfg0.Nx,Tf,H*r,W*r,r)))

                # restrict and replace only refined parents
                U1_back = restrict_avg(U1_full, r)
                for i in range(cfg0.Ny):
                    for j in range(cfg0.Nx):
                        if Bmask[i,j]:
                            y0=i*cfg0.T; x0=j*cfg0.T
                            U0 = U0.at[:, y0:y0+cfg0.T, x0:x0+cfg0.T].set(U1_back[:, y0:y0+cfg0.T, x0:x0+cfg0.T])
        return U0

    # Public API
    def evolve(self, input_fields: Array, params: Dict):
        U = input_fields; self._amr_trace={'depth_maps':[], 'level_masks':[], 'steps':[]}
        outs=[]
        for i in range(self.n_super_step):
            dt = jnp.minimum(self.max_dt, self.timestep(U))
            U  = self._hydrostep_amr(U, params, float(dt), i) if self.use_amr else self._hydrostep_uniform(U, params, float(dt))
            if self.snapshots and (i % self.snapshots == 0): outs.append(U)
        return U, self._amr_trace
