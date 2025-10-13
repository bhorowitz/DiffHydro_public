"""Adaptive mesh refinement hydrodynamics solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from ..common import Array, positivity_fix


def step_tiles_with_halo(hydro, tiles, dt, ax, halo_w: int, dx_o: float, params):
    """Perform an RK2 update on a lattice of tiles with halo exchanges."""
    h = int(halo_w)
    Ny, Nx, C, T, _ = tiles.shape
    ra = 2 if int(ax) == 1 else 1

    def _rhs(U_pad):
        fu = hydro.flux(U_pad, ra, params)
        return fu - jnp.roll(fu, 1, axis=ra)

    if int(ax) == 1:
        def pad_with_halos(tiles_):
            left = jnp.roll(tiles_, 1, axis=1)[..., -h:]
            right = jnp.roll(tiles_, -1, axis=1)[..., :h]
            return jnp.concatenate([left, tiles_, right], axis=-1)

        def crop_interior(arr):
            return arr[..., h:-h]
    elif int(ax) == 2:
        def pad_with_halos(tiles_):
            down = jnp.roll(tiles_, 1, axis=0)[..., -h:, :]
            up = jnp.roll(tiles_, -1, axis=0)[..., :h, :]
            return jnp.concatenate([down, tiles_, up], axis=-2)

        def crop_interior(arr):
            return arr[..., h:-h, :]
    else:
        raise ValueError(f"bad axis {ax}")

    U0_pad = pad_with_halos(tiles)
    rhs0_all = jax.vmap(jax.vmap(_rhs, in_axes=0), in_axes=0)(U0_pad)
    rhs0 = crop_interior(rhs0_all)
    tiles_1 = tiles - (dt / (2.0 * dx_o)) * rhs0
    tiles_1 = tiles_1.at[..., 0, :, :].set(jnp.maximum(tiles_1[..., 0, :, :], 1e-12))

    U1_pad = pad_with_halos(tiles_1)
    rhs1_all = jax.vmap(jax.vmap(_rhs, in_axes=0), in_axes=0)(U1_pad)
    rhs1 = crop_interior(rhs1_all)
    tiles_2 = tiles_1 - (dt / dx_o) * rhs1
    tiles_2 = tiles_2.at[..., 0, :, :].set(jnp.maximum(tiles_2[..., 0, :, :], 1e-12))

    return tiles_2


@dataclass
class LevelConfig:
    Ny: int
    Nx: int
    T: int
    H: int
    W: int
    r: int

    @property
    def shape_tiles(self) -> Tuple[int, int]:
        return (self.Ny, self.Nx)

    @property
    def shape_canvas(self) -> Tuple[int, int]:
        return (self.H, self.W)


def build_level0_config(H: int, W: int, T: int, r: int) -> LevelConfig:
    assert H % T == 0 and W % T == 0, "H and W must be divisible by base tile size T"
    Ny, Nx = H // T, W // T
    return LevelConfig(Ny=Ny, Nx=Nx, T=T, H=H, W=W, r=r)


def extract_tiles(U: Array, cfg: LevelConfig) -> Array:
    C, H, W = U.shape
    Ny, Nx, T = cfg.Ny, cfg.Nx, cfg.T
    return U.reshape(C, Ny, T, Nx, T).transpose(1, 3, 0, 2, 4)


def assemble_from_tiles(tiles: Array, cfg: LevelConfig) -> Array:
    Ny, Nx, C, T, _ = tiles.shape
    return tiles.transpose(2, 0, 3, 1, 4).reshape(C, Ny * T, Nx * T)


def _sobel_like_indicator(rho: Array) -> Array:
    gx = jnp.abs(rho - jnp.roll(rho, 1, axis=-1))
    gy = jnp.abs(rho - jnp.roll(rho, 1, axis=-2))
    return gx + gy


def _dilate_mask(mask, iters: int = 1):
    def step(m):
        neigh = jnp.stack(
            [jnp.roll(jnp.roll(m, di, 0), dj, 1) for di in (-1, 0, 1) for dj in (-1, 0, 1)],
            0,
        )
        return jnp.max(neigh, axis=0)

    return lax.fori_loop(0, iters, lambda _, m: step(m), mask)


def refine_mask_from_indicator_hyst(
    U0: Array,
    cfg: LevelConfig,
    prev_mask: Optional[Array],
    tau_low: float = 0.015,
    tau_high: float = 0.03,
    dilate: int = 2,
) -> Array:
    rho = U0[0]
    ind = _sobel_like_indicator(rho)
    mean_ind = jnp.mean(ind) + 1e-12

    refine_hi = _dilate_mask(ind > (tau_high * mean_ind), iters=dilate)
    refine_lo = _dilate_mask(ind > (tau_low * mean_ind), iters=dilate)

    Cmask_hi = refine_hi.reshape(cfg.Ny, cfg.T, cfg.Nx, cfg.T).transpose(0, 2, 1, 3)
    Cmask_lo = refine_lo.reshape(cfg.Ny, cfg.T, cfg.Nx, cfg.T).transpose(0, 2, 1, 3)

    tile_hi = Cmask_hi.sum(axis=(2, 3)) > 0
    tile_lo = Cmask_lo.sum(axis=(2, 3)) > 0

    if prev_mask is None:
        return tile_hi
    return jnp.where(prev_mask, tile_lo, tile_hi)


def _minmod(a, b):
    s = 0.5 * (jnp.sign(a) + jnp.sign(b))
    return s * jnp.minimum(jnp.abs(a), jnp.abs(b))


def prolong_PLM_minmod(Uc: Array, r: int) -> Array:
    C, Hc, Wc = Uc.shape
    Ux = _minmod(Uc - jnp.roll(Uc, 1, axis=-1), jnp.roll(Uc, -1, axis=-1) - Uc)
    Uy = _minmod(Uc - jnp.roll(Uc, 1, axis=-2), jnp.roll(Uc, -1, axis=-2) - Uc)

    k = jnp.arange(r) + 0.5
    xi = (k / r) - 0.5
    eta = (k / r) - 0.5

    Xi = xi[None, None, None, None, :]
    Eta = eta[None, None, None, :, None]

    u = Uc[:, :, :, None, None]
    sx = Ux[:, :, :, None, None]
    sy = Uy[:, :, :, None, None]

    patches = u + Xi * sx + Eta * sy
    return patches.transpose(0, 1, 3, 2, 4).reshape(Uc.shape[0], Hc * r, Wc * r)


def restrict_avg(Uf: Array, r: int) -> Array:
    C, Hf, Wf = Uf.shape
    Hc, Wc = Hf // r, Wf // r
    return Uf.reshape(C, Hc, r, Wc, r).mean(axis=(2, 4))


class HydroAMR:
    """AMR-enabled hydrodynamics driver."""

    def __init__(
        self,
        fluxes,
        forces=(),
        boundary=None,
        recon=None,
        splitting_schemes=((1, 2, 2, 1), (2, 1, 1, 2)),
        use_amr: bool = True,
        adapt_interval: int = 1,
        refine_ratio: int = 2,
        base_tile: int = 16,
        max_dt: float = 1.0,
        dx: float = 1.0,
        maxjit: bool = False,
        snapshots: Optional[int] = None,
        n_super_step: int = 25,
        halo_width: int = 3,
        tau_low: float = 0.015,
        tau_high: float = 0.03,
        dilate: int = 2,
        seam_reconcile: bool = False,
    ) -> None:
        self.fluxes = list(fluxes)
        self.forces = list(forces)
        self.boundary = boundary
        self.recon = recon
        self.splitting_schemes = tuple(tuple(s) for s in splitting_schemes)
        self.use_amr = bool(use_amr)
        self.adapt_interval = int(adapt_interval)
        self.refine_ratio = int(refine_ratio)
        self.base_tile = int(base_tile)
        self.max_dt = float(max_dt)
        self.dx_o = float(dx)
        self.maxjit = bool(maxjit)
        self.snapshots = snapshots
        self.n_super_step = int(n_super_step)
        self.halo_width = int(halo_width)
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.dilate = int(dilate)
        self.seam_reconcile = bool(seam_reconcile)
        self._amr_trace = {"depth_maps": [], "level_masks": [], "steps": []}
        self._prev_mask = None

    # PDE wrappers -----------------------------------------------------
    def flux(self, sol: Array, ax: int, params: Dict) -> Array:
        total = jnp.zeros_like(sol)
        for flux in self.fluxes:
            total = total + flux.flux(sol, ax, params)
        return total

    def timestep(self, fields: Array) -> Array:
        dts = [flux.timestep(fields) for flux in self.fluxes]
        for force in self.forces:
            dts.append(force.timestep(fields))
        if not dts:
            return jnp.asarray(self.max_dt)
        return jnp.min(jnp.stack(dts))

    # Uniform step -----------------------------------------------------
    def _hydrostep_uniform(self, U: Array, params: Dict, dt: float) -> Array:
        for scheme in self.splitting_schemes:
            for ax in scheme:
                ra = 2 if ax == 1 else 1
                fu1 = self.flux(U, ax, params)
                rhs1 = fu1 - jnp.roll(fu1, 1, axis=ra)
                U = positivity_fix(self, U - (dt / (2.0 * self.dx_o)) * rhs1)
                fu2 = self.flux(U, ax, params)
                rhs2 = fu2 - jnp.roll(fu2, 1, axis=ra)
                U = positivity_fix(self, U - (dt / self.dx_o) * rhs2)
        return U

    # AMR step ---------------------------------------------------------
    def _hydrostep_amr(self, U0_in: Array, params: Dict, dt: float, step_idx: int) -> Array:
        C, H, W = U0_in.shape
        cfg0 = build_level0_config(H, W, self.base_tile, self.refine_ratio)
        tiles0 = extract_tiles(U0_in, cfg0)

        for scheme in self.splitting_schemes:
            dt_ax = dt / len(scheme)
            for ax in scheme:
                tiles0 = step_tiles_with_halo(
                    self, tiles0, dt_ax, ax, self.halo_width, self.dx_o, params
                )

        U0 = positivity_fix(self, assemble_from_tiles(tiles0, cfg0))

        if self.use_amr and (step_idx % self.adapt_interval == 0):
            Bmask = refine_mask_from_indicator_hyst(
                U0,
                cfg0,
                self._prev_mask,
                tau_low=self.tau_low,
                tau_high=self.tau_high,
                dilate=self.dilate,
            )

            self._prev_mask = Bmask
            self._amr_trace.setdefault("n_refined_tiles", []).append(int(Bmask.sum()))
            self._amr_trace.setdefault("steps", []).append(int(step_idx))
            self._amr_trace["depth_maps"].append(Bmask.astype(jnp.int32))
            self._amr_trace["level_masks"].append([Bmask])

            if jnp.any(Bmask):
                r = cfg0.r
                Tf = cfg0.T * r

                U1_full = prolong_PLM_minmod(U0, r)
                U1_full = positivity_fix(self, U1_full)

                cfg1 = LevelConfig(cfg0.Ny, cfg0.Nx, Tf, H * r, W * r, r)
                tiles1 = extract_tiles(U1_full, cfg1)

                for scheme in self.splitting_schemes:
                    for ax in scheme:
                        for _ in range(r):
                            dt_sub = dt / len(scheme) / r
                            dx_f = self.dx_o / r
                            tiles1 = step_tiles_with_halo(
                                self, tiles1, dt_sub, ax, self.halo_width, dx_f, params
                            )

                U1_full = positivity_fix(self, assemble_from_tiles(tiles1, cfg1))
                U0 = restrict_avg(U1_full, r)

        return U0

    # Public API -------------------------------------------------------
    def evolve(self, input_fields: Array, params: Dict):
        U = input_fields
        self._amr_trace = {"depth_maps": [], "level_masks": [], "steps": []}

        for i in range(self.n_super_step):
            dt = jnp.minimum(self.max_dt, self.timestep(U))
            if self.use_amr:
                U = self._hydrostep_amr(U, params, float(dt), i)
            else:
                U = self._hydrostep_uniform(U, params, float(dt))

        return U, self._amr_trace


# Public alias --------------------------------------------------------
hydro = HydroAMR

__all__ = [
    "HydroAMR",
    "assemble_from_tiles",
    "build_level0_config",
    "extract_tiles",
    "hydro",
    "prolong_PLM_minmod",
    "refine_mask_from_indicator_hyst",
    "restrict_avg",
    "step_tiles_with_halo",
]
