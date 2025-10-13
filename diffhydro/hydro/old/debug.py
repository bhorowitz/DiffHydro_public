"""Legacy AMR diagnostics and experimental utilities."""
from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from ..amr.core import LevelConfig, extract_tiles
from ..common import Array, positivity_fix


def check_total_energy(U: Array, name: str = "") -> None:
    mass = float(jnp.sum(U[0]))
    energy = float(jnp.sum(U[3]))
    print(f"{name}: Mass={mass:.6e}, Energy={energy:.6e}")


def diagnose_tile_continuity(tiles: Array, cfg: LevelConfig, name: str = ""):
    """Check for discontinuities at tile boundaries."""
    Ny, Nx, _, _, _ = tiles.shape

    max_jump_x = 0.0
    max_jump_y = 0.0

    for i in range(Ny):
        for j in range(Nx):
            right_edge = tiles[i, j, 3, :, -1]
            left_edge = tiles[i, (j + 1) % Nx, 3, :, 0]
            jump = jnp.max(jnp.abs(right_edge - left_edge))
            max_jump_x = max(max_jump_x, float(jump))

    for i in range(Ny):
        for j in range(Nx):
            bottom_edge = tiles[i, j, 3, -1, :]
            top_edge = tiles[(i + 1) % Ny, j, 3, 0, :]
            jump = jnp.max(jnp.abs(bottom_edge - top_edge))
            max_jump_y = max(max_jump_y, float(jump))

    print(f"{name}: Max jump X={max_jump_x:.3e}, Y={max_jump_y:.3e}")
    return max_jump_x, max_jump_y


def debug_amr_step(self, U0, U1_back, Bmask, cfg0, step_idx):
    """Diagnostic helper kept for reference during AMR development."""
    import matplotlib.pyplot as plt

    mass_coarse = float(jnp.sum(U0[0]))
    mass_fine = float(jnp.sum(U1_back[0]))
    energy_coarse = float(jnp.sum(U0[3]))
    energy_fine = float(jnp.sum(U1_back[3]))

    print(f"\n=== Step {step_idx} Diagnostics ===")
    print(
        f"Mass:   coarse={mass_coarse:.6e}, fine={mass_fine:.6e}, "
        f"error={abs(mass_fine-mass_coarse)/mass_coarse:.3e}"
    )
    print(
        f"Energy: coarse={energy_coarse:.6e}, fine={energy_fine:.6e}, "
        f"error={abs(energy_fine-energy_coarse)/energy_coarse:.3e}"
    )

    tiles0 = extract_tiles(U0, cfg0)
    print("\nCoarse tile discontinuities:")
    diagnose_tile_continuity(tiles0, cfg0, "Coarse")

    tiles1 = extract_tiles(U1_back, cfg0)
    print("\nFine (restricted) tile discontinuities:")
    diagnose_tile_continuity(tiles1, cfg0, "Fine restricted")

    if step_idx % 5 == 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(U0[0], origin="lower")
        axes[0, 0].set_title(f"Coarse Density (step {step_idx})")
        axes[0, 1].imshow(U0[3], origin="lower")
        axes[0, 1].set_title("Coarse Energy")

        T = cfg0.T
        for i in range(cfg0.Ny):
            for j in range(cfg0.Nx):
                if Bmask[i, j]:
                    y0, x0 = i * T, j * T
                    rect = plt.Rectangle((x0 - 0.5, y0 - 0.5), T, T, fill=False, edgecolor="red", linewidth=1)
                    axes[0, 1].add_patch(rect)

        axes[1, 0].imshow(U1_back[0], origin="lower")
        axes[1, 0].set_title("Fine Density (restricted)")
        axes[1, 1].imshow(U1_back[3], origin="lower")
        axes[1, 1].set_title("Fine Energy (restricted)")

        diff = U1_back[3] - U0[3]
        axes[0, 2].imshow(diff, origin="lower", cmap="RdBu")
        axes[0, 2].set_title("Energy Difference (fine - coarse)")

        grad = jnp.abs(jnp.gradient(U0[3])[0]) + jnp.abs(jnp.gradient(U0[3])[1])
        axes[1, 2].imshow(grad, origin="lower")
        axes[1, 2].set_title("Coarse Gradient Magnitude")

        plt.tight_layout()
        plt.savefig(f"amr_debug_step_{step_idx:04d}.png", dpi=150)
        plt.close()


def _roll_axis_from_sweep(ax: int) -> int:
    return 2 if ax == 1 else 1


def _update_interior(U_bc: Array, rhs: Array, halo_w: int, scale: float) -> Array:
    h = int(halo_w)
    if h > 0:
        return U_bc.at[:, h:-h, h:-h].add(-scale * rhs[:, h:-h, h:-h])
    return U_bc - scale * rhs


def solve_step_freeze_ghosts(hydro, U_bc: Array, dt: float, ax: int, halo_w: int, dx_o: float, params: Dict) -> Array:
    ra = _roll_axis_from_sweep(ax)
    fu1 = hydro.flux(U_bc, ax, params)
    rhs1 = fu1 - jnp.roll(fu1, 1, axis=ra)
    U1 = _update_interior(U_bc, rhs1, halo_w, dt / (2.0 * dx_o))
    U1 = positivity_fix(hydro, U1)

    fu2 = hydro.flux(U1, ax, params)
    rhs2 = fu2 - jnp.roll(fu2, 1, axis=ra)
    U2 = _update_interior(U1, rhs2, halo_w, dt / dx_o)
    U2 = positivity_fix(hydro, U2)
    return U2


def make_halos_x(tiles: Array, h: int):
    left_nb = jnp.roll(tiles, 1, axis=1)
    right_nb = jnp.roll(tiles, -1, axis=1)
    return left_nb[..., -h:], right_nb[..., :h]


def make_halos_y(tiles: Array, h: int):
    down_nb = jnp.roll(tiles, 1, axis=0)
    up_nb = jnp.roll(tiles, -1, axis=0)
    return down_nb[..., -h:, :], up_nb[..., :h, :]


def conservative_restriction_with_correction(U_coarse: Array, U_fine_restricted: Array, Bmask: Array, cfg: LevelConfig) -> Array:
    """Blend coarse and restricted fine solutions across refinement boundaries."""
    T = cfg.T
    transition_mask = create_smooth_transition_mask(Bmask, cfg, width=1)
    U_out = U_coarse.copy() if hasattr(U_coarse, "copy") else jnp.array(U_coarse)

    for i in range(cfg.Ny):
        for j in range(cfg.Nx):
            y0, x0 = i * T, j * T
            ye, xe = (i + 1) * T, (j + 1) * T
            alpha = transition_mask[i, j]
            U_out = U_out.at[:, y0:ye, x0:xe].set(
                alpha * U_fine_restricted[:, y0:ye, x0:xe]
                + (1.0 - alpha) * U_coarse[:, y0:ye, x0:xe]
            )

    return U_out


def create_smooth_transition_mask(Bmask: Array, cfg: LevelConfig, width: int = 1) -> Array:
    """Create a smoothed blending mask for conservative restriction."""
    mask = Bmask.astype(float)

    for _ in range(width):
        mask_dilated = mask.copy() if hasattr(mask, "copy") else jnp.array(mask)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                rolled = jnp.roll(jnp.roll(mask, di, axis=0), dj, axis=1)
                mask_dilated = jnp.maximum(mask_dilated, rolled * 0.5)
        mask = mask_dilated

    return mask


__all__ = [
    "check_total_energy",
    "conservative_restriction_with_correction",
    "create_smooth_transition_mask",
    "debug_amr_step",
    "diagnose_tile_continuity",
    "make_halos_x",
    "make_halos_y",
    "solve_step_freeze_ghosts",
]
