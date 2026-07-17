#!/usr/bin/env python3
"""Slice comparison plots (Nyx vs DiffHydro) of gas density, temperature and
LOS velocity, matching the flux-slice convention of postprocess_lya_nhi_compare.py
(yz-plane at the mid x index by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nyx-plotfile", type=Path, required=True)
    p.add_argument("--diffhydro-fields", type=Path, required=True,
                   help="fields npz with gas_dh_final, temp_dh_final, v*_dh_final_cms")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--slice-index", type=int, default=None, help="x index of the yz slice (default: mid-plane)")
    p.add_argument("--los-axis", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--nyx-velocity-divisor", type=float, default=100.0)
    p.add_argument("--vel-unit-cms", type=float, default=1.0e7)
    return p.parse_args()


def _three_panel(a, b, out_path: Path, *, title_a, title_b, cbar_label, cmap="viridis",
                 diverging_diff=True, sym=False):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    rr = bb - aa
    both = np.concatenate([aa.ravel(), bb.ravel()])
    if sym:
        v = float(np.nanpercentile(np.abs(both), 99.5))
        vmin, vmax = -v, v
    else:
        vmin = float(np.nanpercentile(both, 1.0))
        vmax = float(np.nanpercentile(both, 99.0))
    rv = max(float(np.nanpercentile(np.abs(rr), 99.0)), 1.0e-12)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    im0 = axes[0].imshow(aa, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(title_a)
    im1 = axes[1].imshow(bb, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(title_b)
    im2 = axes[2].imshow(rr, origin="lower", cmap="RdBu_r", vmin=-rv, vmax=rv)
    axes[2].set_title("DiffHydro - Nyx")
    for ax in axes:
        ax.set_xlabel("z")
        ax.set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], label=cbar_label)
    fig.colorbar(im1, ax=axes[1], label=cbar_label)
    fig.colorbar(im2, ax=axes[2], label="difference")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = yt.load(str(args.nyx_plotfile.resolve()), hint="NyxDataset")
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    rho_nyx = np.asarray(cg[("boxlib", "density")], dtype=np.float64)
    temp_nyx = np.asarray(cg[("boxlib", "Temp")], dtype=np.float64)
    mom_key = ("xmom", "ymom", "zmom")[int(args.los_axis)]
    vlos_nyx_kms = (
        np.asarray(cg[("boxlib", mom_key)], dtype=np.float64)
        / np.maximum(rho_nyx, 1.0e-30)
        / float(args.nyx_velocity_divisor)
        * float(args.vel_unit_cms)
        / 1.0e5
    )
    gas_nyx = rho_nyx / max(float(np.mean(rho_nyx)), 1.0e-30)

    d = np.load(args.diffhydro_fields)
    gas_dh = np.asarray(d["gas_dh_final"], dtype=np.float64)
    temp_dh = np.asarray(d["temp_dh_final"], dtype=np.float64)
    v_key = ("vx_dh_final_cms", "vy_dh_final_cms", "vz_dh_final_cms")[int(args.los_axis)]
    vlos_dh_kms = np.asarray(d[v_key], dtype=np.float64) / 1.0e5

    if gas_dh.shape != gas_nyx.shape:
        raise ValueError(f"Grid mismatch: Nyx {gas_nyx.shape} vs DiffHydro {gas_dh.shape}")

    ix = int(args.slice_index) if args.slice_index is not None else gas_nyx.shape[0] // 2
    sfx = f"yz slice, x={ix}"

    _three_panel(
        np.log10(np.maximum(gas_nyx[ix], 1.0e-6)),
        np.log10(np.maximum(gas_dh[ix], 1.0e-6)),
        args.output_dir / "density_slice_yz_compare.png",
        title_a=f"Nyx log10 gas overdensity ({sfx})",
        title_b=f"DiffHydro log10 gas overdensity ({sfx})",
        cbar_label="log10(rho/<rho>)",
    )
    _three_panel(
        np.log10(np.maximum(temp_nyx[ix], 1.0)),
        np.log10(np.maximum(temp_dh[ix], 1.0)),
        args.output_dir / "temperature_slice_yz_compare.png",
        title_a=f"Nyx log10 T [K] ({sfx})",
        title_b=f"DiffHydro log10 T [K] ({sfx})",
        cbar_label="log10 T [K]",
        cmap="magma",
    )
    _three_panel(
        vlos_nyx_kms[ix],
        vlos_dh_kms[ix],
        args.output_dir / "vlos_slice_yz_compare.png",
        title_a=f"Nyx LOS velocity [km/s] ({sfx})",
        title_b=f"DiffHydro LOS velocity [km/s] ({sfx})",
        cbar_label="v_los [km/s]",
        cmap="RdBu_r",
        sym=True,
    )
    for label, a, b in (
        ("gas overdensity (log10)", np.log10(np.maximum(gas_nyx, 1e-6)), np.log10(np.maximum(gas_dh, 1e-6))),
        ("temperature (log10 K)", np.log10(np.maximum(temp_nyx, 1.0)), np.log10(np.maximum(temp_dh, 1.0))),
        ("LOS velocity (km/s)", vlos_nyx_kms, vlos_dh_kms),
    ):
        aa = a.ravel() - a.mean()
        bb = b.ravel() - b.mean()
        r = float(aa @ bb / max(np.linalg.norm(aa) * np.linalg.norm(bb), 1e-30))
        print(f"  3D pearson {label}: {r:+.4f}")


if __name__ == "__main__":
    main()
