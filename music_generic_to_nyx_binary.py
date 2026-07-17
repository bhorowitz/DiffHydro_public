#!/usr/bin/env python3
"""Convert MUSIC2 generic HDF5 output to a Nyx BinaryFile particle IC.

Nyx BinaryFile layout:
  - int64: num_particles
  - int32: ndim (=3)
  - int32: nextra (=4)  # mass + 3 velocities
  - per-particle payload (float64 by default): x, y, z, m, vx, vy, vz

This converter assumes single-level (unigrid) MUSIC output and uses the same
scaling Nyx applies for MUSIC-style velocity fields:
  x = (x_cell + dx) * L[Mpc]
  v = v_field * L[Mpc/h]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import h5py
import numpy as np


RHO_CRIT_0_H2 = 2.77536627e11  # Msun / Mpc^3 for h=1


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert MUSIC generic HDF5 output to Nyx BinaryFile IC."
    )
    p.add_argument(
        "--music-h5",
        type=Path,
        required=True,
        help="Input MUSIC generic HDF5 file.",
    )
    p.add_argument(
        "--output-nyx",
        type=Path,
        required=True,
        help="Output Nyx binary particle file.",
    )
    p.add_argument(
        "--box-size-mpch",
        type=float,
        required=True,
        help="Comoving box length in Mpc/h.",
    )
    p.add_argument("--h", type=float, required=True, help="Hubble parameter h.")
    p.add_argument("--omega-m", type=float, required=True, help="Total matter density parameter.")
    p.add_argument(
        "--omega-b",
        type=float,
        default=0.0,
        help="Baryon density parameter (used only if --mass-convention=dm_only).",
    )
    p.add_argument(
        "--mass-convention",
        choices=["total_matter", "dm_only"],
        default="total_matter",
        help=(
            "Particle mass convention. For Nyx LyA with do_santa_barbara=1, "
            "total_matter matches bundled IC behavior."
        ),
    )
    p.add_argument(
        "--level",
        type=int,
        default=None,
        help="Explicit MUSIC level index. Default: header levelmax.",
    )
    p.add_argument(
        "--dtype",
        choices=["f8", "f4"],
        default="f8",
        help="Payload precision (Nyx LyA binaries in this repo use f8).",
    )
    return p


def _load_level_fields(path: Path, level: int | None) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        levelmax = int(f["header"].attrs["levelmax"])
        lev = levelmax if level is None else int(level)
        if "header/grid_len_x" not in f:
            raise KeyError("Missing /header/grid_len_x in MUSIC generic file.")
        n_grid = int(np.asarray(f["header/grid_len_x"])[-1])

        out: dict[str, np.ndarray] = {"level": np.array(lev), "n_grid": np.array(n_grid)}
        for key in ("DM_dx", "DM_dy", "DM_dz", "DM_vx", "DM_vy", "DM_vz"):
            dset = f.get(f"level_{lev:03d}_{key}")
            if dset is None:
                raise KeyError(f"Missing dataset: level_{lev:03d}_{key}")
            arr = np.asarray(dset, dtype=np.float64)
            if arr.ndim != 3:
                raise ValueError(f"{dset.name} must be 3D, got shape {arr.shape}")
            if not (arr.shape[0] == arr.shape[1] == arr.shape[2]):
                raise ValueError(f"{dset.name} must be cubic, got shape {arr.shape}")
            nb = (arr.shape[0] - n_grid) // 2
            if n_grid + 2 * nb != arr.shape[0] or nb < 0:
                raise ValueError(
                    f"Could not infer ghost width for {dset.name}: n_grid={n_grid}, shape={arr.shape}"
                )
            if nb > 0:
                arr = arr[nb : nb + n_grid, nb : nb + n_grid, nb : nb + n_grid]
            out[key] = arr
    return out


def _mass_per_particle_msun(
    n_grid: int, box_size_mpc: float, h: float, omega_m: float, omega_b: float, convention: str
) -> float:
    omega = omega_m if convention == "total_matter" else (omega_m - omega_b)
    if omega <= 0.0:
        raise ValueError(f"Non-positive omega for mass convention {convention}: {omega}")
    rho_crit = RHO_CRIT_0_H2 * (h * h)  # Msun / Mpc^3
    rho = omega * rho_crit
    cell_volume = (box_size_mpc / float(n_grid)) ** 3
    return float(rho * cell_volume)


def _build_particle_array(
    fields: dict[str, np.ndarray],
    box_size_mpch: float,
    h: float,
    omega_m: float,
    omega_b: float,
    mass_convention: str,
    out_dtype: np.dtype,
) -> np.ndarray:
    n_grid = int(fields["n_grid"])
    box_size_mpc = box_size_mpch / h

    i = np.arange(n_grid, dtype=np.float64)
    x0, y0, z0 = np.meshgrid(i, i, i, indexing="ij")
    x0 = (x0 + 0.5) / float(n_grid)
    y0 = (y0 + 0.5) / float(n_grid)
    z0 = (z0 + 0.5) / float(n_grid)

    x = np.mod(x0 + fields["DM_dx"], 1.0) * box_size_mpc
    y = np.mod(y0 + fields["DM_dy"], 1.0) * box_size_mpc
    z = np.mod(z0 + fields["DM_dz"], 1.0) * box_size_mpc

    # For MUSIC-style velocity fields: v[km/s] = v_field * L[Mpc/h].
    vx = fields["DM_vx"] * box_size_mpch
    vy = fields["DM_vy"] * box_size_mpch
    vz = fields["DM_vz"] * box_size_mpch

    m = _mass_per_particle_msun(
        n_grid=n_grid,
        box_size_mpc=box_size_mpc,
        h=h,
        omega_m=omega_m,
        omega_b=omega_b,
        convention=mass_convention,
    )
    mass = np.full_like(x, m, dtype=np.float64)

    data = np.stack(
        [
            x.ravel(order="C"),
            y.ravel(order="C"),
            z.ravel(order="C"),
            mass.ravel(order="C"),
            vx.ravel(order="C"),
            vy.ravel(order="C"),
            vz.ravel(order="C"),
        ],
        axis=1,
    ).astype(out_dtype, copy=False)
    return data


def _write_nyx_binary(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    npart = int(data.shape[0])
    with path.open("wb") as f:
        f.write(struct.pack("q", npart))
        f.write(struct.pack("i", 3))
        f.write(struct.pack("i", 4))
        data.tofile(f)


def main() -> int:
    args = _parser().parse_args()
    music_h5 = args.music_h5.resolve()
    out_path = args.output_nyx.resolve()
    if not music_h5.exists():
        raise FileNotFoundError(music_h5)

    out_dtype = np.float64 if args.dtype == "f8" else np.float32
    fields = _load_level_fields(music_h5, args.level)
    data = _build_particle_array(
        fields=fields,
        box_size_mpch=float(args.box_size_mpch),
        h=float(args.h),
        omega_m=float(args.omega_m),
        omega_b=float(args.omega_b),
        mass_convention=str(args.mass_convention),
        out_dtype=out_dtype,
    )
    _write_nyx_binary(out_path, data)

    names = ("x", "y", "z", "m", "vx", "vy", "vz")
    print(f"[ok] wrote {out_path}")
    print(
        f"      particles={data.shape[0]} level={int(fields['level'])} n_grid={int(fields['n_grid'])} "
        f"dtype={data.dtype}"
    )
    for i, name in enumerate(names):
        v = data[:, i]
        print(f"      {name:>2}: min={v.min():.6e} max={v.max():.6e} mean={v.mean():.6e} std={v.std():.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

