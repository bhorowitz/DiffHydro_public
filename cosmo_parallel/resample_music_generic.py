#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom


FIELDS = ("DM_dx", "DM_dy", "DM_dz", "DM_rho", "DM_vx", "DM_vy", "DM_vz")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resample a periodic MUSIC generic HDF5 cube to a new grid size.")
    p.add_argument("--input-h5", type=Path, required=True)
    p.add_argument("--output-h5", type=Path, required=True)
    p.add_argument("--output-n", type=int, required=True)
    p.add_argument("--source-level", type=int, default=None)
    return p


def _crop_core(arr: np.ndarray, n_grid: int) -> tuple[np.ndarray, int]:
    nb = (arr.shape[0] - n_grid) // 2
    if nb < 0 or n_grid + 2 * nb != arr.shape[0]:
        raise ValueError(f"Could not infer ghost width for shape={arr.shape}, n_grid={n_grid}")
    if nb == 0:
        return arr.copy(), 0
    return arr[nb : nb + n_grid, nb : nb + n_grid, nb : nb + n_grid].copy(), nb


def _resample_periodic(arr: np.ndarray, out_n: int) -> np.ndarray:
    # One-cell wrap padding makes the interpolation periodic across the box seam.
    arrp = np.pad(arr, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    scale = float(out_n) / float(arr.shape[0])
    out = zoom(arrp, zoom=(scale, scale, scale), order=1, grid_mode=True)
    return np.asarray(out[:out_n, :out_n, :out_n], dtype=np.float64)


def _pad_periodic(arr: np.ndarray, nb: int) -> np.ndarray:
    if nb <= 0:
        return arr
    return np.pad(arr, ((nb, nb), (nb, nb), (nb, nb)), mode="wrap")


def main() -> int:
    args = _parser().parse_args()
    with h5py.File(args.input_h5, "r") as fin:
        hdr = fin["header"]
        level = int(hdr.attrs["levelmax"]) if args.source_level is None else int(args.source_level)
        n_grid = int(np.asarray(hdr["grid_len_x"])[-1])
        ghost = None
        arrays: dict[str, np.ndarray] = {}
        for field in FIELDS:
            key = f"level_{level:03d}_{field}"
            if key not in fin:
                raise KeyError(f"Missing dataset {key}")
            core, nb = _crop_core(np.asarray(fin[key], dtype=np.float64), n_grid)
            arrays[field] = _resample_periodic(core, int(args.output_n))
            ghost = nb if ghost is None else ghost

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    if args.output_h5.exists():
        args.output_h5.unlink()

    out_level = level
    with h5py.File(args.output_h5, "w") as fout:
        hdr_out = fout.create_group("header")
        hdr_out.attrs["levelmin"] = np.uint32(out_level)
        hdr_out.attrs["levelmax"] = np.uint32(out_level)
        hdr_out.create_dataset("grid_len_x", data=np.asarray([args.output_n], dtype=np.int64))
        hdr_out.create_dataset("grid_len_y", data=np.asarray([args.output_n], dtype=np.int64))
        hdr_out.create_dataset("grid_len_z", data=np.asarray([args.output_n], dtype=np.int64))
        hdr_out.create_dataset("grid_off_x", data=np.asarray([0], dtype=np.int64))
        hdr_out.create_dataset("grid_off_y", data=np.asarray([0], dtype=np.int64))
        hdr_out.create_dataset("grid_off_z", data=np.asarray([0], dtype=np.int64))
        for field, core in arrays.items():
            fout.create_dataset(f"level_{out_level:03d}_{field}", data=_pad_periodic(core, int(ghost)))
    print(f"[ok] wrote resampled generic {args.output_h5}")
    print(f"     source_n={n_grid} out_n={args.output_n} level={out_level} ghost={ghost}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
