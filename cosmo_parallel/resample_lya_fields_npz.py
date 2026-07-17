#!/usr/bin/env python3
"""Fourier-resample a Lya-compare fields npz (gas_dh_final, temp_dh_final,
v*_dh_final_cms, a_final_dh) to a different cubic grid size.

Strictly-positive fields (density, temperature) are resampled in log space and
all fields get a Gaussian anti-aliasing taper at the target Nyquist scale.
Plain sharp truncation of these very skewed fields produces Gibbs ringing that
overshoots to negative values around dense structures; clipped to floors, those
cells turn into zero-thermal-broadening speckle in the Lya flux."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _k2_grid(n: int) -> np.ndarray:
    k = np.fft.fftfreq(n)  # cycles per cell in the source grid
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij", sparse=True)
    return kx**2 + ky**2 + kz**2


def fourier_resample(field: np.ndarray, n_out: int, *, taper_cells: float = 1.0) -> np.ndarray:
    """Band-limit to the target grid with a Gaussian taper of width
    `taper_cells` target cells, then sample onto n_out^3."""
    n_in = field.shape[0]
    fk = np.fft.fftn(field.astype(np.float64))
    if taper_cells > 0.0:
        # smoothing radius in source-cell units
        r = float(taper_cells) * n_in / n_out
        fk *= np.exp(-2.0 * (np.pi * r) ** 2 * _k2_grid(n_in))
    if n_in == n_out:
        return np.fft.ifftn(fk).real.astype(np.float32)
    out_k = np.zeros((n_out, n_out, n_out), dtype=complex)
    h = min(n_in, n_out) // 2
    sl_lo = slice(0, h)
    for si, so in ((sl_lo, sl_lo), (slice(n_in - h, n_in), slice(n_out - h, n_out))):
        for sj, tj in ((sl_lo, sl_lo), (slice(n_in - h, n_in), slice(n_out - h, n_out))):
            for sk, tk in ((sl_lo, sl_lo), (slice(n_in - h, n_in), slice(n_out - h, n_out))):
                out_k[so, tj, tk] = fk[si, sj, sk]
    return (np.fft.ifftn(out_k).real * (n_out**3 / n_in**3)).astype(np.float32)


def resample_positive(field: np.ndarray, n_out: int, *, floor: float, taper_cells: float) -> np.ndarray:
    logf = np.log(np.maximum(field.astype(np.float64), floor))
    out = np.exp(fourier_resample(logf, n_out, taper_cells=taper_cells))
    return np.maximum(out, floor).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-npz", type=Path, required=True)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--n-out", type=int, required=True)
    p.add_argument("--taper-cells", type=float, default=1.0,
                   help="Gaussian anti-aliasing width in target cells (0 disables).")
    args = p.parse_args()

    d = np.load(args.input_npz)
    out = {}
    for k in d.files:
        arr = np.asarray(d[k])
        if arr.ndim != 3:
            out[k] = arr
        elif k == "gas_dh_final":
            g = resample_positive(arr, args.n_out, floor=1.0e-6, taper_cells=args.taper_cells)
            out[k] = (g / max(float(np.mean(g)), 1.0e-30)).astype(np.float32)
        elif k == "temp_dh_final":
            out[k] = resample_positive(arr, args.n_out, floor=1.0, taper_cells=args.taper_cells)
        else:
            out[k] = fourier_resample(arr, args.n_out, taper_cells=args.taper_cells)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    g = out["gas_dh_final"]
    t = out["temp_dh_final"]
    print(f"[done] wrote {args.output_npz} at n={args.n_out} (taper {args.taper_cells} cells)")
    print(f"  gas_norm mean={float(np.mean(g)):.4f} std={float(np.std(g)):.4f} "
          f"frac@floor={float(np.mean(g <= 1.0e-6)):.2e}")
    print(f"  temp_k min={float(np.min(t)):.2f} frac@floor={float(np.mean(t <= 1.0)):.2e}")


if __name__ == "__main__":
    main()
