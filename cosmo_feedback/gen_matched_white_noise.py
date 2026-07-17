#!/usr/bin/env python3
"""Generate phase-matched white noise across resolutions for a true convergence test.

A fresh `randn((N,N,N))` per resolution gives INDEPENDENT realizations (same power
spectrum, different phases). To compare the same ICs at 128^3 and 256^3 the low-k
Fourier phases must be identical: we draw the high-res field once and band-limit it
to the low-res grid by a sharp Fourier-space truncation (variance-preserving), so the
128^3 field is exactly the large-scale part of the 256^3 field.

    python -m cosmo_feedback.gen_matched_white_noise --hi 256 --lo 128 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "ics"


def fourier_truncate(wn_hi: np.ndarray, m: int) -> np.ndarray:
    """Band-limit a unit-variance white-noise cube (N^3) to M^3, keeping the low-|k|
    modes (shared phases) and rescaling so the result is still unit-variance per cell."""
    n = wn_hi.shape[0]
    assert m < n and m % 2 == 0 and n % 2 == 0
    X = np.fft.fftshift(np.fft.fftn(wn_hi))          # freq 0 at center
    c = n // 2
    h = m // 2
    Xc = X[c - h:c + h, c - h:c + h, c - h:c + h]    # central M^3 (low |k|)
    wn_lo = np.fft.ifftn(np.fft.ifftshift(Xc)).real
    wn_lo *= (m / n) ** 1.5                           # preserve unit variance per cell
    return np.ascontiguousarray(wn_lo, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hi", type=int, default=256)
    p.add_argument("--lo", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    wn_hi = rng.standard_normal((args.hi, args.hi, args.hi)).astype(np.float32)
    wn_lo = fourier_truncate(wn_hi, args.lo)

    print(f"wn_hi {args.hi}^3: mean={wn_hi.mean():+.4f} var={wn_hi.var():.4f}")
    print(f"wn_lo {args.lo}^3: mean={wn_lo.mean():+.4f} var={wn_lo.var():.4f}  (target var~1)")

    hi_path = OUT / f"wn_seed{args.seed}_n{args.hi}.npy"
    lo_path = OUT / f"wn_seed{args.seed}_n{args.lo}.npy"
    np.save(hi_path, wn_hi)
    np.save(lo_path, wn_lo)
    print("wrote", hi_path)
    print("wrote", lo_path)


if __name__ == "__main__":
    main()
