"""
halo_gas_ic.py
==============
Helpers for building gas+DM single-halo validation ICs from the
``cluster_generator`` hydrostatic-equilibrium model.
"""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np

from merger.halo_reference import HaloReference


def load_hse_profile_tables(ref: HaloReference):
    """Export the HSE model once and read the radial profile tables back."""
    fd, path = tempfile.mkstemp(suffix=".h5", prefix="diffhydro_stage2_")
    os.close(fd)
    try:
        ref.hse.write_model_to_h5(path, overwrite=True)
        with h5py.File(path, "r") as f:
            g = f["fields"]
            out = {
                "radius": np.asarray(g["radius"], dtype=np.float64),
                "gas_density": np.asarray(g["density"], dtype=np.float64),
                "pressure": np.asarray(g["pressure"], dtype=np.float64),
                "temperature": np.asarray(g["temperature"], dtype=np.float64),
                "entropy": np.asarray(g["entropy"], dtype=np.float64),
                "gravitational_field": np.asarray(g["gravitational_field"], dtype=np.float64),
                "gravitational_potential": np.asarray(g["gravitational_potential"], dtype=np.float64),
                "total_density": np.asarray(g["total_density"], dtype=np.float64),
                "total_mass": np.asarray(g["total_mass"], dtype=np.float64),
                "dark_matter_density": np.asarray(g["dark_matter_density"], dtype=np.float64),
                "dark_matter_mass": np.asarray(g["dark_matter_mass"], dtype=np.float64),
                "stellar_density": np.asarray(g["stellar_density"], dtype=np.float64),
                "stellar_mass": np.asarray(g["stellar_mass"], dtype=np.float64),
                "gas_mass": np.asarray(g["gas_mass"], dtype=np.float64),
                "gas_fraction": np.asarray(g["gas_fraction"], dtype=np.float64),
                "electron_number_density": np.asarray(g["electron_number_density"], dtype=np.float64),
            }
    finally:
        if os.path.exists(path):
            os.remove(path)
    return out


def radial_to_grid(radius, values, r3d, fill_outer=None):
    """Map a 1D radial profile onto a 3D Cartesian grid."""
    radius = np.asarray(radius, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if fill_outer is None:
        fill_outer = float(values[-1])
    return np.interp(r3d.ravel(), radius, values, left=float(values[0]), right=fill_outer).reshape(r3d.shape)


def grid_coordinates(n_grid, l_box):
    dx = float(l_box) / int(n_grid)
    x = (np.arange(n_grid, dtype=np.float64) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    center = np.array([0.5 * l_box, 0.5 * l_box, 0.5 * l_box], dtype=np.float64)
    r = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
    return X, Y, Z, r, center, dx


def shell_profile_from_grid(field, r3d, r_bins, statistic="mean", weights=None):
    """Radial shell profile from a 3D grid field."""
    rf = np.asarray(r3d, dtype=np.float64).ravel()
    ff = np.asarray(field, dtype=np.float64).ravel()
    if weights is not None:
        wf = np.asarray(weights, dtype=np.float64).ravel()
    else:
        wf = None

    if statistic == "mean":
        num, _ = np.histogram(rf, bins=r_bins, weights=ff if wf is None else ff * wf)
        den, _ = np.histogram(rf, bins=r_bins, weights=None if wf is None else wf)
        if wf is None:
            den = np.histogram(rf, bins=r_bins)[0]
        out = np.full(len(r_bins) - 1, np.nan, dtype=np.float64)
        mask = den > 0
        out[mask] = num[mask] / den[mask]
        return out

    if statistic == "sum":
        vals, _ = np.histogram(rf, bins=r_bins, weights=ff if wf is None else ff * wf)
        return vals

    raise ValueError(f"Unsupported statistic={statistic}")


def build_stage2_gas_grids(ref: HaloReference, n_grid: int, l_box: float):
    """Build gas and stellar fields on a Cartesian mesh from the HSE tables."""
    prof = load_hse_profile_tables(ref)
    X, Y, Z, r3d, center, dx = grid_coordinates(n_grid, l_box)

    rho_g = radial_to_grid(prof["radius"], prof["gas_density"], r3d)
    p_g = radial_to_grid(prof["radius"], prof["pressure"], r3d)
    t_g = radial_to_grid(prof["radius"], prof["temperature"], r3d)
    s_g = radial_to_grid(prof["radius"], prof["entropy"], r3d)
    rho_star = radial_to_grid(prof["radius"], prof["stellar_density"], r3d, fill_outer=0.0)
    rho_tot = radial_to_grid(prof["radius"], prof["total_density"], r3d, fill_outer=0.0)
    g_r = radial_to_grid(prof["radius"], prof["gravitational_field"], r3d, fill_outer=0.0)

    return {
        "profiles": prof,
        "X": X,
        "Y": Y,
        "Z": Z,
        "r3d": r3d,
        "center": center,
        "dx": dx,
        "rho_g": rho_g,
        "p_g": p_g,
        "T_g": t_g,
        "entropy": s_g,
        "rho_star": rho_star,
        "rho_total": rho_tot,
        "g_r": g_r,
    }

