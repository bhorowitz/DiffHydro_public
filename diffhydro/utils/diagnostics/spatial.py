"""Spatial diagnostics on simulation fields."""

from __future__ import annotations

import numpy as np


def isotropy_score(field, nbins: int = 36) -> float:
    """Measure angular isotropy of a 2-D field."""

    nx, ny = field.shape
    x = (np.arange(nx) + 0.5) - 0.5 * nx
    y = (np.arange(ny) + 0.5) - 0.5 * ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    theta = (np.arctan2(Y, X) + 2 * np.pi) % (2 * np.pi)

    field_np = np.asarray(field)
    base = np.median(field_np)
    mask = field_np >= base + 0.1 * (field_np.max() - base)
    if not np.any(mask):
        return float("inf")

    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    radii = []
    for i in range(nbins):
        sel = (theta >= edges[i]) & (theta < edges[i + 1]) & mask
        if np.any(sel):
            radii.append(np.percentile(r[sel], 95))
    if len(radii) < 3:
        return float("inf")
    radii = np.asarray(radii)
    return float(np.std(radii) / (np.mean(radii) + 1e-12))


def blob_radius(field, background: float) -> float:
    """Estimate radius containing hottest material of a 3-D field."""

    nx, ny, nz = field.shape
    x = (np.arange(nx) + 0.5) - 0.5 * nx
    y = (np.arange(ny) + 0.5) - 0.5 * ny
    z = (np.arange(nz) + 0.5) - 0.5 * nz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)

    threshold = background + 0.3 * (field.max() - background)
    mask = field >= threshold
    if not np.any(mask):
        return 0.0
    return float(np.percentile(np.asarray(r)[mask], 95))


__all__ = ["blob_radius", "isotropy_score"]

