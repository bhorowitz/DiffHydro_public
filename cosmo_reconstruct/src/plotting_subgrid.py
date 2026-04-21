from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _slice2d(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field)
    if arr.ndim == 3:
        return np.asarray(arr[:, :, arr.shape[2] // 2], dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def save_summary_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_multi_panel_slices(
    out_path: Path,
    fields: list[tuple[str, np.ndarray]],
    *,
    ncols: int = 3,
    log_fields: set[str] | None = None,
) -> None:
    log_fields = set() if log_fields is None else set(log_fields)
    n = len(fields)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, field) in zip(axes, fields, strict=False):
        arr = _slice2d(field)
        if title in log_fields:
            arr = np.log10(np.clip(arr, 1.0e-8, None))
        im = ax.imshow(arr, origin="lower", cmap="magma")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n:]:
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_histograms(
    out_path: Path,
    fields: list[tuple[str, np.ndarray]],
    *,
    bins: int = 128,
    log_fields: set[str] | None = None,
) -> None:
    log_fields = set() if log_fields is None else set(log_fields)
    fig, axes = plt.subplots(1, len(fields), figsize=(5.0 * len(fields), 4.0), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, field) in zip(axes, fields, strict=True):
        vals = np.asarray(field, dtype=np.float32).ravel()
        if title in log_fields:
            vals = np.log10(np.clip(vals, 1.0e-8, None))
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=2.0)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_timeseries(
    out_path: Path,
    series: list[tuple[str, np.ndarray]],
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), constrained_layout=True)
    for label, values in series:
        arr = np.asarray(values, dtype=np.float64)
        ax.semilogy(np.arange(arr.size), arr, lw=2.0, label=label)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_residual_panel(
    out_path: Path,
    reference: np.ndarray,
    reconstructed: np.ndarray,
    *,
    title_prefix: str = "",
) -> None:
    ref = _slice2d(reference)
    rec = _slice2d(reconstructed)
    res = rec - ref
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for ax, arr, title, cmap in (
        (axes[0], ref, f"{title_prefix} reference", "magma"),
        (axes[1], rec, f"{title_prefix} reconstructed", "magma"),
        (axes[2], res, f"{title_prefix} residual", "coolwarm"),
    ):
        im = ax.imshow(arr, origin="lower", cmap=cmap)
        ax.set_title(title.strip())
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
