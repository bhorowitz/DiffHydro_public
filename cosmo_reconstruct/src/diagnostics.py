from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jaxpm.utils import cross_correlation_coefficients, power_spectrum


def downsample_mean(field: np.ndarray, out_n: int) -> np.ndarray:
    n = int(field.shape[0])
    if field.shape != (n, n, n):
        raise ValueError(f"Expected cubic field, got shape={field.shape}")
    if out_n == n:
        return field.astype(np.float32, copy=False)
    if n % out_n != 0:
        raise ValueError(f"Cannot downsample {n} -> {out_n} with integer block averaging")

    f = n // out_n
    reshaped = field.reshape(out_n, f, out_n, f, out_n, f)
    out = reshaped.mean(axis=(1, 3, 5))
    return out.astype(np.float32)


def _gaussian_smooth_fft_np(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0.0:
        return field.astype(np.float32, copy=False)

    nx, ny, nz = field.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    kernel = np.exp(-0.5 * sigma_cells**2 * k2)

    field_hat = np.fft.rfftn(field)
    smooth = np.fft.irfftn(field_hat * kernel, s=field.shape)
    return smooth.astype(np.float32)


def load_cv0_fields(
    mgas_path: str,
    temp_path: str,
    field_index: int = 0,
    mesh_n: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mgas = np.asarray(np.load(mgas_path, mmap_mode="r")[field_index], dtype=np.float32)
    temp = np.asarray(np.load(temp_path, mmap_mode="r")[field_index], dtype=np.float32)

    mgas_norm = mgas / (float(mgas.mean()) + 1.0e-8)
    if mesh_n is not None and mgas_norm.shape[0] != mesh_n:
        mgas_norm = downsample_mean(mgas_norm, mesh_n)
        temp = downsample_mean(temp, mesh_n)

    return mgas_norm.astype(np.float32), temp.astype(np.float32)


def load_reference_fields(reference_npz: str, mesh_n: int | None = None) -> dict[str, np.ndarray]:
    ref = np.load(reference_npz)
    out: dict[str, np.ndarray] = {}

    for k in ref.files:
        arr = np.asarray(ref[k], dtype=np.float32)
        if mesh_n is not None and arr.ndim == 3 and arr.shape[0] != mesh_n:
            arr = downsample_mean(arr, mesh_n)
        out[k] = arr

    return out


def fit_gas_mapping_from_reference(
    dm_field: np.ndarray,
    gas_field: np.ndarray,
    sigma_grid: np.ndarray | None = None,
) -> dict[str, float]:
    if sigma_grid is None:
        sigma_grid = np.linspace(0.0, 3.0, 13, dtype=np.float32)

    log_dm = np.log(np.clip(dm_field, 1.0e-6, None)).astype(np.float32)
    log_g = np.log(np.clip(gas_field, 1.0e-6, None)).astype(np.float32)
    y = log_g - float(log_g.mean())

    best: dict[str, float] | None = None
    for sigma in sigma_grid:
        x = _gaussian_smooth_fft_np(log_dm, float(sigma))
        x = x - float(x.mean())
        q = x * x - float(np.mean(x * x))

        a = np.stack([x.ravel(), q.ravel()], axis=1).astype(np.float64)
        coef = np.linalg.lstsq(a, y.ravel().astype(np.float64), rcond=None)[0]

        pred = (coef[0] * x + coef[1] * q).astype(np.float32)
        mse = float(np.mean((pred - y) ** 2))

        rec = {
            "smooth_sigma_cells": float(sigma),
            "bias_linear": float(coef[0]),
            "bias_quadratic": float(coef[1]),
            "fit_mse": mse,
        }
        if best is None or rec["fit_mse"] < best["fit_mse"]:
            best = rec

    if best is None:
        raise RuntimeError("Gas mapping fit failed")
    return best


def fit_temperature_mapping_from_reference(
    gas_field: np.ndarray,
    temp_field: np.ndarray,
    temp_init_field: np.ndarray,
) -> dict[str, float]:
    x = np.log(np.clip(gas_field, 1.0e-6, None)).astype(np.float32)
    x = x - float(x.mean())
    q = x * x - float(np.mean(x * x))
    y = np.log(np.clip(temp_field, 1.0e-3, None)).astype(np.float32)

    a = np.stack([np.ones(x.size, dtype=np.float64), x.ravel().astype(np.float64), q.ravel().astype(np.float64)], axis=1)
    coef = np.linalg.lstsq(a, y.ravel().astype(np.float64), rcond=None)[0]

    temp_init_mean = float(np.mean(temp_init_field))
    temp_final_mean = float(np.mean(temp_field))
    heat_gain = float(temp_final_mean / (temp_init_mean + 1.0e-12))

    return {
        "temp_init_k": temp_init_mean,
        "temp_heat_gain": heat_gain,
        "temp_slope": float(coef[1]),
        "temp_quadratic": float(coef[2]),
    }


def field_stats(field: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(field)),
        "std": float(np.std(field)),
        "min": float(np.min(field)),
        "max": float(np.max(field)),
        "p01": float(np.percentile(field, 1.0)),
        "p05": float(np.percentile(field, 5.0)),
        "p50": float(np.percentile(field, 50.0)),
        "p95": float(np.percentile(field, 95.0)),
        "p99": float(np.percentile(field, 99.0)),
    }


def compute_power_and_cross(
    field_ref: np.ndarray,
    field_pred: np.ndarray,
    box_size_mpc_h: float,
) -> dict[str, np.ndarray | float]:
    box = np.array([box_size_mpc_h, box_size_mpc_h, box_size_mpc_h], dtype=np.float64)
    kmin = np.pi / box_size_mpc_h
    dk = 2.0 * np.pi / box_size_mpc_h

    k_ref, pk_ref = power_spectrum(field_ref, boxsize=box, kmin=kmin, dk=dk)
    k_pred, pk_pred = power_spectrum(field_pred, boxsize=box, kmin=kmin, dk=dk)
    k_cross, cross = cross_correlation_coefficients(field_ref, field_pred, boxsize=box, kmin=kmin, dk=dk)
    cross = cross / np.sqrt(np.asarray(pk_ref) * np.asarray(pk_pred) + 1.0e-12)

    return {
        "k_ref": np.asarray(k_ref),
        "pk_ref": np.asarray(pk_ref),
        "k_pred": np.asarray(k_pred),
        "pk_pred": np.asarray(pk_pred),
        "k_cross": np.asarray(k_cross),
        "cross": np.asarray(cross),
        "median_cross": float(np.median(np.asarray(cross))),
    }


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _plot_slice_triplet(
    out_path: Path,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    titles: tuple[str, str, str],
    cmap: str = "magma",
    log10: bool = False,
) -> None:
    z = a.shape[2] // 2
    sa = np.asarray(a[:, :, z])
    sb = np.asarray(b[:, :, z])
    sc = np.asarray(c[:, :, z])

    if log10:
        sa = np.log10(np.clip(sa, 1.0e-6, None))
        sb = np.log10(np.clip(sb, 1.0e-6, None))
        sc = np.log10(np.clip(sc, 1.0e-6, None))

    vmin = float(min(sa.min(), sb.min()))
    vmax = float(max(sa.max(), sb.max()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    im0 = axes[0].imshow(sa, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(titles[0])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sb, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(sc, origin="lower", cmap="coolwarm")
    axes[2].set_title(titles[2])
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_projection_triplet(
    out_path: Path,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    titles: tuple[str, str, str],
) -> None:
    pa = np.asarray(a.sum(axis=2))
    pb = np.asarray(b.sum(axis=2))
    pc = np.asarray(c.sum(axis=2))

    vmin = float(min(pa.min(), pb.min()))
    vmax = float(max(pa.max(), pb.max()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    im0 = axes[0].imshow(pa, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title(titles[0])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pb, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pc, origin="lower", cmap="coolwarm")
    axes[2].set_title(titles[2])
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_hist_density(out_path: Path, fields: list[np.ndarray], labels: list[str], title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    bins = np.linspace(-3.0, 6.0, 240)
    for field, label in zip(fields, labels, strict=True):
        vals = np.log10(np.clip(field.ravel(), 1.0e-6, None))
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=2.0, label=label)
    ax.set_xlabel("log10(field)")
    ax.set_ylabel("PDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_hist_temperature(out_path: Path, fields: list[np.ndarray], labels: list[str]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    bins = np.linspace(3.0, 7.3, 220)
    for field, label in zip(fields, labels, strict=True):
        vals = np.log10(np.clip(field.ravel(), 1.0e-3, None))
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=2.0, label=label)
    ax.set_xlabel("log10(T/K)")
    ax.set_ylabel("PDF")
    ax.set_title("Temperature distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_scatter(out_path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = x.size
    m = min(n, 200_000)
    idx = rng.choice(n, size=m, replace=False)

    xv = np.log10(np.clip(x.ravel()[idx], 1.0e-6, None))
    yv = np.log10(np.clip(y.ravel()[idx], 1.0e-6, None))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), constrained_layout=True)
    ax.scatter(xv, yv, s=1.0, alpha=0.08, rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_power(out_path: Path, spectra: dict[str, np.ndarray | float], title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].loglog(spectra["k_ref"], spectra["pk_ref"], "k--", lw=2.0, label="reference")
    axes[0].loglog(spectra["k_pred"], spectra["pk_pred"], "r", lw=2.0, label="model")
    axes[0].set_xlabel("k [h/Mpc]")
    axes[0].set_ylabel("P(k)")
    axes[0].set_title(title)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].semilogx(spectra["k_cross"], spectra["cross"], "b", lw=2.0)
    axes[1].axhline(1.0, color="k", ls="--", lw=1.0)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("k [h/Mpc]")
    axes[1].set_ylabel("Cross-correlation")
    axes[1].set_title(f"Median r(k)={float(spectra['median_cross']):.3f}")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_forward_plots(
    out_dir: Path,
    target_rho_cv0: np.ndarray,
    target_temp_cv0: np.ndarray,
    ref_gas: np.ndarray,
    ref_temp: np.ndarray,
    pred_rho: np.ndarray,
    pred_temp: np.ndarray,
    pred_dm: np.ndarray,
    spectra_cv0: dict[str, np.ndarray | float],
    spectra_ref: dict[str, np.ndarray | float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    res_cv0 = pred_rho - target_rho_cv0
    res_ref = pred_rho - ref_gas

    _plot_slice_triplet(
        out_dir / "density_slice_cv0.png",
        target_rho_cv0,
        pred_rho,
        res_cv0,
        ("CV0 gas density slice", "Model gas density slice", "Residual (model - CV0)"),
        log10=True,
    )
    _plot_slice_triplet(
        out_dir / "density_slice_ref.png",
        ref_gas,
        pred_rho,
        res_ref,
        ("Reference gas density slice", "Model gas density slice", "Residual (model - ref)"),
        log10=True,
    )
    _plot_projection_triplet(
        out_dir / "density_projection_cv0.png",
        target_rho_cv0,
        pred_rho,
        res_cv0,
        ("CV0 z-projection", "Model z-projection", "Residual projection"),
    )
    _plot_projection_triplet(
        out_dir / "density_projection_ref.png",
        ref_gas,
        pred_rho,
        res_ref,
        ("Reference z-projection", "Model z-projection", "Residual projection"),
    )

    _plot_hist_density(
        out_dir / "density_logpdf.png",
        [target_rho_cv0, ref_gas, pred_rho, pred_dm],
        ["CV0 gas", "Ref gas", "Model gas", "Model DM"],
        "Density log-PDF",
    )
    _plot_hist_temperature(
        out_dir / "temperature_logpdf.png",
        [target_temp_cv0, ref_temp, pred_temp],
        ["CV0 T", "Ref T", "Model T"],
    )

    _plot_scatter(
        out_dir / "dm_vs_gas_model_scatter.png",
        pred_dm,
        pred_rho,
        "log10(model DM density)",
        "log10(model gas density)",
        seed=7,
    )
    _plot_scatter(
        out_dir / "gas_vs_temp_model_scatter.png",
        pred_rho,
        pred_temp,
        "log10(model gas density)",
        "log10(model temperature)",
        seed=11,
    )

    _plot_power(out_dir / "pk_cross_cv0.png", spectra_cv0, "Gas density: CV0 vs model")
    _plot_power(out_dir / "pk_cross_reference.png", spectra_ref, "Gas density: reference vs model")


def make_observable_plots(
    out_dir: Path,
    *,
    target_obs: np.ndarray,
    pred_obs: np.ndarray,
    observable_name: str,
    compare_space: str,
    box_size_mpc_h: float | None = None,
) -> dict[str, np.ndarray | float] | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    targ = np.asarray(target_obs, dtype=np.float32)
    pred = np.asarray(pred_obs, dtype=np.float32)
    if targ.shape != pred.shape:
        raise ValueError(f"Observable shape mismatch: target={targ.shape} pred={pred.shape}")

    resid = pred - targ
    obs_label = str(observable_name)

    if targ.ndim in (2, 3):
        if targ.ndim == 3:
            iz = targ.shape[2] // 2
            ta = np.asarray(targ[:, :, iz], dtype=np.float32)
            pa = np.asarray(pred[:, :, iz], dtype=np.float32)
            ra = np.asarray(resid[:, :, iz], dtype=np.float32)
            slice_tag = f"(z-slice={iz})"
        else:
            ta = np.asarray(targ, dtype=np.float32)
            pa = np.asarray(pred, dtype=np.float32)
            ra = np.asarray(resid, dtype=np.float32)
            slice_tag = "(map)"

        vmin = float(min(np.nanmin(ta), np.nanmin(pa)))
        vmax = float(max(np.nanmax(ta), np.nanmax(pa)))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        im0 = axes[0].imshow(ta, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Target {obs_label} {slice_tag}")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(pa, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Model {obs_label} {slice_tag}")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(ra, origin="lower", cmap="coolwarm")
        axes[2].set_title("Residual (model-target)")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        fig.savefig(out_dir / "observable_map_compare.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    t_flat = targ.ravel().astype(np.float64)
    p_flat = pred.ravel().astype(np.float64)
    if str(compare_space).lower() == "log":
        t_plot = np.log10(np.clip(t_flat, 1.0e-20, None))
        p_plot = np.log10(np.clip(p_flat, 1.0e-20, None))
        xlabel = f"log10({obs_label})"
    else:
        t_plot = t_flat
        p_plot = p_flat
        xlabel = str(obs_label)
    all_vals = np.concatenate([t_plot, p_plot])
    v0 = float(np.nanpercentile(all_vals, 0.5))
    v1 = float(np.nanpercentile(all_vals, 99.5))
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
        v0, v1 = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
        v0, v1 = -1.0, 1.0
    bins = np.linspace(v0, v1, 220)
    ax.hist(t_plot, bins=bins, density=True, histtype="step", lw=2.0, label="target")
    ax.hist(p_plot, bins=bins, density=True, histtype="step", lw=2.0, label="model")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PDF")
    ax.set_title(f"{obs_label} distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "observable_hist.png", dpi=180)
    plt.close(fig)

    spectra_obs = None
    if (
        box_size_mpc_h is not None
        and targ.ndim == 3
        and targ.shape[0] == targ.shape[1] == targ.shape[2]
    ):
        spectra_obs = compute_power_and_cross(targ, pred, float(box_size_mpc_h))
        _plot_power(
            out_dir / "observable_pk_cross.png",
            spectra_obs,
            f"{obs_label}: target vs model",
        )

    return spectra_obs


def plot_optimization_history(out_path: Path, history: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].plot(history[:, 0], history[:, 1], lw=2.0)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("total loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history[:, 0], history[:, 2], lw=2.0, color="tab:green")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("data term")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history[:, 0], history[:, 3], lw=2.0, color="tab:orange", label="prior")
    axes[2].plot(history[:, 0], history[:, 4], lw=2.0, color="tab:red", label="|grad|")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("prior / grad")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
