from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np


def _to_analysis_field(field: jnp.ndarray, *, space: str, eps: float) -> jnp.ndarray:
    f = jnp.asarray(field, dtype=jnp.float32)
    mode = str(space).lower()
    if mode == "log":
        return jnp.log(jnp.clip(f, jnp.asarray(float(eps), dtype=f.dtype), None))
    if mode == "linear":
        return f
    raise ValueError(f"Unsupported power-spectrum field space: {space}")


def make_temperature_power_spectrum_loss(
    target_temp: np.ndarray,
    *,
    box_size_mpc_h: float,
    n_bins: int = 32,
    field_space: str = "log",
    eps: float = 1.0e-8,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], dict[str, np.ndarray | float | int | str]]:
    """
    Build a differentiable loss term matching binned log-power spectra of temperature fields.

    The returned loss is:
      mean[(log P_pred(k) - log P_target(k))^2]
    """
    targ = np.asarray(target_temp, dtype=np.float32)
    if targ.ndim != 3 or targ.shape[0] != targ.shape[1] or targ.shape[1] != targ.shape[2]:
        raise ValueError(f"Expected cubic 3D target temperature field, got shape={targ.shape}")

    n = int(targ.shape[0])
    n_bins = int(max(4, n_bins))
    lx = float(box_size_mpc_h)
    dx = lx / float(n)

    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    kmag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).astype(np.float32)

    nonzero = kmag[kmag > 0.0]
    if nonzero.size == 0:
        raise ValueError("No non-zero Fourier modes available for power-spectrum loss")
    k_min = float(nonzero.min())
    k_max = float(nonzero.max())
    k_edges = np.linspace(k_min, k_max, n_bins + 1, dtype=np.float32)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    flat_k = kmag.reshape(-1)
    bin_ids = np.digitize(flat_k, k_edges, right=False) - 1
    valid = (flat_k > 0.0) & (bin_ids >= 0) & (bin_ids < n_bins)
    flat_valid_idx = np.where(valid)[0].astype(np.int32)
    flat_bin_ids = bin_ids[valid].astype(np.int32)
    counts = np.bincount(flat_bin_ids, minlength=n_bins).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0).astype(np.float32)

    flat_valid_idx_j = jnp.asarray(flat_valid_idx, dtype=jnp.int32)
    flat_bin_ids_j = jnp.asarray(flat_bin_ids, dtype=jnp.int32)
    safe_counts_j = jnp.asarray(safe_counts, dtype=jnp.float32)
    eps_j = jnp.asarray(float(eps), dtype=jnp.float32)

    def _binned_logpk(field: jnp.ndarray) -> jnp.ndarray:
        f = _to_analysis_field(field, space=field_space, eps=float(eps))
        delta = f - jnp.mean(f)
        fhat = jnp.fft.rfftn(delta)
        power_flat = jnp.real(fhat * jnp.conj(fhat)).reshape((-1,))
        power_valid = power_flat[flat_valid_idx_j]
        p_sum = jnp.bincount(flat_bin_ids_j, weights=power_valid, length=n_bins)
        pk = p_sum / safe_counts_j
        return jnp.log(jnp.clip(pk, eps_j, None))

    target_logpk = np.asarray(_binned_logpk(jnp.asarray(targ, dtype=jnp.float32)), dtype=np.float32)
    target_logpk_j = jnp.asarray(target_logpk, dtype=jnp.float32)

    def loss_fn(pred_temp: jnp.ndarray) -> jnp.ndarray:
        pred_logpk = _binned_logpk(pred_temp)
        return jnp.mean((pred_logpk - target_logpk_j) ** 2)

    meta = {
        "enabled": True,
        "n_bins": int(n_bins),
        "field_space": str(field_space).lower(),
        "eps": float(eps),
        "k_min": float(k_min),
        "k_max": float(k_max),
        "k_centers": np.asarray(k_centers, dtype=np.float32),
        "target_logpk": np.asarray(target_logpk, dtype=np.float32),
    }
    return loss_fn, meta
