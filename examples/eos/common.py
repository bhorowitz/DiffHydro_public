from __future__ import annotations

import os
from typing import Callable, Sequence

# Keep example runs lightweight by default; users can override externally.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import diffhydro as dh


def build_hydro(
    eq,
    *,
    forces=None,
    n_super_step: int = 4096,
    max_dt: float = 0.01,
    limiter: str = "VANLEER",
    pmesh_shape=None,
):
    if pmesh_shape is None:
        pmesh_shape = (jax.device_count(), 1, 1)
    solver = dh.HLLC(equation_manager=eq, signal_speed=dh.signal_speed_Rusanov)
    flux = dh.ConvectiveFlux(eq, solver, dh.MUSCL3(limiter=limiter))
    return dh.hydro(
        n_super_step=n_super_step,
        max_dt=max_dt,
        fluxes=[flux],
        forces=list(forces or []),
        use_mol=True,
        boundary=dh.NoBoundary,
        pmesh_shape=pmesh_shape,
    )


def run_timeslices(sim, U0, times: Sequence[float], params=None):
    if len(times) < 1:
        raise ValueError("times must contain at least one entry.")
    if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
        raise ValueError("times must be strictly increasing.")

    fields = np.asarray(U0)
    p = {} if params is None else dict(params)
    outputs = [fields]
    t_prev = float(times[0])

    for t_now in times[1:]:
        dt_segment = float(t_now - t_prev)
        fields, p, _, _, _ = sim.evolve_till_time(
            fields,
            p,
            t_target=dt_segment,
            max_steps=sim.n_super_step,
        )
        fields = np.asarray(jax.device_get(fields))
        outputs.append(fields)
        t_prev = t_now

    return outputs


def density_slice(U):
    arr = np.asarray(U)
    return arr[0, :, :, 0]


def temperature_slice(U, eq):
    U_jax = jnp.asarray(U)
    W = eq.get_primitives_from_conservatives(U_jax)
    rho = W[eq.mass_ids]
    p = W[eq.energy_ids]
    # Use raw p/rho temperature proxy so plotting is not masked by EOS clip floors.
    T = p / (jnp.maximum(rho, eq.eps) * getattr(eq, "R", 1.0) + eq.eps)
    return np.asarray(T[:, :, 0])


def _safe_limits(values: np.ndarray):
    finite = np.isfinite(values)
    if np.any(finite):
        vals = values[finite]
        vmin, vmax = np.nanpercentile(vals, [1.0, 99.0])
        mean = float(np.nanmean(vals))
        span = float(vmax - vmin)
        # Stabilize color scaling for nearly constant fields (e.g. isothermal T).
        if span <= 1.0e-6 * max(1.0, abs(mean)):
            delta = 1.0e-3 * max(1.0, abs(mean))
            return mean - delta, mean + delta
    else:
        vmin, vmax = -1.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -1.0, 1.0
    return float(vmin), float(vmax)


def plot_comparison_grid(
    adiabatic_states,
    isothermal_states,
    times: Sequence[float],
    *,
    field_fn: Callable[[np.ndarray], np.ndarray] = density_slice,
    title: str,
    output_path: str,
    cmap: str = "magma",
    include_temperature_rows: bool = False,
    eq_adiabatic=None,
    eq_isothermal=None,
    temperature_cmap: str = "inferno",
):
    ad_fields = [field_fn(s) for s in adiabatic_states]
    iso_fields = [field_fn(s) for s in isothermal_states]

    rows = [
        ("Adiabatic", ad_fields, cmap),
        ("Isothermal", iso_fields, cmap),
    ]
    if include_temperature_rows:
        if eq_adiabatic is None or eq_isothermal is None:
            raise ValueError("eq_adiabatic and eq_isothermal are required for temperature rows.")
        ad_temp = [temperature_slice(s, eq_adiabatic) for s in adiabatic_states]
        iso_temp = [temperature_slice(s, eq_isothermal) for s in isothermal_states]
        rows.extend([
            ("Adiabatic T", ad_temp, temperature_cmap),
            ("Isothermal T", iso_temp, temperature_cmap),
        ])

    ncols = len(times)
    nrows = len(rows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 2.8 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    for r, (label, fields_r, cmap_r) in enumerate(rows):
        values = np.concatenate([np.asarray(f).ravel() for f in fields_r])
        vmin, vmax = _safe_limits(values)
        im = None
        for j, t in enumerate(times):
            plot_arr = np.nan_to_num(np.asarray(fields_r[j]), nan=0.0, posinf=vmax, neginf=vmin)
            im = axes[r, j].imshow(plot_arr, origin="lower", cmap=cmap_r, vmin=vmin, vmax=vmax)
            axes[r, j].set_title(f"{label}  t={t:.2f}")
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
        if im is not None:
            fig.colorbar(im, ax=axes[r, :].tolist(), shrink=0.9, pad=0.01)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
