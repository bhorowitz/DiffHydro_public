from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from jax import checkpoint, lax, nn
import jax.numpy as jnp
import numpy as np

from diffhydro.cosmology.nyx_cooling_table import NyxCoolingRateInterpolator, NyxCoolingTableData, rho_b_mean_cgs_jax
from diffhydro.postprocess.lya import (
    CosmologyParams,
    build_nyx_eos_interpolator,
    compute_nhi_number_density,
    flux_from_tau,
    tau_cube_from_nhi,
)


def _normalize_mean(field: jnp.ndarray, eps: float = 1.0e-8) -> jnp.ndarray:
    f = jnp.asarray(field, dtype=jnp.float32)
    return f / (jnp.mean(f) + jnp.asarray(float(eps), dtype=f.dtype))


def _smooth_positive(x: jnp.ndarray, scale: float = 1.0e-35) -> jnp.ndarray:
    # Stable soft-ReLU; avoids NaNs that can appear with sqrt-based smooth max
    # at exactly x=0 in float32.
    xx = jnp.asarray(x, dtype=jnp.float32)
    ss = jnp.asarray(float(scale), dtype=xx.dtype)
    return ss * nn.softplus(xx / ss)


def _smooth_abs(x: jnp.ndarray, eps: float = 1.0e-30) -> jnp.ndarray:
    xx = jnp.asarray(x, dtype=jnp.float32)
    ee = jnp.asarray(float(eps), dtype=xx.dtype)
    return jnp.abs(xx) + ee


@contextmanager
def stage_treecool(treecool_file: Path):
    treecool_file = Path(treecool_file).resolve()
    if not treecool_file.exists():
        raise FileNotFoundError(f"TREECOOL file not found: {treecool_file}")

    link = Path("TREECOOL_middle")
    created = False
    copied = False
    if not link.exists():
        try:
            link.symlink_to(treecool_file)
            created = True
        except Exception:
            import shutil

            shutil.copy2(treecool_file, link)
            copied = True

    try:
        yield
    finally:
        if created or copied:
            try:
                link.unlink()
            except Exception:
                pass


def project_observable(field: jnp.ndarray, projection: str, los_axis: int) -> jnp.ndarray:
    f = jnp.asarray(field, dtype=jnp.float32)
    axis = int(los_axis)
    mode = str(projection).lower()
    if mode == "3d":
        return f
    if mode == "mean_los":
        return jnp.mean(f, axis=axis)
    if mode == "sum_los":
        return jnp.sum(f, axis=axis)
    raise ValueError(f"Unknown projection mode: {projection}")


def make_observable_mapper(
    *,
    observable: str,
    projection: str,
    los_axis: int,
    z_target: float,
    h: float,
    omega_m: float,
    omega_b: float,
    box_size_mpc_h: float,
    lya_num_integ_pixels: int,
    lya_delta_floor: float,
    lya_temp_floor_k: float,
    lya_eos_grid_size: int,
    lya_treecool_file: str,
    lya_logdelta_min: float,
    lya_logdelta_max: float,
    lya_logt_min: float,
    lya_logt_max: float,
    lya_eos_path: str,
    xray_cooling_table_npz: str,
    xray_temp_floor_k: float,
    xray_proxy_kind: str,
    lya_skewer_batch_size: int = 256,
    fgpa_tau_a: float = 1.0,
    fgpa_tau_b: float = 1.6,
    fgpa_rho_floor: float = 1.0e-6,
):
    obs_name = str(observable).lower()
    proj_name = str(projection).lower()
    axis = int(los_axis)

    if obs_name == "density":

        def mapper(rho_norm, temp_k, v_los_cms=None):
            del temp_k, v_los_cms
            rho = _normalize_mean(jnp.asarray(rho_norm, dtype=jnp.float32))
            return project_observable(rho, proj_name, axis)

        meta = {
            "observable": "density",
            "projection": proj_name,
            "los_axis": axis,
        }
        return mapper, meta

    if obs_name == "fgpa_flux_dm":
        tau_a = jnp.asarray(max(float(fgpa_tau_a), 1.0e-12), dtype=jnp.float32)
        tau_b = jnp.asarray(float(fgpa_tau_b), dtype=jnp.float32)
        rho_floor = float(fgpa_rho_floor)

        def mapper(rho_norm, temp_k, v_los_cms=None):
            del temp_k, v_los_cms
            rho = jnp.maximum(_normalize_mean(jnp.asarray(rho_norm, dtype=jnp.float32)), rho_floor)
            tau = tau_a * jnp.power(rho, tau_b)
            flux = jnp.exp(-tau)
            return project_observable(jnp.asarray(flux, dtype=jnp.float32), proj_name, axis)

        meta = {
            "observable": "fgpa_flux_dm",
            "projection": proj_name,
            "los_axis": axis,
            "fgpa_tau_a": float(tau_a),
            "fgpa_tau_b": float(tau_b),
            "fgpa_rho_floor": rho_floor,
        }
        return mapper, meta

    if obs_name == "lya_flux":
        cosmo = CosmologyParams(
            h=float(h),
            omega_m=float(omega_m),
            omega_b=float(omega_b),
            omega_l=None,
        )
        boxsize_comoving_mpc = float(box_size_mpc_h) / max(float(h), 1.0e-12)

        with stage_treecool(Path(lya_treecool_file)):
            eos_interp = build_nyx_eos_interpolator(
                z=float(z_target),
                cosmo=cosmo,
                grid_size=int(max(8, lya_eos_grid_size)),
                logdelta_limits=(float(lya_logdelta_min), float(lya_logdelta_max)),
                logt_limits=(float(lya_logt_min), float(lya_logt_max)),
                eos_search_paths=[Path(lya_eos_path)],
            )

        def _lya_forward(delta, temp, v_los):
            n_hi = compute_nhi_number_density(delta, temp, eos_interp)
            tau = tau_cube_from_nhi(
                n_hi,
                temp,
                v_los,
                z=float(z_target),
                cosmo=cosmo,
                boxsize_comoving_mpc=boxsize_comoving_mpc,
                axis_los=axis,
                num_integ_pixels=int(max(1, lya_num_integ_pixels)),
                skewer_batch_size=int(max(1, lya_skewer_batch_size)),
            )
            tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float32), 0.0)
            flux = flux_from_tau(tau)
            return project_observable(jnp.asarray(flux, dtype=jnp.float32), proj_name, axis)

        # Rematerialize the expensive LyA mapping during backward to lower peak memory.
        _lya_forward = checkpoint(_lya_forward)

        def mapper(rho_norm, temp_k, v_los_cms=None):
            delta = jnp.maximum(_normalize_mean(jnp.asarray(rho_norm, dtype=jnp.float32)), float(lya_delta_floor))
            temp = jnp.maximum(jnp.asarray(temp_k, dtype=jnp.float32), float(lya_temp_floor_k))
            if v_los_cms is None:
                v_los = jnp.zeros_like(delta)
            else:
                v_los = jnp.asarray(v_los_cms, dtype=jnp.float32)
            return _lya_forward(delta, temp, v_los)

        meta = {
            "observable": "lya_flux",
            "projection": proj_name,
            "los_axis": axis,
            "z_target": float(z_target),
            "boxsize_comoving_mpc": boxsize_comoving_mpc,
            "lya_num_integ_pixels": int(max(1, lya_num_integ_pixels)),
            "lya_delta_floor": float(lya_delta_floor),
            "lya_temp_floor_k": float(lya_temp_floor_k),
            "lya_eos_grid_size": int(max(8, lya_eos_grid_size)),
            "lya_logdelta_min": float(lya_logdelta_min),
            "lya_logdelta_max": float(lya_logdelta_max),
            "lya_logt_min": float(lya_logt_min),
            "lya_logt_max": float(lya_logt_max),
            "lya_skewer_batch_size": int(max(1, lya_skewer_batch_size)),
            "lya_treecool_file": str(Path(lya_treecool_file).resolve()),
            "lya_eos_path": str(Path(lya_eos_path).resolve()),
        }
        return mapper, meta

    if obs_name == "xray_proxy":
        table_path = Path(xray_cooling_table_npz).resolve()
        if not table_path.exists():
            raise FileNotFoundError(f"Nyx cooling table not found: {table_path}")
        table = NyxCoolingTableData.load(table_path)
        interp = NyxCoolingRateInterpolator(table)
        z_arr = jnp.asarray(float(z_target), dtype=jnp.float32)
        rho_mean = rho_b_mean_cgs_jax(z_arr, h=float(h), omega_b=float(omega_b))
        proxy_kind = str(xray_proxy_kind).lower()

        def mapper(rho_norm, temp_k, v_los_cms=None):
            del v_los_cms
            delta = jnp.maximum(_normalize_mean(jnp.asarray(rho_norm, dtype=jnp.float32)), 1.0e-8)
            temp = jnp.maximum(jnp.asarray(temp_k, dtype=jnp.float32), float(xray_temp_floor_k))
            rho_cgs = delta * jnp.asarray(rho_mean, dtype=jnp.float32)
            heat_cgs, cool_cgs, net_cgs = interp.evaluate(rho_cgs, temp, z_arr)
            if proxy_kind == "cool":
                proxy3d = _smooth_positive(cool_cgs)
            elif proxy_kind == "net_cooling":
                proxy3d = _smooth_positive(-net_cgs)
            elif proxy_kind == "abs_net":
                proxy3d = _smooth_abs(net_cgs)
            else:
                raise ValueError(f"Unknown xray_proxy_kind={xray_proxy_kind}")
            proxy3d = jnp.nan_to_num(proxy3d, nan=0.0, posinf=0.0, neginf=0.0)
            # Normalize with a detached scale to avoid unstable gradients when
            # cooling rates are extremely small.
            scale = lax.stop_gradient(jnp.maximum(jnp.mean(proxy3d), 1.0e-26))
            proxy3d = proxy3d / scale
            proxy3d = proxy3d + 1.0e-20
            return project_observable(jnp.asarray(proxy3d, dtype=jnp.float32), proj_name, axis)

        meta = {
            "observable": "xray_proxy",
            "projection": proj_name,
            "los_axis": axis,
            "z_target": float(z_target),
            "xray_proxy_kind": proxy_kind,
            "xray_temp_floor_k": float(xray_temp_floor_k),
            "xray_cooling_table_npz": str(table_path),
        }
        return mapper, meta

    raise ValueError(f"Unknown observable: {observable}")
