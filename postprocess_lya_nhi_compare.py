#!/usr/bin/env python3
"""Compare Nyx vs DiffHydro LyA post-processing outputs at matched redshift."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yt

import diffhydro as dh
from diffhydro.cosmology import SuperComovingEquationManager
from diffhydro.cosmology import conversions as cosmo_conv
from diffhydro.postprocess.lya import (
    CosmologyParams,
    build_nyx_eos_interpolator,
    column_density_map_nhi,
    compute_nhi_number_density,
    flux_from_tau,
    lya_tau_prefactor,
    mean_map_along_los,
    tau_cube_from_nhi,
    velocity_domain_cmps,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nyx-plotfile",
        type=Path,
        default=Path("Nyx/Exec/LyA/music25_runs/n64_z2/plt00354"),
        help="Nyx plotfile for target redshift.",
    )
    p.add_argument(
        "--diffhydro-fields",
        type=Path,
        default=Path("diffhydro_music25/n64_z2/snapshots/fields_ic_final.npz"),
        help="DiffHydro snapshot fields NPZ from diffhydro_jaxpm_nyx.py.",
    )
    p.add_argument(
        "--diffhydro-state",
        type=Path,
        default=Path("diffhydro_music25/n64_z2/snapshots/final_state_dh.npz.npy"),
        help=(
            "Final DiffHydro conservative state array. Required for RSD unless "
            "--diffhydro-fields already provides velocity fields."
        ),
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=Path("diffhydro_music25/n64_z2/metrics.npz"),
        help=(
            "Metrics NPZ containing final scale factor. Required for RSD when "
            "loading velocity from --diffhydro-state."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diffhydro_music25/n64_z2/lya_postprocess"),
        help="Output directory for plots and summary metrics.",
    )
    p.add_argument("--los-axis", type=int, default=2, choices=[0, 1, 2], help="LOS axis index.")
    p.add_argument("--h", type=float, default=0.71, help="Cosmological h.")
    p.add_argument("--omega-m", type=float, default=0.27, help="Cosmological Omega_m.")
    p.add_argument("--omega-b", type=float, default=0.045, help="Cosmological Omega_b.")
    p.add_argument("--omega-l", type=float, default=None, help="Cosmological Omega_lambda (optional).")
    p.add_argument(
        "--boxsize-comoving-mpc",
        type=float,
        default=None,
        help="Comoving box size in Mpc. Default: infer from Nyx plotfile domain width.",
    )
    p.add_argument(
        "--eos-grid-size",
        type=int,
        default=192,
        help="Grid size for Nyx EOS interpolation table.",
    )
    p.add_argument(
        "--num-integ-pixels",
        type=int,
        default=20,
        help="Tau-kernel integration half-width in pixels.",
    )
    p.add_argument(
        "--nyx-velocity-divisor",
        type=float,
        default=100.0,
        help="Nyx conversion used in diffhydro_jaxpm_nyx.py (v_code = (xmom/rho)/divisor).",
    )
    p.add_argument(
        "--vel-unit-cms",
        type=float,
        default=1.0e7,
        help="Velocity unit used by diffhydro_jaxpm_nyx.py.",
    )
    p.add_argument(
        "--temp-floor-k",
        type=float,
        default=1.0,
        help="Floor for temperature before NHI interpolation.",
    )
    p.add_argument(
        "--delta-floor",
        type=float,
        default=1.0e-6,
        help="Floor for baryon overdensity before NHI interpolation.",
    )
    p.add_argument(
        "--treecool-file",
        type=Path,
        default=Path("Nyx/Exec/LyA/music25_runs/n64_z2/TREECOOL_middle"),
        help="Path to TREECOOL file used by eos_t backend.",
    )
    p.add_argument(
        "--realspace-only",
        action="store_true",
        default=False,
        help="Skip velocity/RSD products and only compute real-space LyA observables.",
    )
    return p.parse_args()


def _read_comoving_a(plotfile: Path) -> float:
    path = plotfile / "comoving_a"
    if not path.exists():
        raise FileNotFoundError(f"Missing comoving_a in {plotfile}")
    return float(path.read_text().strip())


@contextmanager
def _stage_treecool(treecool_file: Path):
    treecool_file = treecool_file.resolve()
    if not treecool_file.exists():
        raise FileNotFoundError(f"TREECOOL file not found: {treecool_file}")

    link_name = Path("TREECOOL_middle")
    created = False
    copied = False
    if not link_name.exists():
        try:
            link_name.symlink_to(treecool_file)
            created = True
        except Exception:
            shutil.copy2(treecool_file, link_name)
            copied = True
    try:
        yield
    finally:
        if created or copied:
            try:
                link_name.unlink()
            except Exception:
                pass


def _load_nyx_plotfile(plotfile: Path):
    if not (plotfile / "Header").exists():
        raise FileNotFoundError(f"Missing Nyx plotfile header: {plotfile / 'Header'}")
    ds = yt.load(str(plotfile), hint="NyxDataset")
    cg = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    density = np.asarray(cg[("boxlib", "density")], dtype=np.float32)
    xmom = np.asarray(cg[("boxlib", "xmom")], dtype=np.float32)
    ymom = np.asarray(cg[("boxlib", "ymom")], dtype=np.float32)
    zmom = np.asarray(cg[("boxlib", "zmom")], dtype=np.float32)
    temp = np.asarray(cg[("boxlib", "Temp")], dtype=np.float32)
    boxsize = float(ds.domain_width[0].to_value("code_length"))
    a = _read_comoving_a(plotfile)
    z = (1.0 / a) - 1.0
    return {
        "density": density,
        "xmom": xmom,
        "ymom": ymom,
        "zmom": zmom,
        "temp_k": temp,
        "boxsize_comoving_mpc": boxsize,
        "a": a,
        "z": z,
    }


def _load_diffhydro_fields(path: Path):
    d = np.load(path)

    def pick(name, candidates, *, required=True):
        for key in candidates:
            if key in d:
                return np.asarray(d[key], dtype=np.float32), key
        if required:
            raise KeyError(f"Missing '{name}' in {path}. Tried keys: {candidates}")
        return None, None

    gas_density_norm, gas_key = pick(
        "gas density field",
        ("gas_dh_final", "model_rho", "gas_density_norm"),
        required=True,
    )
    temp_k, temp_key = pick(
        "temperature field",
        ("temp_dh_final", "model_temp", "temp_k"),
        required=True,
    )
    vx_cms, vx_key = pick("vx field", ("vx_dh_final_cms", "model_vx_cms"), required=False)
    vy_cms, vy_key = pick("vy field", ("vy_dh_final_cms", "model_vy_cms"), required=False)
    vz_cms, vz_key = pick("vz field", ("vz_dh_final_cms", "model_vz_cms"), required=False)
    a_final_arr, a_final_key = pick("final scale factor", ("a_final_dh", "final_scale_factor"), required=False)
    if a_final_arr is None:
        a_final = float("nan")
    else:
        a_final = float(np.asarray(a_final_arr).reshape(-1)[0])

    return {
        "gas_density_norm": gas_density_norm,
        "temp_k": temp_k,
        "vx_cms": vx_cms,
        "vy_cms": vy_cms,
        "vz_cms": vz_cms,
        "a_final": a_final,
        "source_keys": {
            "gas_density_norm": gas_key,
            "temp_k": temp_key,
            "vx_cms": vx_key,
            "vy_cms": vy_key,
            "vz_cms": vz_key,
            "a_final": a_final_key,
        },
    }


def _load_diffhydro_velocity_cms(
    state_path: Path,
    metrics_path: Path,
    *,
    h: float,
    boxsize_comoving_mpc: float,
    vel_unit_cms: float,
):
    U = jnp.asarray(np.load(state_path), dtype=jnp.float32)
    if U.ndim != 4 or U.shape[0] < 4:
        raise ValueError(f"Unexpected DiffHydro state shape: {U.shape}")
    n_grid = int(U.shape[1])

    m = np.load(metrics_path, allow_pickle=True)
    a_final = float(m["a_final_dh"])

    base_eq = dh.equationmanager.EquationManager()
    base_eq.mesh_shape = [n_grid, n_grid, n_grid]
    eq = SuperComovingEquationManager(base_eq, enforce_gamma_53=True)
    W = eq.get_primitives_from_conservatives(U)

    vx_phys = cosmo_conv.velocity_code_to_phys(W[1], a_final)
    vy_phys = cosmo_conv.velocity_code_to_phys(W[2], a_final)
    vz_phys = cosmo_conv.velocity_code_to_phys(W[3], a_final)

    hydro_velocity_scale = (100.0 * float(h)) * float(n_grid) / max(float(boxsize_comoving_mpc), 1.0e-30)
    hydro_vel_unit_cms = float(vel_unit_cms) / max(hydro_velocity_scale, 1.0e-30)
    return (
        np.asarray(vx_phys, dtype=np.float32) * hydro_vel_unit_cms,
        np.asarray(vy_phys, dtype=np.float32) * hydro_vel_unit_cms,
        np.asarray(vz_phys, dtype=np.float32) * hydro_vel_unit_cms,
        a_final,
    )


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(m) < 8:
        return float("nan")
    a = a[m] - np.mean(a[m])
    b = b[m] - np.mean(b[m])
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 0.0:
        return float("nan")
    return float((a @ b) / den)


def _imshow_three_panel(
    a,
    b,
    out_path: Path,
    *,
    title_a: str,
    title_b: str,
    cmap: str = "viridis",
    log10: bool = False,
    eps: float = 1.0e-30,
):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if log10:
        aa = np.log10(np.maximum(aa, eps))
        bb = np.log10(np.maximum(bb, eps))
        cbar_label = "log10"
    else:
        cbar_label = "value"
    rr = bb - aa

    vmin = float(np.nanpercentile(np.concatenate([aa.ravel(), bb.ravel()]), 1.0))
    vmax = float(np.nanpercentile(np.concatenate([aa.ravel(), bb.ravel()]), 99.0))
    rv = float(np.nanpercentile(np.abs(rr.ravel()), 99.0))
    rv = max(rv, 1.0e-12)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    im0 = axes[0].imshow(aa, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(title_a)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=cbar_label)

    im1 = axes[1].imshow(bb, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(title_b)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=cbar_label)

    im2 = axes[2].imshow(rr, origin="lower", cmap="coolwarm", vmin=-rv, vmax=rv)
    axes[2].set_title("DiffHydro - Nyx")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="difference")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _select_los_component(vx, vy, vz, axis_los: int):
    if axis_los == 0:
        return vx
    if axis_los == 1:
        return vy
    return vz


def _flux_pdf(flux_cube, bins=120):
    ff = np.asarray(flux_cube, dtype=np.float64).ravel()
    ff = ff[np.isfinite(ff)]
    hist, edges = np.histogram(ff, bins=bins, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def _los_power_1d(flux_cube, axis_los: int, dv_kms: float):
    flux_cube = np.asarray(flux_cube, dtype=np.float64)
    fbar = float(np.mean(flux_cube))
    delta_f = flux_cube / max(fbar, 1.0e-30) - 1.0
    d = np.moveaxis(delta_f, axis_los, -1)
    n = d.shape[-1]
    d2 = d.reshape((-1, n))

    ft = np.fft.rfft(d2, axis=-1)
    pk = np.mean(np.abs(ft) ** 2, axis=0) / float(n * n)
    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=float(dv_kms))
    return k[1:], pk[1:]


def _cube_stats(name: str, tau_cube, flux_cube):
    tau = np.asarray(tau_cube, dtype=np.float64)
    flux = np.asarray(flux_cube, dtype=np.float64)
    out = {
        f"{name}_tau_min": float(np.min(tau)),
        f"{name}_tau_p50": float(np.percentile(tau, 50.0)),
        f"{name}_tau_p95": float(np.percentile(tau, 95.0)),
        f"{name}_tau_p99": float(np.percentile(tau, 99.0)),
        f"{name}_tau_max": float(np.max(tau)),
        f"{name}_flux_min": float(np.min(flux)),
        f"{name}_flux_p01": float(np.percentile(flux, 1.0)),
        f"{name}_flux_p05": float(np.percentile(flux, 5.0)),
        f"{name}_flux_p50": float(np.percentile(flux, 50.0)),
        f"{name}_frac_flux_lt_1e-3": float(np.mean(flux < 1.0e-3)),
        f"{name}_frac_flux_lt_1e-2": float(np.mean(flux < 1.0e-2)),
        f"{name}_frac_flux_lt_5e-2": float(np.mean(flux < 5.0e-2)),
        f"{name}_frac_flux_lt_1e-1": float(np.mean(flux < 1.0e-1)),
    }
    return out


def _plot_flux_central_slices(flux_nyx, flux_dh, out_dir: Path, prefix: str):
    flux_nyx = np.asarray(flux_nyx, dtype=np.float64)
    flux_dh = np.asarray(flux_dh, dtype=np.float64)
    if flux_nyx.ndim != 3 or flux_dh.ndim != 3:
        raise ValueError(f"Expected 3D flux cubes, got {flux_nyx.shape} and {flux_dh.shape}")
    if flux_nyx.shape != flux_dh.shape:
        raise ValueError(f"Flux cube shape mismatch: {flux_nyx.shape} vs {flux_dh.shape}")

    ix = flux_nyx.shape[0] // 2
    iy = flux_nyx.shape[1] // 2
    iz = flux_nyx.shape[2] // 2

    planes = (
        ("xy", flux_nyx[:, :, iz], flux_dh[:, :, iz], f"z={iz}"),
        ("yz", flux_nyx[ix, :, :], flux_dh[ix, :, :], f"x={ix}"),
        ("xz", flux_nyx[:, iy, :], flux_dh[:, iy, :], f"y={iy}"),
    )
    axis_label = {"xy": "x-y", "yz": "y-z", "xz": "x-z"}
    pretty = prefix.replace("_", " ")
    for plane_id, plane_nyx, plane_dh, fixed_label in planes:
        _imshow_three_panel(
            plane_nyx,
            plane_dh,
            out_dir / f"{prefix}_slice_{plane_id}_compare.png",
            title_a=f"Nyx {pretty} ({axis_label[plane_id]}, {fixed_label})",
            title_b=f"DiffHydro {pretty} ({axis_label[plane_id]}, {fixed_label})",
            log10=False,
        )


def main():
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nyx = _load_nyx_plotfile(args.nyx_plotfile.resolve())
    fields = _load_diffhydro_fields(args.diffhydro_fields.resolve())

    boxsize_mpc = float(args.boxsize_comoving_mpc or nyx["boxsize_comoving_mpc"])
    z_eval = float(nyx["z"])
    n_grid = int(nyx["density"].shape[0])
    if tuple(fields["gas_density_norm"].shape) != tuple(nyx["density"].shape):
        raise ValueError(
            "Shape mismatch between Nyx and DiffHydro density grids: "
            f"{nyx['density'].shape} vs {fields['gas_density_norm'].shape}"
        )

    vx_dh = fields["vx_cms"]
    vy_dh = fields["vy_cms"]
    vz_dh = fields["vz_cms"]
    a_dh = float(fields["a_final"])
    use_field_velocities = (vx_dh is not None) and (vy_dh is not None) and (vz_dh is not None)
    if args.realspace_only:
        print("[info] --realspace-only enabled: skipping RSD and velocity-dependent products.")
    elif use_field_velocities:
        print(
            "[info] using DiffHydro velocity fields from --diffhydro-fields "
            f"(keys: {fields['source_keys']['vx_cms']}, {fields['source_keys']['vy_cms']}, {fields['source_keys']['vz_cms']})"
        )
    else:
        if not args.diffhydro_state.resolve().exists():
            raise FileNotFoundError(
                "DiffHydro velocity fields are missing in --diffhydro-fields, and "
                f"--diffhydro-state does not exist: {args.diffhydro_state.resolve()}"
            )
        if not args.metrics.resolve().exists():
            raise FileNotFoundError(
                "DiffHydro velocity fields are missing in --diffhydro-fields, and "
                f"--metrics does not exist: {args.metrics.resolve()}"
            )
        vx_dh, vy_dh, vz_dh, a_dh = _load_diffhydro_velocity_cms(
            args.diffhydro_state.resolve(),
            args.metrics.resolve(),
            h=float(args.h),
            boxsize_comoving_mpc=boxsize_mpc,
            vel_unit_cms=float(args.vel_unit_cms),
        )
        print("[info] loaded DiffHydro velocity from --diffhydro-state + --metrics.")

    cosmo = CosmologyParams(
        h=float(args.h),
        omega_m=float(args.omega_m),
        omega_b=float(args.omega_b),
        omega_l=(None if args.omega_l is None else float(args.omega_l)),
    )

    gas_nyx_norm = np.asarray(nyx["density"] / np.maximum(np.mean(nyx["density"]), 1.0e-30), dtype=np.float32)
    temp_nyx = np.maximum(np.asarray(nyx["temp_k"], dtype=np.float32), float(args.temp_floor_k))
    gas_dh_norm = np.maximum(np.asarray(fields["gas_density_norm"], dtype=np.float32), float(args.delta_floor))
    temp_dh = np.maximum(np.asarray(fields["temp_k"], dtype=np.float32), float(args.temp_floor_k))

    with _stage_treecool(args.treecool_file):
        eos_interp = build_nyx_eos_interpolator(
            z=z_eval,
            cosmo=cosmo,
            grid_size=int(args.eos_grid_size),
            logdelta_limits=(-3.0, 4.0),
            logt_limits=(0.0, 8.0),
        )

    nhi_nyx = compute_nhi_number_density(
        jnp.asarray(np.maximum(gas_nyx_norm, float(args.delta_floor)), dtype=jnp.float32),
        jnp.asarray(temp_nyx, dtype=jnp.float32),
        eos_interp,
    )
    nhi_dh = compute_nhi_number_density(
        jnp.asarray(gas_dh_norm, dtype=jnp.float32),
        jnp.asarray(temp_dh, dtype=jnp.float32),
        eos_interp,
    )

    zeros_nyx = jnp.zeros_like(jnp.asarray(gas_nyx_norm, dtype=jnp.float32))
    zeros_dh = jnp.zeros_like(jnp.asarray(gas_dh_norm, dtype=jnp.float32))

    tau_real_nyx = tau_cube_from_nhi(
        nhi_nyx,
        temp_nyx,
        zeros_nyx,
        z=z_eval,
        cosmo=cosmo,
        boxsize_comoving_mpc=boxsize_mpc,
        axis_los=int(args.los_axis),
        num_integ_pixels=int(args.num_integ_pixels),
    )
    tau_real_dh = tau_cube_from_nhi(
        nhi_dh,
        temp_dh,
        zeros_dh,
        z=z_eval,
        cosmo=cosmo,
        boxsize_comoving_mpc=boxsize_mpc,
        axis_los=int(args.los_axis),
        num_integ_pixels=int(args.num_integ_pixels),
    )

    if args.realspace_only:
        tau_rsd_nyx = None
        tau_rsd_dh = None
    else:
        vx_nyx = nyx["xmom"] / np.maximum(nyx["density"], 1.0e-30)
        vy_nyx = nyx["ymom"] / np.maximum(nyx["density"], 1.0e-30)
        vz_nyx = nyx["zmom"] / np.maximum(nyx["density"], 1.0e-30)
        vlos_nyx_cms = _select_los_component(vx_nyx, vy_nyx, vz_nyx, int(args.los_axis))
        vlos_nyx_cms = (vlos_nyx_cms / float(args.nyx_velocity_divisor)) * float(args.vel_unit_cms)
        vlos_dh_cms = _select_los_component(vx_dh, vy_dh, vz_dh, int(args.los_axis))
        tau_rsd_nyx = tau_cube_from_nhi(
            nhi_nyx,
            temp_nyx,
            vlos_nyx_cms,
            z=z_eval,
            cosmo=cosmo,
            boxsize_comoving_mpc=boxsize_mpc,
            axis_los=int(args.los_axis),
            num_integ_pixels=int(args.num_integ_pixels),
        )
        tau_rsd_dh = tau_cube_from_nhi(
            nhi_dh,
            temp_dh,
            vlos_dh_cms,
            z=z_eval,
            cosmo=cosmo,
            boxsize_comoving_mpc=boxsize_mpc,
            axis_los=int(args.los_axis),
            num_integ_pixels=int(args.num_integ_pixels),
        )

    flux_real_nyx = flux_from_tau(tau_real_nyx)
    flux_real_dh = flux_from_tau(tau_real_dh)
    if tau_rsd_nyx is None or tau_rsd_dh is None:
        flux_rsd_nyx = None
        flux_rsd_dh = None
    else:
        flux_rsd_nyx = flux_from_tau(tau_rsd_nyx)
        flux_rsd_dh = flux_from_tau(tau_rsd_dh)

    map_nhi_nyx = np.asarray(
        column_density_map_nhi(
            nhi_nyx,
            boxsize_comoving_mpc=boxsize_mpc,
            z=z_eval,
            axis_los=int(args.los_axis),
        )
    )
    map_nhi_dh = np.asarray(
        column_density_map_nhi(
            nhi_dh,
            boxsize_comoving_mpc=boxsize_mpc,
            z=z_eval,
            axis_los=int(args.los_axis),
        )
    )
    map_tau_real_nyx = np.asarray(mean_map_along_los(tau_real_nyx, axis_los=int(args.los_axis)))
    map_tau_real_dh = np.asarray(mean_map_along_los(tau_real_dh, axis_los=int(args.los_axis)))
    map_flux_real_nyx = np.asarray(mean_map_along_los(flux_real_nyx, axis_los=int(args.los_axis)))
    map_flux_real_dh = np.asarray(mean_map_along_los(flux_real_dh, axis_los=int(args.los_axis)))
    map_fluxmin_real_nyx = np.min(np.asarray(flux_real_nyx), axis=int(args.los_axis))
    map_fluxmin_real_dh = np.min(np.asarray(flux_real_dh), axis=int(args.los_axis))
    if flux_rsd_nyx is None or flux_rsd_dh is None:
        map_tau_rsd_nyx = None
        map_tau_rsd_dh = None
        map_flux_rsd_nyx = None
        map_flux_rsd_dh = None
        map_fluxmin_rsd_nyx = None
        map_fluxmin_rsd_dh = None
    else:
        map_tau_rsd_nyx = np.asarray(mean_map_along_los(tau_rsd_nyx, axis_los=int(args.los_axis)))
        map_tau_rsd_dh = np.asarray(mean_map_along_los(tau_rsd_dh, axis_los=int(args.los_axis)))
        map_flux_rsd_nyx = np.asarray(mean_map_along_los(flux_rsd_nyx, axis_los=int(args.los_axis)))
        map_flux_rsd_dh = np.asarray(mean_map_along_los(flux_rsd_dh, axis_los=int(args.los_axis)))
        map_fluxmin_rsd_nyx = np.min(np.asarray(flux_rsd_nyx), axis=int(args.los_axis))
        map_fluxmin_rsd_dh = np.min(np.asarray(flux_rsd_dh), axis=int(args.los_axis))

    print(map_nhi_nyx.shape, map_nhi_dh.shape)
    _imshow_three_panel(
        map_nhi_nyx,
        map_nhi_dh,
        args.output_dir / "nhi_column_map_compare.png",
        title_a="Nyx log10(NHI column)",
        title_b="DiffHydro log10(NHI column)",
        log10=True,
    )
    _imshow_three_panel(
        map_tau_real_nyx,
        map_tau_real_dh,
        args.output_dir / "tau_realspace_mean_map_compare.png",
        title_a="Nyx log10(<tau> real)",
        title_b="DiffHydro log10(<tau> real)",
        log10=True,
    )
    i_slice = n_grid // 2
    tau_real_slice_nyx = np.take(np.asarray(tau_real_nyx), i_slice, axis=int(args.los_axis))
    tau_real_slice_dh = np.take(np.asarray(tau_real_dh), i_slice, axis=int(args.los_axis))

    _imshow_three_panel(
        tau_real_slice_nyx,
        tau_real_slice_dh,
        args.output_dir / "tau_realspace_slice_compare.png",
        title_a=f"Nyx log10(tau real) slice={i_slice}",
        title_b=f"DiffHydro log10(tau real) slice={i_slice}",
        log10=True,
    )
    _imshow_three_panel(
        map_flux_real_nyx,
        map_flux_real_dh,
        args.output_dir / "flux_realspace_mean_map_compare.png",
        title_a="Nyx <F> real",
        title_b="DiffHydro <F> real",
        log10=False,
    )
    _imshow_three_panel(
        map_fluxmin_real_nyx,
        map_fluxmin_real_dh,
        args.output_dir / "flux_realspace_minlos_map_compare.png",
        title_a="Nyx min_LOS(F) real",
        title_b="DiffHydro min_LOS(F) real",
        log10=False,
    )
    _plot_flux_central_slices(
        flux_real_nyx,
        flux_real_dh,
        args.output_dir,
        prefix="flux_realspace",
    )

    c1, h1 = _flux_pdf(flux_real_nyx)
    c2, h2 = _flux_pdf(flux_real_dh)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(c1, h1, lw=2, label="Nyx")
    ax.plot(c2, h2, lw=2, label="DiffHydro")
    ax.set_xlabel("Flux F")
    ax.set_ylabel("PDF(F)")
    ax.set_title("Flux PDF (real space)")
    ax.legend()
    fig.savefig(args.output_dir / "flux_pdf_realspace.png", dpi=170)
    plt.close(fig)

    vdom_kms = float(velocity_domain_cmps(cosmo, z_eval, boxsize_mpc) / 1.0e5)
    dv_kms = vdom_kms / float(n_grid)
    k_nyx_real, p_nyx_real = _los_power_1d(flux_real_nyx, int(args.los_axis), dv_kms)
    k_dh_real, p_dh_real = _los_power_1d(flux_real_dh, int(args.los_axis), dv_kms)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.loglog(k_nyx_real, p_nyx_real, lw=2, label="Nyx")
    ax.loglog(k_dh_real, p_dh_real, lw=2, label="DiffHydro")
    ax.set_xlabel("k [s/km]")
    ax.set_ylabel("P_F(k)")
    ax.set_title("1D Flux Power (real space)")
    ax.legend()
    fig.savefig(args.output_dir / "flux_power_realspace.png", dpi=170)
    plt.close(fig)

    if not args.realspace_only:
        tau_rsd_slice_nyx = np.take(np.asarray(tau_rsd_nyx), i_slice, axis=int(args.los_axis))
        tau_rsd_slice_dh = np.take(np.asarray(tau_rsd_dh), i_slice, axis=int(args.los_axis))
        _imshow_three_panel(
            map_tau_rsd_nyx,
            map_tau_rsd_dh,
            args.output_dir / "tau_redshiftspace_mean_map_compare.png",
            title_a="Nyx log10(<tau> rsd)",
            title_b="DiffHydro log10(<tau> rsd)",
            log10=True,
        )
        _imshow_three_panel(
            tau_rsd_slice_nyx,
            tau_rsd_slice_dh,
            args.output_dir / "tau_redshiftspace_slice_compare.png",
            title_a=f"Nyx log10(tau rsd) slice={i_slice}",
            title_b=f"DiffHydro log10(tau rsd) slice={i_slice}",
            log10=True,
        )
        _imshow_three_panel(
            map_flux_rsd_nyx,
            map_flux_rsd_dh,
            args.output_dir / "flux_redshiftspace_mean_map_compare.png",
            title_a="Nyx <F> rsd",
            title_b="DiffHydro <F> rsd",
            log10=False,
        )
        _imshow_three_panel(
            map_fluxmin_rsd_nyx,
            map_fluxmin_rsd_dh,
            args.output_dir / "flux_redshiftspace_minlos_map_compare.png",
            title_a="Nyx min_LOS(F) rsd",
            title_b="DiffHydro min_LOS(F) rsd",
            log10=False,
        )
        _plot_flux_central_slices(
            flux_rsd_nyx,
            flux_rsd_dh,
            args.output_dir,
            prefix="flux_redshiftspace",
        )

        c1, h1 = _flux_pdf(flux_rsd_nyx)
        c2, h2 = _flux_pdf(flux_rsd_dh)
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.plot(c1, h1, lw=2, label="Nyx")
        ax.plot(c2, h2, lw=2, label="DiffHydro")
        ax.set_xlabel("Flux F")
        ax.set_ylabel("PDF(F)")
        ax.set_title("Flux PDF (redshift space)")
        ax.legend()
        fig.savefig(args.output_dir / "flux_pdf_redshiftspace.png", dpi=170)
        plt.close(fig)

        k_nyx_rsd, p_nyx_rsd = _los_power_1d(flux_rsd_nyx, int(args.los_axis), dv_kms)
        k_dh_rsd, p_dh_rsd = _los_power_1d(flux_rsd_dh, int(args.los_axis), dv_kms)
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.loglog(k_nyx_rsd, p_nyx_rsd, lw=2, label="Nyx")
        ax.loglog(k_dh_rsd, p_dh_rsd, lw=2, label="DiffHydro")
        ax.set_xlabel("k [s/km]")
        ax.set_ylabel("P_F(k)")
        ax.set_title("1D Flux Power (redshift space)")
        ax.legend()
        fig.savefig(args.output_dir / "flux_power_redshiftspace.png", dpi=170)
        plt.close(fig)
    else:
        k_nyx_rsd = np.asarray([], dtype=np.float64)
        p_nyx_rsd = np.asarray([], dtype=np.float64)
        k_dh_rsd = np.asarray([], dtype=np.float64)
        p_dh_rsd = np.asarray([], dtype=np.float64)

    metrics = {
        "z_eval": z_eval,
        "a_nyx": float(nyx["a"]),
        "a_diffhydro": float(a_dh),
        "los_axis": int(args.los_axis),
        "boxsize_comoving_mpc": boxsize_mpc,
        "pearson_log10_nhi_column": _pearson(np.log10(np.maximum(map_nhi_nyx, 1.0e-40)), np.log10(np.maximum(map_nhi_dh, 1.0e-40))),
        "pearson_log10_tau_real_mean": _pearson(np.log10(np.maximum(map_tau_real_nyx, 1.0e-40)), np.log10(np.maximum(map_tau_real_dh, 1.0e-40))),
        "pearson_log10_tau_rsd_mean": (
            float("nan")
            if args.realspace_only
            else _pearson(np.log10(np.maximum(map_tau_rsd_nyx, 1.0e-40)), np.log10(np.maximum(map_tau_rsd_dh, 1.0e-40)))
        ),
        "mean_flux_real_nyx": float(np.mean(np.asarray(flux_real_nyx))),
        "mean_flux_real_diffhydro": float(np.mean(np.asarray(flux_real_dh))),
        "mean_flux_rsd_nyx": (float("nan") if args.realspace_only else float(np.mean(np.asarray(flux_rsd_nyx)))),
        "mean_flux_rsd_diffhydro": (float("nan") if args.realspace_only else float(np.mean(np.asarray(flux_rsd_dh)))),
        "realspace_only": bool(args.realspace_only),
        "used_diffhydro_velocity_from_fields": bool(use_field_velocities),
    }
    metrics.update(_cube_stats("nyx_real", tau_real_nyx, flux_real_nyx))
    metrics.update(_cube_stats("dh_real", tau_real_dh, flux_real_dh))
    if not args.realspace_only:
        metrics.update(_cube_stats("nyx_rsd", tau_rsd_nyx, flux_rsd_nyx))
        metrics.update(_cube_stats("dh_rsd", tau_rsd_dh, flux_rsd_dh))
    metrics.update(
        {
            "od_factor": float(lya_tau_prefactor(cosmo, z_eval)),
            "v_domain_kmps": float(velocity_domain_cmps(cosmo, z_eval, boxsize_mpc) / 1.0e5),
            "cell_dv_kmps": float(velocity_domain_cmps(cosmo, z_eval, boxsize_mpc) / 1.0e5 / float(n_grid)),
            "cell_proper_kpc": float(boxsize_mpc * 1.0e3 / ((1.0 + z_eval) * float(n_grid))),
            "nhi_column_min_cm2_nyx": float(np.min(map_nhi_nyx)),
            "nhi_column_p99_cm2_nyx": float(np.percentile(map_nhi_nyx, 99.0)),
            "nhi_column_max_cm2_nyx": float(np.max(map_nhi_nyx)),
            "nhi_column_min_cm2_dh": float(np.min(map_nhi_dh)),
            "nhi_column_p99_cm2_dh": float(np.percentile(map_nhi_dh, 99.0)),
            "nhi_column_max_cm2_dh": float(np.max(map_nhi_dh)),
        }
    )

    with (args.output_dir / "summary_metrics.txt").open("w", encoding="utf-8") as f:
        for k in sorted(metrics):
            f.write(f"{k}: {metrics[k]}\n")

    save_payload = dict(
        map_nhi_nyx=map_nhi_nyx,
        map_nhi_dh=map_nhi_dh,
        map_tau_real_nyx=map_tau_real_nyx,
        map_tau_real_dh=map_tau_real_dh,
        map_flux_real_nyx=map_flux_real_nyx,
        map_flux_real_dh=map_flux_real_dh,
        k_nyx_real=k_nyx_real,
        p_nyx_real=p_nyx_real,
        k_dh_real=k_dh_real,
        p_dh_real=p_dh_real,
    )
    if not args.realspace_only:
        save_payload.update(
            map_tau_rsd_nyx=map_tau_rsd_nyx,
            map_tau_rsd_dh=map_tau_rsd_dh,
            map_flux_rsd_nyx=map_flux_rsd_nyx,
            map_flux_rsd_dh=map_flux_rsd_dh,
            k_nyx_rsd=k_nyx_rsd,
            p_nyx_rsd=p_nyx_rsd,
            k_dh_rsd=k_dh_rsd,
            p_dh_rsd=p_dh_rsd,
            tau_rsd_nyx=tau_rsd_nyx,
            tau_rsd_dh=tau_rsd_dh,
        )
    save_payload.update(
        tau_real_nyx=tau_real_nyx,
        tau_real_dh=tau_real_dh,
    )
    np.savez_compressed(args.output_dir / "lya_products.npz", **save_payload)

    print(f"[ok] Wrote LyA post-processing outputs to: {args.output_dir}")
    print(f"[ok] Summary: {args.output_dir / 'summary_metrics.txt'}")


if __name__ == "__main__":
    main()
