#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 1: build a differentiable white-noise -> DM -> gas(+T) forward model at z~2, "
            "calibrate mapping from reference fields, and generate diagnostics/plots."
        )
    )

    p.add_argument("--gpu", type=str, default="3", help="CUDA_VISIBLE_DEVICES value.")
    p.add_argument("--xla-preallocate", action="store_true", default=False)

    p.add_argument(
        "--cv0-mgas-path",
        type=str,
        default="/gpfs02/work/diffusion/IllustrisTNG/Grids_Mgas_IllustrisTNG_CV_128_z=2.0.npy",
    )
    p.add_argument(
        "--cv0-temp-path",
        type=str,
        default="/gpfs02/work/diffusion/IllustrisTNG/Grids_T_IllustrisTNG_CV_128_z=2.0.npy",
    )
    p.add_argument("--cv0-field-index", type=int, default=0, help="Use first CV realization by default.")
    p.add_argument(
        "--reference-fields-npz",
        type=str,
        default=(
            "/home/ben.horowitz/DiffHydro_public/"
            "diffhydro_gadgetic_n128_z127to2_hll/snapshots/fields_ic_final.npz"
        ),
    )

    p.add_argument("--mesh-n", type=int, default=128)
    p.add_argument("--box-size-mpc-h", type=float, default=25.0)
    p.add_argument("--z-init", type=float, default=127.0)
    p.add_argument("--z-target", type=float, default=2.0)
    p.add_argument("--kdk-steps", type=int, default=64)
    p.add_argument("--checkpoint", dest="checkpoint", action="store_true", default=True)
    p.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    p.add_argument("--checkpoint-every", type=int, default=4)

    p.add_argument("--h", type=float, default=0.6711)
    p.add_argument("--omega-m", type=float, default=0.3)
    p.add_argument("--omega-b", type=float, default=0.045)
    p.add_argument("--sigma8", type=float, default=0.8)
    p.add_argument("--n-s", type=float, default=0.9624)
    p.add_argument(
        "--ic-power-suppression",
        type=float,
        default=1.0,
        help=(
            "Multiplicative factor on initial-guess IC power, applied only to random init "
            "(equivalent amplitude scaling by sqrt(factor))."
        ),
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--init-random-scale",
        type=float,
        default=1.0,
        help="Scale factor on unit-variance white-noise IC parameters.",
    )

    p.add_argument("--override-smooth-sigma", type=float, default=None)
    p.add_argument("--override-bias-linear", type=float, default=None)
    p.add_argument("--override-bias-quadratic", type=float, default=None)
    p.add_argument("--override-temp-heat-gain", type=float, default=None)
    p.add_argument("--override-temp-slope", type=float, default=None)
    p.add_argument("--override-temp-quadratic", type=float, default=None)

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cosmo_reconstruct/outputs/step1_forward"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.ic_power_suppression < 0.0:
        raise ValueError("--ic-power-suppression must be >= 0.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if args.xla_preallocate else "false"
    os.environ.setdefault("MPLBACKEND", "Agg")

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np

    from src.diagnostics import (
        compute_power_and_cross,
        field_stats,
        fit_gas_mapping_from_reference,
        fit_temperature_mapping_from_reference,
        load_cv0_fields,
        load_reference_fields,
        make_forward_plots,
        save_json,
    )
    from src.forward_model import (
        ForwardModelConfig,
        GasModelParams,
        build_cosmology,
        forward_fields,
        make_lattice_positions,
        make_pk_sqrt,
        prime_growth_cache,
    )

    if not any(dev.platform == "gpu" for dev in jax.devices()):
        raise RuntimeError("GPU backend is required")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    cv0_rho, cv0_temp = load_cv0_fields(
        args.cv0_mgas_path,
        args.cv0_temp_path,
        field_index=args.cv0_field_index,
        mesh_n=args.mesh_n,
    )

    ref = load_reference_fields(args.reference_fields_npz, mesh_n=args.mesh_n)
    ref_gas = np.asarray(ref["gas_dh_final"], dtype=np.float32)
    ref_dm = np.asarray(ref["dm_dh_final"], dtype=np.float32)
    ref_temp = np.asarray(ref["temp_dh_final"], dtype=np.float32)
    ref_temp_ic = np.asarray(ref["temp_dh_ic"], dtype=np.float32)

    gas_fit = fit_gas_mapping_from_reference(ref_dm, ref_gas)
    temp_fit = fit_temperature_mapping_from_reference(ref_gas, ref_temp, ref_temp_ic)

    smooth_sigma = args.override_smooth_sigma if args.override_smooth_sigma is not None else gas_fit["smooth_sigma_cells"]
    bias_linear = args.override_bias_linear if args.override_bias_linear is not None else gas_fit["bias_linear"]
    bias_quadratic = args.override_bias_quadratic if args.override_bias_quadratic is not None else gas_fit["bias_quadratic"]
    temp_heat_gain = args.override_temp_heat_gain if args.override_temp_heat_gain is not None else temp_fit["temp_heat_gain"]
    temp_slope = args.override_temp_slope if args.override_temp_slope is not None else temp_fit["temp_slope"]
    temp_quadratic = args.override_temp_quadratic if args.override_temp_quadratic is not None else temp_fit["temp_quadratic"]

    cfg = ForwardModelConfig(
        mesh_n=args.mesh_n,
        box_size_mpc_h=args.box_size_mpc_h,
        z_init=args.z_init,
        z_target=args.z_target,
        kdk_steps=args.kdk_steps,
        omega_m=args.omega_m,
        omega_b=args.omega_b,
        h=args.h,
        n_s=args.n_s,
        sigma8=args.sigma8,
        checkpoint=bool(args.checkpoint),
        checkpoint_every=max(1, int(args.checkpoint_every)),
    )
    a_init = 1.0 / (1.0 + cfg.z_init)
    gas_params = GasModelParams(
        smooth_sigma_cells=float(smooth_sigma),
        bias_linear=float(bias_linear),
        bias_quadratic=float(bias_quadratic),
        gas_mean=1.0,
        temp_init_k=float(temp_fit["temp_init_k"]),
        temp_heat_gain=float(temp_heat_gain),
        temp_slope=float(temp_slope),
        temp_quadratic=float(temp_quadratic),
    )

    # Keep PK and dynamical growth caches separated (jaxpm.growth cache layout differs
    # from jax_cosmo.background cache used by linear_matter_power).
    cosmo_pk = build_cosmology(cfg)
    cosmo_dyn = build_cosmology(cfg)
    grid_pos = make_lattice_positions(cfg.mesh_n)
    pk_sqrt = make_pk_sqrt(cosmo_pk, cfg)
    prime_growth_cache(cosmo_dyn, a_init)

    key = jr.PRNGKey(args.seed)
    init_guess_amplitude_scale = float(np.sqrt(args.ic_power_suppression))
    white_noise = (
        args.init_random_scale
        * init_guess_amplitude_scale
        * jr.normal(key, (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n), dtype=jnp.float32)
    )

    run_forward = jax.jit(lambda wn: forward_fields(wn, pk_sqrt, grid_pos, cosmo_dyn, cfg, gas_params))

    t_compile = time.perf_counter()
    rho_dm_j, rho_gas_j, temp_j, _, init_mesh_j = run_forward(white_noise)
    rho_dm_j.block_until_ready()
    compile_and_first_s = time.perf_counter() - t_compile

    t_exec = time.perf_counter()
    rho_dm_j, rho_gas_j, temp_j, _, init_mesh_j = run_forward(white_noise)
    rho_dm_j.block_until_ready()
    exec_only_s = time.perf_counter() - t_exec

    rho_dm = np.asarray(rho_dm_j, dtype=np.float32)
    rho_gas = np.asarray(rho_gas_j, dtype=np.float32)
    temp_gas = np.asarray(temp_j, dtype=np.float32)
    init_mesh = np.asarray(init_mesh_j, dtype=np.float32)

    spectra_cv0 = compute_power_and_cross(cv0_rho, rho_gas, cfg.box_size_mpc_h)
    spectra_ref = compute_power_and_cross(ref_gas, rho_gas, cfg.box_size_mpc_h)

    make_forward_plots(
        out_dir,
        target_rho_cv0=cv0_rho,
        target_temp_cv0=cv0_temp,
        ref_gas=ref_gas,
        ref_temp=ref_temp,
        pred_rho=rho_gas,
        pred_temp=temp_gas,
        pred_dm=rho_dm,
        spectra_cv0=spectra_cv0,
        spectra_ref=spectra_ref,
    )

    stats = {
        "run": {
            "mesh_n": cfg.mesh_n,
            "box_size_mpc_h": cfg.box_size_mpc_h,
            "z_init": cfg.z_init,
            "z_target": cfg.z_target,
            "kdk_steps": cfg.kdk_steps,
            "ic_power_suppression": float(args.ic_power_suppression),
            "compile_and_first_forward_s": compile_and_first_s,
            "forward_exec_only_s": exec_only_s,
            "total_elapsed_s": time.perf_counter() - t0,
        },
        "gas_mapping_fit": gas_fit,
        "temperature_mapping_fit": temp_fit,
        "gas_model_params_used": {
            "smooth_sigma_cells": gas_params.smooth_sigma_cells,
            "bias_linear": gas_params.bias_linear,
            "bias_quadratic": gas_params.bias_quadratic,
            "temp_init_k": gas_params.temp_init_k,
            "temp_heat_gain": gas_params.temp_heat_gain,
            "temp_slope": gas_params.temp_slope,
            "temp_quadratic": gas_params.temp_quadratic,
        },
        "stats": {
            "cv0_rho": field_stats(cv0_rho),
            "reference_rho": field_stats(ref_gas),
            "model_rho": field_stats(rho_gas),
            "cv0_temp": field_stats(cv0_temp),
            "reference_temp": field_stats(ref_temp),
            "model_temp": field_stats(temp_gas),
            "model_dm": field_stats(rho_dm),
        },
        "spectra": {
            "cv0_vs_model_median_cross": float(spectra_cv0["median_cross"]),
            "reference_vs_model_median_cross": float(spectra_ref["median_cross"]),
        },
    }

    save_json(out_dir / "forward_stats.json", stats)
    np.savez_compressed(
        out_dir / "forward_fields.npz",
        cv0_rho=cv0_rho,
        cv0_temp=cv0_temp,
        ref_rho=ref_gas,
        ref_temp=ref_temp,
        model_dm=rho_dm,
        model_rho=rho_gas,
        model_temp=temp_gas,
        init_mesh=init_mesh,
        white_noise=np.asarray(white_noise, dtype=np.float32),
        k_cv0=spectra_cv0["k_ref"],
        pk_cv0=spectra_cv0["pk_ref"],
        k_model_cv0=spectra_cv0["k_pred"],
        pk_model_cv0=spectra_cv0["pk_pred"],
        cross_cv0=spectra_cv0["cross"],
        k_cross_cv0=spectra_cv0["k_cross"],
        k_ref=spectra_ref["k_ref"],
        pk_ref=spectra_ref["pk_ref"],
        k_model_ref=spectra_ref["k_pred"],
        pk_model_ref=spectra_ref["pk_pred"],
        cross_ref=spectra_ref["cross"],
        k_cross_ref=spectra_ref["k_cross"],
    )
    np.savez_compressed(
        out_dir / "fields_ic_final.npz",
        gas_dh_final=rho_gas,
        temp_dh_final=temp_gas,
        dm_dh_final=rho_dm,
        gas_dh_ic=init_mesh,
    )

    print(f"Saved outputs to: {out_dir}")
    print(
        f"Model rho mean/std={stats['stats']['model_rho']['mean']:.5f}/{stats['stats']['model_rho']['std']:.5f} | "
        f"Ref rho mean/std={stats['stats']['reference_rho']['mean']:.5f}/{stats['stats']['reference_rho']['std']:.5f} | "
        f"CV0 rho mean/std={stats['stats']['cv0_rho']['mean']:.5f}/{stats['stats']['cv0_rho']['std']:.5f}"
    )
    print(
        f"Model T mean={stats['stats']['model_temp']['mean']:.2f} K | "
        f"Ref T mean={stats['stats']['reference_temp']['mean']:.2f} K | "
        f"CV0 T mean={stats['stats']['cv0_temp']['mean']:.2f} K"
    )
    print(
        f"Median cross r(k): CV0 vs model={stats['spectra']['cv0_vs_model_median_cross']:.4f}, "
        f"reference vs model={stats['spectra']['reference_vs_model_median_cross']:.4f}"
    )


if __name__ == "__main__":
    main()
