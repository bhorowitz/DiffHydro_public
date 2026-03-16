#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 2/3: optimize white-noise ICs by maximizing an observable-space likelihood "
            "against CV0 (or a self-consistent synthetic target), with full backprop through "
            "LPT + KDK + DM->gas map."
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
    p.add_argument("--cv0-field-index", type=int, default=0)
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
            "Multiplicative factor on initial-guess IC power, applied only when using "
            "random white-noise initialization (equivalent amplitude scaling by sqrt(factor))."
        ),
    )

    p.add_argument("--compare-space", choices=["log", "linear"], default="log")
    p.add_argument(
        "--noise-sigma",
        type=float,
        default=0.01,
        help="Gaussian likelihood sigma in the chosen compare space.",
    )
    p.add_argument(
        "--prior-weight",
        type=float,
        default=1.0,
        help="Weight on unit Gaussian white-noise prior term.",
    )
    p.add_argument(
        "--temp-ps-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional additive weight on log-power-spectrum MSE for the temperature field. "
            "0 disables this term."
        ),
    )
    p.add_argument(
        "--temp-ps-loss-nbins",
        type=int,
        default=32,
        help="Number of k-bins for temperature power-spectrum loss.",
    )
    p.add_argument(
        "--temp-ps-loss-space",
        choices=["log", "linear"],
        default="log",
        help="Field transform for temperature power-spectrum loss.",
    )
    p.add_argument(
        "--temp-ps-loss-eps",
        type=float,
        default=1.0e-8,
        help="Numerical epsilon for temperature power-spectrum loss.",
    )
    p.add_argument(
        "--observable",
        choices=["density", "lya_flux", "xray_proxy", "fgpa_flux_dm"],
        default="density",
        help="Observable used in the data term.",
    )
    p.add_argument(
        "--observable-projection",
        choices=["3d", "mean_los", "sum_los"],
        default="3d",
        help="Projection for observable-space fitting.",
    )
    p.add_argument("--los-axis", type=int, choices=[0, 1, 2], default=2)
    p.add_argument(
        "--target-observable-npy",
        type=str,
        default=None,
        help=(
            "Optional externally prepared target observable (.npy or .npz). "
            "If provided, overrides internally computed target observable and "
            "self-target noise injection."
        ),
    )
    p.add_argument(
        "--target-source",
        choices=["cv0", "self_consistent"],
        default="cv0",
        help="Target observable source: CV0 fields or internally generated self-consistent forward model.",
    )
    p.add_argument(
        "--self-target-white-noise-npy",
        type=str,
        default=None,
        help="Optional white-noise field used to generate --target-source=self_consistent target.",
    )
    p.add_argument("--self-target-seed", type=int, default=123)
    p.add_argument("--self-target-scale", type=float, default=1.0)
    p.add_argument(
        "--self-target-noise-sigma",
        type=float,
        default=0.0,
        help="Optional additive Gaussian noise in compare space for self-consistent targets.",
    )
    p.add_argument("--lya-num-integ-pixels", type=int, default=20)
    p.add_argument("--lya-delta-floor", type=float, default=1.0e-8)
    p.add_argument("--lya-temp-floor-k", type=float, default=1.0)
    p.add_argument("--lya-eos-grid-size", type=int, default=96)
    p.add_argument("--lya-treecool-file", type=str, default="diffhydro/nyx_eos/TREECOOL_middle")
    p.add_argument("--lya-eos-path", type=str, default="diffhydro/nyx_eos")
    p.add_argument("--lya-logdelta-min", type=float, default=-3.0)
    p.add_argument("--lya-logdelta-max", type=float, default=4.0)
    p.add_argument("--lya-logt-min", type=float, default=0.0)
    p.add_argument("--lya-logt-max", type=float, default=8.0)
    p.add_argument("--fgpa-tau-a", type=float, default=1.0)
    p.add_argument("--fgpa-tau-b", type=float, default=1.6)
    p.add_argument("--fgpa-rho-floor", type=float, default=1.0e-6)
    p.add_argument(
        "--lya-skewer-batch-size",
        type=int,
        default=256,
        help=(
            "Batch size for LyA LOS skewers in tau computation. Smaller values reduce peak "
            "memory during backprop at extra runtime."
        ),
    )
    p.add_argument(
        "--xray-cooling-table-npz",
        type=str,
        default="diffhydro_gadgetic_n128_z127to2_hll/nyx_cooling_table.npz",
    )
    p.add_argument("--xray-temp-floor-k", type=float, default=1.0)
    p.add_argument(
        "--xray-proxy-kind",
        choices=["cool", "net_cooling", "abs_net"],
        default="cool",
        help="X-ray proxy derived from Nyx cooling table rates.",
    )

    p.add_argument("--n-iters", type=int, default=20)
    p.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    p.add_argument("--adam-lr", type=float, default=1.5e-2)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--lbfgs-lr", type=float, default=1.0)
    p.add_argument("--lbfgs-memory", type=int, default=4)
    p.add_argument(
        "--lbfgs-linesearch",
        choices=["none", "zoom"],
        default="zoom",
        help=(
            "Line-search mode for L-BFGS. "
            "'zoom' matches Optax default behavior and often improves convergence; "
            "'none' disables line-search."
        ),
    )
    p.add_argument(
        "--lbfgs-max-linesearch-steps",
        type=int,
        default=15,
        help="Maximum line-search steps when --lbfgs-linesearch=zoom.",
    )
    p.add_argument(
        "--lbfgs-scale-init-precond",
        dest="lbfgs_scale_init_precond",
        action="store_true",
        default=True,
        help="Enable scaling of initial inverse-Hessian estimate in Optax L-BFGS.",
    )
    p.add_argument(
        "--no-lbfgs-scale-init-precond",
        dest="lbfgs_scale_init_precond",
        action="store_false",
        help="Disable scaling of initial inverse-Hessian estimate in Optax L-BFGS.",
    )
    p.add_argument(
        "--stop-grad-norm",
        type=float,
        default=0.0,
        help="Early stop when ||grad|| <= threshold (0 disables).",
    )
    p.add_argument(
        "--stop-rel-loss",
        type=float,
        default=0.0,
        help="Early stop when relative loss improvement drops below threshold (0 disables).",
    )
    p.add_argument(
        "--stop-patience",
        type=int,
        default=0,
        help=(
            "Patience for --stop-rel-loss. 0 means stop immediately when criterion is met; "
            ">0 requires this many consecutive non-improving iterations."
        ),
    )
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help=(
            "If >0, write full snapshot outputs every N iterations into "
            "<output-dir>/checkpoints/iter_XXXXXX. This is expensive."
        ),
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-random-scale", type=float, default=1.0)
    p.add_argument("--init-white-noise-npy", type=str, default=None)

    p.add_argument("--override-smooth-sigma", type=float, default=None)
    p.add_argument("--override-bias-linear", type=float, default=None)
    p.add_argument("--override-bias-quadratic", type=float, default=None)
    p.add_argument("--override-temp-heat-gain", type=float, default=None)
    p.add_argument("--override-temp-slope", type=float, default=None)
    p.add_argument("--override-temp-quadratic", type=float, default=None)

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cosmo_reconstruct/outputs/step2_3_optimize"),
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.ic_power_suppression < 0.0:
        raise ValueError("--ic-power-suppression must be >= 0.")
    n_iters = max(1, int(args.n_iters))
    save_every = max(0, int(args.save_every))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if args.xla_preallocate else "false"
    os.environ.setdefault("MPLBACKEND", "Agg")

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np
    import optax

    from src.diagnostics import (
        compute_power_and_cross,
        field_stats,
        fit_gas_mapping_from_reference,
        fit_temperature_mapping_from_reference,
        load_cv0_fields,
        load_reference_fields,
        make_forward_plots,
        make_observable_plots,
        plot_optimization_history,
        save_json,
    )
    from src.observable_utils import make_observable_mapper
    from src.power_loss import make_temperature_power_spectrum_loss
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

    if args.init_white_noise_npy is not None:
        params = jnp.asarray(np.load(args.init_white_noise_npy), dtype=jnp.float32)
        if params.shape != (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n):
            raise ValueError(
                f"Init white-noise shape mismatch: got {params.shape}, "
                f"expected {(cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)}"
            )
    else:
        key = jr.PRNGKey(args.seed)
        init_guess_amplitude_scale = float(np.sqrt(args.ic_power_suppression))
        params = (
            args.init_random_scale
            * init_guess_amplitude_scale
            * jr.normal(key, (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n), dtype=jnp.float32)
        )

    # Keep optimization and evaluation cosmology objects separate.
    cosmo_eval = build_cosmology(cfg)
    prime_growth_cache(cosmo_eval, a_init)
    run_forward_eval = jax.jit(lambda wn: forward_fields(wn, pk_sqrt, grid_pos, cosmo_eval, cfg, gas_params))
    use_dm_observable = str(args.observable).lower() == "fgpa_flux_dm"

    observable_fn, observable_meta = make_observable_mapper(
        observable=args.observable,
        projection=args.observable_projection,
        los_axis=args.los_axis,
        z_target=cfg.z_target,
        h=cfg.h,
        omega_m=cfg.omega_m,
        omega_b=cfg.omega_b,
        box_size_mpc_h=cfg.box_size_mpc_h,
        lya_num_integ_pixels=args.lya_num_integ_pixels,
        lya_delta_floor=args.lya_delta_floor,
        lya_temp_floor_k=args.lya_temp_floor_k,
        lya_eos_grid_size=args.lya_eos_grid_size,
        lya_treecool_file=args.lya_treecool_file,
        lya_logdelta_min=args.lya_logdelta_min,
        lya_logdelta_max=args.lya_logdelta_max,
        lya_logt_min=args.lya_logt_min,
        lya_logt_max=args.lya_logt_max,
        lya_eos_path=args.lya_eos_path,
        xray_cooling_table_npz=args.xray_cooling_table_npz,
        xray_temp_floor_k=args.xray_temp_floor_k,
        xray_proxy_kind=args.xray_proxy_kind,
        lya_skewer_batch_size=args.lya_skewer_batch_size,
        fgpa_tau_a=args.fgpa_tau_a,
        fgpa_tau_b=args.fgpa_tau_b,
        fgpa_rho_floor=args.fgpa_rho_floor,
    )

    target_source = str(args.target_source)
    if target_source == "cv0":
        if use_dm_observable and args.target_observable_npy is None:
            raise ValueError(
                "observable=fgpa_flux_dm with target_source=cv0 requires --target-observable-npy "
                "(or use --target-source=self_consistent) because CV0 DM target is unavailable."
            )
        if use_dm_observable and args.target_observable_npy is not None:
            print(
                "[objective] observable=fgpa_flux_dm + target_source=cv0: "
                "using CV0 gas field only for diagnostic placeholders."
            )
        target_rho_source = np.asarray(cv0_rho, dtype=np.float32)
        target_temp_source = np.asarray(cv0_temp, dtype=np.float32)
        target_white_noise = None
    elif target_source == "self_consistent":
        if args.self_target_white_noise_npy is not None:
            target_white_noise = np.asarray(np.load(args.self_target_white_noise_npy), dtype=np.float32)
            if target_white_noise.shape != (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n):
                raise ValueError(
                    f"Self-target white-noise shape mismatch: got {target_white_noise.shape}, "
                    f"expected {(cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)}"
                )
            target_wn = jnp.asarray(target_white_noise, dtype=jnp.float32)
        else:
            key_target = jr.PRNGKey(int(args.self_target_seed))
            target_wn = jnp.asarray(
                float(args.self_target_scale)
                * jr.normal(key_target, (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n), dtype=jnp.float32),
                dtype=jnp.float32,
            )
            target_white_noise = np.asarray(target_wn, dtype=np.float32)
        rho_dm_t, rho_gas_t, temp_t, _, _ = run_forward_eval(target_wn)
        rho_dm_t.block_until_ready()
        target_rho_source = np.asarray(rho_dm_t if use_dm_observable else rho_gas_t, dtype=np.float32)
        target_temp_source = np.asarray(temp_t, dtype=np.float32)
    else:
        raise ValueError(f"Unknown target_source={args.target_source}")

    target_obs_j = observable_fn(
        jnp.asarray(target_rho_source, dtype=jnp.float32),
        jnp.asarray(target_temp_source, dtype=jnp.float32),
        None,
    )
    target_obs_j.block_until_ready()
    target_obs_pre_noise = np.asarray(target_obs_j, dtype=np.float32)
    target_obs = np.asarray(target_obs_pre_noise, dtype=np.float32)

    def _load_target_observable(path: str) -> np.ndarray:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            return np.asarray(loaded, dtype=np.float32)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                for key in ("target_observable", "target_observable_pre_noise", "arr_0"):
                    if key in loaded:
                        return np.asarray(loaded[key], dtype=np.float32)
                raise ValueError(
                    f"Could not find a target observable array in {path}. "
                    f"Tried keys: target_observable, target_observable_pre_noise, arr_0. "
                    f"Available keys: {loaded.files}"
                )
            finally:
                loaded.close()
        raise TypeError(f"Unsupported target observable format at {path}")

    if args.target_observable_npy is not None:
        loaded_target_obs = _load_target_observable(args.target_observable_npy)
        if loaded_target_obs.shape != target_obs.shape:
            raise ValueError(
                f"target_observable_npy shape mismatch: got {loaded_target_obs.shape}, expected {target_obs.shape}"
            )
        target_obs = loaded_target_obs
        if float(args.self_target_noise_sigma) > 0.0:
            print("[objective] --target-observable-npy provided; ignoring --self-target-noise-sigma.")
    elif target_source == "self_consistent" and float(args.self_target_noise_sigma) > 0.0:
        rng = np.random.default_rng(int(args.self_target_seed) + 17)
        if str(args.compare_space).lower() == "log":
            noisy = np.log(np.clip(target_obs, 1.0e-20, None)) + float(args.self_target_noise_sigma) * rng.standard_normal(
                size=target_obs.shape
            )
            target_obs = np.exp(noisy).astype(np.float32)
        else:
            target_obs = (
                target_obs + float(args.self_target_noise_sigma) * rng.standard_normal(size=target_obs.shape)
            ).astype(np.float32)
            #target_obs = np.maximum(target_obs, 0.0).astype(np.float32)

    print(
        f"[objective] observable={args.observable} projection={args.observable_projection} "
        f"target_source={target_source} target_shape={target_obs.shape}"
    )
    if target_source == "self_consistent":
        print(
            f"[objective] self-consistent target prepared "
            f"(noise_sigma={float(args.self_target_noise_sigma):.3e})"
        )

    temp_ps_loss_weight = float(max(0.0, args.temp_ps_loss_weight))
    temp_ps_loss_fn = None
    temp_ps_loss_meta: dict[str, object] = {
        "enabled": False,
        "weight": temp_ps_loss_weight,
    }
    if temp_ps_loss_weight > 0.0:
        temp_ps_loss_fn, ps_meta = make_temperature_power_spectrum_loss(
            np.asarray(target_temp_source, dtype=np.float32),
            box_size_mpc_h=float(cfg.box_size_mpc_h),
            n_bins=int(max(4, args.temp_ps_loss_nbins)),
            field_space=str(args.temp_ps_loss_space),
            eps=float(args.temp_ps_loss_eps),
        )
        temp_ps_loss_meta = dict(ps_meta)
        temp_ps_loss_meta["weight"] = temp_ps_loss_weight
        print(
            "[objective] enabled temperature power-spectrum loss: "
            f"weight={temp_ps_loss_weight:.3e}, bins={int(max(4, args.temp_ps_loss_nbins))}, "
            f"space={str(args.temp_ps_loss_space)}"
        )

    target_obs_jax = jnp.asarray(target_obs, dtype=jnp.float32)
    sigma = jnp.maximum(jnp.asarray(float(args.noise_sigma), dtype=jnp.float32), 1.0e-8)
    prior_w = jnp.asarray(float(args.prior_weight), dtype=jnp.float32)

    if str(args.compare_space).lower() == "log":

        def _to_compare(x):
            return jnp.log(jnp.clip(jnp.asarray(x, dtype=jnp.float32), 1.0e-20, None))

    elif str(args.compare_space).lower() == "linear":

        def _to_compare(x):
            return jnp.asarray(x, dtype=jnp.float32)

    else:
        raise ValueError(f"Unknown compare_space={args.compare_space}")

    def nlogpost_terms(wn):
        rho_dm, rho_gas, temp_gas, _, _ = forward_fields(wn, pk_sqrt, grid_pos, cosmo_dyn, cfg, gas_params)
        obs_rho = rho_dm if use_dm_observable else rho_gas
        pred_obs = observable_fn(obs_rho, temp_gas, None)
        resid = _to_compare(pred_obs) - _to_compare(target_obs_jax)
        data_nll = 0.5 * jnp.mean((resid / sigma) ** 2)
        if temp_ps_loss_fn is not None:
            data_nll = data_nll + jnp.asarray(temp_ps_loss_weight, dtype=jnp.float32) * temp_ps_loss_fn(temp_gas)
        prior_nll = 0.5 * prior_w * jnp.mean(wn**2)
        loss = data_nll + prior_nll
        return loss, (data_nll, prior_nll)

    value_grad_fn = jax.value_and_grad(nlogpost_terms, has_aux=True)

    def write_outputs(
        write_dir: Path,
        params_snapshot,
        history_snapshot,
        *,
        stop_reason_snapshot: str | None,
        optimization_wall_s_snapshot: float,
        total_elapsed_s_snapshot: float,
        is_periodic_snapshot: bool,
    ):
        write_dir.mkdir(parents=True, exist_ok=True)

        rho_dm_j, rho_gas_j, temp_j, _, init_mesh_j = run_forward_eval(params_snapshot)
        rho_dm_j.block_until_ready()

        rho_dm = np.asarray(rho_dm_j, dtype=np.float32)
        rho_gas = np.asarray(rho_gas_j, dtype=np.float32)
        temp_gas = np.asarray(temp_j, dtype=np.float32)
        obs_rho_j = rho_dm_j if use_dm_observable else rho_gas_j
        pred_obs_j = observable_fn(obs_rho_j, temp_j, None)
        pred_obs = np.asarray(pred_obs_j, dtype=np.float32)
        init_mesh = np.asarray(init_mesh_j, dtype=np.float32)
        history_np = np.asarray(history_snapshot, dtype=np.float64)
        temp_ps_loss_value = None
        if temp_ps_loss_fn is not None:
            temp_ps_loss_value = float(temp_ps_loss_fn(jnp.asarray(temp_gas, dtype=jnp.float32)))
        temp_ps_k_centers = np.asarray(temp_ps_loss_meta.get("k_centers", np.asarray([], dtype=np.float32)))
        temp_ps_target_logpk = np.asarray(temp_ps_loss_meta.get("target_logpk", np.asarray([], dtype=np.float32)))

        spectra_cv0 = compute_power_and_cross(cv0_rho, rho_gas, cfg.box_size_mpc_h)
        spectra_ref = compute_power_and_cross(ref_gas, rho_gas, cfg.box_size_mpc_h)

        make_forward_plots(
            write_dir,
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
        spectra_obs = make_observable_plots(
            write_dir,
            target_obs=target_obs,
            pred_obs=pred_obs,
            observable_name=args.observable,
            compare_space=args.compare_space,
            box_size_mpc_h=cfg.box_size_mpc_h,
        )
        plot_optimization_history(write_dir / "optimization_history.png", history_np)

        stats = {
            "run": {
                "mesh_n": cfg.mesh_n,
                "box_size_mpc_h": cfg.box_size_mpc_h,
                "z_init": cfg.z_init,
                "z_target": cfg.z_target,
                "kdk_steps": cfg.kdk_steps,
                "ic_power_suppression": float(args.ic_power_suppression),
                "n_iters": n_iters,
                "n_iters_completed": int(history_np[-1, 0]),
                "optimizer": args.optimizer,
                "adam_lr": args.adam_lr,
                "lbfgs_lr": args.lbfgs_lr,
                "lbfgs_memory": args.lbfgs_memory,
                "lbfgs_linesearch": args.lbfgs_linesearch,
                "lbfgs_max_linesearch_steps": args.lbfgs_max_linesearch_steps,
                "lbfgs_scale_init_precond": bool(args.lbfgs_scale_init_precond),
                "stop_grad_norm": args.stop_grad_norm,
                "stop_rel_loss": args.stop_rel_loss,
                "stop_patience": args.stop_patience,
                "stop_reason": stop_reason_snapshot,
                "noise_sigma": args.noise_sigma,
                "prior_weight": args.prior_weight,
                "temp_ps_loss_weight": temp_ps_loss_weight,
                "temp_ps_loss_nbins": int(max(4, args.temp_ps_loss_nbins)),
                "temp_ps_loss_space": str(args.temp_ps_loss_space),
                "temp_ps_loss_eps": float(args.temp_ps_loss_eps),
                "compare_space": args.compare_space,
                "observable": args.observable,
                "observable_projection": args.observable_projection,
                "los_axis": int(args.los_axis),
                "target_source": target_source,
                "target_observable_npy": args.target_observable_npy,
                "self_target_noise_sigma": float(args.self_target_noise_sigma),
                "compile_and_first_step_s": compile_and_first_s,
                "optimization_wall_s": optimization_wall_s_snapshot,
                "total_elapsed_s": total_elapsed_s_snapshot,
                "save_every": save_every,
                "is_periodic_snapshot": bool(is_periodic_snapshot),
                "snapshot_iteration": int(history_np[-1, 0]),
            },
            "gas_mapping_fit": gas_fit,
            "temperature_mapping_fit": temp_fit,
            "observable_meta": observable_meta,
            "temperature_power_loss": {
                "enabled": bool(temp_ps_loss_fn is not None),
                "weight": temp_ps_loss_weight,
                "field_space": str(args.temp_ps_loss_space),
                "n_bins": int(max(4, args.temp_ps_loss_nbins)),
                "eps": float(args.temp_ps_loss_eps),
                "k_min": float(temp_ps_loss_meta.get("k_min", 0.0)),
                "k_max": float(temp_ps_loss_meta.get("k_max", 0.0)),
                "value": temp_ps_loss_value,
            },
            "gas_model_params_used": {
                "smooth_sigma_cells": gas_params.smooth_sigma_cells,
                "bias_linear": gas_params.bias_linear,
                "bias_quadratic": gas_params.bias_quadratic,
                "temp_init_k": gas_params.temp_init_k,
                "temp_heat_gain": gas_params.temp_heat_gain,
                "temp_slope": gas_params.temp_slope,
                "temp_quadratic": gas_params.temp_quadratic,
            },
            "loss": {
                "initial_total": float(history_np[0, 1]),
                "final_total": float(history_np[-1, 1]),
                "initial_data": float(history_np[0, 2]),
                "final_data": float(history_np[-1, 2]),
                "initial_prior": float(history_np[0, 3]),
                "final_prior": float(history_np[-1, 3]),
                "initial_grad_norm": float(history_np[0, 4]),
                "final_grad_norm": float(history_np[-1, 4]),
            },
            "stats": {
                "cv0_rho": field_stats(cv0_rho),
                "reference_rho": field_stats(ref_gas),
                "model_rho": field_stats(rho_gas),
                "cv0_temp": field_stats(cv0_temp),
                "reference_temp": field_stats(ref_temp),
                "model_temp": field_stats(temp_gas),
                "model_dm": field_stats(rho_dm),
                "target_observable_pre_noise": field_stats(target_obs_pre_noise),
                "target_observable": field_stats(target_obs),
                "model_observable": field_stats(pred_obs),
            },
            "spectra": {
                "cv0_vs_model_median_cross": float(spectra_cv0["median_cross"]),
                "reference_vs_model_median_cross": float(spectra_ref["median_cross"]),
                "observable_target_vs_model_median_cross": (
                    None if spectra_obs is None else float(spectra_obs["median_cross"])
                ),
            },
        }

        save_json(write_dir / "optimize_stats.json", stats)
        np.savez_compressed(
            write_dir / "optimize_outputs.npz",
            history=history_np,
            white_noise=np.asarray(params_snapshot, dtype=np.float32),
            init_mesh=init_mesh,
            cv0_rho=cv0_rho,
            cv0_temp=cv0_temp,
            ref_rho=ref_gas,
            ref_temp=ref_temp,
            model_dm=rho_dm,
            model_rho=rho_gas,
            model_temp=temp_gas,
            target_rho_objective=np.asarray(target_rho_source, dtype=np.float32),
            target_temp_objective=np.asarray(target_temp_source, dtype=np.float32),
            target_observable_pre_noise=np.asarray(target_obs_pre_noise, dtype=np.float32),
            target_observable=np.asarray(target_obs, dtype=np.float32),
            model_observable=np.asarray(pred_obs, dtype=np.float32),
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
            temp_ps_k_centers=temp_ps_k_centers.astype(np.float32),
            temp_ps_target_logpk=temp_ps_target_logpk.astype(np.float32),
            temp_ps_loss_value=np.asarray(np.nan if temp_ps_loss_value is None else temp_ps_loss_value, dtype=np.float32),
        )
        np.savez_compressed(
            write_dir / "fields_ic_final.npz",
            gas_dh_final=rho_gas,
            temp_dh_final=temp_gas,
            dm_dh_final=rho_dm,
            gas_dh_ic=init_mesh,
        )
        np.save(write_dir / "optimized_white_noise.npy", np.asarray(params_snapshot, dtype=np.float32))
        if target_white_noise is not None:
            np.save(write_dir / "target_white_noise.npy", np.asarray(target_white_noise, dtype=np.float32))
        return stats

    if args.optimizer == "adam":
        if args.grad_clip_norm > 0:
            opt = optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optax.adam(args.adam_lr))
        else:
            opt = optax.adam(args.adam_lr)
    elif args.optimizer == "lbfgs":
        if args.grad_clip_norm > 0:
            print("Note: --grad-clip-norm is ignored for optimizer=lbfgs.")
        if args.lbfgs_linesearch == "zoom":
            linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=args.lbfgs_max_linesearch_steps)
        else:
            linesearch = None
        opt = optax.lbfgs(
            learning_rate=args.lbfgs_lr,
            memory_size=args.lbfgs_memory,
            scale_init_precond=bool(args.lbfgs_scale_init_precond),
            linesearch=linesearch,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    opt_state = opt.init(params)

    if args.optimizer == "adam":

        @jax.jit
        def step(p, s):
            (loss, (data_nll, prior_nll)), grads = value_grad_fn(p)
            updates, s_new = opt.update(grads, s, p)
            p_new = optax.apply_updates(p, updates)
            grad_norm = optax.global_norm(grads)
            return p_new, s_new, loss, data_nll, prior_nll, grad_norm

    else:

        def value_only_fn(x):
            return nlogpost_terms(x)[0]

        @jax.jit
        def step(p, s):
            (loss, (data_nll, prior_nll)), grads = value_grad_fn(p)
            if args.lbfgs_linesearch == "zoom":
                updates, s_new = opt.update(
                    grads,
                    s,
                    p,
                    value=loss,
                    grad=grads,
                    value_fn=value_only_fn,
                )
            else:
                updates, s_new = opt.update(grads, s, p)
            p_new = optax.apply_updates(p, updates)
            grad_norm = optax.global_norm(grads)
            return p_new, s_new, loss, data_nll, prior_nll, grad_norm

    history = []
    stop_reason: str | None = None
    compile_and_first_s = 0.0

    t_startup_eval = time.perf_counter()
    startup_loss, (startup_data, startup_prior) = nlogpost_terms(params)
    jax.block_until_ready(startup_loss)
    startup_eval_s = time.perf_counter() - t_startup_eval
    startup_history = [[0, float(startup_loss), float(startup_data), float(startup_prior), 0.0]]
    startup_dir = out_dir / "checkpoints" / "iter_000000"
    _ = write_outputs(
        startup_dir,
        params,
        startup_history,
        stop_reason_snapshot="startup_initial_guess",
        optimization_wall_s_snapshot=0.0,
        total_elapsed_s_snapshot=time.perf_counter() - t0,
        is_periodic_snapshot=True,
    )
    print(f"[snapshot] wrote startup artifacts: {startup_dir} (eval {startup_eval_s:.2f}s)")

    t_compile = time.perf_counter()
    params, opt_state, loss, data_nll, prior_nll, grad_norm = step(params, opt_state)
    jax.block_until_ready(loss)
    compile_and_first_s = time.perf_counter() - t_compile
    history.append([1, float(loss), float(data_nll), float(prior_nll), float(grad_norm)])
    print(
        f"[{args.optimizer}] iter=1/{n_iters} loss={float(loss):.6e} data={float(data_nll):.6e} "
        f"prior={float(prior_nll):.6e} |grad|={float(grad_norm):.6e} "
        f"(compile+step {compile_and_first_s:.2f}s)"
    )

    t_opt = time.perf_counter()
    last_loss = float(loss)
    no_improve_count = 0

    if save_every > 0 and n_iters > 1 and (1 % save_every == 0):
        ckpt_dir = out_dir / "checkpoints" / "iter_000001"
        _ = write_outputs(
            ckpt_dir,
            params,
            history,
            stop_reason_snapshot=stop_reason,
            optimization_wall_s_snapshot=0.0,
            total_elapsed_s_snapshot=time.perf_counter() - t0,
            is_periodic_snapshot=True,
        )
        print(f"[snapshot] wrote periodic artifacts: {ckpt_dir}")

    for i in range(2, n_iters + 1):
        params, opt_state, loss, data_nll, prior_nll, grad_norm = step(params, opt_state)
        loss_f = float(loss)
        grad_norm_f = float(grad_norm)
        rel_improve = (last_loss - loss_f) / max(abs(last_loss), 1.0e-12)

        if args.log_every > 0 and (i % args.log_every == 0 or i == n_iters):
            print(
                f"[{args.optimizer}] iter={i}/{n_iters} loss={loss_f:.6e} data={float(data_nll):.6e} "
                f"prior={float(prior_nll):.6e} |grad|={grad_norm_f:.6e} rel_improve={rel_improve:.3e}"
            )
        history.append([i, loss_f, float(data_nll), float(prior_nll), grad_norm_f])

        if args.stop_grad_norm > 0.0 and grad_norm_f <= args.stop_grad_norm:
            stop_reason = f"grad_norm <= {args.stop_grad_norm}"
            break

        if args.stop_rel_loss > 0.0:
            if rel_improve < args.stop_rel_loss:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if args.stop_patience <= 0 and rel_improve < args.stop_rel_loss:
                stop_reason = f"rel_improve < {args.stop_rel_loss}"
                break
            if args.stop_patience > 0 and no_improve_count >= args.stop_patience:
                stop_reason = (
                    f"rel_improve < {args.stop_rel_loss} for {args.stop_patience} consecutive iterations"
                )
                break

        if save_every > 0 and (i % save_every == 0) and (i < n_iters):
            ckpt_dir = out_dir / "checkpoints" / f"iter_{i:06d}"
            _ = write_outputs(
                ckpt_dir,
                params,
                history,
                stop_reason_snapshot=stop_reason,
                optimization_wall_s_snapshot=time.perf_counter() - t_opt,
                total_elapsed_s_snapshot=time.perf_counter() - t0,
                is_periodic_snapshot=True,
            )
            print(f"[snapshot] wrote periodic artifacts: {ckpt_dir}")

        last_loss = loss_f
    opt_wall_s = time.perf_counter() - t_opt
    if stop_reason is not None:
        print(f"[{args.optimizer}] early stop at iter={int(history[-1][0])}: {stop_reason}")
    stats = write_outputs(
        out_dir,
        params,
        history,
        stop_reason_snapshot=stop_reason,
        optimization_wall_s_snapshot=opt_wall_s,
        total_elapsed_s_snapshot=time.perf_counter() - t0,
        is_periodic_snapshot=False,
    )

    print(f"Saved outputs to: {out_dir}")
    print(
        f"Loss: initial={stats['loss']['initial_total']:.6e}, final={stats['loss']['final_total']:.6e} | "
        f"data initial/final={stats['loss']['initial_data']:.6e}/{stats['loss']['final_data']:.6e}"
    )
    print(
        f"Model rho mean/std={stats['stats']['model_rho']['mean']:.5f}/{stats['stats']['model_rho']['std']:.5f} | "
        f"CV0 rho mean/std={stats['stats']['cv0_rho']['mean']:.5f}/{stats['stats']['cv0_rho']['std']:.5f}"
    )
    print(
        f"Median cross r(k): CV0 vs model={stats['spectra']['cv0_vs_model_median_cross']:.4f}, "
        f"reference vs model={stats['spectra']['reference_vs_model_median_cross']:.4f}"
    )
    print(
        f"Observable ({args.observable}) mean/std target={stats['stats']['target_observable']['mean']:.5e}/"
        f"{stats['stats']['target_observable']['std']:.5e}, model={stats['stats']['model_observable']['mean']:.5e}/"
        f"{stats['stats']['model_observable']['std']:.5e}"
    )


if __name__ == "__main__":
    main()
