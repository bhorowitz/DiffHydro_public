#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Optimize white-noise ICs against CV0 (or a self-consistent synthetic target) "
            "in a chosen observable space using full differentiable JaxPM+DiffHydro coupled "
            "evolution (hydro + gravity)."
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
            "diffhydro_gadgetic_n128_z127to2_hllcd/snapshots/fields_ic_final.npz"
        ),
    )

    p.add_argument("--mesh-n", type=int, default=128)
    p.add_argument("--box-size-mpc-h", type=float, default=25.0)
    p.add_argument("--z-init", type=float, default=127.0)
    p.add_argument("--z-target", type=float, default=2.0)
    p.add_argument("--lpt-order", type=int, choices=[1, 2], default=1)

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

    p.add_argument("--hydro-steps", type=int, default=128)
    p.add_argument("--dtau-min", type=float, default=2.0e-7)
    p.add_argument("--dtau-max", type=float, default=8.0e-2)
    p.add_argument("--solver", choices=["hll", "hllc", "lf", "laxfriedrichs", "nyx"], default="hllc")
    p.add_argument("--state-floor", type=float, default=6.0e-8)
    p.add_argument("--pressure-floor", type=float, default=6.0e-8)
    p.add_argument("--hydro-temp-floor-k", type=float, default=0.0)
    p.add_argument("--force-eps", type=float, default=1.0e-8)
    p.add_argument("--checkpoint", dest="checkpoint", action="store_true", default=True)
    p.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Hydro rematerialization block size. >1 uses block checkpointing in the hydro scan "
            "to reduce backprop memory at extra recompute cost."
        ),
    )
    p.add_argument("--dual-energy", dest="dual_energy", action="store_true", default=True)
    p.add_argument("--no-dual-energy", dest="dual_energy", action="store_false")

    p.add_argument("--temperature0", type=float, default=1.0e4)
    p.add_argument("--temperature-gamma", type=float, default=2.0 / 3.0)
    p.add_argument("--gas-mean-fraction", type=float, default=1.58e-1)
    p.add_argument("--dm-kick-scale", type=float, default=1.0)
    p.add_argument("--gas-kick-scale", type=float, default=1.0)
    p.add_argument("--gas-kick-factor", type=float, default=None)
    p.add_argument("--enable-cooling", action="store_true", default=False)
    p.add_argument("--cooling-model", choices=["legacy", "nyx_table"], default="nyx_table")
    p.add_argument("--cooling-stop-gradient", dest="cooling_stop_gradient", action="store_true", default=True)
    p.add_argument("--no-cooling-stop-gradient", dest="cooling_stop_gradient", action="store_false")
    p.add_argument("--hydro-flux-stop-gradient", dest="hydro_flux_stop_gradient", action="store_true", default=False,
                   help="Stop gradient through the Riemann solver (eliminates ~5-20 GiB backward memory at 128^3).")
    p.add_argument("--gravity-stop-gradient-source", dest="gravity_stop_gradient_source", action="store_true", default=False)
    p.add_argument("--cooling-table", type=str, default="data/m-00.cie")
    p.add_argument("--heating-rate-per-h", type=float, default=1.0e-33)
    p.add_argument("--nyx-heating-scale", type=float, default=1.2)
    p.add_argument("--cooling-rate-scale", type=float, default=1.0)
    p.add_argument("--cooling-temp-floor-k", type=float, default=1.0)
    p.add_argument("--cooling-subcycles", type=int, default=8)
    p.add_argument("--cooling-dtmax-s", type=float, default=1.0e16)
    p.add_argument("--nyx-cooling-table-npz", type=str, default="data/nyx_cooling_table.npz")
    p.add_argument("--nyx-cooling-treecool", type=str, default="diffhydro/nyx_eos/TREECOOL_middle")
    p.add_argument(
        "--nyx-cooling-z-nodes",
        type=str,
        default="2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100",
    )
    p.add_argument("--nyx-cooling-logdelta-min", type=float, default=-3.0)
    p.add_argument("--nyx-cooling-logdelta-max", type=float, default=3.0)
    p.add_argument("--nyx-cooling-logdelta-n", type=int, default=96)
    p.add_argument("--nyx-cooling-logt-min", type=float, default=0.0)
    p.add_argument("--nyx-cooling-logt-max", type=float, default=8.0)
    p.add_argument("--nyx-cooling-logt-n", type=int, default=120)
    p.add_argument("--nyx-cooling-rebuild", dest="nyx_cooling_rebuild", action="store_true", default=False)
    p.add_argument("--no-nyx-cooling-rebuild", dest="nyx_cooling_rebuild", action="store_false")
    p.add_argument("--nyx-cooling-eos-path", type=str, default="diffhydro/nyx_eos")
    p.add_argument("--nyx-auto-rho-unit", dest="nyx_auto_rho_unit", action="store_true", default=False)
    p.add_argument("--no-nyx-auto-rho-unit", dest="nyx_auto_rho_unit", action="store_false")
    p.add_argument("--tau-time-unit-s", type=float, default=3.085677581e19)
    p.add_argument("--rho-unit-cgs", type=float, default=1.6e-24)
    p.add_argument("--vel-unit-cms", type=float, default=1.0e7)
    p.add_argument("--mu-hydrogen", type=float, default=1.0)
    p.add_argument("--h-species", type=float, default=0.76)

    p.add_argument("--compare-space", choices=["log", "linear"], default="log")
    p.add_argument("--noise-sigma", type=float, default=0.05)
    p.add_argument("--prior-weight", type=float, default=1.0)
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
        choices=["density", "lya_flux", "xray_proxy"],
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
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip optimization and only evaluate/save first-guess diagnostics.",
    )
    p.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    p.add_argument("--adam-lr", type=float, default=1.0e-3)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--sanitize-nonfinite-grads", dest="sanitize_nonfinite_grads", action="store_true", default=True)
    p.add_argument("--no-sanitize-nonfinite-grads", dest="sanitize_nonfinite_grads", action="store_false")
    p.add_argument("--stop-on-nonfinite-grads", dest="stop_on_nonfinite_grads", action="store_true", default=True)
    p.add_argument("--no-stop-on-nonfinite-grads", dest="stop_on_nonfinite_grads", action="store_false")
    p.add_argument("--lbfgs-lr", type=float, default=1.0)
    p.add_argument("--lbfgs-memory", type=int, default=10)
    p.add_argument(
        "--lbfgs-linesearch",
        choices=["none", "zoom"],
        default="zoom",
        help="L-BFGS line-search mode. 'zoom' usually converges faster near minima.",
    )
    p.add_argument("--lbfgs-max-linesearch-steps", type=int, default=15)
    p.add_argument(
        "--lbfgs-scale-init-precond",
        dest="lbfgs_scale_init_precond",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-lbfgs-scale-init-precond",
        dest="lbfgs_scale_init_precond",
        action="store_false",
    )
    p.add_argument(
        "--max-rollbacks",
        type=int,
        default=5,
        help=(
            "Maximum automatic rollbacks on unstable iterations. "
            "Set <=0 to disable rollback and preserve legacy behavior."
        ),
    )
    p.add_argument(
        "--rollback-loss-factor",
        type=float,
        default=4.0,
        help=(
            "Rollback when current loss exceeds this factor times the previous accepted loss. "
            "Set <=0 to disable loss-spike rollback."
        ),
    )
    p.add_argument("--stop-grad-norm", type=float, default=0.0)
    p.add_argument("--stop-rel-loss", type=float, default=0.0)
    p.add_argument("--stop-patience", type=int, default=0)
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
    p.add_argument(
        "--init-white-noise-npy",
        type=str,
        default=None,
        help="Warm-start white-noise file, e.g. simple model's optimized_white_noise.npy",
    )
    p.add_argument(
        "--allow-init-config-mismatch",
        action="store_true",
        default=False,
        help=(
            "Allow loading init white-noise from a full-hydro checkpoint with different "
            "simulation settings. By default, mismatches raise an error."
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cosmo_reconstruct/outputs/step2_3_optimize_full_hydro"),
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.ic_power_suppression < 0.0:
        raise ValueError("--ic-power-suppression must be >= 0.")
    n_iters_requested = max(0, int(args.n_iters))
    dry_run = bool(args.dry_run or n_iters_requested == 0)
    n_iters = n_iters_requested
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
        load_cv0_fields,
        load_reference_fields,
        make_forward_plots,
        make_observable_plots,
        plot_optimization_history,
        save_json,
    )
    from src.observable_utils import make_observable_mapper
    from src.power_loss import make_temperature_power_spectrum_loss
    from src.forward_model import make_lattice_positions, make_pk_sqrt
    from src.full_hydro_model import (
        FullHydroConfig,
        build_full_hydro_system,
        build_lpt_cosmology,
        forward_fields_full_hydro,
        prime_system_growth_cache,
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

    cfg = FullHydroConfig(
        mesh_n=args.mesh_n,
        box_size_mpc_h=args.box_size_mpc_h,
        z_init=args.z_init,
        z_target=args.z_target,
        lpt_order=args.lpt_order,
        omega_m=args.omega_m,
        omega_b=args.omega_b,
        h=args.h,
        n_s=args.n_s,
        sigma8=args.sigma8,
        hydro_steps=args.hydro_steps,
        dtau_min=args.dtau_min,
        dtau_max=args.dtau_max,
        solver=str(args.solver),
        state_floor=args.state_floor,
        pressure_floor=args.pressure_floor,
        hydro_temp_floor_k=args.hydro_temp_floor_k,
        force_eps=args.force_eps,
        checkpoint=bool(args.checkpoint),
        checkpoint_every=max(1, int(args.checkpoint_every)),
        dual_energy=bool(args.dual_energy),
        temperature0=args.temperature0,
        temperature_gamma=args.temperature_gamma,
        gas_mean_fraction=args.gas_mean_fraction,
        dm_kick_scale=args.dm_kick_scale,
        gas_kick_scale=args.gas_kick_scale,
        gas_kick_factor=args.gas_kick_factor,
        hydro_flux_stop_gradient=bool(args.hydro_flux_stop_gradient),
        gravity_stop_gradient_source=bool(args.gravity_stop_gradient_source),
        enable_cooling=bool(args.enable_cooling),
        cooling_model=str(args.cooling_model),
        cooling_stop_gradient=bool(args.cooling_stop_gradient),
        cooling_table=str(args.cooling_table),
        heating_rate_per_h=args.heating_rate_per_h,
        nyx_heating_scale=args.nyx_heating_scale,
        cooling_rate_scale=args.cooling_rate_scale,
        cooling_temp_floor_k=args.cooling_temp_floor_k,
        cooling_subcycles=args.cooling_subcycles,
        cooling_dtmax_s=args.cooling_dtmax_s,
        nyx_cooling_table_npz=str(args.nyx_cooling_table_npz),
        nyx_cooling_treecool=str(args.nyx_cooling_treecool),
        nyx_cooling_z_nodes=str(args.nyx_cooling_z_nodes),
        nyx_cooling_logdelta_min=args.nyx_cooling_logdelta_min,
        nyx_cooling_logdelta_max=args.nyx_cooling_logdelta_max,
        nyx_cooling_logdelta_n=args.nyx_cooling_logdelta_n,
        nyx_cooling_logt_min=args.nyx_cooling_logt_min,
        nyx_cooling_logt_max=args.nyx_cooling_logt_max,
        nyx_cooling_logt_n=args.nyx_cooling_logt_n,
        nyx_cooling_rebuild=bool(args.nyx_cooling_rebuild),
        nyx_cooling_eos_path=str(args.nyx_cooling_eos_path),
        nyx_auto_rho_unit=bool(args.nyx_auto_rho_unit),
        tau_time_unit_s=args.tau_time_unit_s,
        rho_unit_cgs=args.rho_unit_cgs,
        vel_unit_cms=args.vel_unit_cms,
        mu_hydrogen=args.mu_hydrogen,
        h_species=args.h_species,
    )

    cosmo_pk = build_lpt_cosmology(cfg)
    cosmo_dyn = build_lpt_cosmology(cfg)
    system = build_full_hydro_system(cfg, cosmo_dyn)
    prime_system_growth_cache(system, cfg)

    grid_pos = make_lattice_positions(cfg.mesh_n)
    pk_sqrt = make_pk_sqrt(cosmo_pk, cfg)

    if args.init_white_noise_npy is not None:
        init_wn_path = Path(args.init_white_noise_npy)
        init_stats_path = init_wn_path.with_name("optimize_stats.json")
        if init_stats_path.exists():
            try:
                with init_stats_path.open("r", encoding="utf-8") as f:
                    init_stats = json.load(f)
            except Exception as exc:  # pragma: no cover - defensive IO path
                print(f"[init-checkpoint] warning: failed to read {init_stats_path}: {exc}")
                init_stats = None

            run_cfg = init_stats.get("run", {}) if isinstance(init_stats, dict) else {}
            # Enforce strict checks only for full-hydro checkpoints.
            if isinstance(run_cfg, dict) and ("hydro_steps" in run_cfg):
                missing = object()
                checks = [
                    ("mesh_n", cfg.mesh_n),
                    ("box_size_mpc_h", cfg.box_size_mpc_h),
                    ("z_init", cfg.z_init),
                    ("z_target", cfg.z_target),
                    ("lpt_order", cfg.lpt_order),
                    ("hydro_steps", cfg.hydro_steps),
                    ("checkpoint_every", cfg.checkpoint_every),
                    ("solver", cfg.solver),
                    ("dual_energy", bool(cfg.dual_energy)),
                    ("temperature0", cfg.temperature0),
                    ("temperature_gamma", cfg.temperature_gamma),
                    ("gas_mean_fraction", cfg.gas_mean_fraction),
                    ("dm_kick_scale", cfg.dm_kick_scale),
                    ("gas_kick_scale", cfg.gas_kick_scale),
                    ("gas_kick_factor", cfg.gas_kick_factor),
                    ("enable_cooling", bool(cfg.enable_cooling)),
                    ("cooling_model", cfg.cooling_model),
                    ("cooling_stop_gradient", bool(cfg.cooling_stop_gradient)),
                    ("rho_unit_cgs", cfg.rho_unit_cgs),
                    ("vel_unit_cms", cfg.vel_unit_cms),
                    ("tau_time_unit_s", cfg.tau_time_unit_s),
                    ("nyx_cooling_table_npz", str(cfg.nyx_cooling_table_npz)),
                    ("nyx_cooling_treecool", str(cfg.nyx_cooling_treecool)),
                    ("nyx_cooling_z_nodes", str(cfg.nyx_cooling_z_nodes)),
                    ("nyx_heating_scale", cfg.nyx_heating_scale),
                    ("cooling_rate_scale", cfg.cooling_rate_scale),
                ]
                mismatches: list[tuple[str, object, object]] = []
                for key, current_value in checks:
                    saved_value = run_cfg.get(key, missing)
                    if saved_value is missing:
                        continue
                    if isinstance(current_value, float) and isinstance(saved_value, (float, int)):
                        if not bool(np.isclose(current_value, float(saved_value), rtol=1.0e-6, atol=1.0e-8)):
                            mismatches.append((key, current_value, saved_value))
                    else:
                        if current_value != saved_value:
                            mismatches.append((key, current_value, saved_value))

                if mismatches:
                    diff_lines = "\n".join(
                        f"  - {k}: current={cv!r}, checkpoint={sv!r}" for (k, cv, sv) in mismatches
                    )
                    msg = (
                        f"[init-checkpoint] full-hydro config mismatch for {init_wn_path} "
                        f"(compared to {init_stats_path}):\n{diff_lines}"
                    )
                    if args.allow_init_config_mismatch:
                        print(f"{msg}\n[init-checkpoint] continuing because --allow-init-config-mismatch is set.")
                    else:
                        raise ValueError(
                            f"{msg}\nPass --allow-init-config-mismatch to override."
                        )
                else:
                    print(f"[init-checkpoint] config matches source checkpoint: {init_stats_path}")
            else:
                print(
                    f"[init-checkpoint] info: {init_stats_path} does not look like a full-hydro checkpoint; "
                    "skipping strict compatibility checks."
                )
        else:
            print(
                f"[init-checkpoint] info: {init_stats_path} not found; "
                "skipping checkpoint compatibility checks."
            )

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

    # Keep optimization and evaluation system objects separated.
    cosmo_eval = build_lpt_cosmology(cfg)
    system_eval = build_full_hydro_system(cfg, cosmo_eval)
    prime_system_growth_cache(system_eval, cfg)
    run_forward_eval = jax.jit(
        lambda wn: forward_fields_full_hydro(
            wn,
            pk_sqrt,
            grid_pos,
            system_eval,
            cfg,
        )
    )

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
    )

    target_source = str(args.target_source)
    if target_source == "cv0":
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
        _, rho_gas_t, temp_t, _, _, _, _, _, _ = run_forward_eval(target_wn)
        rho_gas_t.block_until_ready()
        target_rho_source = np.asarray(rho_gas_t, dtype=np.float32)
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
         #  target_obs = np.maximum(target_obs, 0.0).astype(np.float32)

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
        _, rho_gas, temp_gas, _, _, _, _, _, _ = forward_fields_full_hydro(
            wn,
            pk_sqrt,
            grid_pos,
            system,
            cfg,
        )
        pred_obs = observable_fn(rho_gas, temp_gas, None)
        resid = _to_compare(pred_obs) - _to_compare(target_obs_jax)
        data_nll = 0.5 * jnp.mean((resid / sigma) ** 2)
        if temp_ps_loss_fn is not None:
            data_nll = data_nll + jnp.asarray(temp_ps_loss_weight, dtype=jnp.float32) * temp_ps_loss_fn(temp_gas)
        prior_nll = 0.5 * prior_w * jnp.mean(wn**2)
        loss = data_nll + prior_nll
        return loss, (data_nll, prior_nll)

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

        rho_dm_j, rho_gas_j, temp_j, _, init_mesh_j, aux_j, vx_j, vy_j, vz_j = run_forward_eval(params_snapshot)
        rho_dm_j.block_until_ready()

        rho_dm = np.asarray(rho_dm_j, dtype=np.float32)
        rho_gas = np.asarray(rho_gas_j, dtype=np.float32)
        temp_gas = np.asarray(temp_j, dtype=np.float32)
        pred_obs_j = observable_fn(rho_gas_j, temp_j, None)
        pred_obs = np.asarray(pred_obs_j, dtype=np.float32)
        vx_cms = np.asarray(vx_j, dtype=np.float32)
        vy_cms = np.asarray(vy_j, dtype=np.float32)
        vz_cms = np.asarray(vz_j, dtype=np.float32)
        init_mesh = np.asarray(init_mesh_j, dtype=np.float32)
        aux = np.asarray(aux_j, dtype=np.float32)
        a_final, dtau_mean = float(aux[0]), float(aux[1])
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
                "lpt_order": cfg.lpt_order,
                "ic_power_suppression": float(args.ic_power_suppression),
                "hydro_steps": cfg.hydro_steps,
                "dtau_min": cfg.dtau_min,
                "dtau_max": cfg.dtau_max,
                "solver": cfg.solver,
                "state_floor": cfg.state_floor,
                "pressure_floor": cfg.pressure_floor,
                "hydro_temp_floor_k": cfg.hydro_temp_floor_k,
                "force_eps": cfg.force_eps,
                "dual_energy": bool(cfg.dual_energy),
                "checkpoint_every": cfg.checkpoint_every,
                "temperature0": cfg.temperature0,
                "temperature_gamma": cfg.temperature_gamma,
                "gas_mean_fraction": cfg.gas_mean_fraction,
                "dm_kick_scale": cfg.dm_kick_scale,
                "gas_kick_scale": cfg.gas_kick_scale,
                "gas_kick_factor": cfg.gas_kick_factor,
                "enable_cooling": bool(cfg.enable_cooling),
                "cooling_model": cfg.cooling_model,
                "cooling_stop_gradient": bool(cfg.cooling_stop_gradient),
                "cooling_table": cfg.cooling_table,
                "heating_rate_per_h": cfg.heating_rate_per_h,
                "nyx_heating_scale": cfg.nyx_heating_scale,
                "cooling_rate_scale": cfg.cooling_rate_scale,
                "cooling_temp_floor_k": cfg.cooling_temp_floor_k,
                "cooling_subcycles": cfg.cooling_subcycles,
                "cooling_dtmax_s": cfg.cooling_dtmax_s,
                "nyx_cooling_table_npz": cfg.nyx_cooling_table_npz,
                "nyx_cooling_treecool": cfg.nyx_cooling_treecool,
                "nyx_cooling_z_nodes": cfg.nyx_cooling_z_nodes,
                "nyx_cooling_logdelta_min": cfg.nyx_cooling_logdelta_min,
                "nyx_cooling_logdelta_max": cfg.nyx_cooling_logdelta_max,
                "nyx_cooling_logdelta_n": cfg.nyx_cooling_logdelta_n,
                "nyx_cooling_logt_min": cfg.nyx_cooling_logt_min,
                "nyx_cooling_logt_max": cfg.nyx_cooling_logt_max,
                "nyx_cooling_logt_n": cfg.nyx_cooling_logt_n,
                "nyx_cooling_rebuild": bool(cfg.nyx_cooling_rebuild),
                "nyx_cooling_eos_path": cfg.nyx_cooling_eos_path,
                "nyx_auto_rho_unit": bool(cfg.nyx_auto_rho_unit),
                "tau_time_unit_s": cfg.tau_time_unit_s,
                "rho_unit_cgs": cfg.rho_unit_cgs,
                "vel_unit_cms": cfg.vel_unit_cms,
                "mu_hydrogen": cfg.mu_hydrogen,
                "h_species": cfg.h_species,
                "dry_run": dry_run,
                "n_iters_requested": n_iters_requested,
                "n_iters": n_iters,
                "n_iters_completed": int(history_np[-1, 0]),
                "dry_run_gradients_computed": (not dry_run),
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
                "sanitize_nonfinite_grads": bool(args.sanitize_nonfinite_grads),
                "stop_on_nonfinite_grads": bool(args.stop_on_nonfinite_grads),
                "max_rollbacks": int(args.max_rollbacks),
                "rollback_loss_factor": float(args.rollback_loss_factor),
                "rollback_enabled": bool(rollback_enabled),
                "rollbacks_used": int(n_rollbacks_total),
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
                "final_scale_factor": a_final,
                "mean_dtau": dtau_mean,
            },
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
            "loss": {
                "initial_total": float(history_np[0, 1]),
                "final_total": float(history_np[-1, 1]),
                "initial_data": float(history_np[0, 2]),
                "final_data": float(history_np[-1, 2]),
                "initial_prior": float(history_np[0, 3]),
                "final_prior": float(history_np[-1, 3]),
                "initial_grad_norm": (None if dry_run else float(history_np[0, 4])),
                "final_grad_norm": (None if dry_run else float(history_np[-1, 4])),
            },
            "stats": {
                "cv0_rho": field_stats(cv0_rho),
                "reference_rho": field_stats(ref_gas),
                "model_rho": field_stats(rho_gas),
                "cv0_temp": field_stats(cv0_temp),
                "reference_temp": field_stats(ref_temp),
                "model_temp": field_stats(temp_gas),
                "model_dm": field_stats(rho_dm),
                "reference_dm": field_stats(ref_dm),
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
            model_vx_cms=vx_cms,
            model_vy_cms=vy_cms,
            model_vz_cms=vz_cms,
            final_scale_factor=np.asarray(a_final, dtype=np.float32),
            mean_dtau=np.asarray(dtau_mean, dtype=np.float32),
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
            vx_dh_final_cms=vx_cms,
            vy_dh_final_cms=vy_cms,
            vz_dh_final_cms=vz_cms,
            gas_dh_ic=init_mesh,
            a_final_dh=np.asarray(a_final, dtype=np.float32),
        )
        np.save(write_dir / "optimized_white_noise.npy", np.asarray(params_snapshot, dtype=np.float32))
        if target_white_noise is not None:
            np.save(write_dir / "target_white_noise.npy", np.asarray(target_white_noise, dtype=np.float32))
        return stats

    def count_nonfinite_tree(tree):
        n_nonfinite = jnp.asarray(0, dtype=jnp.int32)
        n_total = jnp.asarray(0, dtype=jnp.int32)
        for g in jax.tree_util.tree_leaves(tree):
            n_nonfinite = n_nonfinite + jnp.sum(~jnp.isfinite(g))
            n_total = n_total + jnp.asarray(g.size, dtype=jnp.int32)
        return n_nonfinite, n_total

    rollback_enabled = int(args.max_rollbacks) > 0
    rollback_loss_factor = float(args.rollback_loss_factor)
    max_rollbacks = max(0, int(args.max_rollbacks))
    rollback_count = 0
    n_rollbacks_total = 0

    def instability_reason(
        *,
        loss_v: float,
        data_v: float,
        prior_v: float,
        grad_norm_v: float,
        n_nonfinite_i: int,
        n_total_i: int,
        last_good_loss_v: float,
    ) -> str | None:
        if n_nonfinite_i > 0:
            return f"non-finite gradients ({n_nonfinite_i}/{max(n_total_i, 1)})"
        if not np.isfinite(loss_v):
            return "non-finite loss"
        if not np.isfinite(data_v):
            return "non-finite data term"
        if not np.isfinite(prior_v):
            return "non-finite prior term"
        if not np.isfinite(grad_norm_v):
            return "non-finite gradient norm"
        if rollback_loss_factor > 0.0 and np.isfinite(last_good_loss_v):
            denom = max(abs(last_good_loss_v), 1.0e-12)
            if loss_v > (rollback_loss_factor * denom):
                return (
                    f"loss spike: {loss_v:.6e} > {rollback_loss_factor:.3g} x "
                    f"{last_good_loss_v:.6e}"
                )
        return None

    history = []
    stop_reason: str | None = None
    compile_and_first_s = 0.0
    opt_wall_s = 0.0
    startup_loss = None
    startup_data = None
    startup_prior = None

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

    if dry_run:
        compile_and_first_s = startup_eval_s
        history.append([0, float(startup_loss), float(startup_data), float(startup_prior), 0.0])
        stop_reason = "dry_run" if args.dry_run else "n_iters=0"
        print(
            f"[dry-run] loss={float(startup_loss):.6e} data={float(startup_data):.6e} "
            f"prior={float(startup_prior):.6e} |grad|=not-computed "
            f"(compile+eval {compile_and_first_s:.2f}s)"
        )
    elif args.optimizer == "adam":
        value_grad_fn = jax.value_and_grad(nlogpost_terms, has_aux=True)
        if args.grad_clip_norm > 0:
            opt = optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optax.adam(args.adam_lr))
        else:
            opt = optax.adam(args.adam_lr)
        opt_state = opt.init(params)

        @jax.jit
        def step(p, s):
            (loss, (data_nll, prior_nll)), grads = value_grad_fn(p)
            n_nonfinite, n_total = count_nonfinite_tree(grads)
            if args.sanitize_nonfinite_grads:
                grads = jax.tree_util.tree_map(
                    lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
                    grads,
                )
            apply_update = jnp.asarray(True)
            if args.stop_on_nonfinite_grads:
                apply_update = n_nonfinite == 0

            def do_update(_):
                updates, s_new = opt.update(grads, s, p)
                p_new = optax.apply_updates(p, updates)
                return p_new, s_new

            def skip_update(_):
                return p, s

            p_new, s_new = jax.lax.cond(apply_update, do_update, skip_update, operand=None)
            grad_norm = optax.global_norm(grads)
            return p_new, s_new, loss, data_nll, prior_nll, grad_norm, n_nonfinite, n_total

        t_compile = time.perf_counter()
        params_cur, opt_state_cur = params, opt_state
        params_new, opt_state_new, loss, data_nll, prior_nll, grad_norm, n_nonfinite, n_total = step(params_cur, opt_state_cur)
        jax.block_until_ready(loss)
        compile_and_first_s = time.perf_counter() - t_compile
        loss_f = float(loss)
        data_f = float(data_nll)
        prior_f = float(prior_nll)
        grad_norm_f = float(grad_norm)
        n_nonfinite_i = int(n_nonfinite)
        n_total_i = max(int(n_total), 1)
        nonfinite_frac = float(n_nonfinite_i) / float(n_total_i)
        reason = instability_reason(
            loss_v=loss_f,
            data_v=data_f,
            prior_v=prior_f,
            grad_norm_v=grad_norm_f,
            n_nonfinite_i=n_nonfinite_i,
            n_total_i=n_total_i,
            last_good_loss_v=float(startup_loss),
        )

        # Keep track of the last accepted/evaluated stable state for rollback.
        last_good_params = params_cur
        last_good_loss = float(startup_loss)
        last_good_data = float(startup_data)
        last_good_prior = float(startup_prior)

        if reason is not None and rollback_enabled:
            n_rollbacks_total += 1
            rollback_count += 1
            history.append([1, last_good_loss, last_good_data, last_good_prior, 0.0])
            params = last_good_params
            opt_state = opt.init(params)
            print(
                f"[{args.optimizer}] iter=1/{n_iters} rollback: {reason}; "
                f"reverting to last accepted state and resetting optimizer history "
                f"(rollbacks {n_rollbacks_total}/{max_rollbacks}) "
                f"(compile+step {compile_and_first_s:.2f}s)"
            )
            if rollback_count >= max_rollbacks:
                stop_reason = f"rollback limit reached ({n_rollbacks_total}/{max_rollbacks})"
        else:
            rollback_count = 0
            history.append([1, loss_f, data_f, prior_f, grad_norm_f])
            # Mark current state as accepted; next iter can roll back to it.
            last_good_params = params_cur
            last_good_loss = loss_f
            last_good_data = data_f
            last_good_prior = prior_f
            params = params_new
            opt_state = opt_state_new

        print(
            f"[{args.optimizer}] iter=1/{n_iters} loss={loss_f:.6e} data={data_f:.6e} "
            f"prior={prior_f:.6e} |grad|={grad_norm_f:.6e} "
            f"(compile+step {compile_and_first_s:.2f}s)"
        )
        if n_nonfinite_i > 0:
            print(
                f"[{args.optimizer}] non-finite gradients: {n_nonfinite_i}/{n_total_i} "
                f"({nonfinite_frac:.3%})"
            )
            if args.stop_on_nonfinite_grads and (not rollback_enabled):
                stop_reason = f"non-finite gradients ({n_nonfinite_i}/{n_total_i})"

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

        t_opt = time.perf_counter()
        last_loss = float(history[-1][1])
        no_improve_count = 0

        for i in range(2, n_iters + 1):
            if stop_reason is not None:
                break
            params_cur, opt_state_cur = params, opt_state
            params_new, opt_state_new, loss, data_nll, prior_nll, grad_norm, n_nonfinite, n_total = step(
                params_cur, opt_state_cur
            )
            loss_f = float(loss)
            data_f = float(data_nll)
            prior_f = float(prior_nll)
            grad_norm_f = float(grad_norm)
            n_nonfinite_i = int(n_nonfinite)
            n_total_i = max(int(n_total), 1)
            nonfinite_frac = float(n_nonfinite_i) / float(n_total_i)
            reason = instability_reason(
                loss_v=loss_f,
                data_v=data_f,
                prior_v=prior_f,
                grad_norm_v=grad_norm_f,
                n_nonfinite_i=n_nonfinite_i,
                n_total_i=n_total_i,
                last_good_loss_v=last_good_loss,
            )
            if reason is not None and rollback_enabled:
                n_rollbacks_total += 1
                rollback_count += 1
                params = last_good_params
                opt_state = opt.init(params)
                history.append([i, last_good_loss, last_good_data, last_good_prior, 0.0])
                print(
                    f"[{args.optimizer}] iter={i}/{n_iters} rollback: {reason}; "
                    f"reverting to iter with loss={last_good_loss:.6e} and resetting optimizer history "
                    f"(rollbacks {n_rollbacks_total}/{max_rollbacks})"
                )
                if rollback_count >= max_rollbacks:
                    stop_reason = f"rollback limit reached ({n_rollbacks_total}/{max_rollbacks})"
                    break
                continue

            rollback_count = 0
            rel_improve = (last_loss - loss_f) / max(abs(last_loss), 1.0e-12)

            if args.log_every > 0 and (i % args.log_every == 0 or i == n_iters):
                print(
                    f"[{args.optimizer}] iter={i}/{n_iters} loss={loss_f:.6e} data={data_f:.6e} "
                    f"prior={prior_f:.6e} |grad|={grad_norm_f:.6e} rel_improve={rel_improve:.3e}"
                )
            history.append([i, loss_f, data_f, prior_f, grad_norm_f])
            last_good_params = params_cur
            last_good_loss = loss_f
            last_good_data = data_f
            last_good_prior = prior_f
            params = params_new
            opt_state = opt_state_new
            if n_nonfinite_i > 0:
                print(
                    f"[{args.optimizer}] non-finite gradients: {n_nonfinite_i}/{n_total_i} "
                    f"({nonfinite_frac:.3%})"
                )
                if args.stop_on_nonfinite_grads and (not rollback_enabled):
                    stop_reason = f"non-finite gradients ({n_nonfinite_i}/{n_total_i})"
                    break

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
    elif args.optimizer == "lbfgs":
        value_grad_fn = jax.value_and_grad(nlogpost_terms, has_aux=True)
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
        opt_state = opt.init(params)

        def value_only_fn(x):
            return nlogpost_terms(x)[0]

        @jax.jit
        def step(p, s):
            (loss, (data_nll, prior_nll)), grads = value_grad_fn(p)
            n_nonfinite, n_total = count_nonfinite_tree(grads)
            if args.sanitize_nonfinite_grads:
                grads = jax.tree_util.tree_map(
                    lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
                    grads,
                )
            apply_update = jnp.asarray(True)
            if args.stop_on_nonfinite_grads:
                apply_update = n_nonfinite == 0

            def do_update(_):
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
                return p_new, s_new

            def skip_update(_):
                return p, s

            p_new, s_new = jax.lax.cond(apply_update, do_update, skip_update, operand=None)
            grad_norm = optax.global_norm(grads)
            return p_new, s_new, loss, data_nll, prior_nll, grad_norm, n_nonfinite, n_total

        t_compile = time.perf_counter()
        params_cur, opt_state_cur = params, opt_state
        params_new, opt_state_new, loss, data_nll, prior_nll, grad_norm, n_nonfinite, n_total = step(
            params_cur, opt_state_cur
        )
        jax.block_until_ready(loss)
        compile_and_first_s = time.perf_counter() - t_compile
        loss_f = float(loss)
        data_f = float(data_nll)
        prior_f = float(prior_nll)
        grad_norm_f = float(grad_norm)
        n_nonfinite_i = int(n_nonfinite)
        n_total_i = max(int(n_total), 1)
        nonfinite_frac = float(n_nonfinite_i) / float(n_total_i)
        reason = instability_reason(
            loss_v=loss_f,
            data_v=data_f,
            prior_v=prior_f,
            grad_norm_v=grad_norm_f,
            n_nonfinite_i=n_nonfinite_i,
            n_total_i=n_total_i,
            last_good_loss_v=float(startup_loss),
        )

        # Keep track of the last accepted/evaluated stable state for rollback.
        last_good_params = params_cur
        last_good_loss = float(startup_loss)
        last_good_data = float(startup_data)
        last_good_prior = float(startup_prior)

        if reason is not None and rollback_enabled:
            n_rollbacks_total += 1
            rollback_count += 1
            history.append([1, last_good_loss, last_good_data, last_good_prior, 0.0])
            params = last_good_params
            opt_state = opt.init(params)
            print(
                f"[{args.optimizer}] iter=1/{n_iters} rollback: {reason}; "
                f"reverting to last accepted state and resetting optimizer history "
                f"(rollbacks {n_rollbacks_total}/{max_rollbacks}) "
                f"(compile+step {compile_and_first_s:.2f}s)"
            )
            if rollback_count >= max_rollbacks:
                stop_reason = f"rollback limit reached ({n_rollbacks_total}/{max_rollbacks})"
        else:
            rollback_count = 0
            history.append([1, loss_f, data_f, prior_f, grad_norm_f])
            # Mark current state as accepted; next iter can roll back to it.
            last_good_params = params_cur
            last_good_loss = loss_f
            last_good_data = data_f
            last_good_prior = prior_f
            params = params_new
            opt_state = opt_state_new

        print(
            f"[{args.optimizer}] iter=1/{n_iters} loss={loss_f:.6e} data={data_f:.6e} "
            f"prior={prior_f:.6e} |grad|={grad_norm_f:.6e} "
            f"(compile+step {compile_and_first_s:.2f}s)"
        )
        if n_nonfinite_i > 0:
            print(
                f"[{args.optimizer}] non-finite gradients: {n_nonfinite_i}/{n_total_i} "
                f"({nonfinite_frac:.3%})"
            )
            if args.stop_on_nonfinite_grads and (not rollback_enabled):
                stop_reason = f"non-finite gradients ({n_nonfinite_i}/{n_total_i})"

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

        t_opt = time.perf_counter()
        last_loss = float(history[-1][1])
        no_improve_count = 0

        for i in range(2, n_iters + 1):
            if stop_reason is not None:
                break
            params_cur, opt_state_cur = params, opt_state
            params_new, opt_state_new, loss, data_nll, prior_nll, grad_norm, n_nonfinite, n_total = step(
                params_cur, opt_state_cur
            )
            loss_f = float(loss)
            data_f = float(data_nll)
            prior_f = float(prior_nll)
            grad_norm_f = float(grad_norm)
            n_nonfinite_i = int(n_nonfinite)
            n_total_i = max(int(n_total), 1)
            nonfinite_frac = float(n_nonfinite_i) / float(n_total_i)
            reason = instability_reason(
                loss_v=loss_f,
                data_v=data_f,
                prior_v=prior_f,
                grad_norm_v=grad_norm_f,
                n_nonfinite_i=n_nonfinite_i,
                n_total_i=n_total_i,
                last_good_loss_v=last_good_loss,
            )
            if reason is not None and rollback_enabled:
                n_rollbacks_total += 1
                rollback_count += 1
                params = last_good_params
                opt_state = opt.init(params)
                history.append([i, last_good_loss, last_good_data, last_good_prior, 0.0])
                print(
                    f"[{args.optimizer}] iter={i}/{n_iters} rollback: {reason}; "
                    f"reverting to iter with loss={last_good_loss:.6e} and resetting optimizer history "
                    f"(rollbacks {n_rollbacks_total}/{max_rollbacks})"
                )
                if rollback_count >= max_rollbacks:
                    stop_reason = f"rollback limit reached ({n_rollbacks_total}/{max_rollbacks})"
                    break
                continue

            rollback_count = 0
            rel_improve = (last_loss - loss_f) / max(abs(last_loss), 1.0e-12)

            if args.log_every > 0 and (i % args.log_every == 0 or i == n_iters):
                print(
                    f"[{args.optimizer}] iter={i}/{n_iters} loss={loss_f:.6e} data={data_f:.6e} "
                    f"prior={prior_f:.6e} |grad|={grad_norm_f:.6e} rel_improve={rel_improve:.3e}"
                )
            history.append([i, loss_f, data_f, prior_f, grad_norm_f])
            last_good_params = params_cur
            last_good_loss = loss_f
            last_good_data = data_f
            last_good_prior = prior_f
            params = params_new
            opt_state = opt_state_new
            if n_nonfinite_i > 0:
                print(
                    f"[{args.optimizer}] non-finite gradients: {n_nonfinite_i}/{n_total_i} "
                    f"({nonfinite_frac:.3%})"
                )
                if args.stop_on_nonfinite_grads and (not rollback_enabled):
                    stop_reason = f"non-finite gradients ({n_nonfinite_i}/{n_total_i})"
                    break

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
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

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
