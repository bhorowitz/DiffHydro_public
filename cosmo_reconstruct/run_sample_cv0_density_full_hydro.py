#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sample the posterior over full-hydro initial conditions against an observable-space "
            "target using BlackJAX. This is the sampling analogue of "
            "run_optimize_cv0_density_full_hydro.py."
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
            "random white-noise initialization."
        ),
    )

    p.add_argument("--hydro-steps", type=int, default=128)
    p.add_argument("--dtau-min", type=float, default=2.0e-7)
    p.add_argument("--dtau-max", type=float, default=8.0e-2)
    p.add_argument("--solver", choices=["hll", "hllc", "lf", "laxfriedrichs", "nyx"], default="hllc")
    p.add_argument("--hydro-limiter", choices=["MINMOD", "VANLEER", "MC", "SUPERBEE"], default="MINMOD",
                   help="Reconstruction slope limiter. 'MINMOD' (default) or smoother 'VANLEER' "
                        "for a less-kinky gradient.")
    p.add_argument("--hydro-riemann-smooth", type=float, default=0.0,
                   help="Soft-HLL smoothing (fraction of sound speed). 0 = exact HLL (default). "
                        ">0 (e.g. 0.1) replaces the HLL min/max/where kinks with softplus blends "
                        "for a C-infinity flux; only applies when --solver hll.")
    p.add_argument("--state-floor", type=float, default=6.0e-8)
    p.add_argument("--pressure-floor", type=float, default=6.0e-8)
    p.add_argument("--hydro-temp-floor-k", type=float, default=0.0)
    p.add_argument("--force-eps", type=float, default=1.0e-8)
    p.add_argument("--checkpoint", dest="checkpoint", action="store_true", default=True)
    p.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--dual-energy", dest="dual_energy", action="store_true", default=True)
    p.add_argument("--no-dual-energy", dest="dual_energy", action="store_false")
    p.add_argument("--dual-energy-smooth", type=float, default=0.0,
                   help=">0 replaces the dual-energy pressure selector's hard where() with "
                        "a C-inf sigmoid blend of this relative width around eta (e.g. 0.5). "
                        "The hard switch is a VALUE discontinuity that breaks force-vs-value "
                        "consistency for MH samplers.")

    p.add_argument("--temperature0", type=float, default=1.0e4)
    p.add_argument("--temperature-gamma", type=float, default=2.0 / 3.0)
    p.add_argument("--gas-mean-fraction", type=float, default=1.58e-1)
    p.add_argument("--dm-kick-scale", type=float, default=1.0)
    p.add_argument("--gas-kick-scale", type=float, default=1.0)
    p.add_argument("--gas-kick-factor", type=float, default=None)
    p.add_argument("--enable-cooling", dest="enable_cooling", action="store_true", default=False)
    p.add_argument("--no-enable-cooling", dest="enable_cooling", action="store_false",
                   help="Override an earlier --enable-cooling (the phase4 wrapper always "
                        "passes it; this lets probe runs disable cooling for bisection).")
    p.add_argument("--cooling-model", choices=["legacy", "nyx_table"], default="nyx_table")
    p.add_argument("--cooling-stop-gradient", dest="cooling_stop_gradient", action="store_true", default=True)
    p.add_argument("--no-cooling-stop-gradient", dest="cooling_stop_gradient", action="store_false")
    p.add_argument("--cooling-table", type=str, default="data/m-00.cie")
    p.add_argument("--heating-rate-per-h", type=float, default=1.0e-33)
    p.add_argument("--nyx-heating-scale", type=float, default=1.2)
    p.add_argument("--cooling-rate-scale", type=float, default=1.0)
    p.add_argument("--cooling-temp-floor-k", type=float, default=1.0)
    p.add_argument("--cooling-subcycles", type=int, default=8)
    p.add_argument("--cooling-integrator", choices=["explicit", "exponential", "exponential_exact", "implicit_adjoint"], default="explicit",
                   help="Cooling sub-step integrator: 'explicit' (forward Euler, default), "
                        "'exponential' (analytic relaxation, smooth but biased-gradient surrogate), "
                        "'exponential_exact' (same forward value as 'exponential' but the gradient "
                        "flows through the phi/k rate factors -> conservative force for MH samplers; "
                        "requires --cooling-smooth-table), or "
                        "'implicit_adjoint' (backward-Euler + IFT adjoint; NOTE its f_e<=0 limiter and "
                        "finite-guards make the force strongly non-conservative in the typical set).")
    p.add_argument("--cooling-smooth-table", dest="cooling_smooth_table", action="store_true", default=False,
                   help="C1 smoothstep interpolation of the Nyx cooling table (removes bilinear kink comb; "
                        "recommended with --cooling-integrator implicit_adjoint).")
    p.add_argument("--cooling-nn-npz", type=str, default=None,
                   help="Trained NN cooling-rate surrogate npz (src/nn_cooling.py). "
                        "Replaces the tabulated rates with a C-infinity bounded-curvature "
                        "MLP; pair with --cooling-integrator exponential_exact for a "
                        "conservative, leapfrog-integrable potential.")
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
        "--posterior-scale",
        choices=["mean", "sum", "neff"],
        default="sum",
        help=(
            "Scale used for the BlackJAX log posterior. 'sum' samples the usual "
            "field-level Gaussian likelihood/prior while checkpoint histories still "
            "report per-voxel mean losses. 'mean' preserves the old hot posterior. "
            "'neff' uses --posterior-neff for both data and field-prior mean terms."
        ),
    )
    p.add_argument(
        "--posterior-neff",
        type=float,
        default=None,
        help="Effective sample count used when --posterior-scale=neff.",
    )
    p.add_argument("--temp-ps-loss-weight", type=float, default=0.0)
    p.add_argument("--temp-ps-loss-nbins", type=int, default=32)
    p.add_argument("--temp-ps-loss-space", choices=["log", "linear"], default="log")
    p.add_argument("--temp-ps-loss-eps", type=float, default=1.0e-8)
    p.add_argument("--observable", choices=["density", "lya_flux", "xray_proxy"], default="density")
    p.add_argument("--observable-projection", choices=["3d", "mean_los", "sum_los"], default="3d")
    p.add_argument("--los-axis", type=int, choices=[0, 1, 2], default=2)
    p.add_argument("--target-observable-npy", type=str, default=None)
    p.add_argument("--target-source", choices=["cv0", "self_consistent"], default="cv0")
    p.add_argument("--self-target-white-noise-npy", type=str, default=None)
    p.add_argument("--self-target-seed", type=int, default=123)
    p.add_argument("--self-target-scale", type=float, default=1.0)
    p.add_argument("--self-target-noise-sigma", type=float, default=0.0)
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
    p.add_argument("--lya-skewer-batch-size", type=int, default=256)
    p.add_argument("--lya-differentiable", dest="lya_differentiable", action="store_true", default=False,
                   help="Use the C-infinity RSD paint (fixed-grid, no integer pixel gather) + softplus "
                        "floors in the LyA flux forward -> smooth gradient for HMC (default off).")
    p.add_argument("--lya-smooth-eos", dest="lya_smooth_eos", action="store_true", default=False,
                   help="C1 smoothstep EOS table interpolation for the LyA nHI lookup (default off).")
    p.add_argument(
        "--xray-cooling-table-npz",
        type=str,
        default="diffhydro_gadgetic_n128_z127to2_hll/nyx_cooling_table.npz",
    )
    p.add_argument("--xray-temp-floor-k", type=float, default=1.0)
    p.add_argument("--xray-proxy-kind", choices=["cool", "net_cooling", "abs_net"], default="cool")

    # ---- LyA Kaiser posterior preconditioning (initial_phases sampling) ----
    p.add_argument(
        "--lya-precond",
        choices=["none", "kaiser", "hessian"],
        default="none",
        help=(
            "IC-field preconditioning. 'none' = prior-whitening only (legacy). "
            "'kaiser' reparametrizes by the sqrt of a linear-LyA Gaussian posterior "
            "covariance so the posterior is ~unit-Gaussian in sampling coordinates. "
            "'hessian' loads an empirical posterior scale_eff(k) measured at the MAP "
            "via Hutchinson HVPs (--lya-precond-scale-npz); captures the true Hessian's "
            "high-k collapse that the analytic Kaiser model misses."
        ),
    )
    p.add_argument("--lya-precond-scale-npz", type=str, default=None,
                   help="Path to an npz with key 'scale_eff' (rfft layout, mesh^3) "
                        "used as the preconditioner scale when --lya-precond hessian.")
    p.add_argument("--lya-precond-bf", type=float, default=-0.15,
                   help="Linear LyA flux bias b_F used by the Kaiser preconditioner.")
    p.add_argument("--lya-precond-beta", type=float, default=1.4,
                   help="Linear LyA redshift-space distortion beta_F for the preconditioner.")
    p.add_argument("--lya-precond-neff", type=float, default=None,
                   help="Effective inverse per-voxel flux noise variance for the "
                        "preconditioner. Defaults to 1/noise_sigma**2 when unset.")
    p.add_argument("--lya-precond-los-axis", type=int, choices=[0, 1, 2], default=None,
                   help="LOS axis for the Kaiser mu^2 factor. Defaults to --los-axis.")
    p.add_argument(
        "--grad-smooth-kcut",
        type=float,
        default=0.0,
        help=(
            "If >0, low-pass the white-noise gradient in Fourier on the reverse pass "
            "(custom_vjp): exact forward log-density, smoothed surrogate force. Cutoff "
            "is this fraction of the Nyquist (e.g. 0.5 = half-Nyquist). ONLY valid with "
            "a Metropolis-adjusted sampler (nuts/hmc/mams), where the MH step keeps the "
            "target exact; it biases unadjusted mclmc. <=0 disables."
        ),
    )

    p.add_argument("--sampler", choices=["mclmc", "mams", "nuts", "hmc", "ghmc"], default="mclmc")
    p.add_argument(
        "--sample-space",
        choices=["initial_phases", "initial_phases_and_model_params"],
        default="initial_phases",
        help=(
            "initial_phases samples only the white-noise IC field. "
            "initial_phases_and_model_params also samples Omega_M and sigma8 with "
            "Gaussian priors in standardized coordinates."
        ),
    )
    p.add_argument(
        "--omega-m-prior-frac",
        type=float,
        default=0.10,
        help="Fractional Gaussian prior width for Omega_M in joint sampling.",
    )
    p.add_argument(
        "--sigma8-prior-frac",
        type=float,
        default=0.10,
        help="Fractional Gaussian prior width for sigma8 in joint sampling.",
    )
    p.add_argument(
        "--cosmo-pk-points",
        type=int,
        default=256,
        help="Number of k nodes used when rebuilding the differentiable IC transfer in joint sampling.",
    )
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--num-warmup", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help=(
            "If >0, write expensive full forward-model checkpoint artifacts every N posterior "
            "samples into <output-dir>/checkpoints/iter_XXXXXX."
        ),
    )
    p.add_argument("--nuts-target-acceptance-rate", type=float, default=0.8)
    p.add_argument("--nuts-initial-step-size", type=float, default=1.0e-2)
    p.add_argument("--nuts-max-num-doublings", type=int, default=4)
    p.add_argument(
        "--nuts-mass-adaptation",
        choices=["auto", "window", "identity"],
        default="auto",
        help=(
            "NUTS warmup metric. 'window' = blackjax window adaptation (adapts step "
            "AND diagonal mass matrix). 'identity' = step-size-only dual averaging "
            "with a fixed identity mass matrix (correct when an analytic preconditioner "
            "already sets the metric; avoids the mass-matrix/step collapse). 'auto' "
            "uses 'identity' when --lya-precond is active, else 'window'."
        ),
    )
    p.add_argument("--hmc-num-integration-steps", type=int, default=16)
    p.add_argument(
        "--hmc-reset-state-after-warmup",
        dest="hmc_reset_state_after_warmup",
        action="store_true",
        default=True,
        help="After HMC step-size/mass adaptation, restart sampling from the requested initial position.",
    )
    p.add_argument(
        "--no-hmc-reset-state-after-warmup",
        dest="hmc_reset_state_after_warmup",
        action="store_false",
    )
    p.add_argument(
        "--ghmc-step-size",
        type=float,
        default=None,
        help=(
            "GHMC leapfrog step size (ONE leapfrog step per MH proposal). "
            "Defaults to --nuts-initial-step-size. Seed from the "
            "--energy-probe dh table (eps where one-step dH std ~ 0.5-1 nat)."
        ),
    )
    p.add_argument(
        "--ghmc-alpha",
        type=float,
        default=0.01,
        help=(
            "GHMC momentum partial-refresh noise in (0,1]. Persistent-momentum "
            "decoherence length ~ step_size/alpha; match the working MCLMC "
            "L~200 via alpha ~ step_size/200."
        ),
    )
    p.add_argument(
        "--ghmc-delta",
        type=float,
        default=0.1,
        help="GHMC slice-variable refresh noise (blackjax ghmc 'delta').",
    )
    p.add_argument(
        "--euclidean-integrator",
        choices=["velocity_verlet", "mclachlan", "omelyan", "yoshida"],
        default="velocity_verlet",
        help=(
            "Symplectic integrator for NUTS/HMC (blackjax.mcmc.integrators). "
            "Higher-order (mclachlan/omelyan/yoshida) buys a ~2-3x larger stable "
            "step for 2-3 gradient evals per step. Not used by ghmc/mclmc/mams."
        ),
    )
    p.add_argument("--mclmc-desired-energy-var", type=float, default=5.0e-4)
    p.add_argument("--mclmc-diagonal-preconditioning", dest="mclmc_diagonal_preconditioning", action="store_true", default=True)
    p.add_argument("--no-mclmc-diagonal-preconditioning", dest="mclmc_diagonal_preconditioning", action="store_false")
    p.add_argument(
        "--mclmc-initial-step-size",
        type=float,
        default=0.005,
        help=(
            "Initial step size passed to mclmc_find_L_and_step_size. "
            "BlackJAX's default (sqrt(dim)*0.25 ≈ 128 for 64^3) immediately NaNs "
            "the full-hydro forward model; this overrides it to a safe starting value."
        ),
    )
    p.add_argument(
        "--mclmc-initial-L",
        type=float,
        default=1.0,
        help="Initial trajectory length L passed to mclmc_find_L_and_step_size.",
    )
    p.add_argument(
        "--mclmc-skip-tuning",
        dest="mclmc_skip_tuning",
        action="store_true",
        default=False,
        help=(
            "Skip mclmc_find_L_and_step_size entirely and run MCLMC with the fixed "
            "--mclmc-initial-L / --mclmc-initial-step-size (inverse_mass_matrix=ones). "
            "Use with the exact implicit-adjoint cooling: the tuner wanders off-MAP "
            "where the exact gradient can go non-finite and returns degenerate "
            "L=step=0, killing the (unadjusted, no-reject) chain."
        ),
    )
    p.add_argument(
        "--diagnostic-only",
        action="store_true",
        default=False,
        help=(
            "Stop before adaptation/sampling and run fixed-step NUTS timing probes. "
            "This is useful when window adaptation is too opaque or slow."
        ),
    )
    p.add_argument(
        "--diagnostic-step-sizes",
        type=str,
        default="2e-6,1e-6,5e-7",
        help="Comma-separated NUTS step sizes to test when --diagnostic-only is set.",
    )
    p.add_argument(
        "--diagnostic-max-doublings",
        type=str,
        default="4,6,8",
        help="Comma-separated max_num_doublings values to test when --diagnostic-only is set.",
    )
    p.add_argument(
        "--diagnostic-transitions",
        type=int,
        default=1,
        help="Number of fixed NUTS transitions to run per diagnostic setting.",
    )

    # ---- Phase-0 leapfrog energy probes (see _leapfrog_energy_probe.py) ----
    p.add_argument(
        "--energy-probe",
        choices=["repro", "force", "dh"],
        default=None,
        help=(
            "Run a leapfrog energy diagnostic on the exact sampled log-density and "
            "exit (writes <output-dir>/energy_probe_<mode>.json). 'repro' = U-value "
            "reproducibility in nats (MH corruption by GPU scatter-add atomics); "
            "'force' = conservativity of the AD force along leapfrog trajectories "
            "(detects value/gradient inconsistency, e.g. the exponential cooling "
            "surrogate's stop_gradient); 'dh' = dH(eps,L) profile with implied "
            "acceptance, seeds GHMC/HMC step sizes."
        ),
    )
    p.add_argument("--energy-probe-eps", type=str, default="3e-4,1e-3,3e-3",
                   help="Comma-separated leapfrog step sizes for --energy-probe force/dh.")
    p.add_argument("--energy-probe-steps", type=int, default=50,
                   help="Trajectory length (steps) for --energy-probe force.")
    p.add_argument("--energy-probe-lengths", type=str, default="1,8,32",
                   help="Comma-separated trajectory lengths L for --energy-probe dh.")
    p.add_argument("--energy-probe-repeats", type=int, default=10,
                   help="Re-evaluations per point for --energy-probe repro.")
    p.add_argument("--energy-probe-jitter", type=str, default="1e-3",
                   help="Comma-separated jitter scales for --energy-probe repro points.")
    p.add_argument("--energy-probe-draws", type=int, default=3,
                   help="Momentum draws per (eps,L) for --energy-probe dh.")

    p.add_argument(
        "--deterministic-paint",
        action="store_true",
        default=False,
        help=(
            "Replace jaxpm's atomic CIC scatter-add with a bitwise-deterministic "
            "sorted compensated-scan paint (src/deterministic_paint.py). Removes "
            "the ~0.4-nat run-to-run log-density noise that corrupts Metropolis "
            "accept/reject, at ~2x forward cost. Pair with --lya-differentiable "
            "to also remove the LyA pixel-scatter nondeterminism."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-random-scale", type=float, default=1.0)
    p.add_argument("--init-white-noise-npy", type=str, default=None)
    p.add_argument("--allow-init-config-mismatch", action="store_true", default=False)

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cosmo_reconstruct/outputs/step2_3_sample_full_hydro"),
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.ic_power_suppression < 0.0:
        raise ValueError("--ic-power-suppression must be >= 0.")
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be > 0.")
    joint_cosmo_sampling = str(args.sample_space) == "initial_phases_and_model_params"
    if float(args.omega_m_prior_frac) <= 0.0:
        raise ValueError("--omega-m-prior-frac must be > 0.")
    if float(args.sigma8_prior_frac) <= 0.0:
        raise ValueError("--sigma8-prior-frac must be > 0.")

    save_every = max(0, int(args.save_every))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if args.xla_preallocate else "false"
    os.environ.setdefault("MPLBACKEND", "Agg")

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import jax_cosmo as jc
    import matplotlib.pyplot as plt
    import numpy as np
    from jax.flatten_util import ravel_pytree

    try:
        import blackjax
    except ImportError as exc:  # pragma: no cover - environment-dependent path
        raise ImportError(
            "blackjax is required for run_sample_cv0_density_full_hydro.py. "
            "Install it into the active environment, e.g. `conda activate jax-gpu && pip install blackjax`."
        ) from exc

    if bool(args.deterministic_paint):
        # Must precede the src.* imports below: they bind jaxpm's cic_paint at
        # import time.
        from src.deterministic_paint import apply_deterministic_paint_patch

        apply_deterministic_paint_patch()

    from src.diagnostics import (
        compute_power_and_cross,
        field_stats,
        load_cv0_fields,
        load_reference_fields,
        make_forward_plots,
        make_observable_plots,
        save_json,
    )
    from src.observable_utils import make_observable_mapper
    from src.power_loss import make_temperature_power_spectrum_loss
    from src.forward_model import (
        make_lattice_positions,
        make_pk_sqrt,
        make_lya_kaiser_scale,
        make_pk_sqrt_post,
        rfft_hermitian_weights,
    )
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
        hydro_limiter=str(args.hydro_limiter),
        hydro_riemann_smooth=float(args.hydro_riemann_smooth),
        state_floor=args.state_floor,
        pressure_floor=args.pressure_floor,
        hydro_temp_floor_k=args.hydro_temp_floor_k,
        force_eps=args.force_eps,
        checkpoint=bool(args.checkpoint),
        checkpoint_every=max(1, int(args.checkpoint_every)),
        dual_energy=bool(args.dual_energy),
        dual_energy_smooth=float(args.dual_energy_smooth),
        temperature0=args.temperature0,
        temperature_gamma=args.temperature_gamma,
        gas_mean_fraction=args.gas_mean_fraction,
        dm_kick_scale=args.dm_kick_scale,
        gas_kick_scale=args.gas_kick_scale,
        gas_kick_factor=args.gas_kick_factor,
        enable_cooling=bool(args.enable_cooling),
        cooling_model=str(args.cooling_model),
        cooling_stop_gradient=bool(args.cooling_stop_gradient),
        cooling_table=str(args.cooling_table),
        heating_rate_per_h=args.heating_rate_per_h,
        nyx_heating_scale=args.nyx_heating_scale,
        cooling_rate_scale=args.cooling_rate_scale,
        cooling_temp_floor_k=args.cooling_temp_floor_k,
        cooling_subcycles=args.cooling_subcycles,
        cooling_integrator=args.cooling_integrator,
        cooling_nn_npz=args.cooling_nn_npz,
        cooling_smooth_table=bool(args.cooling_smooth_table),
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

    # ---- LyA Kaiser posterior preconditioning -----------------------------
    # When enabled, sampling happens in coordinates where the IC->white-noise
    # reparametrization folds in a linear-LyA posterior covariance, so the
    # likelihood-dominated directions become ~unit-variance and NUTS can take
    # non-divergent steps. pk_sqrt_sample replaces pk_sqrt as the transfer
    # applied to the sampled white noise; the field prior is re-weighted by
    # 1/scale**2 (in Fourier) to keep the stationary distribution exact.
    lya_precond_active = str(args.lya_precond) in ("kaiser", "hessian")
    precond_los_axis = int(args.lya_precond_los_axis) if args.lya_precond_los_axis is not None else int(args.los_axis)
    precond_neff = (
        float(args.lya_precond_neff)
        if args.lya_precond_neff is not None
        else 1.0 / max(float(args.noise_sigma) ** 2, 1.0e-12)
    )
    if lya_precond_active:
        if str(args.lya_precond) == "hessian":
            if args.lya_precond_scale_npz is None:
                raise ValueError("--lya-precond hessian requires --lya-precond-scale-npz")
            _hd = np.load(args.lya_precond_scale_npz)
            precond_scale = jnp.asarray(_hd["scale_eff"], dtype=jnp.float32)
            exp_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n // 2 + 1)
            if tuple(precond_scale.shape) != exp_shape:
                raise ValueError(
                    f"scale_eff shape {tuple(precond_scale.shape)} != expected rfft {exp_shape}")
            print(
                f"[precond] lya-hessian active: scale_npz={args.lya_precond_scale_npz} "
                f"scale[min/max]={float(jnp.min(precond_scale)):.4g}/{float(jnp.max(precond_scale)):.4g}",
                flush=True,
            )
        else:
            precond_scale = make_lya_kaiser_scale(
                cosmo_pk,
                cfg,
                b_flux=float(args.lya_precond_bf),
                beta_flux=float(args.lya_precond_beta),
                n_eff=precond_neff,
                los_axis=precond_los_axis,
            )
            print(
                f"[precond] lya-kaiser active: b_F={args.lya_precond_bf} beta_F={args.lya_precond_beta} "
                f"n_eff={precond_neff:.6g} los_axis={precond_los_axis} "
                f"scale[min/max]={float(jnp.min(precond_scale)):.4g}/{float(jnp.max(precond_scale)):.4g}",
                flush=True,
            )
        pk_sqrt_sample = make_pk_sqrt_post(pk_sqrt, precond_scale)
        precond_inv_scale2 = jnp.asarray(1.0 / (precond_scale ** 2), dtype=jnp.float32)
        precond_herm_w = rfft_hermitian_weights((cfg.mesh_n, cfg.mesh_n, cfg.mesh_n))
    else:
        precond_scale = None
        pk_sqrt_sample = pk_sqrt
        precond_inv_scale2 = None
        precond_herm_w = None

    omega_m_prior_sigma = float(args.omega_m) * float(args.omega_m_prior_frac)
    sigma8_prior_sigma = float(args.sigma8) * float(args.sigma8_prior_frac)

    def _position_to_cosmo(position: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return physical Omega_M/sigma8 and their standardized latent values."""
        if joint_cosmo_sampling:
            if not isinstance(position, dict):
                raise TypeError("Joint sample-space positions must be dict pytrees.")
            om_raw = jnp.asarray(position.get("omega_m_raw", 0.0), dtype=jnp.float32)
            s8_raw = jnp.asarray(position.get("sigma8_raw", 0.0), dtype=jnp.float32)
            omega_m = jnp.asarray(float(args.omega_m), dtype=jnp.float32) + jnp.asarray(
                omega_m_prior_sigma, dtype=jnp.float32
            ) * om_raw
            sigma8 = jnp.asarray(float(args.sigma8), dtype=jnp.float32) + jnp.asarray(
                sigma8_prior_sigma, dtype=jnp.float32
            ) * s8_raw
            # Keep pathological warmup excursions away from invalid cosmologies.
            omega_m = jnp.clip(omega_m, jnp.asarray(0.05, dtype=jnp.float32), jnp.asarray(0.95, dtype=jnp.float32))
            sigma8 = jnp.clip(sigma8, jnp.asarray(0.05, dtype=jnp.float32), jnp.asarray(2.5, dtype=jnp.float32))
            return omega_m, sigma8, om_raw, s8_raw
        return (
            jnp.asarray(float(args.omega_m), dtype=jnp.float32),
            jnp.asarray(float(args.sigma8), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _pk_sqrt_for_position(position: Any) -> jnp.ndarray:
        """Differentiable IC transfer for joint cosmology sampling.

        The full hydro system still uses the fiducial expansion/cooling objects built above.
        In joint mode Omega_M and sigma8 enter through the initial transfer normalization
        and shape; the late-time evolution remains the same DiffHydro forward machinery as
        phase 4. This keeps the first phase-5 chain operational while preserving a clear
        upgrade path to fully dynamic hydro background parameters.
        """
        if not joint_cosmo_sampling:
            return pk_sqrt_sample
        omega_m, sigma8, _, _ = _position_to_cosmo(position)
        cosmo_dynamic = jc.Planck15(
            Omega_c=omega_m - jnp.asarray(float(args.omega_b), dtype=jnp.float32),
            Omega_b=jnp.asarray(float(args.omega_b), dtype=jnp.float32),
            h=jnp.asarray(float(args.h), dtype=jnp.float32),
            n_s=jnp.asarray(float(args.n_s), dtype=jnp.float32),
            sigma8=sigma8,
        )
        cfg_dynamic = FullHydroConfig(
            mesh_n=cfg.mesh_n,
            box_size_mpc_h=cfg.box_size_mpc_h,
            z_init=cfg.z_init,
            z_target=cfg.z_target,
            lpt_order=cfg.lpt_order,
            omega_m=omega_m,
            omega_b=cfg.omega_b,
            h=cfg.h,
            n_s=cfg.n_s,
            sigma8=sigma8,
        )
        pk_sqrt_dynamic = make_pk_sqrt(cosmo_dynamic, cfg_dynamic, k_points=int(args.cosmo_pk_points))
        if lya_precond_active:
            # Reuse the fiducial Kaiser scale as a (cosmology-independent) preconditioner.
            return pk_sqrt_dynamic / precond_scale
        return pk_sqrt_dynamic

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
                ]
                mismatches: list[tuple[str, object, object]] = []
                for key, current_value in checks:
                    saved_value = run_cfg.get(key, missing)
                    if saved_value is missing:
                        continue
                    if isinstance(current_value, float) and isinstance(saved_value, (float, int)):
                        if not bool(np.isclose(current_value, float(saved_value), rtol=1.0e-6, atol=1.0e-8)):
                            mismatches.append((key, current_value, saved_value))
                    elif current_value != saved_value:
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
                        raise ValueError(f"{msg}\nPass --allow-init-config-mismatch to override.")
                else:
                    print(f"[init-checkpoint] config matches source checkpoint: {init_stats_path}")
            else:
                print(
                    f"[init-checkpoint] info: {init_stats_path} does not look like a full-hydro checkpoint; "
                    "skipping strict compatibility checks."
                )

        initial_white_noise = jnp.asarray(np.load(args.init_white_noise_npy), dtype=jnp.float32)
        if initial_white_noise.shape != (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n):
            raise ValueError(
                f"Init white-noise shape mismatch: got {initial_white_noise.shape}, "
                f"expected {(cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)}"
            )
        if lya_precond_active:
            # The provided white noise was optimized in raw (prior-whitened)
            # coordinates. Convert to preconditioned coordinates so it maps to the
            # SAME physical IC field under the new reparametrization:
            #   init = irfftn(rfftn(wn_post) * pk_sqrt/scale) = irfftn(rfftn(wn_raw)*pk_sqrt)
            #   => wn_post = irfftn(rfftn(wn_raw) * scale)
            initial_white_noise = jnp.asarray(
                jnp.fft.irfftn(
                    jnp.fft.rfftn(initial_white_noise) * precond_scale,
                    s=initial_white_noise.shape,
                ),
                dtype=jnp.float32,
            )
    else:
        key = jr.PRNGKey(args.seed)
        init_guess_amplitude_scale = float(np.sqrt(args.ic_power_suppression))
        initial_white_noise = (
            args.init_random_scale
            * init_guess_amplitude_scale
            * jr.normal(key, (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n), dtype=jnp.float32)
        )

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

    def _forward_for_position_eval(position):
        return forward_fields_full_hydro(
            _extract_white_noise(position),
            _pk_sqrt_for_position(position),
            grid_pos,
            system_eval,
            cfg,
        )

    run_forward_position_eval = jax.jit(_forward_for_position_eval)

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
        lya_differentiable=bool(args.lya_differentiable),
        lya_smooth_eos=bool(args.lya_smooth_eos),
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

    print(
        f"[objective] observable={args.observable} projection={args.observable_projection} "
        f"target_source={target_source} target_shape={target_obs.shape}"
    )

    temp_ps_loss_weight = float(max(0.0, args.temp_ps_loss_weight))
    temp_ps_loss_fn = None
    temp_ps_loss_meta: dict[str, object] = {"enabled": False, "weight": temp_ps_loss_weight}
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

    target_obs_jax = jnp.asarray(target_obs, dtype=jnp.float32)
    sigma = jnp.maximum(jnp.asarray(float(args.noise_sigma), dtype=jnp.float32), 1.0e-8)
    prior_w = jnp.asarray(float(args.prior_weight), dtype=jnp.float32)
    if args.posterior_scale == "mean":
        data_loss_scale = 1.0
        field_prior_scale = 1.0
    elif args.posterior_scale == "sum":
        data_loss_scale = float(np.size(target_obs))
        field_prior_scale = float(np.size(initial_white_noise))
    else:
        if args.posterior_neff is None or float(args.posterior_neff) <= 0.0:
            raise ValueError("--posterior-neff must be positive when --posterior-scale=neff.")
        data_loss_scale = float(args.posterior_neff)
        field_prior_scale = float(args.posterior_neff)
    data_loss_scale_j = jnp.asarray(data_loss_scale, dtype=jnp.float32)
    field_prior_scale_j = jnp.asarray(field_prior_scale, dtype=jnp.float32)
    print(
        f"[objective] posterior_scale={args.posterior_scale} "
        f"data_loss_scale={data_loss_scale:.6g} field_prior_scale={field_prior_scale:.6g} "
        "(logs/checkpoints report mean loss terms)"
    )

    if str(args.compare_space).lower() == "log":

        def _to_compare(x):
            return jnp.log(jnp.clip(jnp.asarray(x, dtype=jnp.float32), 1.0e-20, None))

    elif str(args.compare_space).lower() == "linear":

        def _to_compare(x):
            return jnp.asarray(x, dtype=jnp.float32)

    else:
        raise ValueError(f"Unknown compare_space={args.compare_space}")

    def nlogpost_terms(position):
        wn = _extract_white_noise(position)
        pk_sqrt_position = _pk_sqrt_for_position(position)
        _, rho_gas, temp_gas, _, _, _, _, _, _ = forward_fields_full_hydro(
            wn,
            pk_sqrt_position,
            grid_pos,
            system,
            cfg,
        )
        pred_obs = observable_fn(rho_gas, temp_gas, None)
        resid = _to_compare(pred_obs) - _to_compare(target_obs_jax)
        data_nll_mean = 0.5 * jnp.mean((resid / sigma) ** 2)
        if temp_ps_loss_fn is not None:
            data_nll_mean = data_nll_mean + jnp.asarray(temp_ps_loss_weight, dtype=jnp.float32) * temp_ps_loss_fn(temp_gas)
        if lya_precond_active:
            # Under the Kaiser reparametrization the implied IC-field prior is no
            # longer white: in sampling coordinates it is Gaussian with per-mode
            # variance scale(k)**2. Compute it in Fourier space with the rfft
            # Hermitian weights so that this reduces EXACTLY to 0.5*mean(wn**2)
            # when scale == 1 (validated numerically by --lya-precond none).
            n_total = jnp.asarray(wn.size, dtype=jnp.float32)
            wn_hat = jnp.fft.rfftn(wn)
            # NB: use re^2 + im^2, NOT jnp.abs(.)**2 — the latter has a NaN
            # gradient at the self-conjugate (DC/Nyquist) modes where imag == 0.
            wn_pow = jnp.real(wn_hat) ** 2 + jnp.imag(wn_hat) ** 2
            quad = (1.0 / n_total) * jnp.sum(
                precond_herm_w * wn_pow * precond_inv_scale2
            )  # == sum(wn**2) when scale == 1 (Parseval)
            field_prior_nll_mean = 0.5 * prior_w * quad / n_total
        else:
            field_prior_nll_mean = 0.5 * prior_w * jnp.mean(wn**2)
        cosmo_prior_nll = jnp.asarray(0.0, dtype=jnp.float32)
        if joint_cosmo_sampling:
            _, _, om_raw, s8_raw = _position_to_cosmo(position)
            cosmo_prior_nll = 0.5 * (om_raw**2 + s8_raw**2)
        scaled_loss = (
            data_loss_scale_j * data_nll_mean
            + field_prior_scale_j * field_prior_nll_mean
            + cosmo_prior_nll
        )
        report_prior_nll = field_prior_nll_mean + cosmo_prior_nll
        report_total_nll = data_nll_mean + report_prior_nll
        return scaled_loss, (data_nll_mean, report_prior_nll, report_total_nll, cosmo_prior_nll)

    nlogpost_terms_jit = jax.jit(nlogpost_terms)
    grad_nlogpost_jit = jax.jit(jax.grad(lambda x: nlogpost_terms(x)[0]))

    # ---- sampling log-density (optionally gradient-smoothed) ---------------
    # The sampler's log-density. When --grad-smooth-kcut>0 we wrap it in a
    # custom_vjp that keeps the EXACT forward value but low-passes the white-noise
    # cotangent on the reverse pass (a smooth surrogate force). With a
    # Metropolis-adjusted sampler the accept/reject uses this exact value, so only
    # proposal efficiency changes, not the stationary distribution.
    _base_logdensity = lambda pos: -nlogpost_terms(pos)[0]
    grad_smooth_active = float(args.grad_smooth_kcut) > 0.0
    if grad_smooth_active:
        from src.forward_model import make_lowpass_filter
        _lowpass = make_lowpass_filter(cfg, float(args.grad_smooth_kcut))

        def _filter_wn_cotangent(cot):
            if isinstance(cot, dict) and "white_noise" in cot:
                wn_c = jnp.asarray(cot["white_noise"], dtype=jnp.float32)
                wn_c = jnp.fft.irfftn(jnp.fft.rfftn(wn_c) * _lowpass, s=wn_c.shape)
                return {**cot, "white_noise": jnp.asarray(wn_c, dtype=jnp.float32)}
            return cot

        @jax.custom_vjp
        def sampling_logdensity(position):
            return _base_logdensity(position)

        def _sld_fwd(position):
            val, vjp_fn = jax.vjp(_base_logdensity, position)
            return val, vjp_fn

        def _sld_bwd(vjp_fn, g):
            (cot,) = vjp_fn(g)
            return (_filter_wn_cotangent(cot),)

        sampling_logdensity.defvjp(_sld_fwd, _sld_bwd)
        if str(args.sampler) == "mclmc":
            print(
                "[grad-smooth] WARNING: --grad-smooth-kcut biases UNADJUSTED mclmc "
                "(no MH correction). Use nuts/hmc/mams for an exact target.",
                flush=True,
            )
        print(
            f"[grad-smooth] reverse-pass low-pass active: k_cut={float(args.grad_smooth_kcut):.3f}*Nyquist "
            f"(exact forward, smoothed surrogate force)",
            flush=True,
        )
    else:
        sampling_logdensity = _base_logdensity

    def _tree_global_norm(tree) -> float:
        sq = 0.0
        for leaf in jax.tree_util.tree_leaves(tree):
            arr = np.asarray(leaf, dtype=np.float64)
            sq += float(np.sum(arr * arr))
        return float(np.sqrt(max(sq, 0.0)))

    def _to_float_scalar(x: Any, default: float = float("nan")) -> float:
        if x is None:
            return default
        try:
            arr = np.asarray(x)
        except Exception:
            return default
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])

    def _to_bool_scalar(x: Any, default: bool = False) -> bool:
        if x is None:
            return default
        try:
            arr = np.asarray(x)
        except Exception:
            return default
        if arr.size == 0:
            return default
        return bool(arr.reshape(-1)[0])

    def _maybe_info_value(info: Any, name: str) -> Any:
        if info is None:
            return None
        if hasattr(info, name):
            return getattr(info, name)
        if isinstance(info, dict):
            return info.get(name)
        return None

    def _extract_position(state: Any):
        if hasattr(state, "position"):
            return state.position
        raise TypeError(f"Unsupported BlackJAX state without .position: {type(state)!r}")

    def _extract_white_noise(position: Any):
        if isinstance(position, dict):
            if "white_noise" not in position:
                raise KeyError("Expected 'white_noise' in sampled position pytree.")
            return position["white_noise"]
        return position

    def _state_grad_norm(state: Any) -> float:
        grad = getattr(state, "logdensity_grad", None)
        if grad is None:
            return float("nan")
        return _tree_global_norm(grad)

    def _history_to_array(history_rows: list[list[float]]) -> np.ndarray:
        if not history_rows:
            return np.zeros((0, 5), dtype=np.float64)
        return np.asarray(history_rows, dtype=np.float64)

    def _plot_sampling_history(
        out_path: Path,
        history_np: np.ndarray,
        acceptance_trace: np.ndarray,
        divergent_trace: np.ndarray,
        integration_steps_trace: np.ndarray,
        energy_proxy_trace: np.ndarray,
    ) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

        axes[0, 0].plot(history_np[:, 0], history_np[:, 1], lw=1.8)
        axes[0, 0].set_xlabel("sample")
        axes[0, 0].set_ylabel("total nlogpost")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history_np[:, 0], history_np[:, 2], lw=1.6, label="data")
        axes[0, 1].plot(history_np[:, 0], history_np[:, 3], lw=1.6, label="prior")
        axes[0, 1].set_xlabel("sample")
        axes[0, 1].set_ylabel("term value")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        if np.isfinite(history_np[:, 4]).any():
            axes[1, 0].plot(history_np[:, 0], history_np[:, 4], lw=1.4, color="tab:red")
            axes[1, 0].set_ylabel("|grad logpost|")
            axes[1, 0].set_yscale("log")
        else:
            axes[1, 0].plot(history_np[:, 0], history_np[:, 1], lw=1.2, color="tab:purple")
            axes[1, 0].set_ylabel("total nlogpost")
        axes[1, 0].set_xlabel("sample")
        axes[1, 0].grid(True, alpha=0.3)

        ax = axes[1, 1]
        plotted = False
        x = history_np[:, 0]
        if np.isfinite(acceptance_trace).any():
            ax.plot(x, acceptance_trace, lw=1.4, label="acceptance", color="tab:green")
            plotted = True
        if np.isfinite(integration_steps_trace).any():
            ax.plot(x, integration_steps_trace, lw=1.2, label="integration steps", color="tab:blue")
            plotted = True
        if np.isfinite(energy_proxy_trace).any():
            ax.plot(x, energy_proxy_trace, lw=1.2, label="energy proxy", color="tab:orange")
            plotted = True
        if divergent_trace.size > 0 and np.any(divergent_trace > 0):
            ymax = float(np.nanmax(acceptance_trace)) if np.isfinite(acceptance_trace).any() else 1.0
            ax.scatter(x[divergent_trace > 0], np.full(np.sum(divergent_trace > 0), ymax), s=8, label="divergent")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No sampler-specific diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("sample")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()

        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    def _serialize_sampler_params(params_obj: Any) -> dict[str, Any]:
        if params_obj is None:
            return {}
        if isinstance(params_obj, dict):
            items = params_obj.items()
        elif hasattr(params_obj, "_asdict"):
            items = params_obj._asdict().items()
        else:
            return {"repr": repr(params_obj)}
        out: dict[str, Any] = {}
        for key, value in items:
            arr = np.asarray(value) if hasattr(value, "shape") or np.isscalar(value) else None
            if arr is not None:
                if arr.ndim == 0:
                    out[str(key)] = float(arr)
                else:
                    out[str(key)] = {
                        "shape": list(arr.shape),
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }
            else:
                out[str(key)] = repr(value)
        return out

    def _as_sampler_param_dict(params_obj: Any) -> dict[str, Any]:
        if params_obj is None:
            return {}
        if isinstance(params_obj, dict):
            return dict(params_obj)
        if hasattr(params_obj, "_asdict"):
            return dict(params_obj._asdict())
        raise TypeError(f"Unsupported sampler-parameter container: {type(params_obj)!r}")

    def write_outputs(
        write_dir: Path,
        params_snapshot,
        history_snapshot,
        *,
        sample_kind: str,
        stop_reason_snapshot: str | None,
        total_elapsed_s_snapshot: float,
        is_periodic_snapshot: bool,
        sampler_params_snapshot: Any,
        sampler_diag_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        write_dir.mkdir(parents=True, exist_ok=True)

        rho_dm_j, rho_gas_j, temp_j, _, init_mesh_j, aux_j, vx_j, vy_j, vz_j = run_forward_position_eval(params_snapshot)
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
        omega_m_snapshot, sigma8_snapshot, om_raw_snapshot, s8_raw_snapshot = _position_to_cosmo(params_snapshot)
        history_np = _history_to_array(history_snapshot)
        acceptance_trace = np.asarray(sampler_diag_snapshot["acceptance_trace"], dtype=np.float64)
        divergent_trace = np.asarray(sampler_diag_snapshot["divergent_trace"], dtype=np.float64)
        integration_steps_trace = np.asarray(sampler_diag_snapshot["integration_steps_trace"], dtype=np.float64)
        energy_proxy_trace = np.asarray(sampler_diag_snapshot["energy_proxy_trace"], dtype=np.float64)

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
        _plot_sampling_history(
            write_dir / "sampling_history.png",
            history_np,
            acceptance_trace,
            divergent_trace,
            integration_steps_trace,
            energy_proxy_trace,
        )

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
                "sampler": args.sampler,
                "sample_space": args.sample_space,
                "omega_m_prior_frac": float(args.omega_m_prior_frac),
                "sigma8_prior_frac": float(args.sigma8_prior_frac),
                "cosmo_pk_points": int(args.cosmo_pk_points),
                "omega_m_snapshot": float(np.asarray(omega_m_snapshot)),
                "sigma8_snapshot": float(np.asarray(sigma8_snapshot)),
                "omega_m_raw_snapshot": float(np.asarray(om_raw_snapshot)),
                "sigma8_raw_snapshot": float(np.asarray(s8_raw_snapshot)),
                "num_samples_requested": int(args.num_samples),
                "num_warmup": int(args.num_warmup),
                "noise_sigma": args.noise_sigma,
                "prior_weight": args.prior_weight,
                "posterior_scale": args.posterior_scale,
                "posterior_neff": args.posterior_neff,
                "data_loss_scale": data_loss_scale,
                "field_prior_scale": field_prior_scale,
                "compare_space": args.compare_space,
                "observable": args.observable,
                "observable_projection": args.observable_projection,
                "los_axis": int(args.los_axis),
                "target_source": target_source,
                "target_observable_npy": args.target_observable_npy,
                "self_target_noise_sigma": float(args.self_target_noise_sigma),
                "save_every": save_every,
                "sample_kind": sample_kind,
                "stop_reason": stop_reason_snapshot,
                "is_periodic_snapshot": bool(is_periodic_snapshot),
                "snapshot_iteration": int(history_np[-1, 0]),
                "total_elapsed_s": total_elapsed_s_snapshot,
                "final_scale_factor": a_final,
                "mean_dtau": dtau_mean,
            },
            "observable_meta": observable_meta,
            "sampler_parameters": _serialize_sampler_params(sampler_params_snapshot),
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
                "initial_grad_norm": float(history_np[0, 4]),
                "final_grad_norm": float(history_np[-1, 4]),
            },
            "sampling": {
                "acceptance_mean": float(np.nanmean(acceptance_trace)) if np.isfinite(acceptance_trace).any() else None,
                "acceptance_last": float(acceptance_trace[-1]) if acceptance_trace.size > 0 and np.isfinite(acceptance_trace[-1]) else None,
                "divergent_count": int(np.sum(divergent_trace > 0)),
                "integration_steps_mean": (
                    float(np.nanmean(integration_steps_trace)) if np.isfinite(integration_steps_trace).any() else None
                ),
                "energy_proxy_mean": float(np.nanmean(energy_proxy_trace)) if np.isfinite(energy_proxy_trace).any() else None,
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
            white_noise=np.asarray(_extract_white_noise(params_snapshot), dtype=np.float32),
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
            acceptance_trace=acceptance_trace.astype(np.float32),
            divergent_trace=divergent_trace.astype(np.float32),
            integration_steps_trace=integration_steps_trace.astype(np.float32),
            energy_proxy_trace=energy_proxy_trace.astype(np.float32),
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
        np.save(write_dir / "sampled_white_noise.npy", np.asarray(_extract_white_noise(params_snapshot), dtype=np.float32))
        np.save(write_dir / "optimized_white_noise.npy", np.asarray(_extract_white_noise(params_snapshot), dtype=np.float32))
        if target_white_noise is not None:
            np.save(write_dir / "target_white_noise.npy", np.asarray(target_white_noise, dtype=np.float32))
        return stats

    initial_position = {"white_noise": initial_white_noise}
    if joint_cosmo_sampling:
        initial_position["omega_m_raw"] = jnp.asarray(0.0, dtype=jnp.float32)
        initial_position["sigma8_raw"] = jnp.asarray(0.0, dtype=jnp.float32)

    if args.energy_probe is not None:
        from _leapfrog_energy_probe import run_energy_probe

        run_energy_probe(
            mode=str(args.energy_probe),
            logdensity_fn=sampling_logdensity,
            nlogpost_terms_fn=nlogpost_terms,
            initial_position=initial_position,
            out_dir=out_dir,
            args=args,
        )
        return

    initial_scaled_loss, (initial_data, initial_prior, initial_report_total, initial_cosmo_prior) = nlogpost_terms_jit(initial_position)
    jax.block_until_ready(initial_scaled_loss)
    initial_grad = grad_nlogpost_jit(initial_position)
    startup_grad_norm = _tree_global_norm(initial_grad)

    history: list[list[float]] = [
        [0.0, float(initial_report_total), float(initial_data), float(initial_prior), float(startup_grad_norm)]
    ]
    acceptance_trace: list[float] = [float("nan")]
    divergent_trace: list[float] = [0.0]
    integration_steps_trace: list[float] = [float("nan")]
    energy_proxy_trace: list[float] = [float("nan")]

    startup_diag = {
        "acceptance_trace": acceptance_trace,
        "divergent_trace": divergent_trace,
        "integration_steps_trace": integration_steps_trace,
        "energy_proxy_trace": energy_proxy_trace,
    }
    startup_dir = out_dir / "checkpoints" / "iter_000000"
    _ = write_outputs(
        startup_dir,
        initial_position,
        history,
        sample_kind="startup_initial_position",
        stop_reason_snapshot="startup_initial_position",
        total_elapsed_s_snapshot=time.perf_counter() - t0,
        is_periodic_snapshot=True,
        sampler_params_snapshot=None,
        sampler_diag_snapshot=startup_diag,
    )
    print(f"[snapshot] wrote startup artifacts: {startup_dir}")

    seed_key = jr.PRNGKey(int(args.seed))
    init_key, warmup_key, sample_key = jr.split(seed_key, 3)

    # Symplectic integrator for the euclidean samplers (nuts/hmc + diagnostics).
    # Higher-order members trade 2-3 gradient evals per step for a ~2-3x larger
    # stable step size -- useful when the step-size ceiling, not per-step cost,
    # is the binding constraint.
    euclid_integrator = getattr(blackjax.mcmc.integrators, str(args.euclidean_integrator))
    if str(args.euclidean_integrator) != "velocity_verlet":
        print(f"[integrator] NUTS/HMC euclidean integrator: {args.euclidean_integrator}", flush=True)

    if bool(args.diagnostic_only):
        def _parse_csv_floats(raw: str) -> list[float]:
            return [float(item.strip()) for item in str(raw).split(",") if item.strip()]

        def _parse_csv_ints(raw: str) -> list[int]:
            return [int(item.strip()) for item in str(raw).split(",") if item.strip()]

        step_sizes = _parse_csv_floats(args.diagnostic_step_sizes)
        max_doublings = _parse_csv_ints(args.diagnostic_max_doublings)
        if not step_sizes:
            raise ValueError("--diagnostic-step-sizes must contain at least one value.")
        if not max_doublings:
            raise ValueError("--diagnostic-max-doublings must contain at least one value.")

        flat_init, _ = ravel_pytree(initial_position)
        inverse_mass_matrix = jnp.ones_like(flat_init)
        n_transitions = max(1, int(args.diagnostic_transitions))
        diag_rows: list[dict[str, Any]] = []
        diag_key = sample_key

        print(
            f"[diagnostic] fixed NUTS probes: dim={int(flat_init.shape[0])} "
            f"step_sizes={step_sizes} max_doublings={max_doublings} "
            f"transitions={n_transitions}",
            flush=True,
        )
        for max_doubling in max_doublings:
            for step_size in step_sizes:
                algorithm_diag = blackjax.nuts(
                    sampling_logdensity,
                    step_size=float(step_size),
                    inverse_mass_matrix=inverse_mass_matrix,
                    max_num_doublings=int(max_doubling),
                    integrator=euclid_integrator,
                )
                state_diag = algorithm_diag.init(initial_position)

                @jax.jit
                def _diag_step(rng_key, sampler_state):
                    return algorithm_diag.step(rng_key, sampler_state)

                # First transition includes the XLA compile for this tree-depth/step-size closure.
                compile_t0 = time.perf_counter()
                diag_key, key0 = jr.split(diag_key)
                state_diag, info = _diag_step(key0, state_diag)
                jax.block_until_ready(state_diag.logdensity)
                compile_and_first_s = time.perf_counter() - compile_t0

                transition_times = [compile_and_first_s]
                infos = [info]
                for _ in range(1, n_transitions):
                    diag_key, key_i = jr.split(diag_key)
                    step_t0 = time.perf_counter()
                    state_diag, info = _diag_step(key_i, state_diag)
                    jax.block_until_ready(state_diag.logdensity)
                    transition_times.append(time.perf_counter() - step_t0)
                    infos.append(info)

                acceptances = [_to_float_scalar(_maybe_info_value(item, "acceptance_rate")) for item in infos]
                n_steps = [_to_float_scalar(_maybe_info_value(item, "num_integration_steps")) for item in infos]
                divergences = [1.0 if _to_bool_scalar(_maybe_info_value(item, "is_divergent")) else 0.0 for item in infos]
                energy = [
                    _to_float_scalar(
                        _maybe_info_value(item, "energy_change"),
                        default=_to_float_scalar(_maybe_info_value(item, "energy")),
                    )
                    for item in infos
                ]
                row = {
                    "step_size": float(step_size),
                    "max_num_doublings": int(max_doubling),
                    "transitions": int(n_transitions),
                    "compile_and_first_transition_s": float(compile_and_first_s),
                    "mean_transition_s": float(np.mean(transition_times)),
                    "post_compile_mean_transition_s": (
                        float(np.mean(transition_times[1:])) if len(transition_times) > 1 else None
                    ),
                    "acceptance_mean": float(np.nanmean(acceptances)),
                    "acceptance_last": float(acceptances[-1]),
                    "num_integration_steps_mean": float(np.nanmean(n_steps)),
                    "num_integration_steps_last": float(n_steps[-1]),
                    "divergent_count": int(np.nansum(divergences)),
                    "energy_proxy_last": float(energy[-1]),
                    "final_logdensity": float(np.asarray(state_diag.logdensity)),
                    "final_grad_norm": float(_state_grad_norm(state_diag)),
                }
                diag_rows.append(row)
                print(
                    "[diagnostic] "
                    f"step_size={step_size:.3e} max_doublings={max_doubling} "
                    f"first_s={compile_and_first_s:.2f} "
                    f"post_compile_s={row['post_compile_mean_transition_s']} "
                    f"accept={row['acceptance_last']:.4f} "
                    f"n_steps={row['num_integration_steps_last']:.1f} "
                    f"divergent={row['divergent_count']}",
                    flush=True,
                )

        diag = {
            "initial_scaled_loss": float(np.asarray(initial_scaled_loss)),
            "initial_report_total": float(initial_report_total),
            "initial_data": float(initial_data),
            "initial_prior": float(initial_prior),
            "startup_grad_norm": float(startup_grad_norm),
            "dimension": int(flat_init.shape[0]),
            "rows": diag_rows,
        }
        diag_path = out_dir / "diagnostic_nuts_probe.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, sort_keys=True)
        print(f"[diagnostic] wrote {diag_path}", flush=True)
        return

    if args.sampler == "mclmc":
        initial_state = blackjax.mcmc.mclmc.init(
            position=initial_position,
            logdensity_fn=sampling_logdensity,
            rng_key=init_key,
        )
        kernel_factory = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=sampling_logdensity,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        tune_t0 = time.perf_counter()
        from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState as _MCLMCAdaptationState
        flat_init, _ = ravel_pytree(initial_position)
        _mclmc_dim = flat_init.shape[0]
        _mclmc_init_params = _MCLMCAdaptationState(
            L=jnp.asarray(float(args.mclmc_initial_L), dtype=jnp.float32),
            step_size=jnp.asarray(float(args.mclmc_initial_step_size), dtype=jnp.float32),
            inverse_mass_matrix=jnp.ones((_mclmc_dim,), dtype=jnp.float32),
        )
        # NOTE: blackjax 1.2.5's mclmc_find_L_and_step_size has no `params=` kwarg
        # (initial L/step_size are set internally from the dimension and tuned).
        # _mclmc_init_params is retained only for the post-tuning safety clamp below.
        if bool(args.mclmc_skip_tuning):
            # Bypass the tuner: it explores off-MAP where the exact implicit-adjoint
            # gradient can go non-finite and returns degenerate L=step=0, killing the
            # unadjusted (no-reject) chain. Run with the fixed initial L/step instead.
            print(
                f"[mclmc] SKIP tuning: using fixed "
                f"L={float(args.mclmc_initial_L):.6e}, "
                f"step_size={float(args.mclmc_initial_step_size):.6e}, imm=ones"
            )
            state = initial_state
            sampler_params = _mclmc_init_params
        else:
            state, sampler_params, _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel_factory,
                num_steps=max(10, int(args.num_warmup)),
                state=initial_state,
                rng_key=warmup_key,
                diagonal_preconditioning=bool(args.mclmc_diagonal_preconditioning),
                desired_energy_var=float(args.mclmc_desired_energy_var),
            )
        warmup_s = time.perf_counter() - tune_t0

        # The diagonal mass matrix is estimated via E[x²]-E[x]², which can go
        # slightly negative from floating-point cancellation when the chain barely
        # moves.  A negative element causes sqrt(inv_mass) = NaN on the first
        # sampling step, killing the entire chain.  Clamp to a safe floor.
        _imm = jnp.asarray(sampler_params.inverse_mass_matrix)
        _imm_bad = ~(jnp.isfinite(_imm) & (_imm > 0))
        if bool(jnp.any(_imm_bad)):
            _n_bad = int(jnp.sum(_imm_bad))
            print(f"[mclmc] WARNING: {_n_bad}/{_mclmc_dim} inverse_mass_matrix elements are non-positive "
                  f"or non-finite after tuning (min={float(jnp.min(_imm)):.3e}); clamping to 1.0")
        _imm_safe = jnp.where(_imm_bad, jnp.ones_like(_imm), _imm)
        sampler_params = _MCLMCAdaptationState(
            L=sampler_params.L,
            step_size=sampler_params.step_size,
            inverse_mass_matrix=_imm_safe,
        )

        # If the tuning left the chain in a NaN state (can happen when the hydro
        # NaN escapes the handle_nans rollback), reset to the known-good initial position.
        _flat_pos, _ = ravel_pytree(state.position)
        if not bool(jnp.all(jnp.isfinite(_flat_pos))):
            print("[mclmc] WARNING: tuned state position has non-finite values; "
                  "re-initializing chain from initial_position")
            state = blackjax.mcmc.mclmc.init(
                position=initial_position,
                logdensity_fn=sampling_logdensity,
                rng_key=init_key,
            )

        algorithm = blackjax.mclmc(
            sampling_logdensity,
            L=sampler_params.L,
            step_size=sampler_params.step_size,
            inverse_mass_matrix=sampler_params.inverse_mass_matrix,
        )
        print(
            f"[mclmc] tuned in {warmup_s:.2f}s with "
            f"L={float(np.asarray(sampler_params.L)):.6e}, "
            f"step_size={float(np.asarray(sampler_params.step_size)):.6e}"
        )
    elif args.sampler == "mams":
        # Metropolis-Adjusted Microcanonical Sampler (adjusted MCLMC). Keeps the
        # microcanonical dynamics that let MCLMC move on this stiff target, but
        # adds an MH accept/reject so the stationary distribution is EXACT and
        # robust to integration / exact-cooling-gradient error (the accept ratio
        # uses the log-density, not the gradient -- the gradient only affects
        # proposal efficiency, not correctness). Diagonal preconditioning is OFF:
        # the Kaiser preconditioner is the metric (same reasoning as MCLMC/NUTS).
        from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
        from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState as _MAMSState
        mams_logdensity = sampling_logdensity
        flat_init, _ = ravel_pytree(initial_position)
        _dim = int(flat_init.shape[0])
        init_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
            position=initial_position,
            logdensity_fn=mams_logdensity,
            random_generator_arg=init_key,
        )
        # Seed L/step_size from --mclmc-initial-L / --mclmc-initial-step-size.
        # CRITICAL for cost: MAMS does ~L/step_size integration steps (grad evals)
        # PER sample, so the FLBench default (L=sqrt(dim)~512) would mean hundreds
        # of expensive hydro grads per proposal. Keep L/step ~ a few (match the
        # unadjusted MCLMC tuned value, L/step~3-4).
        init_config = _MAMSState(
            L=jnp.asarray(float(args.mclmc_initial_L), dtype=jnp.float32),
            step_size=jnp.asarray(float(args.mclmc_initial_step_size), dtype=jnp.float32),
            inverse_mass_matrix=jnp.ones((_dim,), dtype=jnp.float32),
        )
        _int_steps_fn = lambda avg: (lambda k: jnp.ceil(jr.uniform(k) * rescale(avg)))
        _mams_kernel = (
            lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix:
            blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
                integration_steps_fn=_int_steps_fn(avg_num_integration_steps),
                inverse_mass_matrix=inverse_mass_matrix,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            )(
                rng_key=rng_key,
                state=state,
                step_size=step_size,
                logdensity_fn=mams_logdensity,
                L_proposal_factor=jnp.inf,
            )
        )
        tune_t0 = time.perf_counter()
        state, sampler_params, _n_tune = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=_mams_kernel,
            num_steps=max(10, int(args.num_warmup)),
            state=init_state,
            rng_key=warmup_key,
            target=float(args.nuts_target_acceptance_rate),
            frac_tune1=0.1,
            frac_tune2=0.1,
            frac_tune3=0.0,
            diagonal_preconditioning=bool(args.mclmc_diagonal_preconditioning),
            params=init_config,
        )
        warmup_s = time.perf_counter() - tune_t0
        _L = sampler_params.L
        _ss = sampler_params.step_size
        # The adjusted-MCLMC tuning runs dynamic-length trajectories that can hit a
        # hydro NaN on this stiff target; the NaN then poisons the tuned L/step
        # (observed L=nan, step=nan -> frozen chain). Fall back to the finite
        # user-provided initial seed if tuning returned non-finite/non-positive.
        _L_f = float(np.asarray(_L)); _ss_f = float(np.asarray(_ss))
        if not (np.isfinite(_L_f) and np.isfinite(_ss_f) and _L_f > 0.0 and _ss_f > 0.0):
            print(
                f"[mams] WARNING: tuning produced non-finite L/step_size "
                f"(L={_L_f}, step={_ss_f}); falling back to initial seed "
                f"L={float(args.mclmc_initial_L)}, step={float(args.mclmc_initial_step_size)}"
            )
            _L = jnp.asarray(float(args.mclmc_initial_L), dtype=jnp.float32)
            _ss = jnp.asarray(float(args.mclmc_initial_step_size), dtype=jnp.float32)
        # Reset the chain to the known-good initial position if tuning left it NaN.
        _flat_pos, _ = ravel_pytree(state.position)
        if not bool(jnp.all(jnp.isfinite(_flat_pos))):
            print("[mams] WARNING: tuned state has non-finite position; re-initializing from initial_position")
            state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
                position=initial_position, logdensity_fn=mams_logdensity, random_generator_arg=init_key)
        _imm = jnp.asarray(sampler_params.inverse_mass_matrix)
        _imm = jnp.where(jnp.isfinite(_imm) & (_imm > 0), _imm, jnp.ones_like(_imm))
        algorithm = blackjax.adjusted_mclmc_dynamic(
            mams_logdensity,
            step_size=_ss,
            integration_steps_fn=lambda key: jnp.ceil(jr.uniform(key) * rescale(_L / _ss)),
            inverse_mass_matrix=_imm,
            L_proposal_factor=jnp.inf,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )
        print(
            f"[mams] tuned in {warmup_s:.2f}s with L={float(np.asarray(_L)):.6e}, "
            f"step_size={float(np.asarray(_ss)):.6e}, "
            f"target_acc={float(args.nuts_target_acceptance_rate):.2f}, "
            f"avg_int_steps={float(np.asarray(_L / _ss)):.1f}"
        )
    elif args.sampler == "nuts":
        _mass_mode = str(args.nuts_mass_adaptation)
        if _mass_mode == "auto":
            _mass_mode = "identity" if lya_precond_active else "window"
        nuts_logdensity = sampling_logdensity
        tune_t0 = time.perf_counter()
        if _mass_mode == "identity":
            # Step-size-only dual averaging with a FIXED identity mass matrix.
            # The analytic Kaiser preconditioner already sets the metric (posterior
            # ~ unit covariance in sampling coords), so window adaptation's diagonal
            # mass-matrix estimate is redundant AND destructive here: on this stiff
            # target it reads the (initially frozen) chain's ~0 sample variance,
            # inflates the kinetic energy, and collapses the step (observed
            # 1e-10..1e-15). Identity mass + step-only DA keeps the step near the
            # value the curvature probe identifies.
            from blackjax.adaptation.step_size import dual_averaging_adaptation
            flat_init0, _ = ravel_pytree(initial_position)
            inverse_mass = jnp.ones_like(flat_init0)
            max_doubl = int(args.nuts_max_num_doublings)

            def _nuts_at(step):
                return blackjax.nuts(
                    nuts_logdensity, step_size=step,
                    inverse_mass_matrix=inverse_mass, max_num_doublings=max_doubl,
                    integrator=euclid_integrator)

            da_init, da_update, da_final = dual_averaging_adaptation(
                target=float(args.nuts_target_acceptance_rate))
            init_ss = float(args.nuts_initial_step_size)
            da_state0 = da_init(init_ss)
            state0 = _nuts_at(jnp.asarray(init_ss, dtype=jnp.float32)).init(initial_position)

            def _wu_step(carry, key):
                st, das = carry
                ss = jnp.exp(das.log_step_size)
                st, info = _nuts_at(ss).step(key, st)
                acc = jnp.clip(jnp.nan_to_num(info.acceptance_rate, nan=0.0), 0.0, 1.0)
                das = da_update(das, acc)
                return (st, das), (ss, acc, info.is_divergent)

            n_wu = max(10, int(args.num_warmup))
            (state, da_state), (ss_tr, acc_tr, div_tr) = jax.lax.scan(
                jax.jit(_wu_step), (state0, da_state0), jr.split(warmup_key, n_wu))
            adapted_ss = float(da_final(da_state))
            warmup_s = time.perf_counter() - tune_t0
            sampler_params = {
                "step_size": float(adapted_ss),
                "inverse_mass_matrix": inverse_mass,
                "max_num_doublings": max_doubl,
            }
            algorithm = blackjax.nuts(nuts_logdensity, integrator=euclid_integrator, **sampler_params)
            state = algorithm.init(initial_position)
            print(
                f"[nuts] step-only DA warmup ({n_wu} steps) in {warmup_s:.2f}s: "
                f"step_size={adapted_ss:.6e} (identity mass), "
                f"warmup_acc[mean/last]={float(np.nanmean(np.asarray(acc_tr))):.3f}/"
                f"{float(np.asarray(acc_tr)[-1]):.3f}, "
                f"warmup_div={int(np.sum(np.asarray(div_tr)))}, "
                f"max_num_doublings={max_doubl}"
            )
        else:
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                nuts_logdensity,
                progress_bar=False,
                target_acceptance_rate=float(args.nuts_target_acceptance_rate),
                initial_step_size=float(args.nuts_initial_step_size),
                max_num_doublings=int(args.nuts_max_num_doublings),
                integrator=euclid_integrator,
            )
            (state, sampler_params), _ = warmup.run(
                warmup_key,
                initial_position,
                num_steps=max(10, int(args.num_warmup)),
            )
            warmup_s = time.perf_counter() - tune_t0
            sampler_params = _as_sampler_param_dict(sampler_params)
            sampler_params["max_num_doublings"] = int(
                sampler_params.get("max_num_doublings", int(args.nuts_max_num_doublings))
            )
            sampler_params.pop("integrator", None)
            algorithm = blackjax.nuts(nuts_logdensity, integrator=euclid_integrator, **sampler_params)
            print(
                f"[nuts] window warmup in {warmup_s:.2f}s with "
                f"step_size={float(np.asarray(sampler_params['step_size'])):.6e}, "
                f"max_num_doublings={int(sampler_params['max_num_doublings'])}"
            )
    elif args.sampler == "hmc":
        tune_t0 = time.perf_counter()
        if int(args.num_warmup) <= 0:
            flat_init, _ = ravel_pytree(initial_position)
            sampler_params = {
                "step_size": jnp.asarray(float(args.nuts_initial_step_size), dtype=jnp.float32),
                "inverse_mass_matrix": jnp.ones_like(flat_init),
                "num_integration_steps": int(args.hmc_num_integration_steps),
            }
            state = blackjax.hmc(
                sampling_logdensity,
                integrator=euclid_integrator,
                **sampler_params,
            ).init(initial_position)
        else:
            warmup = blackjax.window_adaptation(
                blackjax.hmc,
                sampling_logdensity,
                progress_bar=False,
                target_acceptance_rate=float(args.nuts_target_acceptance_rate),
                initial_step_size=float(args.nuts_initial_step_size),
                num_integration_steps=int(args.hmc_num_integration_steps),
                integrator=euclid_integrator,
            )
            (state, sampler_params), _ = warmup.run(
                warmup_key,
                initial_position,
                num_steps=max(10, int(args.num_warmup)),
            )
            sampler_params = _as_sampler_param_dict(sampler_params)
            sampler_params["num_integration_steps"] = int(
                sampler_params.get("num_integration_steps", int(args.hmc_num_integration_steps))
            )
            sampler_params.pop("integrator", None)
        warmup_s = time.perf_counter() - tune_t0
        algorithm = blackjax.hmc(
            sampling_logdensity,
            integrator=euclid_integrator,
            **sampler_params,
        )
        if int(args.num_warmup) > 0 and bool(args.hmc_reset_state_after_warmup):
            state = algorithm.init(initial_position)
            print("[hmc] state reset to initial position after warmup")
        print(
            f"[hmc] warmup in {warmup_s:.2f}s with "
            f"step_size={float(np.asarray(sampler_params['step_size'])):.6e}, "
            f"num_integration_steps={int(sampler_params['num_integration_steps'])}"
        )
    elif args.sampler == "ghmc":
        # Generalized HMC: ONE leapfrog step per MH proposal with persistent
        # (partially-refreshed) momentum -- the Metropolis-adjusted analog of
        # MCLMC. Per-proposal energy error is a single leapfrog step, so the
        # trajectory-long dH accumulation that freezes NUTS/MAMS on this target
        # (stiffness, value/grad inconsistency of the exp surrogate, float32
        # value noise) never builds up, while the persistent momentum keeps
        # exploration ballistic with decoherence length ~ step_size/alpha.
        # MH rejection also tolerates the heavy-tailed exact implicit-adjoint
        # force that NaN'd unadjusted MCLMC. No adaptation on purpose: the
        # Kaiser preconditioner is the metric and every tuner tried so far
        # collapses on this target; set step/alpha explicitly (seed the step
        # from the --energy-probe dh table).
        ghmc_step_size = (
            float(args.ghmc_step_size)
            if args.ghmc_step_size is not None
            else float(args.nuts_initial_step_size)
        )
        ghmc_alpha = float(args.ghmc_alpha)
        ghmc_delta = float(args.ghmc_delta)
        momentum_inverse_scale = jax.tree_util.tree_map(jnp.ones_like, initial_position)
        algorithm = blackjax.ghmc(
            sampling_logdensity,
            step_size=ghmc_step_size,
            momentum_inverse_scale=momentum_inverse_scale,
            alpha=ghmc_alpha,
            delta=ghmc_delta,
        )
        state = blackjax.mcmc.ghmc.init(initial_position, init_key, sampling_logdensity)
        # Scalars only: momentum_inverse_scale is a pytree and would blow up the
        # stats-JSON serialization; it is identity anyway.
        sampler_params = {
            "step_size": ghmc_step_size,
            "alpha": ghmc_alpha,
            "delta": ghmc_delta,
        }
        warmup_s = 0.0
        print(
            f"[ghmc] fixed params: step_size={ghmc_step_size:.6e} alpha={ghmc_alpha:.3e} "
            f"delta={ghmc_delta:.3e} (identity momentum scale; "
            f"decoherence L ~ step/alpha = {ghmc_step_size / max(ghmc_alpha, 1e-30):.4g})"
        )
    else:  # pragma: no cover - protected by argparse
        raise ValueError(f"Unknown sampler: {args.sampler}")

    @jax.jit
    def sample_step(rng_key, sampler_state):
        return algorithm.step(rng_key, sampler_state)

    sample_keys = jr.split(sample_key, int(args.num_samples))
    best_total = float("inf")
    best_report_total = float("inf")
    best_white_noise = np.asarray(initial_white_noise, dtype=np.float32)
    best_position = initial_position
    best_sample_index = 0
    last_white_noise_np = np.asarray(initial_white_noise, dtype=np.float32)
    last_position = initial_position
    omega_m_trace: list[float] = [float(args.omega_m)]
    sigma8_trace: list[float] = [float(args.sigma8)]
    omega_m_raw_trace: list[float] = [0.0]
    sigma8_raw_trace: list[float] = [0.0]

    running_mean = np.zeros_like(last_white_noise_np, dtype=np.float64)
    running_m2 = np.zeros_like(last_white_noise_np, dtype=np.float64)
    n_running = 0

    stop_reason: str | None = None
    sample_t0 = time.perf_counter()

    for i, step_key in enumerate(np.asarray(sample_keys), start=1):
        state, info = sample_step(jnp.asarray(step_key), state)
        position = _extract_position(state)
        white_noise = _extract_white_noise(position)
        wn_np = np.asarray(white_noise, dtype=np.float32)
        omega_m_j, sigma8_j, om_raw_j, s8_raw_j = _position_to_cosmo(position)
        omega_m_val = float(np.asarray(omega_m_j))
        sigma8_val = float(np.asarray(sigma8_j))
        om_raw_val = float(np.asarray(om_raw_j))
        s8_raw_val = float(np.asarray(s8_raw_j))
        scaled_total = float(-np.asarray(state.logdensity))
        field_prior_mean = float(0.5 * float(args.prior_weight) * np.mean(wn_np.astype(np.float64) ** 2))
        cosmo_prior = 0.0
        if joint_cosmo_sampling:
            cosmo_prior = float(0.5 * (om_raw_val**2 + s8_raw_val**2))
        prior = field_prior_mean + cosmo_prior
        data = float((scaled_total - field_prior_scale * field_prior_mean - cosmo_prior) / data_loss_scale)
        total = data + prior
        grad_norm = _state_grad_norm(state)

        history.append([float(i), total, data, prior, grad_norm])
        acceptance_trace.append(_to_float_scalar(_maybe_info_value(info, "acceptance_rate")))
        divergent_trace.append(1.0 if _to_bool_scalar(_maybe_info_value(info, "is_divergent")) else 0.0)
        integration_steps_trace.append(_to_float_scalar(_maybe_info_value(info, "num_integration_steps")))
        energy_proxy_trace.append(
            _to_float_scalar(_maybe_info_value(info, "energy_change"), default=_to_float_scalar(_maybe_info_value(info, "energy")))
        )
        omega_m_trace.append(omega_m_val)
        sigma8_trace.append(sigma8_val)
        omega_m_raw_trace.append(om_raw_val)
        sigma8_raw_trace.append(s8_raw_val)

        n_running += 1
        delta = wn_np.astype(np.float64) - running_mean
        running_mean = running_mean + delta / float(n_running)
        delta2 = wn_np.astype(np.float64) - running_mean
        running_m2 = running_m2 + delta * delta2

        if scaled_total < best_total:
            best_total = scaled_total
            best_report_total = total
            best_white_noise = np.asarray(wn_np, dtype=np.float32)
            best_position = position
            best_sample_index = i
        last_white_noise_np = np.asarray(wn_np, dtype=np.float32)
        last_position = position

        if args.log_every > 0 and (i % args.log_every == 0 or i == int(args.num_samples)):
            msg = (
                f"[{args.sampler}] sample={i}/{int(args.num_samples)} total={total:.6e} "
                f"data={data:.6e} prior={prior:.6e}"
            )
            if np.isfinite(acceptance_trace[-1]):
                msg += f" acceptance={acceptance_trace[-1]:.4f}"
            if np.isfinite(integration_steps_trace[-1]):
                msg += f" n_steps={integration_steps_trace[-1]:.1f}"
            if joint_cosmo_sampling:
                msg += f" omega_m={omega_m_val:.5f} sigma8={sigma8_val:.5f}"
            if divergent_trace[-1] > 0:
                msg += " divergent=1"
            print(msg)

        if save_every > 0 and (i % save_every == 0) and (i < int(args.num_samples)):
            ckpt_dir = out_dir / "checkpoints" / f"iter_{i:06d}"
            sampler_diag = {
                "acceptance_trace": acceptance_trace,
                "divergent_trace": divergent_trace,
                "integration_steps_trace": integration_steps_trace,
                "energy_proxy_trace": energy_proxy_trace,
            }
            _ = write_outputs(
                ckpt_dir,
                position,
                history,
                sample_kind="periodic_sample",
                stop_reason_snapshot=stop_reason,
                total_elapsed_s_snapshot=time.perf_counter() - t0,
                is_periodic_snapshot=True,
                sampler_params_snapshot=sampler_params,
                sampler_diag_snapshot=sampler_diag,
            )
            print(f"[snapshot] wrote periodic artifacts: {ckpt_dir}")

    sample_wall_s = time.perf_counter() - sample_t0

    posterior_mean = running_mean.astype(np.float32)
    posterior_std = (
        np.sqrt(running_m2 / max(n_running - 1, 1)).astype(np.float32)
        if n_running > 1
        else np.zeros_like(best_white_noise, dtype=np.float32)
    )

    np.save(out_dir / "posterior_mean_white_noise.npy", posterior_mean)
    np.save(out_dir / "posterior_std_white_noise.npy", posterior_std)
    np.save(out_dir / "best_white_noise.npy", best_white_noise)
    np.save(out_dir / "last_sample_white_noise.npy", last_white_noise_np)

    history_np = _history_to_array(history)
    acceptance_np = np.asarray(acceptance_trace, dtype=np.float64)
    divergent_np = np.asarray(divergent_trace, dtype=np.float64)
    integration_steps_np = np.asarray(integration_steps_trace, dtype=np.float64)
    energy_proxy_np = np.asarray(energy_proxy_trace, dtype=np.float64)
    omega_m_np = np.asarray(omega_m_trace, dtype=np.float64)
    sigma8_np = np.asarray(sigma8_trace, dtype=np.float64)
    omega_m_raw_np = np.asarray(omega_m_raw_trace, dtype=np.float64)
    sigma8_raw_np = np.asarray(sigma8_raw_trace, dtype=np.float64)

    np.savez_compressed(
        out_dir / "sampling_diagnostics.npz",
        history=history_np,
        acceptance_trace=acceptance_np.astype(np.float32),
        divergent_trace=divergent_np.astype(np.float32),
        integration_steps_trace=integration_steps_np.astype(np.float32),
        energy_proxy_trace=energy_proxy_np.astype(np.float32),
        posterior_mean_white_noise=posterior_mean,
        posterior_std_white_noise=posterior_std,
        best_white_noise=best_white_noise,
        last_sample_white_noise=last_white_noise_np,
        omega_m_trace=omega_m_np.astype(np.float32),
        sigma8_trace=sigma8_np.astype(np.float32),
        omega_m_raw_trace=omega_m_raw_np.astype(np.float32),
        sigma8_raw_trace=sigma8_raw_np.astype(np.float32),
    )
    np.save(out_dir / "omega_m_trace.npy", omega_m_np.astype(np.float32))
    np.save(out_dir / "sigma8_trace.npy", sigma8_np.astype(np.float32))

    sampler_diag = {
        "acceptance_trace": acceptance_trace,
        "divergent_trace": divergent_trace,
        "integration_steps_trace": integration_steps_trace,
        "energy_proxy_trace": energy_proxy_trace,
    }

    final_stats = write_outputs(
        out_dir,
        last_position,
        history,
        sample_kind="last_sample",
        stop_reason_snapshot=stop_reason,
        total_elapsed_s_snapshot=time.perf_counter() - t0,
        is_periodic_snapshot=False,
        sampler_params_snapshot=sampler_params,
        sampler_diag_snapshot=sampler_diag,
    )
    _ = write_outputs(
        out_dir / "best_sample",
        best_position,
        history,
        sample_kind="best_posterior_sample",
        stop_reason_snapshot=f"best_sample_index={best_sample_index}",
        total_elapsed_s_snapshot=time.perf_counter() - t0,
        is_periodic_snapshot=False,
        sampler_params_snapshot=sampler_params,
        sampler_diag_snapshot=sampler_diag,
    )
    _ = write_outputs(
        out_dir / "posterior_mean",
        (
            {
                "white_noise": jnp.asarray(posterior_mean, dtype=jnp.float32),
                "omega_m_raw": jnp.asarray(float(np.mean(omega_m_raw_np[1:])), dtype=jnp.float32),
                "sigma8_raw": jnp.asarray(float(np.mean(sigma8_raw_np[1:])), dtype=jnp.float32),
            }
            if joint_cosmo_sampling
            else jnp.asarray(posterior_mean, dtype=jnp.float32)
        ),
        history,
        sample_kind="posterior_mean_white_noise",
        stop_reason_snapshot=f"posterior_mean_over_{n_running}_samples",
        total_elapsed_s_snapshot=time.perf_counter() - t0,
        is_periodic_snapshot=False,
        sampler_params_snapshot=sampler_params,
        sampler_diag_snapshot=sampler_diag,
    )

    print(f"Saved outputs to: {out_dir}")
    print(
        f"Last-sample loss: initial={final_stats['loss']['initial_total']:.6e}, "
        f"final={final_stats['loss']['final_total']:.6e}, best_mean={best_report_total:.6e}, "
        f"best_scaled={best_total:.6e}"
    )
    print(
        f"Sampling runtime: warmup={warmup_s:.2f}s, sampling={sample_wall_s:.2f}s, "
        f"total={time.perf_counter() - t0:.2f}s"
    )
    if np.isfinite(acceptance_np).any():
        print(f"Mean acceptance: {float(np.nanmean(acceptance_np)):.4f}")
    if np.any(divergent_np > 0):
        print(f"Divergences: {int(np.sum(divergent_np > 0))}")


if __name__ == "__main__":
    main()
