#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run staged stellar/metallicity validation experiments for the cosmological DiffHydro model."
    )
    p.add_argument(
        "--stage",
        choices=[
            "baseline",
            "layout_refactor",
            "metallicity",
            "star_carrier",
            "transfer",
            "deterministic_sf",
            "gumbel_sf",
            "age_bins",
            "metal_source",
            "cv0",
            "gradient_smoke",
        ],
        default="baseline",
    )
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--xla-preallocate", action="store_true", default=False)
    p.add_argument("--mesh-n", type=int, default=32)
    p.add_argument("--hydro-steps", type=int, default=16)
    p.add_argument("--state-floor", type=float, default=None, help="Physical gas-density floor used by the post-step hydro projection.")
    p.add_argument("--pressure-floor", type=float, default=None, help="Physical gas-pressure floor used by the post-step hydro projection.")
    p.add_argument("--hydro-temp-floor-k", type=float, default=None, help="Explicit hydro temperature floor in Kelvin used by the post-step projection.")
    p.add_argument("--cooling-temp-floor-k", type=float, default=None, help="Cooling temperature floor in Kelvin; used as the effective hydro floor when hydro-temp-floor-k is unset.")
    p.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="If >0, save intermediate field snapshots every N hydro steps.",
    )
    p.add_argument(
        "--snapshots-subdir",
        type=str,
        default="snapshots",
        help="Subdirectory under the stage output root where intermediate snapshots are written.",
    )
    p.add_argument(
        "--save-start-snapshot",
        action="store_true",
        default=False,
        help="Also save a step-0 snapshot before evolution begins.",
    )
    p.add_argument(
        "--save-final-snapshot",
        action="store_true",
        default=True,
        help="Save a final snapshot at the end if it was not already captured by --save-every-steps.",
    )
    p.add_argument(
        "--save-full-state-snapshots",
        action="store_true",
        default=False,
        help="Include full hydro + DM particle state in each saved snapshot so notebooks can reload them interactively.",
    )
    p.add_argument("--box-size-mpc-h", type=float, default=25.0)
    p.add_argument("--z-init", type=float, default=127.0)
    p.add_argument("--z-target", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--cooling-model",
        choices=["legacy", "nyx_table", "grackle_eq"],
        default="nyx_table",
        help="Cooling/heating backend used by the staged full-hydro runs.",
    )
    p.add_argument(
        "--star-step-start",
        type=int,
        default=0,
        help="Delay star formation, metal source, and newly formed-star feedback until this hydro step index.",
    )
    p.add_argument("--track-metallicity", choices=["true", "false"], default=None)
    p.add_argument("--metal-init", type=float, default=None)
    p.add_argument("--enable-star-formation", choices=["true", "false"], default=None)
    p.add_argument("--star-formation-mode", choices=["deterministic", "gumbel"], default=None)
    p.add_argument("--enable-metal-source", choices=["true", "false"], default=None)
    p.add_argument("--store-subgrid-diagnostics", choices=["true", "false"], default=None)
    p.add_argument(
        "--sf-density-threshold-mode",
        choices=["overdensity", "physical_nH", "physical_nH_comoving"],
        default=None,
        help="Interpret sf_density_threshold as overdensity, proper hydrogen number density in cm^-3, or comoving hydrogen number density in cm^-3 scaled by a^3.",
    )
    p.add_argument("--sf-density-threshold", type=float, default=None)
    p.add_argument("--sf-density-width", type=float, default=None)
    p.add_argument("--sf-temperature-threshold-k", type=float, default=None)
    p.add_argument("--sf-temperature-width-k", type=float, default=None)
    p.add_argument("--sf-pi0", type=float, default=None)
    p.add_argument("--sf-tau", type=float, default=None)
    p.add_argument("--sf-mass-scale", type=float, default=None)
    p.add_argument("--sf-max-fraction-per-step", type=float, default=None)
    p.add_argument("--metal-yield", type=float, default=None)
    p.add_argument(
        "--stellar-paint-sigma-threshold",
        type=float,
        default=None,
        help="If set, only particles with star_mass >= mean_positive + N*sigma_positive contribute to the painted stellar mesh.",
    )
    p.add_argument(
        "--stellar-paint-min-mass",
        type=float,
        default=None,
        help="If set, only particles with star_mass >= this absolute threshold contribute to the painted stellar mesh.",
    )
    p.add_argument(
        "--cv0-init-state-npz",
        type=Path,
        default=ROOT / "outputs" / "cv0_external_ic_bundle_v2.npz",
        help="Full-state NPZ bundle used by --stage cv0.",
    )
    p.add_argument("--state-u-key", type=str, default="U_gas_init")
    p.add_argument("--state-dm-x-key", type=str, default="dm_x_init")
    p.add_argument("--state-dm-p-key", type=str, default="dm_p_or_v_init")
    p.add_argument("--state-dm-mass-key", type=str, default="dm_mass_init")
    p.add_argument("--state-a-key", type=str, default="a_init")
    p.add_argument("--state-init-mesh-key", type=str, default="init_mesh")
    p.add_argument("--override-a-init", type=float, default=None)
    p.add_argument(
        "--snf-energy",
        type=float,
        default=0.0,
        help="Single-event feedback energy injected per newly formed stellar mass in code energy-density units.",
    )
    p.add_argument(
        "--stellar-wind-energy",
        type=float,
        default=0.0,
        help="Sustained low-level feedback energy rate per existing stellar mass in code energy-density units per dtau.",
    )
    p.add_argument(
        "--feedback-energy-clip-fraction",
        type=float,
        default=None,
        help="If set, clip thermal feedback injected into each cell to at most this fraction of the pre-injection cell total energy.",
    )
    p.add_argument(
        "--enable-vacuum-momentum-cap",
        choices=["true", "false"],
        default=None,
        help="If set, cap momentum in near-vacuum hydro cells immediately after the flux solve and before reverse forces.",
    )
    p.add_argument(
        "--vacuum-momentum-rho-guard",
        type=float,
        default=None,
        help="Code-density threshold below which the near-vacuum momentum cap is applied.",
    )
    p.add_argument(
        "--vacuum-momentum-kinetic-to-thermal-max",
        type=float,
        default=None,
        help="Maximum allowed kinetic-to-thermal energy ratio for cells under the near-vacuum guard.",
    )
    p.add_argument(
        "--vacuum-momentum-internal-floor",
        type=float,
        default=None,
        help="Small internal-energy floor used by the near-vacuum momentum cap.",
    )
    p.add_argument(
        "--enable-hydro-state-repair",
        choices=["true", "false"],
        default=None,
        help="If set, repair invalid post-hydro cells to a small valid conservative state before reverse forces.",
    )
    p.add_argument(
        "--hydro-repair-rho-floor",
        type=float,
        default=None,
        help="Code-density floor used by the post-hydro invalid-cell repair.",
    )
    p.add_argument(
        "--hydro-repair-pressure-floor",
        type=float,
        default=None,
        help="Code-pressure floor used by the post-hydro invalid-cell repair.",
    )
    p.add_argument(
        "--hydro-repair-max-kinetic-to-thermal-ratio",
        type=float,
        default=None,
        help="Repair post-hydro cells when kinetic/internal energy exceeds this ratio, even if the conservative state is still finite.",
    )
    p.add_argument(
        "--feedback-deposition-mode",
        choices=["local", "kernel"],
        default="local",
        help="How metals and thermal feedback are deposited onto the gas mesh.",
    )
    p.add_argument(
        "--feedback-kernel-kind",
        type=str,
        default="3x3x3",
        help="Compact kernel family used when --feedback-deposition-mode=kernel.",
    )
    p.add_argument(
        "--disable-unified-feedback-kernel",
        action="store_true",
        default=False,
        help="Use separate kernel widths for metal, thermal, and momentum deposition instead of one shared width.",
    )
    p.add_argument(
        "--metal-kernel-width-cells",
        type=float,
        default=1.0,
        help="Kernel width in cell units for broadened metal deposition.",
    )
    p.add_argument(
        "--thermal-kernel-width-cells",
        type=float,
        default=1.0,
        help="Kernel width in cell units for broadened thermal feedback deposition.",
    )
    p.add_argument(
        "--momentum-kernel-width-cells",
        type=float,
        default=1.0,
        help="Kernel width in cell units for the smoothed source used by momentum feedback.",
    )
    p.add_argument(
        "--enable-momentum-feedback",
        action="store_true",
        default=False,
        help="Enable symmetric shell-like momentum deposition derived from the smoothed stellar source field.",
    )
    p.add_argument(
        "--feedback-momentum-scale",
        type=float,
        default=0.0,
        help="Dimensionless scale factor applied to the discrete momentum kick field.",
    )
    p.add_argument(
        "--grackle-table-npz",
        type=Path,
        default=ROOT.parent / "data" / "grackle_eq_cooling_table.npz",
        help="Prebuilt or to-be-built Grackle cooling table NPZ.",
    )
    p.add_argument(
        "--grackle-table-rebuild",
        action="store_true",
        default=False,
        help="Rebuild the Grackle cooling table NPZ with gracklepy before running.",
    )
    p.add_argument(
        "--grackle-data-file",
        type=Path,
        default=ROOT.parent / "data" / "grackle_data_files" / "input" / "CloudyData_UVB=HM2012.h5",
        help="Cloudy/Grackle HDF5 data file used by the grackle_eq backend.",
    )
    p.add_argument(
        "--grackle-use-mu-table",
        action="store_true",
        default=False,
        help="Use the tabulated Grackle mean-molecular-weight field during runtime cooling updates.",
    )
    p.add_argument(
        "--grackle-z-nodes",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit Grackle redshift nodes, e.g. --grackle-z-nodes 2 3 5 8 12",
    )
    p.add_argument("--grackle-lognH-min", type=float, default=None, help="Optional Grackle log10(n_H/cm^-3) lower bound.")
    p.add_argument("--grackle-lognH-max", type=float, default=None, help="Optional Grackle log10(n_H/cm^-3) upper bound.")
    p.add_argument("--grackle-lognH-n", type=int, default=None, help="Optional number of Grackle lognH nodes.")
    p.add_argument("--grackle-logt-min", type=float, default=None, help="Optional Grackle log10(T/K) lower bound.")
    p.add_argument("--grackle-logt-max", type=float, default=None, help="Optional Grackle log10(T/K) upper bound.")
    p.add_argument("--grackle-logt-n", type=int, default=None, help="Optional number of Grackle logT nodes.")
    p.add_argument("--output-root", type=Path, default=ROOT / "feedback" / "plots")
    return p.parse_args()


def _history_append(history: dict[str, list[float]], snapshot: dict, dtau: float) -> None:
    history["a"].append(float(snapshot["a"]))
    history["z"].append(float(snapshot["z"]))
    history["dtau"].append(float(dtau))
    history["gas_mass"].append(float(np.sum(np.asarray(snapshot["gas_density"], dtype=np.float64))))
    history["mean_temperature_k"].append(float(np.mean(np.asarray(snapshot["gas_temperature_k"], dtype=np.float64))))
    history["dm_mass"].append(float(np.sum(np.asarray(snapshot["dm_density"], dtype=np.float64))))
    if "metal_density" in snapshot:
        history["metal_mass"].append(float(np.sum(np.asarray(snapshot["metal_density"], dtype=np.float64))))
        history["mean_metallicity"].append(float(np.mean(np.asarray(snapshot["metal_fraction"], dtype=np.float64))))
    if "stellar_density" in snapshot:
        history["stellar_mass_mesh"].append(float(np.sum(np.asarray(snapshot["stellar_density"], dtype=np.float64))))
    if "particle_star_mass" in snapshot:
        history["stellar_mass_particles"].append(float(np.sum(np.asarray(snapshot["particle_star_mass"], dtype=np.float64))))
    if "sf_pi0" in snapshot:
        history["sf_mass_increment"].append(float(np.sum(np.asarray(snapshot["sf_delta_star_grid"], dtype=np.float64))))
        history["sf_pi0_mean"].append(float(np.mean(np.asarray(snapshot["sf_pi0"], dtype=np.float64))))
    if "sf_feedback_total_energy" in snapshot:
        history["feedback_total_energy"].append(float(np.sum(np.asarray(snapshot["sf_feedback_total_energy"], dtype=np.float64))))
        history["feedback_snf_energy"].append(float(np.sum(np.asarray(snapshot["sf_feedback_snf_energy"], dtype=np.float64))))
        history["feedback_stellar_wind_energy"].append(float(np.sum(np.asarray(snapshot["sf_feedback_stellar_wind_energy"], dtype=np.float64))))
    if "sf_feedback_kinetic_energy" in snapshot:
        history["feedback_kinetic_energy"].append(float(np.sum(np.asarray(snapshot["sf_feedback_kinetic_energy"], dtype=np.float64))))


def _bundle_value(snapshot: dict, key: str):
    return np.asarray(snapshot[key])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _save_stage_snapshot(
    path: Path,
    snapshot: dict,
    *,
    step_idx: int,
    dtau_value: float | None,
    U_state=None,
    params_state: dict | None = None,
    init_mesh=None,
) -> None:
    zero_mesh = np.zeros(np.asarray(snapshot["gas_density"]).shape, dtype=np.float32)
    payload = dict(
        step=np.asarray(step_idx, dtype=np.int32),
        a=np.asarray(snapshot["a"], dtype=np.float32),
        z=np.asarray(snapshot["z"], dtype=np.float32),
        dtau=np.asarray(np.nan if dtau_value is None else dtau_value, dtype=np.float32),
        gas_density=np.asarray(snapshot["gas_density"], dtype=np.float32),
        gas_temperature_k=np.asarray(snapshot["gas_temperature_k"], dtype=np.float32),
        dm_density=np.asarray(snapshot["dm_density"], dtype=np.float32),
        stellar_density=np.asarray(snapshot.get("stellar_density", zero_mesh), dtype=np.float32),
        metal_density=np.asarray(snapshot.get("metal_density", zero_mesh), dtype=np.float32),
        metal_fraction=np.asarray(snapshot.get("metal_fraction", zero_mesh), dtype=np.float32),
        sf_pi0=np.asarray(snapshot.get("sf_pi0", zero_mesh), dtype=np.float32),
        sf_z_hat=np.asarray(snapshot.get("sf_z_hat", zero_mesh), dtype=np.float32),
        sf_delta_star_grid=np.asarray(snapshot.get("sf_delta_star_grid", zero_mesh), dtype=np.float32),
        sf_metal_deposition_grid=np.asarray(snapshot.get("sf_metal_deposition_grid", zero_mesh), dtype=np.float32),
        sf_thermal_deposition_grid=np.asarray(snapshot.get("sf_thermal_deposition_grid", zero_mesh), dtype=np.float32),
        sf_feedback_momentum_x=np.asarray(snapshot.get("sf_feedback_momentum_x", zero_mesh), dtype=np.float32),
        sf_feedback_momentum_y=np.asarray(snapshot.get("sf_feedback_momentum_y", zero_mesh), dtype=np.float32),
        sf_feedback_momentum_z=np.asarray(snapshot.get("sf_feedback_momentum_z", zero_mesh), dtype=np.float32),
        sf_feedback_kinetic_energy=np.asarray(snapshot.get("sf_feedback_kinetic_energy", zero_mesh), dtype=np.float32),
        sf_feedback_total_energy=np.asarray(snapshot.get("sf_feedback_total_energy", zero_mesh), dtype=np.float32),
    )
    for key, value in snapshot.items():
        if key.startswith("sf_") and key not in payload:
            payload[key] = np.asarray(value, dtype=np.float32)
    if U_state is not None and params_state is not None:
        dm_state = params_state.get("dm", {})
        payload["U_gas_state"] = np.asarray(U_state, dtype=np.float32)
        payload["dm_x_state"] = np.asarray(dm_state.get("x"), dtype=np.float32)
        payload["dm_p_or_v_state"] = np.asarray(dm_state.get("p_or_v"), dtype=np.float32)
        payload["dm_mass_state"] = np.asarray(dm_state.get("mass"), dtype=np.float32)
        if "star_mass" in dm_state:
            payload["dm_star_mass_state"] = np.asarray(dm_state["star_mass"], dtype=np.float32)
        if "star_age_bins" in dm_state:
            payload["dm_star_age_bins_state"] = np.asarray(dm_state["star_age_bins"], dtype=np.float32)
        if "rng_sf" in params_state:
            payload["rng_sf_state"] = np.asarray(params_state["rng_sf"], dtype=np.uint32)
        if init_mesh is not None:
            payload["init_mesh_state"] = np.asarray(init_mesh, dtype=np.float32)
        payload["has_full_state"] = np.asarray(True)
    else:
        payload["has_full_state"] = np.asarray(False)
    np.savez_compressed(path, **payload)


def _build_config(args, stage: str):
    from src.full_hydro_model import FullHydroConfig

    def _opt_bool(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        return str(v).lower() == "true"

    cfg = FullHydroConfig(
        mesh_n=int(args.mesh_n),
        hydro_steps=int(args.hydro_steps),
        box_size_mpc_h=float(args.box_size_mpc_h),
        z_init=float(args.z_init),
        z_target=float(args.z_target),
        checkpoint=True,
        checkpoint_every=1,
        solver="hll",
        enable_cooling=True,
        cooling_model=str(args.cooling_model),
        temperature0=2.0e4,
        star_step_start=int(args.star_step_start),
        snf_energy=float(args.snf_energy),
        stellar_wind_energy=float(args.stellar_wind_energy),
        feedback_energy_clip_fraction=args.feedback_energy_clip_fraction,
        feedback_deposition_mode=str(args.feedback_deposition_mode),
        unified_feedback_kernel=not bool(args.disable_unified_feedback_kernel),
        feedback_kernel_kind=str(args.feedback_kernel_kind),
        metal_kernel_width_cells=float(args.metal_kernel_width_cells),
        thermal_kernel_width_cells=float(args.thermal_kernel_width_cells),
        momentum_kernel_width_cells=float(args.momentum_kernel_width_cells),
        enable_momentum_feedback=bool(args.enable_momentum_feedback),
        feedback_momentum_scale=float(args.feedback_momentum_scale),
        stellar_paint_sigma_threshold=args.stellar_paint_sigma_threshold,
        stellar_paint_min_mass=args.stellar_paint_min_mass,
        grackle_table_npz=str(Path(args.grackle_table_npz).resolve()),
        grackle_table_rebuild=bool(args.grackle_table_rebuild),
        grackle_data_file=str(Path(args.grackle_data_file).resolve()),
        grackle_use_mu_table=bool(args.grackle_use_mu_table),
        store_subgrid_diagnostics=bool(
            float(args.snf_energy) > 0.0
            or float(args.stellar_wind_energy) > 0.0
            or str(args.feedback_deposition_mode) == "kernel"
            or bool(args.enable_momentum_feedback)
        ),
    )
    if args.grackle_z_nodes is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "grackle_z_nodes": ",".join(f"{float(v):g}" for v in args.grackle_z_nodes)})
    if args.grackle_lognH_min is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "grackle_lognH_min": float(args.grackle_lognH_min)})
    if args.grackle_lognH_max is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "grackle_lognH_max": float(args.grackle_lognH_max)})
    if args.grackle_lognH_n is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "grackle_lognH_n": int(args.grackle_lognH_n)})
    if args.grackle_logt_min is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "grackle_logt_min": float(args.grackle_logt_min)})
    if args.grackle_logt_max is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "grackle_logt_max": float(args.grackle_logt_max)})
    if args.grackle_logt_n is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "grackle_logt_n": int(args.grackle_logt_n)})

    if stage in {"metallicity"}:
        cfg = FullHydroConfig(**{**cfg.__dict__, "track_metallicity": True, "metal_init": 0.02, "metal_blob_amplitude": 0.15})
    if stage in {"metal_source", "cv0"}:
        cfg = FullHydroConfig(**{**cfg.__dict__, "track_metallicity": True, "metal_init": 0.02, "metal_blob_amplitude": 0.0})
    if stage == "star_carrier":
        cfg = FullHydroConfig(
            **{
                **cfg.__dict__,
                "synthetic_star_mass_kind": "gaussian",
                "synthetic_star_mass_amplitude": 0.05,
            }
        )
    if stage == "deterministic_sf":
        cfg = FullHydroConfig(
            **{
                **cfg.__dict__,
                "enable_star_formation": True,
                "store_subgrid_diagnostics": True,
                "sf_pi0": 0.2,
                "sf_mass_scale": 0.1,
                "sf_density_threshold": 2.0,
                "sf_density_width": 0.8,
                "sf_temperature_threshold_k": 3.0e5,
                "sf_temperature_width_k": 8.0e3,
            }
        )
    if stage == "gumbel_sf":
        cfg = FullHydroConfig(
            **{
                **cfg.__dict__,
                "enable_star_formation": True,
                "star_formation_mode": "gumbel",
                "store_subgrid_diagnostics": True,
                "sf_pi0": 0.2,
                "sf_tau": 0.3,
                "sf_mass_scale": 100.0,
                "sf_density_threshold": 3.0,
                "sf_density_width": 0.8,
                "sf_temperature_threshold_k": 3.0e5,
                "sf_temperature_width_k": 8.0e3,
                "sf_seed": int(args.seed),
            }
        )
    if stage == "age_bins":
        cfg = FullHydroConfig(
            **{
                **cfg.__dict__,
                "enable_star_formation": True,
                "store_subgrid_diagnostics": True,
                "sf_pi0": 0.2,
                "sf_mass_scale": 0.1,
                "sf_density_threshold": 4.0,
                "sf_density_width": 0.8,
                "sf_temperature_threshold_k": 3.0e4,
                "sf_temperature_width_k": 8.0e3,
                "star_age_bins": 4,
                "star_age_shift_steps": max(1, int(args.hydro_steps) // 4),
            }
        )
    if stage in {"metal_source", "cv0"}:
        cfg = FullHydroConfig(
            **{
                **cfg.__dict__,
                "track_metallicity": True,
                "star_formation_mode": "gumbel",
                "metal_init": 0.01,
                "enable_star_formation": True,
                "enable_metal_source": True,
                "store_subgrid_diagnostics": True,
                "sf_pi0": 0.1,
                "sf_tau": 0.001,
                "sf_mass_scale": 100.0,
                "sf_density_threshold": 7.0,
                "sf_density_width": 0.8,
                "sf_temperature_threshold_k": 3.0e5,
                "sf_temperature_width_k": 8.0e3,
                "metal_yield": 0.4,
            }
        )

    # Explicit CLI/config overrides should win over stage presets.
    if args.sf_density_threshold_mode is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "sf_density_threshold_mode": str(args.sf_density_threshold_mode)})
    if args.state_floor is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "state_floor": float(args.state_floor)})
    if args.pressure_floor is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "pressure_floor": float(args.pressure_floor)})
    if args.hydro_temp_floor_k is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "hydro_temp_floor_k": float(args.hydro_temp_floor_k)})
    if args.cooling_temp_floor_k is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "cooling_temp_floor_k": float(args.cooling_temp_floor_k)})
    if args.sf_density_threshold is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "sf_density_threshold": float(args.sf_density_threshold)})
    if args.sf_density_width is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "sf_density_width": float(args.sf_density_width)})
    if args.sf_temperature_threshold_k is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "sf_temperature_threshold_k": float(args.sf_temperature_threshold_k)})
    if args.sf_temperature_width_k is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "sf_temperature_width_k": float(args.sf_temperature_width_k)})
    if args.sf_pi0 is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "sf_pi0": float(args.sf_pi0)})
    if args.sf_tau is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "sf_tau": float(args.sf_tau)})
    if args.sf_mass_scale is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "sf_mass_scale": float(args.sf_mass_scale)})
    if args.sf_max_fraction_per_step is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "sf_max_fraction_per_step": float(args.sf_max_fraction_per_step)})
    if args.metal_yield is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "metal_yield": float(args.metal_yield)})
    if args.feedback_energy_clip_fraction is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "feedback_energy_clip_fraction": float(args.feedback_energy_clip_fraction)})
    if args.enable_vacuum_momentum_cap is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "enable_vacuum_momentum_cap": _opt_bool(args.enable_vacuum_momentum_cap)})
    if args.vacuum_momentum_rho_guard is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "vacuum_momentum_rho_guard": float(args.vacuum_momentum_rho_guard)})
    if args.vacuum_momentum_kinetic_to_thermal_max is not None:
        cfg = FullHydroConfig(
            **{
                **cfg.__dict__,
                "vacuum_momentum_kinetic_to_thermal_max": float(args.vacuum_momentum_kinetic_to_thermal_max),
            }
        )
    if args.vacuum_momentum_internal_floor is not None:
        cfg = FullHydroConfig(
            **{
                **cfg.__dict__,
                "vacuum_momentum_internal_floor": float(args.vacuum_momentum_internal_floor),
            }
        )
    if args.enable_hydro_state_repair is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "enable_hydro_state_repair": _opt_bool(args.enable_hydro_state_repair)})
    if args.hydro_repair_rho_floor is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "hydro_repair_rho_floor": float(args.hydro_repair_rho_floor)})
    if args.hydro_repair_pressure_floor is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "hydro_repair_pressure_floor": float(args.hydro_repair_pressure_floor)})
    if args.hydro_repair_max_kinetic_to_thermal_ratio is not None:
        cfg = FullHydroConfig(
            **{
                **cfg.__dict__,
                "hydro_repair_max_kinetic_to_thermal_ratio": float(args.hydro_repair_max_kinetic_to_thermal_ratio),
            }
        )
    if args.stellar_paint_sigma_threshold is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "stellar_paint_sigma_threshold": float(args.stellar_paint_sigma_threshold)})
    if args.stellar_paint_min_mass is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "stellar_paint_min_mass": float(args.stellar_paint_min_mass)})
    if _opt_bool(args.track_metallicity) is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "track_metallicity": bool(_opt_bool(args.track_metallicity))})
    if args.metal_init is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "metal_init": float(args.metal_init)})
    if _opt_bool(args.enable_star_formation) is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "enable_star_formation": bool(_opt_bool(args.enable_star_formation))})
    if args.star_formation_mode is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "star_formation_mode": str(args.star_formation_mode)})
    if _opt_bool(args.enable_metal_source) is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "enable_metal_source": bool(_opt_bool(args.enable_metal_source))})
    if _opt_bool(args.store_subgrid_diagnostics) is not None:
        cfg = FullHydroConfig(**{**cfg.__dict__, "store_subgrid_diagnostics": bool(_opt_bool(args.store_subgrid_diagnostics))})
    return cfg


def _load_full_state_from_npz(args, system, cfg):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from src.forward_model import a_from_z
    from src.metallicity import initialize_metal_density
    from src.star_formation import initialize_star_particle_fields

    ic_path = Path(args.cv0_init_state_npz).resolve()
    if not ic_path.exists():
        raise FileNotFoundError(f"CV0 init-state bundle not found: {ic_path}")
    d = np.load(ic_path)

    for key in (args.state_u_key, args.state_dm_x_key, args.state_dm_p_key):
        if key not in d:
            raise KeyError(f"Missing '{key}' in {ic_path}")

    U = jnp.asarray(np.asarray(d[args.state_u_key], dtype=np.float32))
    if U.ndim != 4 or U.shape[1:] != (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n):
        raise ValueError(f"Unexpected gas conservative shape {tuple(U.shape)} for mesh-n={cfg.mesh_n}")

    dm_x = jnp.asarray(np.asarray(d[args.state_dm_x_key], dtype=np.float32))
    dm_p = jnp.asarray(np.asarray(d[args.state_dm_p_key], dtype=np.float32))
    if dm_x.ndim != 2 or dm_x.shape[1] != 3:
        raise ValueError(f"Unexpected dm_x shape {tuple(dm_x.shape)}")
    if dm_p.shape != dm_x.shape:
        raise ValueError(f"dm_p_or_v shape {tuple(dm_p.shape)} does not match dm_x {tuple(dm_x.shape)}")

    if args.state_dm_mass_key in d:
        dm_mass = jnp.asarray(np.asarray(d[args.state_dm_mass_key], dtype=np.float32))
    else:
        default_mass = 1.0 - float(cfg.gas_mean_fraction)
        dm_mass = jnp.ones((dm_x.shape[0],), dtype=jnp.float32) * jnp.asarray(default_mass, dtype=jnp.float32)
    if dm_mass.ndim == 0:
        dm_mass = jnp.ones((dm_x.shape[0],), dtype=jnp.float32) * dm_mass
    if dm_mass.ndim != 1 or dm_mass.shape[0] != dm_x.shape[0]:
        raise ValueError(f"Unexpected dm_mass shape {tuple(dm_mass.shape)}")

    if args.override_a_init is not None:
        a_init = float(args.override_a_init)
    elif args.state_a_key in d:
        a_init = float(np.asarray(d[args.state_a_key]).reshape(-1)[0])
    else:
        a_init = float(a_from_z(cfg.z_init))

    if bool(cfg.track_metallicity) and getattr(system.eq, "metal_density_ids", None) is not None:
        rhoZ0 = initialize_metal_density(
            U[system.eq.mass_ids],
            metal_init=float(cfg.metal_init),
            metal_floor=float(cfg.metal_floor),
            blob_amplitude=float(cfg.metal_blob_amplitude),
            blob_sigma_cells=float(cfg.metal_blob_sigma_cells),
        )
        U = U.at[int(system.eq.metal_density_ids)].set(rhoZ0)

    omega_m = float(system.cosmo_lpt.Omega_b + system.cosmo_lpt.Omega_c)
    dm_params = {
        "x": dm_x,
        "p_or_v": dm_p,
        "mass": dm_mass,
        "drift_factor": jnp.asarray(system.background.H0, dtype=jnp.float32),
        "kick_prefactor": jnp.asarray(
            1.5 * omega_m * system.background.H0 * float(cfg.dm_kick_scale),
            dtype=jnp.float32,
        ),
    }
    if cfg.gas_kick_factor is None:
        dm_params["gas_kick_prefactor"] = jnp.asarray(
            1.5 * omega_m * (system.background.H0**2) * float(cfg.gas_kick_scale),
            dtype=jnp.float32,
        )
    else:
        dm_params["gas_kick_factor"] = jnp.asarray(float(cfg.gas_kick_factor), dtype=jnp.float32)

    dm_params.update(
        initialize_star_particle_fields(
            dm_x,
            amplitude=float(cfg.synthetic_star_mass_amplitude),
            kind=str(cfg.synthetic_star_mass_kind),
            seed=int(cfg.sf_seed),
            n_age_bins=int(cfg.star_age_bins),
        )
    )
    params = {"a": jnp.asarray(a_init, dtype=jnp.float32), "dm": dm_params}
    if bool(cfg.enable_star_formation):
        params["rng_sf"] = jax.random.PRNGKey(int(cfg.sf_seed))

    init_mesh = np.asarray(d[args.state_init_mesh_key], dtype=np.float32) if args.state_init_mesh_key in d else None
    return U, params, init_mesh, str(ic_path)


def _fallback_init_state(white_noise, pk_sqrt, grid_pos, system, cfg):
    import jax
    import jax.numpy as jnp
    from diffhydro.cosmology import conversions as cosmo_conv
    from src.forward_model import a_from_z, white_noise_to_init_mesh
    from src.metallicity import initialize_metal_density
    from src.star_formation import initialize_star_particle_fields

    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    a_init = a_from_z(cfg.z_init)
    init_mesh = white_noise_to_init_mesh(white_noise, pk_sqrt)
    dm_x = jnp.asarray(grid_pos, dtype=jnp.float32)
    p_lpt = jnp.zeros_like(dm_x)
    rho_dm_norm = jnp.exp(0.15 * (init_mesh - jnp.mean(init_mesh)))

    gas_mean = jnp.asarray(float(cfg.gas_mean_fraction), dtype=jnp.float32)
    rho_gas_phys = gas_mean * rho_dm_norm
    T0_k = jnp.asarray(float(cfg.temperature0), dtype=jnp.float32)
    T_k = T0_k * jnp.power(jnp.maximum(rho_gas_phys / jnp.maximum(gas_mean, 1.0e-8), 1.0e-4), float(cfg.temperature_gamma))
    T_phys = jnp.maximum(T_k * jnp.asarray(float(system.kelvin_to_code_temp), dtype=jnp.float32), jnp.asarray(1.0e-12, dtype=jnp.float32))
    p_gas_phys = rho_gas_phys * system.eq.R * T_phys
    zeros = jnp.zeros_like(rho_gas_phys)
    w_code = cosmo_conv.primitives_phys_to_code(rho_gas_phys, zeros, zeros, zeros, p_gas_phys, a_init)
    U0 = system.eq.get_conservatives_from_primitives(w_code)
    if bool(cfg.track_metallicity) and getattr(system.eq, "metal_density_ids", None) is not None:
        rhoZ0 = initialize_metal_density(
            U0[system.eq.mass_ids],
            metal_init=float(cfg.metal_init),
            metal_floor=float(cfg.metal_floor),
            blob_amplitude=float(cfg.metal_blob_amplitude),
            blob_sigma_cells=float(cfg.metal_blob_sigma_cells),
        )
        U0 = U0.at[int(system.eq.metal_density_ids)].set(rhoZ0)

    dm_mass_fraction = float(max(1.0e-6, min(0.999999, 1.0 - float(cfg.gas_mean_fraction))))
    omega_m = float(system.cosmo_lpt.Omega_b + system.cosmo_lpt.Omega_c)
    dm_params = {
        "x": dm_x,
        "p_or_v": p_lpt,
        "mass": jnp.ones((dm_x.shape[0],), dtype=jnp.float32) * jnp.asarray(dm_mass_fraction, dtype=jnp.float32),
        "drift_factor": jnp.asarray(system.background.H0, dtype=jnp.float32),
        "kick_prefactor": jnp.asarray(1.5 * omega_m * system.background.H0 * float(cfg.dm_kick_scale), dtype=jnp.float32),
        "gas_kick_prefactor": jnp.asarray(1.5 * omega_m * (system.background.H0**2) * float(cfg.gas_kick_scale), dtype=jnp.float32),
    }
    dm_params.update(
        initialize_star_particle_fields(
            dm_x,
            amplitude=float(cfg.synthetic_star_mass_amplitude),
            kind=str(cfg.synthetic_star_mass_kind),
            seed=int(cfg.sf_seed),
            n_age_bins=int(cfg.star_age_bins),
        )
    )
    params0 = {"a": jnp.asarray(a_init, dtype=jnp.float32), "dm": dm_params}
    if bool(cfg.enable_star_formation):
        params0["rng_sf"] = jax.random.PRNGKey(int(cfg.sf_seed))
    return U0, params0, init_mesh


def _save_standard_outputs(stage_dir: Path, stage: str, initial: dict, final: dict, history: dict[str, list[float]], summary: dict) -> None:
    from src.plotting_subgrid import (
        save_histograms,
        save_multi_panel_slices,
        save_residual_panel,
        save_summary_json,
        save_timeseries,
    )

    stage_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        ("gas_density", final["gas_density"]),
        ("gas_temperature_k", final["gas_temperature_k"]),
        ("dm_density", final["dm_density"]),
    ]
    if "metal_fraction" in final:
        fields.extend([("metal_density", final["metal_density"]), ("metal_fraction", final["metal_fraction"])])
    if "stellar_density" in final:
        fields.append(("stellar_density", final["stellar_density"]))
    if "sf_pi0" in final:
        fields.extend([("sf_pi0", final["sf_pi0"]), ("sf_z_hat", final["sf_z_hat"]), ("sf_delta_star_grid", final["sf_delta_star_grid"])])
    if "sf_feedback_total_energy" in final:
        fields.extend(
            [
                ("sf_feedback_snf_energy", final["sf_feedback_snf_energy"]),
                ("sf_feedback_stellar_wind_energy", final["sf_feedback_stellar_wind_energy"]),
                ("sf_feedback_kinetic_energy", final.get("sf_feedback_kinetic_energy", np.zeros_like(final["sf_feedback_total_energy"]))),
                ("sf_feedback_total_energy", final["sf_feedback_total_energy"]),
            ]
        )
    if "sf_metal_deposition_grid" in final:
        fields.append(("sf_metal_deposition_grid", final["sf_metal_deposition_grid"]))
    if "sf_thermal_deposition_grid" in final:
        fields.append(("sf_thermal_deposition_grid", final["sf_thermal_deposition_grid"]))
    save_multi_panel_slices(
        stage_dir / "summary_panel.png",
        fields[:9],
        log_fields={
            "gas_density",
            "gas_temperature_k",
            "dm_density",
            "metal_density",
            "stellar_density",
            "sf_metal_deposition_grid",
            "sf_thermal_deposition_grid",
            "sf_feedback_kinetic_energy",
            "sf_feedback_total_energy",
        },
    )

    hist_fields = [
        ("gas_density", final["gas_density"]),
        ("gas_temperature_k", final["gas_temperature_k"]),
    ]
    if "metal_fraction" in final:
        hist_fields.append(("metal_fraction", final["metal_fraction"]))
    if "particle_star_mass" in final:
        hist_fields.append(("particle_star_mass", final["particle_star_mass"]))
    if "sf_pi0" in final:
        hist_fields.append(("sf_pi0", final["sf_pi0"]))
    if "sf_feedback_total_energy" in final:
        hist_fields.append(("sf_feedback_total_energy", final["sf_feedback_total_energy"]))
    if "sf_feedback_kinetic_energy" in final:
        hist_fields.append(("sf_feedback_kinetic_energy", final["sf_feedback_kinetic_energy"]))
    save_histograms(
        stage_dir / "histograms.png",
        hist_fields[:4],
        log_fields={"gas_density", "gas_temperature_k", "particle_star_mass", "sf_feedback_total_energy", "sf_feedback_kinetic_energy"},
    )

    series = [
        ("gas_mass", np.asarray(history["gas_mass"], dtype=np.float64)),
        ("mean_temperature_k", np.asarray(history["mean_temperature_k"], dtype=np.float64)),
        ("dtau", np.asarray(history["dtau"], dtype=np.float64)),
    ]
    if history.get("metal_mass"):
        series.append(("metal_mass", np.asarray(history["metal_mass"], dtype=np.float64)))
    if history.get("stellar_mass_particles"):
        series.append(("stellar_mass_particles", np.asarray(history["stellar_mass_particles"], dtype=np.float64)))
    if history.get("feedback_total_energy"):
        series.append(("feedback_total_energy", np.asarray(history["feedback_total_energy"], dtype=np.float64)))
    if history.get("feedback_kinetic_energy"):
        series.append(("feedback_kinetic_energy", np.asarray(history["feedback_kinetic_energy"], dtype=np.float64)))
    save_timeseries(stage_dir / "timeseries.png", series)

    if "metal_fraction" in initial and "metal_fraction" in final:
        save_residual_panel(stage_dir / "metallicity_compare.png", initial["metal_fraction"], final["metal_fraction"], title_prefix="Metallicity")
    if "stellar_density" in initial and "stellar_density" in final:
        save_residual_panel(stage_dir / "stellar_compare.png", initial["stellar_density"], final["stellar_density"], title_prefix="Stellar density")
    if "gas_temperature_k" in initial and "gas_temperature_k" in final:
        save_residual_panel(stage_dir / "temperature_compare.png", initial["gas_temperature_k"], final["gas_temperature_k"], title_prefix="Temperature")

    mesh_shape = np.asarray(final["gas_density"]).shape
    zero_mesh = np.zeros(mesh_shape, dtype=np.float32)
    np.savez_compressed(
        stage_dir / "diagnostics_bundle.npz",
        stage=np.asarray(stage),
        gas_density_initial=_bundle_value(initial, "gas_density"),
        gas_density_final=_bundle_value(final, "gas_density"),
        gas_temperature_initial=_bundle_value(initial, "gas_temperature_k"),
        gas_temperature_final=_bundle_value(final, "gas_temperature_k"),
        dm_density_initial=_bundle_value(initial, "dm_density"),
        dm_density_final=_bundle_value(final, "dm_density"),
        a_history=np.asarray(history["a"], dtype=np.float32),
        dtau_history=np.asarray(history["dtau"], dtype=np.float32),
        gas_mass_history=np.asarray(history["gas_mass"], dtype=np.float32),
        mean_temperature_history=np.asarray(history["mean_temperature_k"], dtype=np.float32),
        metal_mass_history=np.asarray(history.get("metal_mass", []), dtype=np.float32),
        stellar_mass_history=np.asarray(history.get("stellar_mass_particles", []), dtype=np.float32),
        sf_pi0_final=np.asarray(final.get("sf_pi0", zero_mesh), dtype=np.float32),
        sf_z_hat_final=np.asarray(final.get("sf_z_hat", zero_mesh), dtype=np.float32),
        sf_delta_star_grid_final=np.asarray(final.get("sf_delta_star_grid", zero_mesh), dtype=np.float32),
        feedback_total_energy_history=np.asarray(history.get("feedback_total_energy", []), dtype=np.float32),
        feedback_kinetic_energy_history=np.asarray(history.get("feedback_kinetic_energy", []), dtype=np.float32),
        feedback_snf_energy_history=np.asarray(history.get("feedback_snf_energy", []), dtype=np.float32),
        feedback_stellar_wind_energy_history=np.asarray(history.get("feedback_stellar_wind_energy", []), dtype=np.float32),
    )
    np.savez_compressed(
        stage_dir / "final_fields.npz",
        stage=np.asarray(stage),
        a_final=np.asarray(final["a"], dtype=np.float32),
        z_final=np.asarray(final["z"], dtype=np.float32),
        gas_density=np.asarray(final["gas_density"], dtype=np.float32),
        gas_temperature_k=np.asarray(final["gas_temperature_k"], dtype=np.float32),
        dm_density=np.asarray(final["dm_density"], dtype=np.float32),
        stellar_density=np.asarray(final.get("stellar_density", zero_mesh), dtype=np.float32),
        metal_density=np.asarray(final.get("metal_density", zero_mesh), dtype=np.float32),
        metal_fraction=np.asarray(final.get("metal_fraction", zero_mesh), dtype=np.float32),
        sf_pi0=np.asarray(final.get("sf_pi0", zero_mesh), dtype=np.float32),
        sf_z_hat=np.asarray(final.get("sf_z_hat", zero_mesh), dtype=np.float32),
        sf_delta_star_grid=np.asarray(final.get("sf_delta_star_grid", zero_mesh), dtype=np.float32),
        sf_metal_deposition_grid=np.asarray(final.get("sf_metal_deposition_grid", zero_mesh), dtype=np.float32),
        sf_thermal_deposition_grid=np.asarray(final.get("sf_thermal_deposition_grid", zero_mesh), dtype=np.float32),
        sf_feedback_momentum_x=np.asarray(final.get("sf_feedback_momentum_x", zero_mesh), dtype=np.float32),
        sf_feedback_momentum_y=np.asarray(final.get("sf_feedback_momentum_y", zero_mesh), dtype=np.float32),
        sf_feedback_momentum_z=np.asarray(final.get("sf_feedback_momentum_z", zero_mesh), dtype=np.float32),
        sf_feedback_kinetic_energy=np.asarray(final.get("sf_feedback_kinetic_energy", zero_mesh), dtype=np.float32),
        sf_feedback_total_energy=np.asarray(final.get("sf_feedback_total_energy", zero_mesh), dtype=np.float32),
    )
    save_summary_json(stage_dir / "summary.json", summary)


def _run_transfer_stage(args, stage_dir: Path) -> None:
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np
    from src.full_hydro_model import _paint_dm_density, _paint_particle_scalar, build_full_hydro_system, build_lpt_cosmology
    from src.plotting_subgrid import save_histograms, save_multi_panel_slices, save_residual_panel, save_summary_json
    from src.star_formation import grid_to_particle_mass_increment

    cfg = _build_config(args, "star_carrier")
    cosmo = build_lpt_cosmology(cfg)
    system = build_full_hydro_system(cfg, cosmo)
    mesh_n = int(cfg.mesh_n)
    grid = np.arange(mesh_n, dtype=np.float32)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    sigma2 = (0.12 * mesh_n) ** 2
    gauss = np.exp(-((gx - 0.5 * mesh_n) ** 2 + (gy - 0.5 * mesh_n) ** 2 + (gz - 0.5 * mesh_n) ** 2) / (2.0 * sigma2)).astype(np.float32)

    rng = jr.PRNGKey(int(args.seed))
    particle_positions = jr.uniform(rng, (mesh_n**3, 3), minval=0.0, maxval=float(mesh_n), dtype=jnp.float32)
    particle_values = grid_to_particle_mass_increment(jnp.asarray(gauss), particle_positions)
    repainted = _paint_particle_scalar(particle_positions, particle_values, (mesh_n, mesh_n, mesh_n))
    residual = np.asarray(repainted) - gauss

    stage_dir.mkdir(parents=True, exist_ok=True)
    save_multi_panel_slices(
        stage_dir / "summary_panel.png",
        [("grid_field", gauss), ("repainted", np.asarray(repainted)), ("residual", residual)],
    )
    save_histograms(stage_dir / "histograms.png", [("particle_readout", np.asarray(particle_values))], log_fields={"particle_readout"})
    save_residual_panel(stage_dir / "transfer_compare.png", gauss, np.asarray(repainted), title_prefix="Transfer")
    summary = {
        "stage": "transfer",
        "roundtrip_l2_rel": float(np.linalg.norm(residual.ravel()) / max(np.linalg.norm(gauss.ravel()), 1.0e-8)),
        "original_total": float(np.sum(gauss)),
        "repainted_total": float(np.sum(np.asarray(repainted))),
        "particle_total": float(np.sum(np.asarray(particle_values))),
        "pass_large_scale_structure": bool(np.linalg.norm(residual.ravel()) / max(np.linalg.norm(gauss.ravel()), 1.0e-8) < 0.6),
    }
    np.savez_compressed(stage_dir / "diagnostics_bundle.npz", original=gauss, repainted=np.asarray(repainted), residual=residual, particle_values=np.asarray(particle_values))
    save_summary_json(stage_dir / "summary.json", summary)


def _run_gradient_smoke(args, stage_dir: Path) -> None:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    import numpy as np

    from src.forward_model import make_lattice_positions, make_pk_sqrt
    from src.full_hydro_model import (
        FullHydroConfig,
        build_full_hydro_system,
        build_lpt_cosmology,
        evolve_full_hydro_state,
        extract_subgrid_state,
    )

    cfg = _build_config(args, "metal_source")
    cfg = FullHydroConfig(**{**cfg.__dict__, "mesh_n": 16, "hydro_steps": 6, "store_subgrid_diagnostics": False})
    cosmo = build_lpt_cosmology(cfg)
    system = build_full_hydro_system(cfg, cosmo)
    grid_pos = make_lattice_positions(cfg.mesh_n)
    pk_sqrt = make_pk_sqrt(cosmo, cfg)
    white_noise = jr.normal(jr.PRNGKey(int(args.seed)), (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n), dtype=jnp.float32)

    target_cfg = FullHydroConfig(**{**cfg.__dict__, "sf_pi0": 0.3, "metal_yield": 0.08})
    target_system = build_full_hydro_system(target_cfg, cosmo)
    _, _, _, Uf_t, params_t, _ = evolve_full_hydro_state(white_noise, pk_sqrt, grid_pos, target_system, target_cfg)
    target_snapshot = extract_subgrid_state(Uf_t, params_t, target_system, target_cfg)
    target_star = jnp.asarray(target_snapshot["stellar_density"], dtype=jnp.float32)
    target_Z = jnp.asarray(target_snapshot["metal_fraction"], dtype=jnp.float32)

    def loss_fn(theta):
        sf_pi0, metal_yield = theta
        cfg_theta = FullHydroConfig(**{**cfg.__dict__, "sf_pi0": float(sf_pi0), "metal_yield": float(metal_yield)})
        system_theta = build_full_hydro_system(cfg_theta, cosmo)
        _, _, _, Uf, paramsf, _ = evolve_full_hydro_state(white_noise, pk_sqrt, grid_pos, system_theta, cfg_theta)
        snap = extract_subgrid_state(Uf, paramsf, system_theta, cfg_theta)
        star = jnp.asarray(snap["stellar_density"], dtype=jnp.float32)
        metal = jnp.asarray(snap["metal_fraction"], dtype=jnp.float32)
        return jnp.mean((star - target_star) ** 2) + jnp.mean((metal - target_Z) ** 2)

    theta0 = jnp.asarray([cfg.sf_pi0, cfg.metal_yield], dtype=jnp.float32)
    grad_auto = jax.grad(loss_fn)(theta0)
    eps = 1.0e-2
    fd = []
    for idx in range(theta0.size):
        step = np.zeros(theta0.shape, dtype=np.float32)
        step[idx] = eps
        l_hi = float(loss_fn(theta0 + step))
        l_lo = float(loss_fn(theta0 - step))
        fd.append((l_hi - l_lo) / (2.0 * eps))
    grad_fd = np.asarray(fd, dtype=np.float32)

    stage_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    labels = ["sf_pi0", "metal_yield"]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, np.asarray(grad_auto), width=0.3, label="autodiff")
    ax.bar(x + 0.15, grad_fd, width=0.3, label="finite_diff")
    ax.set_xticks(x, labels)
    ax.set_ylabel("gradient")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(stage_dir / "gradient_compare.png", dpi=180)
    plt.close(fig)

    summary = {
        "stage": "gradient_smoke",
        "theta0": {"sf_pi0": float(theta0[0]), "metal_yield": float(theta0[1])},
        "loss": float(loss_fn(theta0)),
        "autodiff_grad": {"sf_pi0": float(grad_auto[0]), "metal_yield": float(grad_auto[1])},
        "finite_diff_grad": {"sf_pi0": float(grad_fd[0]), "metal_yield": float(grad_fd[1])},
        "pass_gradients_finite": bool(np.all(np.isfinite(np.asarray(grad_auto))) and np.all(np.isfinite(grad_fd))),
    }
    with (stage_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _run_hydro_stage(args, stage: str, stage_dir: Path) -> None:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np

    from src.forward_model import make_lattice_positions, make_pk_sqrt
    from src.full_hydro_model import (
        advance_full_hydro_step,
        build_full_hydro_system,
        build_lpt_cosmology,
        extract_subgrid_state,
        _init_hydro_state_from_white_noise,
    )

    cfg = _build_config(args, stage)
    cosmo = build_lpt_cosmology(cfg)
    system = build_full_hydro_system(cfg, cosmo)
    ic_path_used = None
    if stage == "cv0":
        U, params, init_mesh, ic_path_used = _load_full_state_from_npz(args, system, cfg)
    else:
        grid_pos = make_lattice_positions(cfg.mesh_n)
        pk_sqrt = make_pk_sqrt(cosmo, cfg)
        key = jr.PRNGKey(int(args.seed))
        white_noise = jr.normal(key, (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n), dtype=jnp.float32)

        try:
            U, params, init_mesh = _init_hydro_state_from_white_noise(white_noise, pk_sqrt, grid_pos, system, cfg)
        except Exception as exc:
            print(f"[warn] falling back to diagnostic non-LPT initializer: {type(exc).__name__}: {exc}")
            U, params, init_mesh = _fallback_init_state(white_noise, pk_sqrt, grid_pos, system, cfg)
    snapshot0 = {k: np.asarray(v) for k, v in extract_subgrid_state(U, params, system, cfg).items()}
    snap_dir = (stage_dir / args.snapshots_subdir).resolve()
    save_snapshots = int(args.save_every_steps) > 0 or bool(args.save_start_snapshot) or bool(args.save_final_snapshot)
    snapshot_manifest = {
        "stage": stage,
        "snapshots_dir": str(snap_dir),
        "save_every_steps": int(args.save_every_steps),
        "save_start_snapshot": bool(args.save_start_snapshot),
        "save_final_snapshot": bool(args.save_final_snapshot),
        "save_full_state_snapshots": bool(args.save_full_state_snapshots),
        "snapshots": [],
    }

    def maybe_save_snapshot(snapshot: dict, *, snap_idx: int, step_idx: int, dtau_value: float | None) -> None:
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"snapshot_{snap_idx:06d}.npz"
        _save_stage_snapshot(
            snap_path,
            snapshot,
            step_idx=step_idx,
            dtau_value=dtau_value,
            U_state=(U if bool(args.save_full_state_snapshots) else None),
            params_state=(params if bool(args.save_full_state_snapshots) else None),
            init_mesh=(init_mesh if bool(args.save_full_state_snapshots) else None),
        )
        snapshot_manifest["snapshots"].append(
            {
                "index": int(snap_idx),
                "filename": snap_path.name,
                "step": int(step_idx),
                "a": float(np.asarray(snapshot["a"])),
                "z": float(np.asarray(snapshot["z"])),
                "dtau": (None if dtau_value is None else float(dtau_value)),
                "has_full_state": bool(args.save_full_state_snapshots),
            }
        )
        print(
            f"[snapshot] idx={snap_idx:06d} step={step_idx} "
            f"a={float(np.asarray(snapshot['a'])):.6f} z={float(np.asarray(snapshot['z'])):.4f} "
            f"saved={snap_path}"
        )

    def maybe_raise_on_nonfinite(*, step_idx: int, dtau_value: float) -> None:
        raw_flag = params.get("_raw_nonfinite_flag", None)
        if raw_flag is None or not bool(np.asarray(raw_flag)):
            return
        failure_snapshot = {k: np.asarray(v) for k, v in extract_subgrid_state(U, params, system, cfg).items()}
        fail_path = stage_dir / f"failure_step_{step_idx:06d}.npz"
        _save_stage_snapshot(
            fail_path,
            failure_snapshot,
            step_idx=step_idx,
            dtau_value=dtau_value,
            U_state=U,
            params_state=params,
            init_mesh=init_mesh,
        )
        counts = {
            "U": int(np.asarray(params.get("_raw_nonfinite_U_count", 0))),
            "rho_nonpositive": int(np.asarray(params.get("_raw_nonpositive_rho_count", 0))),
            "dm_x": int(np.asarray(params.get("_raw_nonfinite_dm_x_count", 0))),
            "dm_p": int(np.asarray(params.get("_raw_nonfinite_dm_p_count", 0))),
            "star_mass": int(np.asarray(params.get("_raw_nonfinite_star_mass_count", 0))),
        }
        raise RuntimeError(
            "Invalid raw hydro state encountered during step "
            f"{step_idx} (dtau={dtau_value:.6e}). "
            f"Counts: U_nonfinite={counts['U']}, rho_nonpositive={counts['rho_nonpositive']}, "
            f"dm_x_nonfinite={counts['dm_x']}, dm_p_nonfinite={counts['dm_p']}, "
            f"star_mass_nonfinite={counts['star_mass']}. "
            f"Raw failure state saved to {fail_path}."
        )

    history = {
        "a": [],
        "z": [],
        "dtau": [],
        "gas_mass": [],
        "mean_temperature_k": [],
        "dm_mass": [],
        "metal_mass": [],
        "mean_metallicity": [],
        "stellar_mass_mesh": [],
        "stellar_mass_particles": [],
        "sf_mass_increment": [],
        "sf_pi0_mean": [],
        "feedback_total_energy": [],
        "feedback_kinetic_energy": [],
        "feedback_snf_energy": [],
        "feedback_stellar_wind_energy": [],
    }
    _history_append(history, snapshot0, float("nan"))
    snapshot_idx = 0
    saved_steps: set[int] = set()
    if save_snapshots and bool(args.save_start_snapshot):
        maybe_save_snapshot(snapshot0, snap_idx=snapshot_idx, step_idx=0, dtau_value=None)
        saved_steps.add(0)
        snapshot_idx += 1

    step_fn = jax.jit(lambda U_in, params_in, i_in: advance_full_hydro_step(U_in, params_in, system, cfg, i_in))
    for i in range(int(cfg.hydro_steps)):
        (U, params), dtau = step_fn(U, params, jnp.asarray(i, dtype=jnp.int32))
        jax.block_until_ready(dtau)
        dtau_f = float(np.asarray(dtau))
        step_i = i + 1
        maybe_raise_on_nonfinite(step_idx=step_i, dtau_value=dtau_f)
        snap = {k: np.asarray(v) for k, v in extract_subgrid_state(U, params, system, cfg).items()}
        _history_append(history, snap, dtau_f)
        if save_snapshots and int(args.save_every_steps) > 0 and (step_i % int(args.save_every_steps)) == 0:
            maybe_save_snapshot(snap, snap_idx=snapshot_idx, step_idx=step_i, dtau_value=dtau_f)
            saved_steps.add(step_i)
            snapshot_idx += 1

    snapshotf = {k: np.asarray(v) for k, v in extract_subgrid_state(U, params, system, cfg).items()}
    if save_snapshots and bool(args.save_final_snapshot) and int(cfg.hydro_steps) not in saved_steps:
        maybe_save_snapshot(snapshotf, snap_idx=snapshot_idx, step_idx=int(cfg.hydro_steps), dtau_value=None)
        snapshot_idx += 1
    if save_snapshots:
        snapshot_manifest["n_snapshots"] = int(len(snapshot_manifest["snapshots"]))
        _write_json(stage_dir / "snapshot_manifest.json", snapshot_manifest)
    gas_mass_hist = np.asarray(history["gas_mass"], dtype=np.float64)
    star_hist = np.asarray(history.get("stellar_mass_particles", []), dtype=np.float64)
    metal_hist = np.asarray(history.get("metal_mass", []), dtype=np.float64)
    summary = {
        "stage": stage,
        "mesh_n": int(cfg.mesh_n),
        "hydro_steps": int(cfg.hydro_steps),
        "ic_mode": "cv0_bundle" if stage == "cv0" else "white_noise",
        "cooling_model": str(cfg.cooling_model),
        "feedback_deposition_mode": str(cfg.feedback_deposition_mode),
        "feedback_kernel_kind": str(cfg.feedback_kernel_kind),
        "unified_feedback_kernel": bool(cfg.unified_feedback_kernel),
        "metal_kernel_width_cells": float(cfg.metal_kernel_width_cells),
        "thermal_kernel_width_cells": float(cfg.thermal_kernel_width_cells),
        "momentum_kernel_width_cells": float(cfg.momentum_kernel_width_cells),
        "enable_momentum_feedback": bool(cfg.enable_momentum_feedback),
        "feedback_momentum_scale": float(cfg.feedback_momentum_scale),
        "stellar_paint_sigma_threshold": cfg.stellar_paint_sigma_threshold,
        "stellar_paint_min_mass": cfg.stellar_paint_min_mass,
        "grackle_table_npz": str(cfg.grackle_table_npz),
        "grackle_data_file": str(cfg.grackle_data_file),
        "gas_mass_initial": float(gas_mass_hist[0]),
        "gas_mass_final": float(gas_mass_hist[-1]),
        "gas_mass_rel_drift": float((gas_mass_hist[-1] - gas_mass_hist[0]) / max(abs(gas_mass_hist[0]), 1.0e-8)),
        "mean_temperature_initial": float(np.asarray(history["mean_temperature_k"], dtype=np.float64)[0]),
        "mean_temperature_final": float(np.asarray(history["mean_temperature_k"], dtype=np.float64)[-1]),
        "has_nan_final": bool(any(np.isnan(np.asarray(v, dtype=np.float32)).any() for v in snapshotf.values() if np.asarray(v).dtype.kind in {"f", "c"})),
    }
    if ic_path_used is not None:
        summary["ic_path"] = ic_path_used
    if metal_hist.size > 0:
        summary["metal_mass_initial"] = float(metal_hist[0])
        summary["metal_mass_final"] = float(metal_hist[-1])
        summary["metal_mass_rel_change"] = float((metal_hist[-1] - metal_hist[0]) / max(abs(metal_hist[0]), 1.0e-8))
    if star_hist.size > 0:
        summary["stellar_mass_initial"] = float(star_hist[0])
        summary["stellar_mass_final"] = float(star_hist[-1])
        summary["stellar_mass_monotonic"] = bool(np.all(np.diff(star_hist) >= -1.0e-6))
    if "sf_pi0" in snapshotf:
        pi0 = np.asarray(snapshotf["sf_pi0"], dtype=np.float32)
        summary["sf_pi0_min"] = float(np.min(pi0))
        summary["sf_pi0_max"] = float(np.max(pi0))
    if history.get("feedback_total_energy"):
        feedback_hist = np.asarray(history["feedback_total_energy"], dtype=np.float64)
        summary["feedback_total_energy_initial"] = float(feedback_hist[0])
        summary["feedback_total_energy_final"] = float(feedback_hist[-1])
        summary["feedback_total_energy_cumulative"] = float(np.sum(feedback_hist))
        if history.get("feedback_kinetic_energy"):
            feedback_kin_hist = np.asarray(history["feedback_kinetic_energy"], dtype=np.float64)
            summary["feedback_kinetic_energy_initial"] = float(feedback_kin_hist[0])
            summary["feedback_kinetic_energy_final"] = float(feedback_kin_hist[-1])
            summary["feedback_kinetic_energy_cumulative"] = float(np.sum(feedback_kin_hist))
        summary["snf_energy_parameter"] = float(cfg.snf_energy)
        summary["stellar_wind_energy_parameter"] = float(cfg.stellar_wind_energy)
    if "star_age_bins" in snapshotf:
        age_bins = np.asarray(snapshotf["star_age_bins"], dtype=np.float32)
        summary["age_bins_match_total"] = bool(
            np.allclose(
                np.sum(age_bins, axis=1),
                np.asarray(snapshotf["particle_star_mass"], dtype=np.float32),
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        )

    if stage in {"baseline", "layout_refactor"}:
        summary["pass_baseline"] = bool((not summary["has_nan_final"]) and abs(summary["gas_mass_rel_drift"]) < 5.0e-2)
    elif stage == "metallicity":
        summary["pass_metal_conservation"] = bool(abs(summary["metal_mass_rel_change"]) < 5.0e-2 and np.min(snapshotf["metal_density"]) >= -1.0e-8)
    elif stage == "star_carrier":
        summary["pass_star_conservation"] = bool(np.allclose(star_hist[-1], star_hist[0], rtol=1.0e-6, atol=1.0e-6))
    elif stage in {"deterministic_sf", "gumbel_sf"}:
        summary["pass_sf_bounds"] = bool(summary["sf_pi0_min"] >= -1.0e-6 and summary["sf_pi0_max"] <= 1.0 + 1.0e-6 and summary.get("stellar_mass_monotonic", False))
    elif stage == "age_bins":
        summary["pass_age_bins"] = bool(summary.get("age_bins_match_total", False) and summary.get("stellar_mass_monotonic", False))
    elif stage in {"metal_source", "cv0"}:
        formed_total = np.asarray(history["stellar_mass_particles"], dtype=np.float64)[-1] - np.asarray(history["stellar_mass_particles"], dtype=np.float64)[0]
        expected_metals = float(cfg.metal_yield) * formed_total
        actual_metals = metal_hist[-1] - metal_hist[0]
        summary["expected_metal_increase"] = float(expected_metals)
        summary["actual_metal_increase"] = float(actual_metals)
        summary["pass_metal_budget"] = bool(abs(actual_metals - expected_metals) / max(abs(expected_metals), 1.0e-8) < 5.0e-2)

    _save_standard_outputs(stage_dir, stage, snapshot0, snapshotf, history, summary)


def main() -> None:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if args.xla_preallocate else "false"
    os.environ.setdefault("MPLBACKEND", "Agg")

    import numpy as np

    t0 = time.perf_counter()
    stage_dir = Path(args.output_root) / {
        "baseline": "stage_00_baseline",
        "layout_refactor": "stage_01_layout_refactor",
        "metallicity": "stage_02_passive_metallicity",
        "star_carrier": "stage_03_dm_stellar_field",
        "transfer": "stage_04_transfer_operators",
        "deterministic_sf": "stage_05_deterministic_sf",
        "gumbel_sf": "stage_06_gumbel_sf",
        "age_bins": "stage_07_age_bins",
        "metal_source": "stage_08_metal_source",
        "cv0": "stage_10_cv0",
        "gradient_smoke": "stage_09_gradient_smoke_tests",
    }[args.stage]

    if args.stage == "transfer":
        _run_transfer_stage(args, stage_dir)
    elif args.stage == "gradient_smoke":
        _run_gradient_smoke(args, stage_dir)
    else:
        _run_hydro_stage(args, args.stage, stage_dir)

    elapsed = time.perf_counter() - t0
    print(f"[ok] stage={args.stage} wrote outputs to {stage_dir}")
    print(f"[ok] elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
