#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Re-run full coupled JaxPM+DiffHydro evolution and dump numbered 3D snapshots "
            "for animation workflows."
        )
    )

    p.add_argument("--gpu", type=str, default="3", help="CUDA_VISIBLE_DEVICES value.")
    p.add_argument("--xla-preallocate", action="store_true", default=False)

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

    p.add_argument("--hydro-steps", type=int, default=128)
    p.add_argument("--dtau-min", type=float, default=2.0e-7)
    p.add_argument("--dtau-max", type=float, default=8.0e-2)
    p.add_argument("--solver", choices=["hll", "hllc", "lf", "laxfriedrichs", "nyx"], default="hll")
    p.add_argument("--state-floor", type=float, default=2.0e-8)
    p.add_argument("--pressure-floor", type=float, default=2.0e-8)
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
            "to reduce memory during differentiable runs."
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

    p.add_argument(
        "--init-white-noise",
        type=Path,
        default=None,
        help="Path to white-noise IC (.npy or .npz). Mutually exclusive with --init-state-npz.",
    )
    p.add_argument(
        "--init-white-noise-key",
        type=str,
        default="white_noise",
        help="Key used when --init-white-noise points to a .npz file.",
    )
    p.add_argument(
        "--init-state-npz",
        type=Path,
        default=None,
        help=(
            "Path to full IC state NPZ. Expected keys include "
            "U_gas_init, dm_x_init, dm_p_or_v_init (and optional dm_mass_init, a_init)."
        ),
    )
    p.add_argument("--state-u-key", type=str, default="U_gas_init")
    p.add_argument("--state-dm-x-key", type=str, default="dm_x_init")
    p.add_argument("--state-dm-p-key", type=str, default="dm_p_or_v_init")
    p.add_argument("--state-dm-mass-key", type=str, default="dm_mass_init")
    p.add_argument("--state-a-key", type=str, default="a_init")
    p.add_argument(
        "--override-a-init",
        type=float,
        default=None,
        help="Override initial scale factor even if IC file contains an a_init key.",
    )
    p.add_argument(
        "--save-initial-state-bundle",
        dest="save_initial_state_bundle",
        action="store_true",
        default=True,
        help="Write a reusable full-state IC bundle to output dir.",
    )
    p.add_argument(
        "--no-save-initial-state-bundle",
        dest="save_initial_state_bundle",
        action="store_false",
    )

    p.add_argument("--output-dir", type=Path, default=Path("cosmo_reconstruct/outputs/animation_snapshots"))
    p.add_argument("--snapshots-subdir", type=str, default="snapshots")
    p.add_argument(
        "--save-every-steps",
        type=int,
        default=1,
        help="Save a snapshot every N hydro steps (N>=1).",
    )
    p.add_argument("--save-start", dest="save_start", action="store_true", default=True)
    p.add_argument("--no-save-start", dest="save_start", action="store_false")
    p.add_argument("--save-final", dest="save_final", action="store_true", default=True)
    p.add_argument("--no-save-final", dest="save_final", action="store_false")
    p.add_argument("--log-every-steps", type=int, default=10)

    return p.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _as_float(x) -> float:
    return float(x)


def main() -> None:
    args = _parse_args()

    if (args.init_white_noise is None) == (args.init_state_npz is None):
        raise ValueError("Specify exactly one of --init-white-noise or --init-state-npz.")
    if args.save_every_steps < 1:
        raise ValueError("--save-every-steps must be >= 1.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if args.xla_preallocate else "false"
    os.environ.setdefault("MPLBACKEND", "Agg")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from diffhydro.cosmology import conversions as cosmo_conv
    from jaxpm.painting import cic_paint

    from src.forward_model import a_from_z, make_lattice_positions, make_pk_sqrt
    from src.full_hydro_model import (
        CosmologicalHydrogenCoolingForce,
        FullHydroConfig,
        NyxTabulatedCoolingForce,
        _init_hydro_state_from_white_noise,
        build_full_hydro_system,
        build_lpt_cosmology,
        prime_system_growth_cache,
    )

    if not any(dev.platform == "gpu" for dev in jax.devices()):
        raise RuntimeError("GPU backend is required")

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

    out_dir = args.output_dir.resolve()
    snap_dir = (out_dir / args.snapshots_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    cosmo_pk = build_lpt_cosmology(cfg)
    cosmo_dyn = build_lpt_cosmology(cfg)
    system = build_full_hydro_system(cfg, cosmo_dyn)
    prime_system_growth_cache(system, cfg)
    grid_pos = make_lattice_positions(cfg.mesh_n)
    pk_sqrt = make_pk_sqrt(cosmo_pk, cfg)

    white_noise_used = None
    init_mesh = None

    if args.init_white_noise is not None:
        ic_path = args.init_white_noise.resolve()
        if ic_path.suffix.lower() == ".npz":
            d = np.load(ic_path)
            if args.init_white_noise_key not in d:
                raise KeyError(f"Missing key '{args.init_white_noise_key}' in {ic_path}")
            white_noise = np.asarray(d[args.init_white_noise_key], dtype=np.float32)
        else:
            white_noise = np.asarray(np.load(ic_path), dtype=np.float32)
        if white_noise.shape != (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n):
            raise ValueError(
                f"White-noise shape mismatch: {white_noise.shape} vs {(cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)}"
            )
        white_noise_used = white_noise
        U, params, init_mesh_j = _init_hydro_state_from_white_noise(
            jnp.asarray(white_noise, dtype=jnp.float32),
            pk_sqrt,
            grid_pos,
            system,
            cfg,
        )
        init_mesh = np.asarray(init_mesh_j, dtype=np.float32)
        ic_mode = "white_noise"
        ic_path_used = str(ic_path)
    else:
        ic_path = args.init_state_npz.resolve()
        d = np.load(ic_path)
        if args.state_u_key not in d:
            raise KeyError(f"Missing '{args.state_u_key}' in {ic_path}")
        if args.state_dm_x_key not in d:
            raise KeyError(f"Missing '{args.state_dm_x_key}' in {ic_path}")
        if args.state_dm_p_key not in d:
            raise KeyError(f"Missing '{args.state_dm_p_key}' in {ic_path}")

        U = jnp.asarray(np.asarray(d[args.state_u_key], dtype=np.float32))
        if U.ndim != 4 or U.shape[1:] != (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n):
            raise ValueError(
                f"Unexpected gas conservative shape {tuple(U.shape)} for mesh-n={cfg.mesh_n}."
            )
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

        params = {
            "a": jnp.asarray(a_init, dtype=jnp.float32),
            "dm": dm_params,
        }
        init_mesh = np.asarray(d["init_mesh"], dtype=np.float32) if "init_mesh" in d else None
        ic_mode = "full_state"
        ic_path_used = str(ic_path)

    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    a_target = jnp.asarray(a_from_z(cfg.z_target), dtype=jnp.float32)
    n_steps = int(cfg.hydro_steps)
    if n_steps < 1:
        raise ValueError("hydro-steps must be >= 1")

    t_floor_k = max(float(cfg.hydro_temp_floor_k), 0.0)
    if t_floor_k <= 0.0 and bool(cfg.enable_cooling):
        t_floor_k = max(float(cfg.cooling_temp_floor_k), 0.0)
    t_floor_code = jnp.asarray(float(system.kelvin_to_code_temp) * float(t_floor_k), dtype=jnp.float32)

    cooling_force = None
    for force in system.sim.forces:
        if isinstance(force, (CosmologicalHydrogenCoolingForce, NyxTabulatedCoolingForce)):
            cooling_force = force
            break

    @jax.jit
    def step_once(U_in, params_in, i_step):
        i_f = jnp.asarray(i_step, dtype=jnp.float32)
        remaining = jnp.maximum(jnp.asarray(float(n_steps), dtype=jnp.float32) - i_f, 1.0)
        a_now = jnp.asarray(params_in["a"], dtype=jnp.float32)
        da_dtau = system.background.da_dtau(a_now)
        dtau_raw = (a_target - a_now) / jnp.maximum(da_dtau * remaining, 1.0e-12)
        dtau = jnp.clip(dtau_raw, float(cfg.dtau_min), float(cfg.dtau_max))

        U_new, params_new = system.sim._hydrostep(i_step, (U_in, params_in), dtau)

        w_new = system.eq.get_primitives_from_conservatives(U_new)
        w_new = jnp.nan_to_num(w_new, nan=0.0, posinf=0.0, neginf=0.0)
        a_new = jnp.asarray(params_new.get("a", a_now), dtype=jnp.float32)
        rho_floor = cosmo_conv.density_phys_to_code(jnp.asarray(float(cfg.state_floor), dtype=w_new.dtype), a_new)
        p_floor = cosmo_conv.pressure_phys_to_code(jnp.asarray(float(cfg.pressure_floor), dtype=w_new.dtype), a_new)
        if float(t_floor_k) > 0.0:
            rho_phys_new = cosmo_conv.density_code_to_phys(w_new[0], a_new)
            rho_phys_new = jnp.nan_to_num(
                rho_phys_new,
                nan=float(cfg.state_floor),
                posinf=float(cfg.state_floor),
                neginf=float(cfg.state_floor),
            )
            p_floor_temp_phys = rho_phys_new * jnp.asarray(float(system.eq.R), dtype=w_new.dtype) * t_floor_code
            p_floor_temp_code = cosmo_conv.pressure_phys_to_code(p_floor_temp_phys, a_new)
            p_floor = jnp.maximum(p_floor, p_floor_temp_code)

        w_new = w_new.at[0].set(jnp.maximum(w_new[0], rho_floor))
        w_new = w_new.at[4].set(jnp.maximum(w_new[4], p_floor))
        for i_passive in tuple(getattr(system.eq, "passive_ids", ())):
            if int(i_passive) >= 5:
                w_new = w_new.at[int(i_passive)].set(jnp.maximum(w_new[int(i_passive)], system.eq.eps))
        U_proj = system.eq.get_conservatives_from_primitives(w_new)
        return U_proj, params_new, dtau

    def snapshot_fields(U_state, params_state):
        a = jnp.asarray(params_state["a"], dtype=jnp.float32)
        w = system.eq.get_primitives_from_conservatives(U_state)
        rho_code = jnp.maximum(w[0], system.eq.eps)
        p_code = jnp.maximum(w[4], system.eq.eps)

        rho_gas_phys = jnp.maximum(cosmo_conv.density_code_to_phys(rho_code, a), 1.0e-30)
        p_gas_phys = jnp.maximum(cosmo_conv.pressure_code_to_phys(p_code, a), 1.0e-30)
        t_code = p_gas_phys / jnp.maximum(rho_gas_phys * system.eq.R, 1.0e-30)
        temp_kelvin = t_code * jnp.asarray(float(system.code_to_kelvin_temp), dtype=jnp.float32)

        if cooling_force is not None:
            temp_kelvin = cooling_force.temperature_kelvin_from_code(rho_code, p_code, a)
            cool_rate_cgs = cooling_force.cooling_rate_cgs_from_code(rho_code, p_code, a)
            _, _, _, p_cgs = cooling_force._code_thermo_to_cgs(rho_code, p_code, a)
            e_th_cgs = p_cgs / (float(cooling_force.gamma) - 1.0)
            cool_time_s = jnp.abs(e_th_cgs / jnp.maximum(jnp.abs(cool_rate_cgs), 1.0e-40))
        else:
            cool_rate_cgs = jnp.zeros(mesh_shape, dtype=jnp.float32)
            cool_time_s = jnp.zeros(mesh_shape, dtype=jnp.float32)

        vx_phys = cosmo_conv.velocity_code_to_phys(w[1], a)
        vy_phys = cosmo_conv.velocity_code_to_phys(w[2], a)
        vz_phys = cosmo_conv.velocity_code_to_phys(w[3], a)
        v_unit = jnp.asarray(float(system.hydro_vel_unit_cms), dtype=jnp.float32)
        vx_cms = vx_phys * v_unit
        vy_cms = vy_phys * v_unit
        vz_cms = vz_phys * v_unit

        dm_x = jnp.asarray(params_state["dm"]["x"], dtype=jnp.float32)
        dm_mass = jnp.asarray(params_state["dm"].get("mass", 1.0), dtype=jnp.float32)
        if dm_mass.ndim == 0:
            dm_mass = jnp.ones((dm_x.shape[0],), dtype=jnp.float32) * dm_mass
        dm_density = cic_paint(jnp.zeros(mesh_shape, dtype=jnp.float32), dm_x, weight=dm_mass)
        dm_density_norm = dm_density / (jnp.mean(dm_density) + 1.0e-8)

        gas_density_norm = rho_gas_phys / (jnp.mean(rho_gas_phys) + 1.0e-8)

        return {
            "a": _as_float(a),
            "z": _as_float((1.0 / jnp.maximum(a, 1.0e-12)) - 1.0),
            "gas_density_phys": np.asarray(rho_gas_phys, dtype=np.float32),
            "gas_density_norm": np.asarray(gas_density_norm, dtype=np.float32),
            "gas_pressure_phys": np.asarray(p_gas_phys, dtype=np.float32),
            "gas_temperature_kelvin": np.asarray(temp_kelvin, dtype=np.float32),
            "gas_velocity_x_cms": np.asarray(vx_cms, dtype=np.float32),
            "gas_velocity_y_cms": np.asarray(vy_cms, dtype=np.float32),
            "gas_velocity_z_cms": np.asarray(vz_cms, dtype=np.float32),
            "dm_density": np.asarray(dm_density, dtype=np.float32),
            "dm_density_norm": np.asarray(dm_density_norm, dtype=np.float32),
            "cooling_rate_cgs": np.asarray(cool_rate_cgs, dtype=np.float32),
            "cooling_time_s": np.asarray(cool_time_s, dtype=np.float32),
        }

    def save_snapshot(snapshot_idx: int, step_idx: int, dtau_value: float):
        fields = snapshot_fields(U, params)
        snap_path = snap_dir / f"snapshot_{snapshot_idx:06d}.npz"
        np.savez_compressed(
            snap_path,
            snapshot_index=np.asarray(snapshot_idx, dtype=np.int32),
            step=np.asarray(step_idx, dtype=np.int32),
            a=np.asarray(fields["a"], dtype=np.float32),
            z=np.asarray(fields["z"], dtype=np.float32),
            dtau=np.asarray(dtau_value, dtype=np.float32),
            gas_density_phys=fields["gas_density_phys"],
            gas_density_norm=fields["gas_density_norm"],
            gas_pressure_phys=fields["gas_pressure_phys"],
            gas_temperature_kelvin=fields["gas_temperature_kelvin"],
            gas_velocity_x_cms=fields["gas_velocity_x_cms"],
            gas_velocity_y_cms=fields["gas_velocity_y_cms"],
            gas_velocity_z_cms=fields["gas_velocity_z_cms"],
            dm_density=fields["dm_density"],
            dm_density_norm=fields["dm_density_norm"],
            cooling_rate_cgs=fields["cooling_rate_cgs"],
            cooling_time_s=fields["cooling_time_s"],
        )
        return snap_path, fields

    manifest = {
        "run": {
            "mesh_n": cfg.mesh_n,
            "box_size_mpc_h": cfg.box_size_mpc_h,
            "z_init": cfg.z_init,
            "z_target": cfg.z_target,
            "hydro_steps": cfg.hydro_steps,
            "checkpoint_every": cfg.checkpoint_every,
            "save_every_steps": int(args.save_every_steps),
            "save_start": bool(args.save_start),
            "save_final": bool(args.save_final),
            "solver": cfg.solver,
            "enable_cooling": bool(cfg.enable_cooling),
            "cooling_model": cfg.cooling_model,
            "output_dir": str(out_dir),
            "snapshots_dir": str(snap_dir),
        },
        "ics": {
            "mode": ic_mode,
            "path": ic_path_used,
            "has_white_noise": bool(white_noise_used is not None),
            "has_init_mesh": bool(init_mesh is not None),
        },
        "snapshots": [],
    }

    if white_noise_used is not None:
        np.save(out_dir / "initial_white_noise.npy", np.asarray(white_noise_used, dtype=np.float32))
    if args.save_initial_state_bundle:
        dm_bundle = params["dm"]
        np.savez_compressed(
            out_dir / "initial_state_bundle.npz",
            U_gas_init=np.asarray(U, dtype=np.float32),
            dm_x_init=np.asarray(dm_bundle["x"], dtype=np.float32),
            dm_p_or_v_init=np.asarray(dm_bundle["p_or_v"], dtype=np.float32),
            dm_mass_init=np.asarray(dm_bundle.get("mass", np.array([], dtype=np.float32)), dtype=np.float32),
            a_init=np.asarray(float(params["a"]), dtype=np.float32),
            init_mesh=(np.asarray(init_mesh, dtype=np.float32) if init_mesh is not None else np.zeros(mesh_shape, dtype=np.float32)),
        )

    t0 = time.perf_counter()
    snapshot_idx = 0
    saved_steps: set[int] = set()

    if args.save_start:
        snap_path, fields = save_snapshot(snapshot_idx, 0, float("nan"))
        manifest["snapshots"].append(
            {
                "index": snapshot_idx,
                "filename": snap_path.name,
                "step": 0,
                "a": float(fields["a"]),
                "z": float(fields["z"]),
                "dtau": None,
            }
        )
        print(
            f"[snapshot] idx={snapshot_idx:06d} step=0 a={float(fields['a']):.6f} z={float(fields['z']):.4f} "
            f"saved={snap_path}"
        )
        saved_steps.add(0)
        snapshot_idx += 1

    for i in range(n_steps):
        U, params, dtau = step_once(U, params, jnp.asarray(i, dtype=jnp.int32))
        dtau_f = float(dtau)
        step_i = i + 1

        if args.log_every_steps > 0 and (step_i % args.log_every_steps == 0 or step_i == n_steps):
            a_now = float(params["a"])
            z_now = (1.0 / max(a_now, 1.0e-12)) - 1.0
            print(
                f"[evolve] step={step_i}/{n_steps} a={a_now:.6f} z={z_now:.4f} "
                f"dtau={dtau_f:.3e}"
            )

        if (step_i % args.save_every_steps) == 0:
            snap_path, fields = save_snapshot(snapshot_idx, step_i, dtau_f)
            manifest["snapshots"].append(
                {
                    "index": snapshot_idx,
                    "filename": snap_path.name,
                    "step": step_i,
                    "a": float(fields["a"]),
                    "z": float(fields["z"]),
                    "dtau": dtau_f,
                }
            )
            print(
                f"[snapshot] idx={snapshot_idx:06d} step={step_i} a={float(fields['a']):.6f} "
                f"z={float(fields['z']):.4f} saved={snap_path}"
            )
            saved_steps.add(step_i)
            snapshot_idx += 1

    if args.save_final and (n_steps not in saved_steps):
        snap_path, fields = save_snapshot(snapshot_idx, n_steps, float("nan"))
        manifest["snapshots"].append(
            {
                "index": snapshot_idx,
                "filename": snap_path.name,
                "step": n_steps,
                "a": float(fields["a"]),
                "z": float(fields["z"]),
                "dtau": None,
            }
        )
        print(
            f"[snapshot] idx={snapshot_idx:06d} step={n_steps} a={float(fields['a']):.6f} "
            f"z={float(fields['z']):.4f} saved={snap_path}"
        )

    final_a = float(params["a"])
    target_a = float(a_target)
    manifest["run"]["elapsed_s"] = time.perf_counter() - t0
    manifest["run"]["target_a"] = target_a
    manifest["run"]["final_a"] = final_a
    manifest["run"]["target_z"] = cfg.z_target
    manifest["run"]["final_z"] = (1.0 / max(final_a, 1.0e-12)) - 1.0
    manifest["run"]["reached_target_a"] = bool(final_a >= target_a)
    _write_json(out_dir / "snapshot_manifest.json", manifest)
    if final_a < target_a:
        print(
            f"[warn] final scale factor {final_a:.6f} did not reach target {target_a:.6f}. "
            "Consider increasing --hydro-steps or --dtau-max."
        )
    print(f"[ok] wrote {len(manifest['snapshots'])} snapshots to {snap_dir}")
    print(f"[ok] manifest: {out_dir / 'snapshot_manifest.json'}")


if __name__ == "__main__":
    main()
