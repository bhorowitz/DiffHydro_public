#!/usr/bin/env python3
"""Nyx-focused DiffHydro benchmark against Nyx plotfile snapshots.

This script is a Nyx-only variant of the external-IC benchmark:
- Reads Nyx AMReX plotfiles for gas grid fields (density, momenta, Temp)
- Reads Nyx DM particle HDF5 files (x,y,z,m,vx,vy,vz)
- Evolves gas with cosmological background plus optional gravity and cooling
- Compares final DiffHydro state to Nyx final state with metrics + plots

Outputs are written to ./diffhydro_nyx by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nyx-only DiffHydro benchmark with optional gravity/cooling."
    )

    parser.add_argument(
        "--nyx-ic-plotfile",
        type=Path,
        default=Path("Nyx/Exec/LyA/runs_a0p1/2_hydro_grav_nocool/plt00000"),
        help="Nyx IC plotfile directory.",
    )
    parser.add_argument(
        "--nyx-final-plotfile",
        type=Path,
        default=Path("Nyx/Exec/LyA/runs_a0p1/2_hydro_grav_nocool/plt00233"),
        help="Nyx final plotfile directory.",
    )
    parser.add_argument(
        "--nyx-dm-ic-h5",
        type=Path,
        default=Path(
            "Nyx/Exec/LyA/runs_a0p1/2_hydro_grav_nocool/hdf5_particles_ic_final/plt00000_particles.h5"
        ),
        help="Nyx DM particle HDF5 for IC.",
    )
    parser.add_argument(
        "--nyx-dm-final-h5",
        type=Path,
        default=Path(
            "Nyx/Exec/LyA/runs_a0p1/2_hydro_grav_nocool/hdf5_particles_ic_final/plt00233_particles.h5"
        ),
        help="Nyx DM particle HDF5 for final snapshot.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diffhydro_nyx"),
        help="Output directory for metrics and plots.",
    )
    parser.add_argument(
        "--snapshot-fields-dir",
        type=Path,
        default=Path("diffhydro_nyx/snapshots"),
        help="Directory for saved 3D snapshot arrays.",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=None,
        help="Grid size per side. Default: inferred from Nyx IC plotfile.",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=None,
        help="Box size in Nyx code length units. Default: inferred from Nyx IC plotfile.",
    )
    parser.add_argument(
        "--z-init",
        type=float,
        default=None,
        help="Initial redshift override. Default: read from Nyx comoving_a.",
    )
    parser.add_argument(
        "--z-final",
        type=float,
        default=None,
        help="Final redshift override. Default: read from Nyx comoving_a.",
    )

    parser.add_argument("--h", type=float, default=0.71, help="Cosmology h parameter.")
    parser.add_argument("--omega-m", type=float, default=0.27, help="Omega_m.")
    parser.add_argument("--omega-b", type=float, default=0.044, help="Omega_b.")
    parser.add_argument("--omega-k", type=float, default=0.0, help="Omega_k.")
    parser.add_argument("--w0", type=float, default=-1.0, help="Dark-energy w0.")
    parser.add_argument("--wa", type=float, default=0.0, help="Dark-energy wa.")
    parser.add_argument("--sigma8", type=float, default=0.8, help="sigma8.")
    parser.add_argument("--n-s", type=float, default=0.96, help="Scalar spectral index.")

    parser.add_argument(
        "--gas-mean-fraction",
        type=float,
        default=1.58e-1,
        help="Mean gas density fraction in code units.",
    )
    parser.add_argument(
        "--dm-kick-scale",
        type=float,
        default=1.0,
        help="Scale factor for DM kick prefactor.",
    )
    parser.add_argument(
        "--gas-kick-scale",
        type=float,
        default=1.0,
        help="Scale factor for gas kick prefactor under selected gas-kick mode.",
    )
    parser.add_argument(
        "--gas-kick-mode",
        type=str,
        choices=["dm_consistent", "legacy"],
        default="dm_consistent",
        help=(
            "Gas kick prefactor model when --gas-kick-factor is not set. "
            "'dm_consistent' uses 1.5*Omega_m*H0^2 (unit-consistent with gas velocity scaling); "
            "'legacy' uses 1.5*Omega_m*H0*(H0*n_grid/box)."
        ),
    )
    parser.add_argument(
        "--gas-kick-factor",
        type=float,
        default=None,
        help="Absolute gas kick factor override (otherwise prefactor*a is used).",
    )
    parser.add_argument(
        "--force-eps",
        type=float,
        default=1.0e-10,
        help="Numerical epsilon for gravity force source.",
    )
    parser.add_argument(
        "--use-gravity",
        dest="use_gravity",
        action="store_true",
        default=True,
        help="Enable self-gravity coupling (gas+DM PM force).",
    )
    parser.add_argument(
        "--no-gravity",
        dest="use_gravity",
        action="store_false",
        help="Disable gravity coupling to match Nyx do_grav=0 runs.",
    )
    parser.add_argument(
        "--enable-cooling",
        action="store_true",
        default=False,
        help="Enable cosmological hydrogen cooling source term.",
    )
    parser.add_argument(
        "--cooling-model",
        type=str,
        choices=["legacy", "nyx_table"],
        default="legacy",
        help="Cooling backend: legacy n_H^2*Lambda(T) table or Nyx tabulated (z,delta,T) source terms.",
    )
    parser.add_argument(
        "--cooling-table",
        type=Path,
        default=Path("data/m-00.cie"),
        help="Cooling table with columns: log10(T/K), log10(Lambda)-20.",
    )
    parser.add_argument(
        "--heating-rate-per-h",
        type=float,
        default=1.0e-33,
        help="Optional uniform heating rate per H atom [erg s^-1].",
    )
    parser.add_argument(
        "--nyx-heating-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to Nyx tabulated heating branch.",
    )
    parser.add_argument(
        "--cooling-rate-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale factor applied to cooling branch.",
    )
    parser.add_argument(
        "--cooling-temp-floor-k",
        type=float,
        default=1.0,
        help="Temperature floor for cooling integration [K].",
    )
    parser.add_argument(
        "--cooling-subcycles",
        type=int,
        default=8,
        help="Cooling subcycles per hydro force application.",
    )
    parser.add_argument(
        "--cooling-dtmax-s",
        type=float,
        default=1.0e16,
        help="Max physical substep span per cooling update [s].",
    )
    parser.add_argument(
        "--nyx-cooling-table-npz",
        type=Path,
        default=Path("data/nyx_cooling_table.npz"),
        help="Nyx tabulated cooling file (.npz) with (z, logdelta, logT) grid.",
    )
    parser.add_argument(
        "--nyx-cooling-treecool",
        type=Path,
        default=Path("diffhydro/nyx_eos/TREECOOL_middle"),
        help="TREECOOL file used when building Nyx tabulated cooling rates.",
    )
    parser.add_argument(
        "--nyx-cooling-z-nodes",
        type=str,
        default="9,12,15,20,25,30,40,60,100",
        help="Comma-separated redshift nodes used by Nyx cooling table builder.",
    )
    parser.add_argument(
        "--nyx-cooling-logdelta-min",
        type=float,
        default=-3.0,
        help="Minimum log10(delta_b) for Nyx cooling table.",
    )
    parser.add_argument(
        "--nyx-cooling-logdelta-max",
        type=float,
        default=3.0,
        help="Maximum log10(delta_b) for Nyx cooling table.",
    )
    parser.add_argument(
        "--nyx-cooling-logdelta-n",
        type=int,
        default=96,
        help="Number of log10(delta_b) nodes for Nyx cooling table.",
    )
    parser.add_argument(
        "--nyx-cooling-logt-min",
        type=float,
        default=0.0,
        help="Minimum log10(T/K) for Nyx cooling table.",
    )
    parser.add_argument(
        "--nyx-cooling-logt-max",
        type=float,
        default=8.0,
        help="Maximum log10(T/K) for Nyx cooling table.",
    )
    parser.add_argument(
        "--nyx-cooling-logt-n",
        type=int,
        default=120,
        help="Number of log10(T/K) nodes for Nyx cooling table.",
    )
    parser.add_argument(
        "--nyx-cooling-rebuild",
        action="store_true",
        default=True,
        help="Force rebuild of Nyx cooling table even if .npz exists.",
    )
    parser.add_argument(
        "--nyx-cooling-eos-path",
        type=Path,
        default=Path("diffhydro/nyx_eos"),
        help="Directory containing eos_t extension and supporting Nyx EOS files.",
    )
    parser.add_argument(
        "--nyx-auto-rho-unit",
        dest="nyx_auto_rho_unit",
        action="store_true",
        default=False,
        help="For nyx_table cooling, override rho_unit_cgs from cosmology and gas_mean_fraction.",
    )
    parser.add_argument(
        "--no-nyx-auto-rho-unit",
        dest="nyx_auto_rho_unit",
        action="store_false",
        help="Disable automatic rho_unit_cgs override in nyx_table mode.",
    )
    parser.add_argument(
        "--tau-time-unit-s",
        type=float,
        default=3.085677581e19,
        help="Code tau time unit in seconds for dt_phys = a^2 * dtau * tau_time_unit_s.",
    )

    parser.add_argument("--dtau-max", type=float, default=8.0e-2, help="Upper bound for adaptive dtau.")
    parser.add_argument("--dtau-min", type=float, default=2.0e-10, help="Lower bound for adaptive dtau.")
    parser.add_argument(
        "--dtau-safety",
        type=float,
        default=0.3,
        help="Safety prefactor in dtau = safety * da/da_dtau.",
    )
    parser.add_argument(
        "--relative-max-change-a",
        type=float,
        default=0.027,
        help="Nyx-style limiter: max fractional scale-factor change per accepted step (da/a).",
    )
    parser.add_argument(
        "--absolute-max-change-a",
        type=float,
        default=0.0,
        help="Optional absolute da limiter per accepted step (0 disables).",
    )
    parser.add_argument(
        "--dtau-change-max",
        type=float,
        default=0.0,
        help="Nyx-style limiter: max growth factor relative to previous accepted dtau.",
    )
    parser.add_argument(
        "--dtau-init-shrink",
        type=float,
        default=1.0,
        help="Nyx-style initial shrink factor applied to first proposed dtau.",
    )
    parser.add_argument(
        "--solver-dt-safety",
        type=float,
        default=0.9,
        help="Safety factor applied to hydro timestep estimate.",
    )
    parser.add_argument(
        "--disable-solver-dt",
        action="store_true",
        default=False,
        help="Disable extra dt limiter from sim.timestep(U).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50000,
        help="Maximum accepted integration steps.",
    )
    parser.add_argument(
        "--step-retries",
        type=int,
        default=10,
        help="Retries with smaller dtau after non-finite trial step.",
    )
    parser.add_argument(
        "--dtau-retry-factor",
        type=float,
        default=0.5,
        help="dtau multiplier on retry.",
    )
    parser.add_argument(
        "--dtau-retry-cap-relax",
        type=float,
        default=1.1,
        help="Per-step multiplicative relaxation for retry-informed dtau cap.",
    )
    parser.add_argument(
        "--retry-cap-mult",
        type=float,
        default=1.25,
        help="On retried success, enforce retry cap <= retry_cap_mult * accepted trial dtau.",
    )
    parser.add_argument(
        "--retry-fallback-threshold",
        type=int,
        default=8,
        help="With auto fallback, switch HLLC->LF if a single accepted step needs this many retries.",
    )

    parser.add_argument(
        "--state-floor",
        type=float,
        default=2.0e-8,
        help="Density floor used by positivity projection.",
    )
    parser.add_argument(
        "--pressure-floor",
        type=float,
        default=2.0e-8,
        help="Pressure floor used by positivity projection.",
    )
    parser.add_argument(
        "--hydro-temp-floor-k",
        type=float,
        default=0.0,
        help="Kelvin floor for imported Nyx gas temperature and positivity projection.",
    )
    parser.add_argument(
        "--mu-hydrogen",
        type=float,
        default=1.0,
        help="Mean molecular weight for Kelvin conversion.",
    )
    parser.add_argument(
        "--h-species",
        type=float,
        default=0.76,
        help="Hydrogen mass fraction used when mapping rho to n_H in cooling/heating.",
    )
    parser.add_argument(
        "--temp-log-floor-k",
        type=float,
        default=1.0e-6,
        help="Small positive floor used only for log-temperature diagnostics.",
    )
    parser.add_argument(
        "--rho-unit-cgs",
        type=float,
        default=1.0e-29,
        help="Density unit conversion: rho_cgs = rho_phys * rho_unit_cgs.",
    )
    parser.add_argument(
        "--vel-unit-cms",
        type=float,
        default=1.0e7,
        help="Velocity unit in cm/s used by conversion utilities.",
    )
    parser.add_argument(
        "--nyx-velocity-divisor",
        type=float,
        default=100.0,
        help="Convert Nyx xmom/rho velocity to code units via v_code = v_nyx / divisor.",
    )
    parser.add_argument(
        "--ic-import-density-a3",
        action="store_true",
        dest="ic_import_density_a3",
        help="Include an a^-3 factor when mapping imported Nyx density into physical gas density.",
    )
    parser.add_argument(
        "--no-ic-import-density-a3",
        action="store_false",
        dest="ic_import_density_a3",
        help="Disable the extra a^-3 factor when importing Nyx density into physical gas density.",
    )
    parser.set_defaults(ic_import_density_a3=False)

    parser.add_argument(
        "--solver",
        type=str,
        choices=["hllc", "lf", "hll", "nyx_riemann", "nyx"],
        default="hllc",
        help="Riemann solver for hydro fluxes.",
    )
    parser.add_argument(
        "--nyx-interface-energy-mode",
        type=str,
        choices=["eos", "selected"],
        default="eos",
        help=(
            "NyxRiemann interface-energy mode: 'eos' enforces e=p/(gamma-1) "
            "at faces; 'selected' keeps selected/star internal energy."
        ),
    )
    parser.add_argument(
        "--dual-energy",
        dest="dual_energy",
        action="store_true",
        default=False,
        help=(
            "Enable 6-channel hydro state with passive dual-energy channel "
            "(rho*e at conservative index 5)."
        ),
    )
    parser.add_argument(
        "--no-dual-energy",
        dest="dual_energy",
        action="store_false",
        help="Use legacy 5-channel hydro state.",
    )
    parser.add_argument(
        "--auto-lf-fallback",
        action="store_true",
        default=False,
        help="If repeated NaN retries occur with HLLC, switch to LaxFriedrichs.",
    )
    parser.add_argument(
        "--no-auto-lf-fallback",
        dest="auto_lf_fallback",
        action="store_false",
        help="Disable automatic HLLC->LaxFriedrichs fallback.",
    )
    parser.add_argument(
        "--nan-fallback-threshold",
        type=int,
        default=1,
        help="Consecutive failed steps before solver fallback.",
    )

    parser.add_argument(
        "--min-gas-rho-corr",
        type=float,
        default=0.2,
        help="Pass threshold for final gas-density Pearson correlation.",
    )
    parser.add_argument(
        "--min-gas-temp-corr",
        type=float,
        default=0.2,
        help="Pass threshold for final gas-temperature Pearson correlation.",
    )
    parser.add_argument(
        "--min-dm-corr",
        type=float,
        default=0.2,
        help="Pass threshold for final DM-density Pearson correlation.",
    )

    parser.add_argument(
        "--debug-step-stats",
        type=int,
        default=0,
        help="If >0, print flow diagnostics for first N accepted steps.",
    )
    parser.add_argument(
        "--phase-history-every",
        type=int,
        default=0,
        help="If >0, record DiffHydro phase-fit diagnostics every N accepted steps.",
    )
    parser.add_argument(
        "--nyx-phase-history-stride",
        type=int,
        default=1,
        help="Stride for Nyx plotfile series used in phase-history diagnostics.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="1",
        help="CUDA_VISIBLE_DEVICES value set before importing JAX.",
    )
    parser.add_argument(
        "--xla-preallocate",
        action="store_true",
        default=False,
        help="Enable XLA GPU preallocation.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if args.xla_preallocate else "false"
    os.environ.setdefault("MPLBACKEND", "Agg")

    import h5py
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, SymLogNorm
    import numpy as np
    import yt

    import diffhydro as dh
    from diffhydro.cosmology import (
        BackgroundExpansionForce,
        JaxPMCoupledGravityForce,
        LCDMBackground,
        SuperComovingEquationManager,
    )
    from diffhydro.cosmology import conversions as cosmo_conv
    from diffhydro.cosmology.nyx_cooling_table import (
        NyxCoolingRateInterpolator,
        NyxCoolingTableData,
        build_nyx_cooling_table,
        parse_float_list,
        rho_crit0_cgs,
    )
    from jaxpm.pm import cic_paint
    from powerspectra import cross_correlation, power_spectrum

    def _require_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing file/path: {path}")

    def _read_comoving_a(plotfile: Path) -> float:
        a_path = plotfile / "comoving_a"
        _require_file(a_path)
        return float(a_path.read_text().strip())

    def _nyx_field_list(ds) -> list[str]:
        return sorted([f[1] for f in ds.field_list if f[0] == "boxlib" and str(f[1]).strip() != ""])

    def _load_nyx_plotfile_fields(plotfile: Path) -> dict[str, Any]:
        _require_file(plotfile / "Header")
        ds = yt.load(str(plotfile), hint="NyxDataset")
        required = ("density", "xmom", "ymom", "zmom", "Temp")
        available = set(_nyx_field_list(ds))
        missing = [f for f in required if f not in available]
        if missing:
            raise KeyError(
                f"Missing required Nyx fields in {plotfile}: {missing}; available={sorted(available)}"
            )

        # NOTE:
        # all_data() returns a 1D cell list whose ordering follows AMReX box
        # storage, not guaranteed C-ordered i/j/k indexing. Reshaping that flat
        # array can create blocky/periodic artifacts. Use covering_grid to pull
        # fields in proper level-0 index order.
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

        box_size = float(ds.domain_width[0].to_value("code_length"))
        dims = tuple(int(v) for v in ds.domain_dimensions)
        if len(set(dims)) != 1:
            raise ValueError(f"Expected cubic Nyx grid, got dimensions {dims}")
        n_grid_local = int(dims[0])
        a = _read_comoving_a(plotfile)
        z = (1.0 / a) - 1.0
        return {
            "density": density,
            "xmom": xmom,
            "ymom": ymom,
            "zmom": zmom,
            "temp_k": temp,
            "n_grid": n_grid_local,
            "box_size": box_size,
            "a": float(a),
            "z": float(z),
            "available_fields": sorted(available),
        }

    def _load_nyx_dm_h5(path: Path, n_grid_local: int, box_size_local: float, a_eval: float) -> dict[str, Any]:
        _require_file(path)
        with h5py.File(path, "r") as f:
            for key in ("x", "y", "z", "vx", "vy", "vz"):
                if key not in f:
                    raise KeyError(f"{path} missing dataset '{key}'")
            x = np.asarray(f["x"], dtype=np.float32)
            y = np.asarray(f["y"], dtype=np.float32)
            z = np.asarray(f["z"], dtype=np.float32)
            vx = np.asarray(f["vx"], dtype=np.float32)
            vy = np.asarray(f["vy"], dtype=np.float32)
            vz = np.asarray(f["vz"], dtype=np.float32)
            m = np.asarray(f["m"], dtype=np.float32) if "m" in f else np.ones_like(x, dtype=np.float32)

        pos = np.stack([x, y, z], axis=1)
        vel = np.stack([vx, vy, vz], axis=1)
        pos_grid = np.mod((pos / float(box_size_local)) * float(n_grid_local), float(n_grid_local))
        p_or_v = (vel / 100.0) * float(a_eval) / float(box_size_local) * float(n_grid_local)
        return {
            "pos_grid": jnp.asarray(pos_grid, dtype=jnp.float32),
            "p_or_v": jnp.asarray(p_or_v, dtype=jnp.float32),
            "mass": jnp.asarray(m, dtype=jnp.float32),
            "n_particles": int(pos.shape[0]),
        }

    def _pearson_corr(field_a: np.ndarray, field_b: np.ndarray) -> float:
        a = np.asarray(field_a, dtype=np.float64).ravel()
        b = np.asarray(field_b, dtype=np.float64).ravel()
        mask = np.isfinite(a) & np.isfinite(b)
        if np.count_nonzero(mask) < 8:
            return float("nan")
        a = a[mask]
        b = b[mask]
        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1.0e-20:
            return float("nan")
        return float((a @ b) / denom)

    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        return float(np.sqrt(np.mean(d * d)))

    def _mae(a: np.ndarray, b: np.ndarray) -> float:
        d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
        return float(np.mean(d))

    def paint_particle_density(positions: jnp.ndarray, masses: jnp.ndarray, mesh_shape: tuple[int, int, int]) -> jnp.ndarray:
        return cic_paint(
            jnp.zeros(mesh_shape, dtype=jnp.float32),
            positions,
            weight=jnp.asarray(masses, dtype=jnp.float32),
        )

    def _jaxpm_p_to_peculiar_velocity_code(
        p_or_v: jnp.ndarray, a_eval: float, box_mpch: float, n_grid_local: int
    ) -> tuple[jnp.ndarray, float]:
        factor = float(box_mpch) / (float(n_grid_local) * max(float(a_eval), 1.0e-12))
        return jnp.asarray(p_or_v, dtype=jnp.float32) * jnp.asarray(factor, dtype=jnp.float32), factor

    def _compute_phys_floor_code(a, rho_floor_phys: float, p_floor_phys: float, dtype):
        a_arr = jnp.asarray(a, dtype=dtype)
        rho_floor_code = cosmo_conv.density_phys_to_code(jnp.asarray(rho_floor_phys, dtype=dtype), a_arr)
        p_floor_code = cosmo_conv.pressure_phys_to_code(jnp.asarray(p_floor_phys, dtype=dtype), a_arr)
        return rho_floor_code, p_floor_code

    def _code_temp_to_kelvin(t_code: jnp.ndarray, mu_hydrogen: float, vel_unit_cms: float):
        mH_cgs = 1.6735575e-24
        kB_cgs = 1.380649e-16
        return jnp.maximum(
            t_code * (mu_hydrogen * mH_cgs / kB_cgs) * (vel_unit_cms * vel_unit_cms),
            1.0e-30,
        )

    def _state_is_finite(U_state, params_state) -> bool:
        ok = bool(jnp.all(jnp.isfinite(U_state)))
        if not ok:
            return False
        a_val = float(params_state.get("a", 1.0))
        if not np.isfinite(a_val):
            return False
        dm_state = params_state.get("dm", None)
        if dm_state is not None:
            if "x" in dm_state and (not bool(jnp.all(jnp.isfinite(dm_state["x"])))):
                return False
            if "p_or_v" in dm_state and (not bool(jnp.all(jnp.isfinite(dm_state["p_or_v"])))):
                return False
        return True

    class CosmologicalHydrogenCoolingForce_OLD:
        """Cooling source term using Kelvin/cgs internally, mapped from supercomoving state."""

        def __init__(
            self,
            equation_manager,
            logT_table,
            logLambda_m20_table,
            *,
            rho_unit_cgs: float,
            vel_unit_cms: float,
            tau_time_unit_s: float,
            mu: float = 1.0,
            h_species: float = 0.76,
            heating_rate_per_h: float = 0.0,
            cooling_rate_scale: float = 1.0,
            temp_floor_k: float = 1.0e-2,
            subcycles: int = 8,
            dtmax_s: float = 1.0e16,
            eps: float = 1.0e-30,
        ):
            self.eq = equation_manager
            self.logT_table = jnp.asarray(logT_table, dtype=jnp.float32)
            self.logLambda_m20_table = jnp.asarray(logLambda_m20_table, dtype=jnp.float32)
            self.rho_unit_cgs = float(rho_unit_cgs)
            self.vel_unit_cms = float(vel_unit_cms)
            self.p_unit_cgs = float(rho_unit_cgs) * float(vel_unit_cms) * float(vel_unit_cms)
            self.tau_time_unit_s = float(tau_time_unit_s)
            self.mu = float(mu)
            self.h_species = float(np.clip(h_species, 0.0, 1.0))
            self.heating_rate_per_h = float(heating_rate_per_h)
            self.cooling_rate_scale = float(max(cooling_rate_scale, 0.0))
            self.temp_floor_k = float(temp_floor_k)
            self.subcycles = int(max(1, subcycles))
            self.dtmax_s = float(dtmax_s)
            self.eps = float(eps)
            self.gamma = float(getattr(self.eq, "gamma", 5.0 / 3.0))
            self.i_rho = self.eq.mass_ids
            self.i_p = self.eq.energy_ids
            self.i_dual = getattr(self.eq, "dual_energy_ids", None)
            self.mH_cgs = 1.6735575e-24
            self.kB_cgs = 1.380649e-16

        def timestep(self, U):
            del U
            return jnp.asarray(1.0e10, dtype=jnp.float32)

        def _interp_log_lambda(self, logT):
            logT_min = self.logT_table[0]
            logT_max = self.logT_table[-1]
            logT_clip = jnp.clip(logT, logT_min, logT_max)
            logLm20_clip = jnp.interp(logT_clip, self.logT_table, self.logLambda_m20_table)
            # Avoid holding Lambda fixed at the first tabulated point (typically 1e4 K).
            # At lower T collisional cooling should rapidly weaken; linear extrapolation
            # in log-log space is a better approximation than hard clipping.
            dlogT = jnp.maximum(self.logT_table[1] - self.logT_table[0], 1.0e-6)
            slope_lo = (self.logLambda_m20_table[1] - self.logLambda_m20_table[0]) / dlogT
            logLm20_lo = self.logLambda_m20_table[0] + slope_lo * (logT - logT_min)
            logLm20 = jnp.where(logT < logT_min, logLm20_lo, logLm20_clip)
            return logLm20 - 20.0

        def _code_thermo_to_cgs(self, rho_code, p_code, a):
            rho_phys = jnp.maximum(cosmo_conv.density_code_to_phys(rho_code, a), self.eps)
            p_phys = jnp.maximum(cosmo_conv.pressure_code_to_phys(p_code, a), self.eps)
            rho_cgs = jnp.maximum(rho_phys * self.rho_unit_cgs, self.eps)
            p_cgs = jnp.maximum(p_phys * self.p_unit_cgs, self.eps)
            return rho_phys, p_phys, rho_cgs, p_cgs

        def temperature_kelvin_from_code(self, rho_code, p_code, a):
            _, _, rho_cgs, p_cgs = self._code_thermo_to_cgs(rho_code, p_code, a)
            return (self.mu * self.mH_cgs / self.kB_cgs) * p_cgs / jnp.maximum(rho_cgs, self.eps)

        def force(self, i_step, U, params, dtau):
            del i_step
            a = jnp.asarray(params.get("a", 1.0), dtype=jnp.float32)
            dtau = jnp.maximum(jnp.asarray(dtau, dtype=jnp.float32), 0.0)
            dt_s = jnp.minimum((a * a) * dtau * self.tau_time_unit_s, self.dtmax_s)
            dt_sub = dt_s / float(self.subcycles)

            W = self.eq.get_primitives_from_conservatives(U)
            rho_code = jnp.maximum(W[self.i_rho], self.eps)
            p_code = jnp.maximum(W[self.i_p], self.eps)

            _, _, rho_cgs, p_cgs = self._code_thermo_to_cgs(rho_code, p_code, a)
            Et = p_cgs / (self.gamma - 1.0)
            Et_floor = rho_cgs * self.kB_cgs * self.temp_floor_k / (
                self.mu * self.mH_cgs * (self.gamma - 1.0)
            )

            def body(_, Et_cur):
                p_cgs_cur = (self.gamma - 1.0) * jnp.maximum(Et_cur, self.eps)
                T_cur = (self.mu * self.mH_cgs / self.kB_cgs) * p_cgs_cur / jnp.maximum(rho_cgs, self.eps)
                logT = jnp.log10(jnp.maximum(T_cur, self.temp_floor_k))
                logLambda = self._interp_log_lambda(logT)
                Lambda = 10.0**logLambda
                nH = self.h_species * rho_cgs / self.mH_cgs
                dotE = -self.cooling_rate_scale * nH * nH * Lambda + nH * self.heating_rate_per_h
                Et_new = jnp.maximum(Et_cur + dotE * dt_sub, Et_floor)
                return Et_new

            Et_final = jax.lax.fori_loop(0, self.subcycles, body, Et)
            p_cgs_new = (self.gamma - 1.0) * jnp.maximum(Et_final, self.eps)
            p_phys_new = p_cgs_new / self.p_unit_cgs
            p_code_new = cosmo_conv.pressure_phys_to_code(p_phys_new, a)

            p_code_new = jnp.maximum(p_code_new, self.eq.eps)
            W_new = W.at[self.i_p].set(p_code_new)
            if self.i_dual is not None and int(self.i_dual) < int(W.shape[0]):
                rhoe_code_new = jnp.maximum(p_code_new / (self.gamma - 1.0), self.eq.eps)
                W_new = W_new.at[int(self.i_dual)].set(rhoe_code_new)
            U_new = self.eq.get_conservatives_from_primitives(W_new)
            return U_new, params

    class NyxTabulatedCoolingForce:
        """Nyx-style tabulated cooling/heating source terms in cgs space."""

        def __init__(
            self,
            equation_manager,
            nyx_interp: NyxCoolingRateInterpolator,
            *,
            rho_unit_cgs: float,
            vel_unit_cms: float,
            tau_time_unit_s: float,
            mu: float = 1.0,
            cooling_rate_scale: float = 1.0,
            heating_rate_scale: float = 1.0,
            temp_floor_k: float = 1.0e-2,
            subcycles: int = 8,
            dtmax_s: float = 1.0e16,
            eps: float = 1.0e-40,
        ):
            self.eq = equation_manager
            self.nyx = nyx_interp
            self.rho_unit_cgs = float(rho_unit_cgs)
            self.vel_unit_cms = float(vel_unit_cms)
            self.p_unit_cgs = float(rho_unit_cgs) * float(vel_unit_cms) * float(vel_unit_cms)
            self.tau_time_unit_s = float(tau_time_unit_s)
            self.mu = float(mu)
            self.cooling_rate_scale = float(max(cooling_rate_scale, 0.0))
            self.heating_rate_scale = float(max(heating_rate_scale, 0.0))
            self.temp_floor_k = float(temp_floor_k)
            self.subcycles = int(max(1, subcycles))
            self.dtmax_s = float(dtmax_s)
            self.eps = float(eps)
            self.gamma = float(getattr(self.eq, "gamma", 5.0 / 3.0))
            self.i_rho = self.eq.mass_ids
            self.i_p = self.eq.energy_ids
            self.i_dual = getattr(self.eq, "dual_energy_ids", None)
            self.mH_cgs = 1.6735575e-24
            self.kB_cgs = 1.380649e-16

        def timestep(self, U):
            del U
            return jnp.asarray(1.0e10, dtype=jnp.float32)

        def _code_thermo_to_cgs(self, rho_code, p_code, a):
            rho_phys = jnp.maximum(cosmo_conv.density_code_to_phys(rho_code, a), self.eps)
            p_phys = jnp.maximum(cosmo_conv.pressure_code_to_phys(p_code, a), self.eps)
            rho_cgs = jnp.maximum(rho_phys * self.rho_unit_cgs, self.eps)
            p_cgs = jnp.maximum(p_phys * self.p_unit_cgs, self.eps)
            return rho_phys, p_phys, rho_cgs, p_cgs

        def force(self, i_step, U, params, dtau):
            del i_step
            a = jnp.asarray(params.get("a", 1.0), dtype=jnp.float32)
            z = (1.0 / jnp.maximum(a, 1.0e-12)) - 1.0
            dtau = jnp.maximum(jnp.asarray(dtau, dtype=jnp.float32), 0.0)
            dt_s = jnp.minimum((a * a) * dtau * self.tau_time_unit_s, self.dtmax_s)
            dt_sub = dt_s / float(self.subcycles)

            W = self.eq.get_primitives_from_conservatives(U)
            rho_code = jnp.maximum(W[self.i_rho], self.eps)
            p_code = jnp.maximum(W[self.i_p], self.eps)

            _, _, rho_cgs, p_cgs = self._code_thermo_to_cgs(rho_code, p_code, a)
            Et = p_cgs / (self.gamma - 1.0)
            Et_floor = rho_cgs * self.kB_cgs * self.temp_floor_k / (
                self.mu * self.mH_cgs * (self.gamma - 1.0)
            )

            def body(_, Et_cur):
                p_cgs_cur = (self.gamma - 1.0) * jnp.maximum(Et_cur, self.eps)
                T_cur = (self.mu * self.mH_cgs / self.kB_cgs) * p_cgs_cur / jnp.maximum(rho_cgs, self.eps)
                heat_cgs, cool_cgs, net = self.nyx.evaluate(rho_cgs, T_cur, z)
                dotE = self.heating_rate_scale * heat_cgs - self.cooling_rate_scale * cool_cgs
                Et_new = jnp.maximum(Et_cur + dotE * dt_sub, Et_floor)
                return Et_new

            Et_final = jax.lax.fori_loop(0, self.subcycles, body, Et)
            p_cgs_new = (self.gamma - 1.0) * jnp.maximum(Et_final, self.eps)
            p_phys_new = p_cgs_new / self.p_unit_cgs
            p_code_new = cosmo_conv.pressure_phys_to_code(p_phys_new, a)
            p_code_new = jnp.maximum(p_code_new, self.eq.eps)

            W_new = W.at[self.i_p].set(p_code_new)
            if self.i_dual is not None and int(self.i_dual) < int(W.shape[0]):
                rhoe_code_new = jnp.maximum(p_code_new / (self.gamma - 1.0), self.eq.eps)
                W_new = W_new.at[int(self.i_dual)].set(rhoe_code_new)
            U_new = self.eq.get_conservatives_from_primitives(W_new)
            return U_new, params

    def _flow_diagnostics(U_state, a_state):
        w = eq.get_primitives_from_conservatives(U_state)
        rho_code = jnp.maximum(w[0], eq.eps)
        p_code = jnp.maximum(w[4], eq.eps)
        vtx, vty, vtz = w[1], w[2], w[3]
        vmag_tilde = jnp.sqrt(vtx * vtx + vty * vty + vtz * vtz)
        cs_tilde = jnp.sqrt(
            jnp.maximum(jnp.asarray(float(eq.gamma * eq.R), dtype=w.dtype) * p_code / rho_code, 1.0e-30)
        )
        mach = vmag_tilde / jnp.maximum(cs_tilde, 1.0e-30)
        a_arr = jnp.asarray(a_state, dtype=w.dtype)
        vmag_phys_code = vmag_tilde / jnp.maximum(a_arr, 1.0e-30)
        speed_kms = vmag_phys_code * (float(hydro_vel_unit_cms) / 1.0e5)
        temp_phys_code = (p_code / jnp.maximum(rho_code, 1.0e-30)) / jnp.maximum(a_arr * a_arr, 1.0e-30)
        temp_kelvin = temp_phys_code * jnp.asarray(code_to_kelvin_temp, dtype=w.dtype)
        return {
            "mach_p95": float(np.asarray(jnp.percentile(mach, 95.0))),
            "mach_p99": float(np.asarray(jnp.percentile(mach, 99.0))),
            "mach_max": float(np.asarray(jnp.max(mach))),
            "speed_kms_p99": float(np.asarray(jnp.percentile(speed_kms, 99.0))),
            "speed_kms_max": float(np.asarray(jnp.max(speed_kms))),
            "tempk_min": float(np.asarray(jnp.min(temp_kelvin))),
            "tempk_p50": float(np.asarray(jnp.percentile(temp_kelvin, 50.0))),
            "tempk_p99": float(np.asarray(jnp.percentile(temp_kelvin, 99.0))),
        }

    def build_hydrosim(
        eq_local,
        forces_local,
        *,
        solver_name: str,
        dx_o: float = 1.0,
        nyx_interface_energy_mode: str = "eos",
    ):
        ss = dh.signal_speed_Einfeldt
        sname = solver_name.lower()
        if sname == "hllc":
            riemann_solver = dh.HLLC(equation_manager=eq_local, signal_speed=ss)
        elif sname == "laxfriedrichs" or sname == "lf":
            riemann_solver = dh.LaxFriedrichs(equation_manager=eq_local, signal_speed=ss)
        elif sname == "hll":
            riemann_solver = dh.HLL(equation_manager=eq_local, signal_speed=ss)
        elif sname == "nyx_riemann" or sname == "nyx":
            riemann_solver = dh.NyxRiemannEuler(
                equation_manager=eq_local,
                signal_speed=ss,
                eos_consistent_interface_energy=(str(nyx_interface_energy_mode).lower() != "selected"),
            )
        else:
            raise ValueError(f"Unknown solver '{solver_name}'")

        conv_flux = dh.ConvectiveFlux(
            eq_local,
            riemann_solver,
            dh.PPM_CW(limiter="MINMOD",steepen=False), #VANLEER?
            positivity=True,
        )
        conv_flux.dx_o = float(dx_o)
        sim_local = dh.hydro(
            n_super_step=1,
            max_dt=0.2,
            fluxes=[conv_flux],
            forces=list(forces_local),
            use_mol=True,
            pmesh_shape=(1, 1, 1),
        )
        sim_local.dx_o = float(dx_o)
        for flux in sim_local.fluxes:
            if hasattr(flux, "dx_o"):
                flux.dx_o = float(dx_o)
        return sim_local

    def _slice_triplet(arr3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny, nz = arr3d.shape
        sx = arr3d[nx // 2, :, :]
        sy = arr3d[:, ny // 2, :]
        sz = arr3d[:, :, nz // 2]
        return sx, sy, sz

    def _norm_for_plot(arr: np.ndarray, log: bool, diverging: bool = False):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        if diverging:
            vmax = np.percentile(np.abs(finite), 99.0)
            vmax = max(vmax, 1.0e-12)
            return SymLogNorm(linthresh=max(vmax * 1.0e-3, 1.0e-12), vmin=-vmax, vmax=vmax)
        if log and np.all(finite > 0):
            vmin = np.percentile(finite, 1.0)
            vmax = np.percentile(finite, 99.0)
            if vmax <= vmin:
                vmax = vmin * 1.01
            return LogNorm(vmin=max(vmin, 1.0e-12), vmax=max(vmax, 1.01e-12))
        vmin = np.percentile(finite, 1.0)
        vmax = np.percentile(finite, 99.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        return plt.Normalize(vmin=vmin, vmax=vmax)

    def save_3panel(arr3d: np.ndarray, outpath: Path, title: str, *, log: bool = True, diverging: bool = False, cmap: str = "viridis"):
        sx, sy, sz = _slice_triplet(np.asarray(arr3d))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=180, constrained_layout=True)
        data = [sx, sy, sz]
        names = ["x-mid (y,z)", "y-mid (x,z)", "z-mid (x,y)"]
        for ax, arr, label in zip(axes, data, names):
            norm = _norm_for_plot(arr, log=log, diverging=diverging)
            im = ax.imshow(arr, origin="lower", cmap=cmap, norm=norm)
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=7)
        fig.suptitle(title)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    def save_overview_compare(
        outpath: Path,
        dm_nyx: np.ndarray,
        dm_dh: np.ndarray,
        gas_nyx: np.ndarray,
        gas_dh: np.ndarray,
        temp_nyx: np.ndarray,
        temp_dh: np.ndarray,
        title_prefix: str,
    ):
        k = dm_nyx.shape[2] // 2
        fig, axes = plt.subplots(3, 3, figsize=(13, 11), dpi=170, constrained_layout=True)
        rows = [
            ("DM density norm", dm_nyx[:, :, k], dm_dh[:, :, k], (dm_dh - dm_nyx)[:, :, k]),
            ("Gas density norm", gas_nyx[:, :, k], gas_dh[:, :, k], (gas_dh - gas_nyx)[:, :, k]),
            ("Gas Temp [K]", temp_nyx[:, :, k], temp_dh[:, :, k], (temp_dh - temp_nyx)[:, :, k]),
        ]
        for r, (label, a_ref, a_dh, a_res) in enumerate(rows):
            norm_ref = _norm_for_plot(a_ref, log=(r < 2 and np.all(a_ref > 0)))
            norm_dh = _norm_for_plot(a_dh, log=(r < 2 and np.all(a_dh > 0)))
            norm_res = _norm_for_plot(a_res, log=False, diverging=True)

            im0 = axes[r, 0].imshow(a_ref, origin="lower", cmap="viridis", norm=norm_ref)
            im1 = axes[r, 1].imshow(a_dh, origin="lower", cmap="viridis", norm=norm_dh)
            im2 = axes[r, 2].imshow(a_res, origin="lower", cmap="coolwarm", norm=norm_res)
            axes[r, 0].set_title(f"{label}: Nyx")
            axes[r, 1].set_title(f"{label}: DiffHydro")
            axes[r, 2].set_title(f"{label}: DiffHydro - Nyx")
            for c in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
            fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)
            fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)
        fig.suptitle(title_prefix)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    def _shared_norm_for_plot(arr_a: np.ndarray, arr_b: np.ndarray, *, log: bool):
        aa = np.asarray(arr_a, dtype=np.float64).ravel()
        bb = np.asarray(arr_b, dtype=np.float64).ravel()
        both = np.concatenate([aa[np.isfinite(aa)], bb[np.isfinite(bb)]])
        if both.size == 0:
            return None
        if log:
            pos = both[both > 0.0]
            if pos.size >= 8:
                vmin = np.percentile(pos, 1.0)
                vmax = np.percentile(pos, 99.0)
                if vmax <= vmin:
                    vmax = vmin * 1.01
                return LogNorm(vmin=max(vmin, 1.0e-20), vmax=max(vmax, 1.01e-20))
        vmin = np.percentile(both, 1.0)
        vmax = np.percentile(both, 99.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        return plt.Normalize(vmin=vmin, vmax=vmax)

    def save_matched_2x3_compare(
        outpath: Path,
        nyx_arr3d: np.ndarray,
        dh_arr3d: np.ndarray,
        title: str,
        *,
        log: bool = True,
        cmap: str = "viridis",
    ):
        nyx_slices = _slice_triplet(np.asarray(nyx_arr3d))
        dh_slices = _slice_triplet(np.asarray(dh_arr3d))
        norm = _shared_norm_for_plot(np.stack(nyx_slices), np.stack(dh_slices), log=log)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=180, constrained_layout=True)
        labels = ["x-mid (y,z)", "y-mid (x,z)", "z-mid (x,y)"]
        rows = [("Nyx", nyx_slices), ("DiffHydro", dh_slices)]
        im = None
        for r, (row_name, row_data) in enumerate(rows):
            for c, (slc, slabel) in enumerate(zip(row_data, labels)):
                im = axes[r, c].imshow(slc, origin="lower", cmap=cmap, norm=norm)
                axes[r, c].set_title(f"{row_name}: {slabel}")
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        fig.suptitle(title)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    def save_spectra_plot(
        outpath: Path,
        dm_ref: np.ndarray,
        dm_dh: np.ndarray,
        gas_ref: np.ndarray,
        gas_dh: np.ndarray,
        temp_ref: np.ndarray,
        temp_dh: np.ndarray,
        boxsize_3d: np.ndarray,
        *,
        temp_log_floor_k: float,
    ):
        kmin = 0.5
        dk = 0.5
        dm_ref_d = dm_ref - 1.0
        dm_dh_d = dm_dh - 1.0
        gas_ref_d = gas_ref - 1.0
        gas_dh_d = gas_dh - 1.0
        t_ref = np.log10(np.maximum(temp_ref, max(temp_log_floor_k, 1.0e-20)))
        t_dh = np.log10(np.maximum(temp_dh, max(temp_log_floor_k, 1.0e-20)))
        t_ref_d = t_ref - np.mean(t_ref)
        t_dh_d = t_dh - np.mean(t_dh)

        k_dm, p_dm_ref = power_spectrum(dm_ref_d, kmin=kmin, dk=dk, boxsize=boxsize_3d)
        _, p_dm_dh = power_spectrum(dm_dh_d, kmin=kmin, dk=dk, boxsize=boxsize_3d)
        _, r_dm = cross_correlation(dm_ref_d, dm_dh_d, kmin=kmin, dk=dk, boxsize=boxsize_3d)

        k_g, p_g_ref = power_spectrum(gas_ref_d, kmin=kmin, dk=dk, boxsize=boxsize_3d)
        _, p_g_dh = power_spectrum(gas_dh_d, kmin=kmin, dk=dk, boxsize=boxsize_3d)
        _, r_g = cross_correlation(gas_ref_d, gas_dh_d, kmin=kmin, dk=dk, boxsize=boxsize_3d)

        k_t, p_t_ref = power_spectrum(t_ref_d, kmin=kmin, dk=dk, boxsize=boxsize_3d)
        _, p_t_dh = power_spectrum(t_dh_d, kmin=kmin, dk=dk, boxsize=boxsize_3d)
        _, r_t = cross_correlation(t_ref_d, t_dh_d, kmin=kmin, dk=dk, boxsize=boxsize_3d)

        fig, axes = plt.subplots(3, 2, figsize=(12, 13), dpi=170, constrained_layout=True)

        axes[0, 0].loglog(k_dm, np.maximum(np.real(p_dm_ref), 1.0e-30), label="Nyx")
        axes[0, 0].loglog(k_dm, np.maximum(np.real(p_dm_dh), 1.0e-30), label="DiffHydro")
        axes[0, 0].set_title("DM Power Spectrum")
        axes[0, 0].set_xlabel("k")
        axes[0, 0].set_ylabel("P(k)")
        axes[0, 0].grid(alpha=0.2)
        axes[0, 0].legend(frameon=False)

        axes[0, 1].plot(k_dm, np.real(r_dm), label="Nyx vs DiffHydro")
        axes[0, 1].axhline(1.0, color="k", lw=0.8, alpha=0.5)
        axes[0, 1].axhline(0.0, color="k", lw=0.8, alpha=0.3)
        axes[0, 1].set_ylim(-1.05, 1.05)
        axes[0, 1].set_title("DM Cross-Correlation")
        axes[0, 1].set_xlabel("k")
        axes[0, 1].set_ylabel("r(k)")
        axes[0, 1].grid(alpha=0.2)

        axes[1, 0].loglog(k_g, np.maximum(np.real(p_g_ref), 1.0e-30), label="Nyx")
        axes[1, 0].loglog(k_g, np.maximum(np.real(p_g_dh), 1.0e-30), label="DiffHydro")
        axes[1, 0].set_title("Gas Density Power Spectrum")
        axes[1, 0].set_xlabel("k")
        axes[1, 0].set_ylabel("P(k)")
        axes[1, 0].grid(alpha=0.2)
        axes[1, 0].legend(frameon=False)

        axes[1, 1].plot(k_g, np.real(r_g), label="Nyx vs DiffHydro")
        axes[1, 1].axhline(1.0, color="k", lw=0.8, alpha=0.5)
        axes[1, 1].axhline(0.0, color="k", lw=0.8, alpha=0.3)
        axes[1, 1].set_ylim(-1.05, 1.05)
        axes[1, 1].set_title("Gas Density Cross-Correlation")
        axes[1, 1].set_xlabel("k")
        axes[1, 1].set_ylabel("r(k)")
        axes[1, 1].grid(alpha=0.2)

        axes[2, 0].loglog(k_t, np.maximum(np.real(p_t_ref), 1.0e-30), label="Nyx")
        axes[2, 0].loglog(k_t, np.maximum(np.real(p_t_dh), 1.0e-30), label="DiffHydro")
        axes[2, 0].set_title(r"Temperature Power Spectrum (log$_{10}$T)")
        axes[2, 0].set_xlabel("k")
        axes[2, 0].set_ylabel("P(k)")
        axes[2, 0].grid(alpha=0.2)
        axes[2, 0].legend(frameon=False)

        axes[2, 1].plot(k_t, np.real(r_t), label="Nyx vs DiffHydro")
        axes[2, 1].axhline(1.0, color="k", lw=0.8, alpha=0.5)
        axes[2, 1].axhline(0.0, color="k", lw=0.8, alpha=0.3)
        axes[2, 1].set_ylim(-1.05, 1.05)
        axes[2, 1].set_title(r"Temperature Cross-Correlation (log$_{10}$T)")
        axes[2, 1].set_xlabel("k")
        axes[2, 1].set_ylabel("r(k)")
        axes[2, 1].grid(alpha=0.2)

        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    def save_hist_plot(
        outpath: Path,
        dm_ref: np.ndarray,
        dm_dh: np.ndarray,
        gas_ref: np.ndarray,
        gas_dh: np.ndarray,
        temp_ref: np.ndarray,
        temp_dh: np.ndarray,
        *,
        temp_log_floor_k: float,
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=170, constrained_layout=True)
        axes[0].hist(np.ravel(dm_ref), bins=120, density=True, alpha=0.6, label="Nyx")
        axes[0].hist(np.ravel(dm_dh), bins=120, density=True, alpha=0.6, label="DiffHydro")
        axes[0].set_title("DM density norm PDF")
        axes[0].set_yscale("log")
        axes[0].legend(frameon=False)

        axes[1].hist(np.ravel(gas_ref), bins=120, density=True, alpha=0.6, label="Nyx")
        axes[1].hist(np.ravel(gas_dh), bins=120, density=True, alpha=0.6, label="DiffHydro")
        axes[1].set_title("Gas density norm PDF")
        axes[1].set_yscale("log")
        axes[1].legend(frameon=False)

        axes[2].hist(
            np.ravel(np.log10(np.maximum(temp_ref, max(temp_log_floor_k, 1.0e-20)))),
            bins=120,
            density=True,
            alpha=0.6,
            label="Nyx",
        )
        axes[2].hist(
            np.ravel(np.log10(np.maximum(temp_dh, max(temp_log_floor_k, 1.0e-20)))),
            bins=120,
            density=True,
            alpha=0.6,
            label="DiffHydro",
        )
        axes[2].set_title(r"log$_{10}$ Temperature PDF")
        axes[2].set_yscale("log")
        axes[2].legend(frameon=False)

        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    def save_scatter_plot(
        outpath: Path,
        ref: np.ndarray,
        pred: np.ndarray,
        title: str,
        *,
        log10: bool = False,
        max_points: int = 120000,
    ):
        x = np.asarray(ref, dtype=np.float64).ravel()
        y = np.asarray(pred, dtype=np.float64).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if log10:
            x = np.log10(np.maximum(x, 1.0e-20))
            y = np.log10(np.maximum(y, 1.0e-20))
        if x.size > max_points:
            rng = np.random.default_rng(1234)
            idx = rng.choice(x.size, size=max_points, replace=False)
            x = x[idx]
            y = y[idx]
        xmin = min(np.min(x), np.min(y))
        xmax = max(np.max(x), np.max(y))
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), dpi=170, constrained_layout=True)
        ax.scatter(x, y, s=1.0, alpha=0.18, linewidths=0)
        ax.plot([xmin, xmax], [xmin, xmax], "k--", lw=1.0, alpha=0.8)
        ax.set_xlabel("Nyx")
        ax.set_ylabel("DiffHydro")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    def _phase_xy(rho_norm: np.ndarray, temp_k: np.ndarray, temp_floor_k: float):
        x = np.log10(np.maximum(np.asarray(rho_norm, dtype=np.float64).ravel(), 1.0e-20))
        y = np.log10(np.maximum(np.asarray(temp_k, dtype=np.float64).ravel(), max(temp_floor_k, 1.0e-20)))
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    def _phase_fit(x: np.ndarray, y: np.ndarray):
        if x.size < 16:
            return float("nan"), float("nan")
        x0 = x - np.mean(x)
        y0 = y - np.mean(y)
        denom = float(np.dot(x0, x0))
        if denom <= 0.0:
            return float("nan"), float("nan")
        slope = float(np.dot(x0, y0) / denom)
        intercept = float(np.mean(y) - slope * np.mean(x))
        return slope, intercept

    def _binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = 40):
        if x.size < 64:
            return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
        x_lo = np.percentile(x, 0.5)
        x_hi = np.percentile(x, 99.5)
        if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
            return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
        edges = np.linspace(x_lo, x_hi, n_bins + 1)
        idx = np.digitize(x, edges) - 1
        x_mid = 0.5 * (edges[:-1] + edges[1:])
        med = np.full((n_bins,), np.nan, dtype=np.float64)
        for bi in range(n_bins):
            vals = y[idx == bi]
            if vals.size > 8:
                med[bi] = np.median(vals)
        good = np.isfinite(med)
        return x_mid[good], med[good]

    def save_temp_density_phase_plot(
        outpath: Path,
        rho_ref: np.ndarray,
        temp_ref: np.ndarray,
        rho_dh: np.ndarray,
        temp_dh: np.ndarray,
        *,
        temp_log_floor_k: float,
    ):
        xr, yr = _phase_xy(rho_ref, temp_ref, temp_log_floor_k)
        xd, yd = _phase_xy(rho_dh, temp_dh, temp_log_floor_k)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=170, constrained_layout=True)

        hb0 = axes[0].hexbin(
            xr,
            yr,
            gridsize=90,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        axes[0].set_title("Nyx Phase: log10(T) vs log10(rho/rho_bar)")
        axes[0].set_xlabel(r"log$_{10}$ rho / rho_bar")
        axes[0].set_ylabel(r"log$_{10}$ T [K]")
        axes[0].set_xlim(-1.2,2)
        axes[0].set_ylim(3,5.5)
        axes[0].grid(alpha=0.25)

        fig.colorbar(hb0, ax=axes[0], label="counts")

        hb1 = axes[1].hexbin(
            xd,
            yd,
            gridsize=90,
            bins="log",
            mincnt=1,
            cmap="magma",
        )
        axes[1].set_title("DiffHydro Phase: log10(T) vs log10(rho/rho_bar)")
        axes[1].set_xlabel(r"log$_{10}$ rho / rho_bar")
        axes[1].set_ylabel(r"log$_{10}$ T [K]")
        axes[1].set_xlim(-1.2,2)
        axes[1].set_ylim(3,5.5)
        axes[1].grid(alpha=0.25)

        fig.colorbar(hb1, ax=axes[1], label="counts")

        xmr, ymr = _binned_median(xr, yr)
        xmd, ymd = _binned_median(xd, yd)
        if xmr.size > 0:
            axes[2].plot(xmr, ymr, lw=2.0, label="Nyx median", color="#1f77b4")
        if xmd.size > 0:
            axes[2].plot(xmd, ymd, lw=2.0, label="DiffHydro median", color="#d62728")
        if xmr.size > 0 and xmd.size > 0:
            x0 = max(np.min(xmr), np.min(xmd))
            x1 = min(np.max(xmr), np.max(xmd))
            if x1 > x0:
                xs = np.linspace(x0, x1, 200)
                yr_i = np.interp(xs, xmr, ymr)
                yd_i = np.interp(xs, xmd, ymd)
        #        axes[2].plot(xs, yd_i - yr_i, lw=1.2, ls="--", color="k", alpha=0.7, label="delta median")
        axes[2].set_title("Median Phase Relation")
        axes[2].set_xlabel(r"log$_{10}$ rho / rho_bar")
        axes[2].set_ylabel(r"log$_{10}$ T [K] / delta")
        axes[2].grid(alpha=0.25)
        axes[2].legend(frameon=False)

        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    def _phase_diag_summary(rho_norm: np.ndarray, temp_k: np.ndarray, temp_log_floor_k: float) -> dict[str, float]:
        x_phase, y_phase = _phase_xy(rho_norm, temp_k, temp_log_floor_k)
        slope, intercept = _phase_fit(x_phase, y_phase)
        corr = _pearson_corr(x_phase, y_phase)
        t = np.asarray(temp_k, dtype=np.float64).ravel()
        t = t[np.isfinite(t)]
        if t.size == 0:
            return {
                "phase_slope": float("nan"),
                "phase_intercept": float("nan"),
                "phase_corr": float("nan"),
                "temp_min_k": float("nan"),
                "temp_p01_k": float("nan"),
                "temp_p50_k": float("nan"),
                "temp_p99_k": float("nan"),
                "temp_lt_1k_frac": float("nan"),
                "temp_lt_10k_frac": float("nan"),
            }
        return {
            "phase_slope": float(slope),
            "phase_intercept": float(intercept),
            "phase_corr": float(corr),
            "temp_min_k": float(np.min(t)),
            "temp_p01_k": float(np.percentile(t, 1.0)),
            "temp_p50_k": float(np.percentile(t, 50.0)),
            "temp_p99_k": float(np.percentile(t, 99.0)),
            "temp_lt_1k_frac": float(np.mean(t < 1.0)),
            "temp_lt_10k_frac": float(np.mean(t < 10.0)),
        }

    def _write_phase_history_csv(path: Path, rows: list[dict[str, float]]) -> None:
        header = [
            "row_index",
            "step",
            "a",
            "z",
            "phase_slope",
            "phase_intercept",
            "phase_corr",
            "temp_min_k",
            "temp_p01_k",
            "temp_p50_k",
            "temp_p99_k",
            "temp_lt_1k_frac",
            "temp_lt_10k_frac",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for i, row in enumerate(rows):
                vals = [
                    i,
                    int(row.get("step", -1)),
                    float(row.get("a", float("nan"))),
                    float(row.get("z", float("nan"))),
                    float(row.get("phase_slope", float("nan"))),
                    float(row.get("phase_intercept", float("nan"))),
                    float(row.get("phase_corr", float("nan"))),
                    float(row.get("temp_min_k", float("nan"))),
                    float(row.get("temp_p01_k", float("nan"))),
                    float(row.get("temp_p50_k", float("nan"))),
                    float(row.get("temp_p99_k", float("nan"))),
                    float(row.get("temp_lt_1k_frac", float("nan"))),
                    float(row.get("temp_lt_10k_frac", float("nan"))),
                ]
                f.write(",".join(str(v) for v in vals) + "\n")

    def _plotfile_step_id(plotfile: Path) -> int:
        s = plotfile.name
        if s.startswith("plt"):
            tail = s[3:]
            if tail.isdigit():
                return int(tail)
        return -1

    def _load_nyx_density_temp_for_phase(plotfile: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
        _require_file(plotfile / "Header")
        ds = yt.load(str(plotfile), hint="NyxDataset")
        available = set(_nyx_field_list(ds))
        for req in ("density", "Temp"):
            if req not in available:
                raise KeyError(f"Missing Nyx field '{req}' in {plotfile}; available={sorted(available)}")
        cg = ds.covering_grid(
            level=0,
            left_edge=ds.domain_left_edge,
            dims=ds.domain_dimensions,
        )
        density = np.asarray(cg[("boxlib", "density")], dtype=np.float32)
        temp = np.asarray(cg[("boxlib", "Temp")], dtype=np.float32)
        a_pf = _read_comoving_a(plotfile)
        z_pf = (1.0 / a_pf) - 1.0
        return density, temp, float(a_pf), float(z_pf)

    def _collect_nyx_phase_history(
        ic_plotfile: Path,
        final_plotfile: Path,
        *,
        stride: int,
        temp_log_floor_k: float,
    ) -> list[dict[str, float]]:
        stride = max(1, int(stride))
        same_parent = ic_plotfile.parent.resolve() == final_plotfile.parent.resolve()
        if same_parent:
            plotfiles = sorted(
                [p for p in ic_plotfile.parent.glob("plt*") if (p / "Header").exists()],
                key=_plotfile_step_id,
            )
        else:
            plotfiles = [ic_plotfile, final_plotfile]

        if not plotfiles:
            return []

        filtered: list[Path] = []
        for idx, p in enumerate(plotfiles):
            if idx % stride == 0:
                filtered.append(p)
        if plotfiles[-1] not in filtered:
            filtered.append(plotfiles[-1])
        if plotfiles[0] not in filtered:
            filtered.insert(0, plotfiles[0])

        rows: list[dict[str, float]] = []
        for p in filtered:
            density, temp, a_pf, z_pf = _load_nyx_density_temp_for_phase(p)
            rho_norm = density / max(float(np.mean(density)), 1.0e-30)
            diag = _phase_diag_summary(rho_norm, temp, temp_log_floor_k)
            diag["step"] = float(_plotfile_step_id(p))
            diag["a"] = float(a_pf)
            diag["z"] = float(z_pf)
            rows.append(diag)
        rows.sort(key=lambda r: (float(r.get("a", float("nan"))), float(r.get("step", -1.0))))
        return rows

    def save_phase_history_compare_plot(
        outpath: Path,
        dh_rows: list[dict[str, float]],
        nyx_rows: list[dict[str, float]],
    ) -> None:
        if (not dh_rows) and (not nyx_rows):
            return
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=170, constrained_layout=True)
        if nyx_rows:
            a_nyx = np.asarray([float(r["a"]) for r in nyx_rows], dtype=np.float64)
            slope_nyx = np.asarray([float(r["phase_slope"]) for r in nyx_rows], dtype=np.float64)
            corr_nyx = np.asarray([float(r["phase_corr"]) for r in nyx_rows], dtype=np.float64)
            axes[0].plot(a_nyx, slope_nyx, "o-", lw=1.8, ms=4, label="Nyx", color="#1f77b4")
            axes[1].plot(a_nyx, corr_nyx, "o-", lw=1.8, ms=4, label="Nyx", color="#1f77b4")
        if dh_rows:
            a_dh = np.asarray([float(r["a"]) for r in dh_rows], dtype=np.float64)
            slope_dh = np.asarray([float(r["phase_slope"]) for r in dh_rows], dtype=np.float64)
            corr_dh = np.asarray([float(r["phase_corr"]) for r in dh_rows], dtype=np.float64)
            axes[0].plot(a_dh, slope_dh, ".-", lw=1.2, ms=3, label="DiffHydro", color="#d62728", alpha=0.9)
            axes[1].plot(a_dh, corr_dh, ".-", lw=1.2, ms=3, label="DiffHydro", color="#d62728", alpha=0.9)

        axes[0].axhline(0.0, color="k", lw=0.8, alpha=0.35)
        axes[0].set_title(r"Phase Slope vs Scale Factor")
        axes[0].set_xlabel("a")
        axes[0].set_ylabel(r"slope of log$_{10} T$ vs log$_{10}(\rho/\bar\rho)$")
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=False)

        axes[1].axhline(0.0, color="k", lw=0.8, alpha=0.35)
        axes[1].set_title(r"Phase Correlation vs Scale Factor")
        axes[1].set_xlabel("a")
        axes[1].set_ylabel(r"corr(log$_{10}\rho$, log$_{10} T$)")
        axes[1].grid(alpha=0.25)
        axes[1].legend(frameon=False)

        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    # ------------------------ data loading ------------------------
    ic_plot = args.nyx_ic_plotfile.resolve()
    final_plot = args.nyx_final_plotfile.resolve()
    ic_dm_h5 = args.nyx_dm_ic_h5.resolve()
    final_dm_h5 = args.nyx_dm_final_h5.resolve()
    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    snapshot_fields_dir = args.snapshot_fields_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_fields_dir.mkdir(parents=True, exist_ok=True)

    ic_nyx = _load_nyx_plotfile_fields(ic_plot)
    final_nyx = _load_nyx_plotfile_fields(final_plot)

    n_grid = int(args.n_grid) if args.n_grid is not None else int(ic_nyx["n_grid"])
    mesh_shape = (n_grid, n_grid, n_grid)
    if any(int(v) != n_grid for v in ic_nyx["density"].shape):
        raise ValueError(
            f"Nyx IC grid {ic_nyx['density'].shape} does not match requested n-grid={n_grid}"
        )
    if any(int(v) != n_grid for v in final_nyx["density"].shape):
        raise ValueError(
            f"Nyx final grid {final_nyx['density'].shape} does not match requested n-grid={n_grid}"
        )

    box_L = float(args.box_size) if args.box_size is not None else float(ic_nyx["box_size"])
    a_init = 1.0 / (1.0 + float(args.z_init)) if args.z_init is not None else float(ic_nyx["a"])
    a_target = 1.0 / (1.0 + float(args.z_final)) if args.z_final is not None else float(final_nyx["a"])
    if a_target <= a_init:
        raise ValueError(f"Target scale factor must exceed a_init; got a_init={a_init}, a_target={a_target}")

    z_init = (1.0 / a_init) - 1.0
    z_target = (1.0 / a_target) - 1.0
    print(
        f"[info] Nyx benchmark: IC={ic_plot} (a={a_init:.6f}, z={z_init:.6f}), "
        f"final={final_plot} (a={a_target:.6f}, z={z_target:.6f}), n_grid={n_grid}, box={box_L:.6f}"
    )

    dm_ic = _load_nyx_dm_h5(ic_dm_h5, n_grid, box_L, a_init)
    dm_target = _load_nyx_dm_h5(final_dm_h5, n_grid, box_L, a_target)
    print(
        f"[info] Loaded Nyx DM HDF5: IC particles={dm_ic['n_particles']}, final particles={dm_target['n_particles']}"
    )

    # ------------------------ cosmology + physics setup ------------------------
    omega_m = float(args.omega_m)
    omega_b = float(args.omega_b)
    omega_k = float(args.omega_k)
    omega_lambda = 1.0 - omega_m - omega_k
    bg = LCDMBackground(
        h=float(args.h),
        Omega_m=omega_m,
        Omega_b=omega_b,
        Omega_lambda=omega_lambda,
        Omega_k=omega_k,
        w0=float(args.w0),
        wa=float(args.wa),
        sigma8=float(args.sigma8),
        n_s=float(args.n_s),
        use_jax_cosmo=False,
    )

    n_cons_requested = 6 if bool(args.dual_energy) else 5
    base_eq = dh.equationmanager.EquationManager(n_cons=n_cons_requested)
    base_eq.mesh_shape = list(mesh_shape)
    eq = SuperComovingEquationManager(base_eq, enforce_gamma_53=True)
    eq.eps = float(eq.eps)
    print(
        f"[info] Hydro state channels: n_cons={int(eq.n_cons)} "
        f"(dual_energy={'on' if bool(args.dual_energy) else 'off'})"
    )

    gas_mass_fraction = min(max(float(args.gas_mean_fraction), 1.0e-6), 0.999999)
    dm_mass_fraction = 1.0 - gas_mass_fraction

    hydro_velocity_scale = float(bg.H0) * float(n_grid) / max(box_L, 1.0e-30)
    hydro_vel_unit_cms = float(args.vel_unit_cms) / max(hydro_velocity_scale, 1.0e-30)
    mH_cgs = 1.6735575e-24
    kB_cgs = 1.380649e-16
    mu = float(args.mu_hydrogen)
    kelvin_to_code_temp = kB_cgs / (
        max(mu, 1.0e-12) * mH_cgs * float(hydro_vel_unit_cms) * float(hydro_vel_unit_cms)
    )
    code_to_kelvin_temp = 1.0 / max(kelvin_to_code_temp, 1.0e-30)

    print(
        "[info] Hydro scaling: "
        f"H0={float(bg.H0):.6f}, velocity_scale={hydro_velocity_scale:.6e}, "
        f"hydro_vel_unit_cms={hydro_vel_unit_cms:.6e}, "
        f"kelvin_to_code={kelvin_to_code_temp:.6e}"
    )

    # ------------------------ build initial gas state from Nyx grid ------------------------
    density_ic = jnp.asarray(ic_nyx["density"], dtype=jnp.float32)
    xmom_ic = jnp.asarray(ic_nyx["xmom"], dtype=jnp.float32)
    ymom_ic = jnp.asarray(ic_nyx["ymom"], dtype=jnp.float32)
    zmom_ic = jnp.asarray(ic_nyx["zmom"], dtype=jnp.float32)
    tempk_ic = jnp.asarray(ic_nyx["temp_k"], dtype=jnp.float32)

    rho_norm_ic = density_ic / jnp.maximum(jnp.mean(density_ic), 1.0e-30)
    rho_gas_phys_ic = jnp.asarray(gas_mass_fraction, dtype=jnp.float32) * rho_norm_ic
    if bool(args.ic_import_density_a3):
        rho_gas_phys_ic = rho_gas_phys_ic / jnp.maximum(a_init**3, 1.0e-30)

    vel_div = float(args.nyx_velocity_divisor)
    vx_pec = (xmom_ic / jnp.maximum(density_ic, 1.0e-30)) / vel_div
    vy_pec = (ymom_ic / jnp.maximum(density_ic, 1.0e-30)) / vel_div
    vz_pec = (zmom_ic / jnp.maximum(density_ic, 1.0e-30)) / vel_div

    t_floor_k = max(float(args.hydro_temp_floor_k), 0.0)
    if t_floor_k <= 0.0 and bool(args.enable_cooling):
        t_floor_k = max(float(args.cooling_temp_floor_k), 0.0)
    tempk_ic = jnp.maximum(tempk_ic, t_floor_k)
    T_phys_ic = jnp.maximum(tempk_ic * kelvin_to_code_temp, 1.0e-12)
    p_gas_phys_ic = rho_gas_phys_ic * eq.R * T_phys_ic

    # Keep dx_o fixed at 1 by rescaling hydro state velocities.
    v_scale = jnp.asarray(float(hydro_velocity_scale), dtype=jnp.float32)
    vx_pec = vx_pec * v_scale
    vy_pec = vy_pec * v_scale
    vz_pec = vz_pec * v_scale

    speed_code_ic = jnp.sqrt(vx_pec * vx_pec + vy_pec * vy_pec + vz_pec * vz_pec)
    speed_kms_ic = speed_code_ic * (float(hydro_vel_unit_cms) / 1.0e5)
    print(
        "[info] IC gas speed diagnostics [km/s]: "
        f"p50={float(np.percentile(np.asarray(speed_kms_ic), 50.0)):.3e}, "
        f"p95={float(np.percentile(np.asarray(speed_kms_ic), 95.0)):.3e}, "
        f"p99={float(np.percentile(np.asarray(speed_kms_ic), 99.0)):.3e}, "
        f"max={float(np.max(np.asarray(speed_kms_ic))):.3e}"
    )

    w_code = cosmo_conv.primitives_phys_to_code(
        rho_gas_phys_ic,
        vx_pec,
        vy_pec,
        vz_pec,
        p_gas_phys_ic,
        a_init,
    )
    U = eq.get_conservatives_from_primitives(w_code)

    bg_force = BackgroundExpansionForce(bg, a_init=a_init)
    grav_force = None
    if bool(args.use_gravity):
        grav_force = JaxPMCoupledGravityForce(
            eq,
            mesh_shape=mesh_shape,
            subtract_mean=True,
            use_jaxpm=True,
            dm_drift_factor=bg.H0,
            dm_kick_factor=1.0,
            gas_kick_factor=(None if args.gas_kick_factor is None else float(args.gas_kick_factor)),
            eps=float(args.force_eps),
        )

    rho_unit_cgs_used = float(args.rho_unit_cgs)
    if bool(args.enable_cooling) and str(args.cooling_model).lower() == "nyx_table" and bool(args.nyx_auto_rho_unit):
        rho_b0_cgs = rho_crit0_cgs(float(args.h)) * float(args.omega_b)
        rho_unit_auto = rho_b0_cgs / max(float(gas_mass_fraction), 1.0e-30)
        rho_unit_cgs_used = float(rho_unit_auto)
        print("[info] Auto-setting cooling rho unit: ", rho_unit_cgs_used)

    cooling_force = None
    if bool(args.enable_cooling):
        cooling_model = str(args.cooling_model).lower()
        if False:
            if not args.cooling_table.exists():
                raise FileNotFoundError(f"Cooling table not found: {args.cooling_table}")
            cooling_table = np.genfromtxt(args.cooling_table)
            if cooling_table.ndim != 2 or cooling_table.shape[1] < 2:
                raise ValueError(
                    f"Cooling table {args.cooling_table} must have at least 2 columns: logT and logLambda_m20."
                )
            cooling_force = CosmologicalHydrogenCoolingForce(
                eq,
                cooling_table[:, 0],
                cooling_table[:, 1],
                rho_unit_cgs=float(rho_unit_cgs_used),
                vel_unit_cms=float(hydro_vel_unit_cms),
                tau_time_unit_s=float(args.tau_time_unit_s),
                mu=float(args.mu_hydrogen),
                h_species=float(args.h_species),
                heating_rate_per_h=float(args.heating_rate_per_h),
                cooling_rate_scale=float(args.cooling_rate_scale),
                temp_floor_k=float(args.cooling_temp_floor_k),
                subcycles=int(args.cooling_subcycles),
                dtmax_s=float(args.cooling_dtmax_s),
            )
        elif cooling_model == "nyx_table":
            z_nodes = parse_float_list(args.nyx_cooling_z_nodes)
            ld_nodes = np.linspace(
                float(args.nyx_cooling_logdelta_min),
                float(args.nyx_cooling_logdelta_max),
                int(max(2, args.nyx_cooling_logdelta_n)),
                dtype=np.float64,
            )
            lt_nodes = np.linspace(
                float(args.nyx_cooling_logt_min),
                float(args.nyx_cooling_logt_max),
                int(max(2, args.nyx_cooling_logt_n)),
                dtype=np.float64,
            )
            nyx_table_path = args.nyx_cooling_table_npz.resolve()
            if bool(args.nyx_cooling_rebuild) or (not nyx_table_path.exists()):
                print(
                    "[info] Building Nyx cooling table: "
                    f"path={nyx_table_path}, z_nodes={len(z_nodes)}, "
                    f"logdelta_n={ld_nodes.size}, logT_n={lt_nodes.size}"
                )
                build_nyx_cooling_table(
                    nyx_table_path,
                    treecool_file=args.nyx_cooling_treecool.resolve(),
                    z_nodes=z_nodes,
                    logdelta_nodes=ld_nodes,
                    logT_nodes=lt_nodes,
                    h=float(args.h),
                    omega_b=float(args.omega_b),
                    h_species=float(args.h_species),
                    uvb_density_a=1.00, #what is this doing....
                    uvb_density_b=0.0,
                    jh=1,
                    jhe=1,
                    eos_search_paths=[args.nyx_cooling_eos_path.resolve()],
                )
            nyx_table = NyxCoolingTableData.load(nyx_table_path)
            nyx_interp = NyxCoolingRateInterpolator(nyx_table)
            cooling_force = NyxTabulatedCoolingForce(
                eq,
                nyx_interp,
                rho_unit_cgs=float(rho_unit_cgs_used),
                vel_unit_cms=float(hydro_vel_unit_cms),
                tau_time_unit_s=float(args.tau_time_unit_s),
                mu=float(args.mu_hydrogen),
                heating_rate_scale=float(args.nyx_heating_scale),
                cooling_rate_scale=float(args.cooling_rate_scale),
                temp_floor_k=float(args.cooling_temp_floor_k),
                subcycles=int(args.cooling_subcycles),
                dtmax_s=float(args.cooling_dtmax_s),
            )
        else:
            raise ValueError(f"Unsupported cooling model: {args.cooling_model}")

    force_stack = [bg_force]
    if grav_force is not None:
        force_stack.append(grav_force)
    if cooling_force is not None:
        force_stack.append(cooling_force)

    active_solver = str(args.solver).lower()
    sim = build_hydrosim(
        eq,
        force_stack,
        solver_name=active_solver,
        dx_o=1.0,
        nyx_interface_energy_mode=str(args.nyx_interface_energy_mode),
    )
    print(f"[info] Hydro solver: {active_solver}")
    if active_solver in ("nyx", "nyx_riemann"):
        print(f"[info] NyxRiemann interface-energy mode: {str(args.nyx_interface_energy_mode).lower()}")
    print(
        "[info] Forces: "
        f"background=on, gravity={'on' if grav_force is not None else 'off'}, "
        f"cooling={'on' if cooling_force is not None else 'off'}"
    )

    if bool(args.use_gravity):
        dm_kick_prefactor_base = 1.5 * omega_m * bg.H0 * float(args.dm_kick_scale)
        gas_kick_prefactor_base = None
        if args.gas_kick_factor is None:
            if args.gas_kick_mode == "dm_consistent":
                gas_kick_prefactor_base = (
                    1.5 * omega_m * bg.H0 * bg.H0 * float(args.gas_kick_scale)
                )
            else:
                gas_kick_prefactor_base = (
                    1.5
                    * omega_m
                    * bg.H0
                    * float(hydro_velocity_scale)
                    * float(args.gas_kick_scale)
                )
        print(
            "[info] Kick prefactors: "
            f"dm={dm_kick_prefactor_base:.6e}, "
            f"gas_mode={args.gas_kick_mode}, "
            f"gas_pref={('fixed:' + format(float(args.gas_kick_factor), '.6e')) if args.gas_kick_factor is not None else format(float(gas_kick_prefactor_base), '.6e')}"
        )
    else:
        dm_kick_prefactor_base = 0.0
        gas_kick_prefactor_base = 0.0
        print("[info] Gravity disabled: DM/Gas kick prefactors set to 0.")

    if cooling_force is not None:
        if str(args.cooling_model).lower() == "legacy":
            print(
                f"[info] Cooling enabled (legacy): table={args.cooling_table}, T_floor={float(args.cooling_temp_floor_k):.3e} K, "
                f"subcycles={int(args.cooling_subcycles)}, cooling_scale={float(args.cooling_rate_scale):.3e}, "
                f"heating_per_H={float(args.heating_rate_per_h):.3e} erg/s, "
                f"h_species={float(args.h_species):.3f}, rho_unit_cgs={rho_unit_cgs_used:.6e}"
            )
        else:
            print(
                f"[info] Cooling enabled (nyx_table): table_npz={args.nyx_cooling_table_npz}, "
                f"treecool={args.nyx_cooling_treecool}, "
                f"T_floor={float(args.cooling_temp_floor_k):.3e} K, "
                f"subcycles={int(args.cooling_subcycles)}, cooling_scale={float(args.cooling_rate_scale):.3e}, "
                f"heating_scale={float(args.nyx_heating_scale):.3e}, "
                f"h_species={float(args.h_species):.3f}, rho_unit_cgs={rho_unit_cgs_used:.6e}, "
                f"auto_rho_unit={bool(args.nyx_auto_rho_unit)}"
            )

    params = {
        "a": jnp.asarray(a_init, dtype=jnp.float32),
        "dm": {
            "x": dm_ic["pos_grid"],
            "p_or_v": dm_ic["p_or_v"],
            "mass": jnp.ones((dm_ic["pos_grid"].shape[0],), dtype=jnp.float32)
            * jnp.asarray(dm_mass_fraction, dtype=jnp.float32),
            "drift_factor": jnp.asarray(bg.H0, dtype=jnp.float32),
            "kick_prefactor": jnp.asarray(dm_kick_prefactor_base, dtype=jnp.float32),
        },
    }
    if (not bool(args.use_gravity)):
        params["dm"]["drift_factor"] = jnp.asarray(0.0, dtype=jnp.float32)
        params["dm"]["kick_prefactor"] = jnp.asarray(0.0, dtype=jnp.float32)
        params["dm"]["gas_kick_prefactor"] = jnp.asarray(0.0, dtype=jnp.float32)
        params["dm"]["gas_kick_factor"] = jnp.asarray(0.0, dtype=jnp.float32)
    elif args.gas_kick_factor is None:
        params["dm"]["gas_kick_prefactor"] = jnp.asarray(gas_kick_prefactor_base, dtype=jnp.float32)
    else:
        params["dm"]["gas_kick_factor"] = jnp.asarray(float(args.gas_kick_factor), dtype=jnp.float32)

    phase_history_every = int(max(0, args.phase_history_every))
    phase_history_dh: list[dict[str, float]] = []
    temp_log_floor_k = max(float(args.temp_log_floor_k), 1.0e-20)

    def _append_diffhydro_phase_history(step_idx: int) -> None:
        w_state = eq.get_primitives_from_conservatives(U)
        rho_code_state = jnp.maximum(w_state[0], eq.eps)
        p_code_state = jnp.maximum(w_state[4], eq.eps)
        a_state = params.get("a", a_init)
        rho_phys_state = cosmo_conv.density_code_to_phys(rho_code_state, a_state)
        p_phys_state = cosmo_conv.pressure_code_to_phys(p_code_state, a_state)
        t_code_state = p_phys_state / jnp.maximum(rho_phys_state * eq.R, 1.0e-30)
        temp_state_k = _code_temp_to_kelvin(
            t_code_state,
            mu_hydrogen=float(args.mu_hydrogen),
            vel_unit_cms=float(hydro_vel_unit_cms),
        )
        rho_norm_state = np.asarray(
            rho_phys_state / jnp.maximum(jnp.mean(rho_phys_state), 1.0e-30),
            dtype=np.float32,
        )
        temp_state_np = np.asarray(temp_state_k, dtype=np.float32)
        diag = _phase_diag_summary(rho_norm_state, temp_state_np, temp_log_floor_k)
        a_now_state = float(np.asarray(a_state))
        diag["step"] = float(step_idx)
        diag["a"] = float(a_now_state)
        diag["z"] = float((1.0 / max(a_now_state, 1.0e-30)) - 1.0)
        phase_history_dh.append(diag)

    if phase_history_every > 0:
        _append_diffhydro_phase_history(0)

    # ------------------------ integration ------------------------
    step_i = 0
    tol = 1.0e-6
    dtau_retry_cap = float(args.dtau_max)
    consecutive_failed_steps = 0
    prev_accepted_dtau = None
    retry_total = 0
    retry_steps_nonzero = 0
    max_retry_in_step = 0

    if int(args.debug_step_stats) > 0:
        d0 = _flow_diagnostics(U, params["a"])
        print(
            "[debug] init "
            f"Mach[p95={d0['mach_p95']:.3e},p99={d0['mach_p99']:.3e},max={d0['mach_max']:.3e}] "
            f"speed[km/s,p99={d0['speed_kms_p99']:.3e},max={d0['speed_kms_max']:.3e}] "
            f"T[K,min={d0['tempk_min']:.3e},p50={d0['tempk_p50']:.3e},p99={d0['tempk_p99']:.3e}]"
        )

    while float(params["a"]) + tol < float(a_target):
        if step_i >= int(args.max_steps):
            raise RuntimeError(
                f"Reached max steps ({args.max_steps}) before target a={a_target:.6f}."
            )

        a_now = float(params["a"])
        da_dtau = float(bg.da_dtau(a_now))
        remaining = max(float(a_target) - a_now, 0.0)
        da_dtau_safe = max(da_dtau, 1.0e-12)

        dtau_by_target = float(args.dtau_safety) * remaining / da_dtau_safe
        dtau_by_rel_a = float("inf")
        if float(args.relative_max_change_a) > 0.0:
            dtau_by_rel_a = float(args.relative_max_change_a) * max(a_now, 1.0e-12) / da_dtau_safe
        dtau_by_abs_a = float("inf")
        if float(args.absolute_max_change_a) > 0.0:
            dtau_by_abs_a = float(args.absolute_max_change_a) / da_dtau_safe
        dtau_solver = float("inf")
        if not args.disable_solver_dt:
            dtau_solver = float(args.solver_dt_safety) * float(sim.timestep(U))

        dtau = min(
            float(args.dtau_max),
            float(dtau_retry_cap),
            dtau_by_target,
            dtau_by_rel_a,
            dtau_by_abs_a,
            dtau_solver
        )

        if step_i %10 ==0:
            print(step_i,params["a"],dtau,float(args.dtau_max),
            float(dtau_retry_cap),
            dtau_by_target,
            dtau_by_rel_a,
            dtau_by_abs_a,
            dtau_solver)

        if prev_accepted_dtau is not None and float(args.dtau_change_max) > 0.0:
            dtau = min(dtau, float(args.dtau_change_max) * float(prev_accepted_dtau))
        if step_i == 0:
            dtau = dtau * float(args.dtau_init_shrink)
        dtau = max(dtau, float(args.dtau_min))

        params_dm = dict(params["dm"])
        params_dm["drift_factor"] = jnp.asarray(bg.H0 if bool(args.use_gravity) else 0.0, dtype=jnp.float32)
        params_dm["kick_prefactor"] = jnp.asarray(dm_kick_prefactor_base, dtype=jnp.float32)
        if not bool(args.use_gravity):
            params_dm["gas_kick_prefactor"] = jnp.asarray(0.0, dtype=jnp.float32)
            params_dm["gas_kick_factor"] = jnp.asarray(0.0, dtype=jnp.float32)
        elif args.gas_kick_factor is None:
            params_dm["gas_kick_prefactor"] = jnp.asarray(gas_kick_prefactor_base, dtype=jnp.float32)
            params_dm.pop("gas_kick_factor", None)
        else:
            params_dm["gas_kick_factor"] = jnp.asarray(float(args.gas_kick_factor), dtype=jnp.float32)
            params_dm.pop("gas_kick_prefactor", None)
        params = dict(params)
        params["dm"] = params_dm

        accepted = False
        needs_solver_switch = False
        trial_dtau = float(dtau)
        retries_this_step = 0
        for retry_i in range(int(args.step_retries) + 1):
            if retry_i > 0:
                retry_total += 1
                retries_this_step = max(retries_this_step, retry_i)
                print(
                    f"[warn] Retry {retry_i}/{args.step_retries}: step={step_i}, a={a_now:.6f}, "
                    f"dtau={trial_dtau:.3e}, solver={active_solver}"
                )
                if (
                    args.auto_lf_fallback
                    and active_solver == "hllc"
                    and retry_i >= int(max(1, args.retry_fallback_threshold))
                ):
                    active_solver = "lf"
                    sim = build_hydrosim(eq, force_stack, solver_name=active_solver, dx_o=1.0)
                    dtau_retry_cap = float(args.dtau_max)
                    needs_solver_switch = True
                    print(
                        f"[warn] Switched solver from HLLC to LaxFriedrichs after {retry_i} retries "
                        f"at step={step_i}, a={a_now:.6f}; retry cap reset to dtau_max."
                    )
                    break
            U_try_raw, params_try = sim._hydrostep(
                step_i, (U, params), jnp.asarray(trial_dtau, dtype=jnp.float32)
            )
            w_try = eq.get_primitives_from_conservatives(U_try_raw)
            a_try = params_try.get("a", params.get("a", 1.0))
            if True:
                rho_floor, p_floor = _compute_phys_floor_code(
                    a_try,
                    float(args.state_floor),
                    float(args.pressure_floor),
                    w_try.dtype,
                )
                if t_floor_k > 0.0:
                    t_floor_code = jnp.asarray(kelvin_to_code_temp * t_floor_k, dtype=w_try.dtype)
                    rho_phys_try = cosmo_conv.density_code_to_phys(w_try[0], a_try)
                    p_floor_temp_phys = rho_phys_try * jnp.asarray(float(eq.R), dtype=w_try.dtype) * t_floor_code
                    p_floor_temp_code = cosmo_conv.pressure_phys_to_code(p_floor_temp_phys, a_try)
                    p_floor = jnp.maximum(p_floor, p_floor_temp_code)

                w_try = w_try.at[0].set(jnp.maximum(w_try[0], rho_floor))
                w_try = w_try.at[4].set(jnp.maximum(w_try[4], p_floor))
                for i_passive in tuple(getattr(eq, "passive_ids", ())):
                    if int(i_passive) >= 5:
                        w_try = w_try.at[i_passive].set(jnp.maximum(w_try[i_passive], eq.eps))
            U_try_proj = eq.get_conservatives_from_primitives(w_try)

            if _state_is_finite(U_try_proj, params_try):
                U, params = U_try_proj, params_try
                if retry_i > 0:
                    dtau_retry_cap = min(
                        float(dtau_retry_cap),
                        float(trial_dtau) * float(args.retry_cap_mult),
                    )
                else:
                    dtau_retry_cap = min(
                        float(args.dtau_max),
                        float(dtau_retry_cap) * 1.1#float(max(args.dtau_retry_cap_relax, 1.0)),
                    )
                dtau_retry_cap = max(float(args.dtau_min), float(dtau_retry_cap))
                accepted = True
                break

            trial_dtau = trial_dtau * float(args.dtau_retry_factor)
            if trial_dtau < float(args.dtau_min):
                break

        if retries_this_step > 0:
            retry_steps_nonzero += 1
            max_retry_in_step = max(max_retry_in_step, retries_this_step)

        if needs_solver_switch:
            continue

        if not accepted:
            consecutive_failed_steps += 1
            print(
                f"[warn] Failed finite step at step={step_i}, a={a_now:.6f}, "
                f"solver={active_solver}, consecutive_failures={consecutive_failed_steps}"
            )
            if (
                args.auto_lf_fallback
                and active_solver == "hllc"
                and consecutive_failed_steps >= int(max(1, args.nan_fallback_threshold))
            ):
                active_solver = "laxfriedrichs"
                sim = build_hydrosim(eq, force_stack, solver_name=active_solver, dx_o=1.0)
                consecutive_failed_steps = 0
                dtau_retry_cap = float(args.dtau_max)
                print(
                    "[warn] Switched solver from HLLC to LaxFriedrichs after repeated NaN failures; "
                    "retry cap reset to dtau_max."
                )
                continue
            raise RuntimeError(
                f"Failed to take finite step at step={step_i}, a={a_now:.6f}. "
                f"last_dtau={trial_dtau:.3e}, solver={active_solver}"
            )

        consecutive_failed_steps = 0
        prev_accepted_dtau = float(trial_dtau)
        step_i += 1
        if phase_history_every > 0 and (step_i % phase_history_every == 0):
            _append_diffhydro_phase_history(step_i)
        if step_i <= int(max(0, args.debug_step_stats)):
            di = _flow_diagnostics(U, params["a"])
            print(
                f"[debug] step={step_i:d} a={float(params['a']):.6f} "
                f"Mach[p95={di['mach_p95']:.3e},p99={di['mach_p99']:.3e},max={di['mach_max']:.3e}] "
                f"speed[km/s,p99={di['speed_kms_p99']:.3e},max={di['speed_kms_max']:.3e}] "
                f"T[K,min={di['tempk_min']:.3e},p50={di['tempk_p50']:.3e},p99={di['tempk_p99']:.3e}]"
            )

    print(
        f"[info] Integration finished: steps={step_i}, a_final={float(params['a']):.9f}, "
        f"z_final={(1.0/float(params['a'])-1.0):.6f}, solver={active_solver}"
    )
    if phase_history_every > 0:
        if (not phase_history_dh) or (int(phase_history_dh[-1].get("step", -1)) != int(step_i)):
            _append_diffhydro_phase_history(step_i)

    # ------------------------ products + tests ------------------------
    if not _state_is_finite(U, params):
        raise RuntimeError("Final state contains non-finite values.")

    dm_dh_ic = paint_particle_density(dm_ic["pos_grid"], jnp.ones((dm_ic["pos_grid"].shape[0],), dtype=jnp.float32), mesh_shape)
    dm_dh_ic = np.asarray(dm_dh_ic / jnp.maximum(jnp.mean(dm_dh_ic), 1.0e-20), dtype=np.float32)
    dm_dh_final = paint_particle_density(
        params["dm"]["x"], jnp.ones((params["dm"]["x"].shape[0],), dtype=jnp.float32), mesh_shape
    )
    dm_dh_final = np.asarray(dm_dh_final / jnp.maximum(jnp.mean(dm_dh_final), 1.0e-20), dtype=np.float32)

    dm_nyx_final = paint_particle_density(
        dm_target["pos_grid"], jnp.ones((dm_target["pos_grid"].shape[0],), dtype=jnp.float32), mesh_shape
    )
    dm_nyx_final = np.asarray(dm_nyx_final / jnp.maximum(jnp.mean(dm_nyx_final), 1.0e-20), dtype=np.float32)

    w_final = eq.get_primitives_from_conservatives(U)
    rho_code_final = jnp.maximum(w_final[0], eq.eps)
    p_code_final = jnp.maximum(w_final[4], eq.eps)
    rho_gas_final_phys = cosmo_conv.density_code_to_phys(rho_code_final, params["a"])
    p_gas_final_phys = cosmo_conv.pressure_code_to_phys(p_code_final, params["a"])
    t_code_final = p_gas_final_phys / jnp.maximum(rho_gas_final_phys * eq.R, 1.0e-30)
    temp_gas_final_k = _code_temp_to_kelvin(
        t_code_final,
        mu_hydrogen=float(args.mu_hydrogen),
        vel_unit_cms=float(hydro_vel_unit_cms),
    )

    gas_dh_final = np.asarray(
        rho_gas_final_phys / jnp.maximum(jnp.mean(rho_gas_final_phys), 1.0e-30),
        dtype=np.float32,
    )
    temp_dh_final = np.asarray(temp_gas_final_k, dtype=np.float32)

    gas_nyx_ic = np.asarray(ic_nyx["density"] / np.maximum(np.mean(ic_nyx["density"]), 1.0e-30), dtype=np.float32)
    gas_nyx_final = np.asarray(
        final_nyx["density"] / np.maximum(np.mean(final_nyx["density"]), 1.0e-30), dtype=np.float32
    )
    temp_nyx_ic = np.asarray(ic_nyx["temp_k"], dtype=np.float32)
    temp_nyx_final = np.asarray(final_nyx["temp_k"], dtype=np.float32)

    # DiffHydro IC maps for plotting
    gas_dh_ic = np.asarray(
        rho_gas_phys_ic / jnp.maximum(jnp.mean(rho_gas_phys_ic), 1.0e-30),
        dtype=np.float32,
    )
    temp_dh_ic = np.asarray(tempk_ic, dtype=np.float32)
    log_temp_nyx_final = np.log10(np.maximum(temp_nyx_final, temp_log_floor_k))
    log_temp_dh_final = np.log10(np.maximum(temp_dh_final, temp_log_floor_k))

    x_phase_nyx, y_phase_nyx = _phase_xy(gas_nyx_final, temp_nyx_final, temp_log_floor_k)
    x_phase_dh, y_phase_dh = _phase_xy(gas_dh_final, temp_dh_final, temp_log_floor_k)
    phase_slope_nyx, phase_intercept_nyx = _phase_fit(x_phase_nyx, y_phase_nyx)
    phase_slope_dh, phase_intercept_dh = _phase_fit(x_phase_dh, y_phase_dh)
    xmr, ymr = _binned_median(x_phase_nyx, y_phase_nyx)
    xmd, ymd = _binned_median(x_phase_dh, y_phase_dh)
    phase_median_rmse_dex = float("nan")
    if xmr.size > 0 and xmd.size > 0:
        x0 = max(np.min(xmr), np.min(xmd))
        x1 = min(np.max(xmr), np.max(xmd))
        if x1 > x0:
            xs = np.linspace(x0, x1, 200)
            yr_i = np.interp(xs, xmr, ymr)
            yd_i = np.interp(xs, xmd, ymd)
            phase_median_rmse_dex = float(np.sqrt(np.mean((yd_i - yr_i) ** 2)))

    metrics = {
        "gas_density_corr_final": _pearson_corr(gas_nyx_final, gas_dh_final),
        "gas_temp_corr_final": _pearson_corr(log_temp_nyx_final, log_temp_dh_final),
        "dm_density_corr_final": _pearson_corr(dm_nyx_final, dm_dh_final),
        "gas_density_rmse_final": _rmse(gas_nyx_final, gas_dh_final),
        "gas_temp_rmse_final": _rmse(log_temp_nyx_final, log_temp_dh_final),
        "dm_density_rmse_final": _rmse(dm_nyx_final, dm_dh_final),
        "gas_density_mae_final": _mae(gas_nyx_final, gas_dh_final),
        "gas_temp_mae_final": _mae(log_temp_nyx_final, log_temp_dh_final),
        "dm_density_mae_final": _mae(dm_nyx_final, dm_dh_final),
        "phase_logrho_logt_corr_nyx_final": _pearson_corr(x_phase_nyx, y_phase_nyx),
        "phase_logrho_logt_corr_dh_final": _pearson_corr(x_phase_dh, y_phase_dh),
        "phase_slope_nyx_final": phase_slope_nyx,
        "phase_slope_dh_final": phase_slope_dh,
        "phase_intercept_nyx_final": phase_intercept_nyx,
        "phase_intercept_dh_final": phase_intercept_dh,
        "phase_median_rmse_dex_final": phase_median_rmse_dex,
        "temp_lt_1k_frac_nyx_final": float(np.mean(temp_nyx_final < 1.0)),
        "temp_lt_1k_frac_dh_final": float(np.mean(temp_dh_final < 1.0)),
        "temp_lt_10k_frac_nyx_final": float(np.mean(temp_nyx_final < 10.0)),
        "temp_lt_10k_frac_dh_final": float(np.mean(temp_dh_final < 10.0)),
        "temp_log_floor_k": float(temp_log_floor_k),
        "hydro_temp_floor_k": float(t_floor_k),
        "heating_rate_per_h": float(args.heating_rate_per_h),
        "cooling_rate_scale": float(args.cooling_rate_scale),
        "use_gravity": bool(args.use_gravity),
        "enable_cooling": bool(args.enable_cooling),
        "a_init": float(a_init),
        "a_target": float(a_target),
        "a_final_dh": float(np.asarray(params["a"])),
        "z_init": float(z_init),
        "z_target": float(z_target),
        "z_final_dh": float((1.0 / float(np.asarray(params["a"]))) - 1.0),
        "n_steps": int(step_i),
        "solver_final": active_solver,
        "retry_total": int(retry_total),
        "retry_steps_nonzero": int(retry_steps_nonzero),
        "max_retry_in_step": int(max_retry_in_step),
        "dtau_retry_cap_final": float(dtau_retry_cap),
        "dm_kick_prefactor_base": float(dm_kick_prefactor_base),
        "gas_kick_mode": str(args.gas_kick_mode),
        "gas_kick_prefactor_base": (
            float(gas_kick_prefactor_base) if gas_kick_prefactor_base is not None else float("nan")
        ),
        "gas_kick_factor_arg": (
            float(args.gas_kick_factor) if args.gas_kick_factor is not None else float("nan")
        ),
        "gas_kick_factor_init": (
            float(args.gas_kick_factor)
            if args.gas_kick_factor is not None
            else float(gas_kick_prefactor_base) * float(a_init)
        ),
        "gas_kick_factor_final": (
            float(args.gas_kick_factor)
            if args.gas_kick_factor is not None
            else float(gas_kick_prefactor_base) * float(np.asarray(params["a"]))
        ),
        "dual_energy": bool(args.dual_energy),
        "n_cons": int(eq.n_cons),
        "nyx_interface_energy_mode": str(args.nyx_interface_energy_mode).lower(),
    }

    pass_gas_rho = metrics["gas_density_corr_final"] >= float(args.min_gas_rho_corr)
    pass_gas_temp = metrics["gas_temp_corr_final"] >= float(args.min_gas_temp_corr)
    pass_dm = metrics["dm_density_corr_final"] >= float(args.min_dm_corr)
    metrics["pass_gas_rho_corr"] = bool(pass_gas_rho)
    metrics["pass_gas_temp_corr"] = bool(pass_gas_temp)
    metrics["pass_dm_corr"] = bool(pass_dm)
    metrics["pass_all"] = bool(pass_gas_rho and pass_gas_temp and pass_dm)

    np.savez_compressed(output_dir / "metrics.npz", **{k: np.asarray(v) for k, v in metrics.items()})
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        for k in sorted(metrics.keys()):
            f.write(f"{k}: {metrics[k]}\n")

    np.savez_compressed(
        snapshot_fields_dir / "fields_ic_final.npz",
        gas_nyx_ic=gas_nyx_ic,
        gas_nyx_final=gas_nyx_final,
        gas_dh_ic=gas_dh_ic,
        gas_dh_final=gas_dh_final,
        temp_nyx_ic=temp_nyx_ic,
        temp_nyx_final=temp_nyx_final,
        temp_dh_ic=temp_dh_ic,
        temp_dh_final=temp_dh_final,
        dm_dh_ic=dm_dh_ic,
        dm_dh_final=dm_dh_final,
        dm_nyx_final=dm_nyx_final,
    )

    # Consolidated final 3D snapshot bundle for Nyx vs DiffHydro field-level analysis.
    U_final_np = np.asarray(U, dtype=np.float32)
    w_final_np = np.asarray(w_final, dtype=np.float32)
    final_snapshot_payload = {
        "a_final": np.asarray(float(params["a"]), dtype=np.float32),
        "z_final": np.asarray(float((1.0 / float(params["a"])) - 1.0), dtype=np.float32),
        "n_grid": np.asarray(int(n_grid), dtype=np.int32),
        "box_size": np.asarray(float(box_L), dtype=np.float32),
        "n_cons": np.asarray(int(eq.n_cons), dtype=np.int32),
        # Nyx final gas fields (raw plotfile variables + derived normalized density)
        "nyx_density": np.asarray(final_nyx["density"], dtype=np.float32),
        "nyx_xmom": np.asarray(final_nyx["xmom"], dtype=np.float32),
        "nyx_ymom": np.asarray(final_nyx["ymom"], dtype=np.float32),
        "nyx_zmom": np.asarray(final_nyx["zmom"], dtype=np.float32),
        "nyx_temp_k": np.asarray(temp_nyx_final, dtype=np.float32),
        "nyx_gas_density_norm": np.asarray(gas_nyx_final, dtype=np.float32),
        "nyx_dm_density_norm": np.asarray(dm_nyx_final, dtype=np.float32),
        # DiffHydro final state (conservative + primitive + derived)
        "dh_cons_density_code": U_final_np[0],
        "dh_cons_xmom_code": U_final_np[1],
        "dh_cons_ymom_code": U_final_np[2],
        "dh_cons_zmom_code": U_final_np[3],
        "dh_cons_energy_code": U_final_np[4],
        "dh_prim_density_code": w_final_np[0],
        "dh_prim_vx_code": w_final_np[1],
        "dh_prim_vy_code": w_final_np[2],
        "dh_prim_vz_code": w_final_np[3],
        "dh_prim_pressure_code": w_final_np[4],
        "dh_temp_k": np.asarray(temp_dh_final, dtype=np.float32),
        "dh_gas_density_norm": np.asarray(gas_dh_final, dtype=np.float32),
        "dh_dm_density_norm": np.asarray(dm_dh_final, dtype=np.float32),
    }
    if U_final_np.shape[0] > 5:
        final_snapshot_payload["dh_cons_dual_energy_code"] = U_final_np[5]
        final_snapshot_payload["dh_prim_dual_energy_code"] = w_final_np[5]

    np.savez_compressed(
        snapshot_fields_dir / "final_snapshot_3d.npz",
        **final_snapshot_payload,
    )

    np.save(snapshot_fields_dir / "final_state_dh.npz", U)

    nyx_phase_history: list[dict[str, float]] = []
    if phase_history_every > 0:
        _write_phase_history_csv(output_dir / "phase_history_diffhydro.csv", phase_history_dh)
        try:
            nyx_phase_history = _collect_nyx_phase_history(
                ic_plot,
                final_plot,
                stride=int(args.nyx_phase_history_stride),
                temp_log_floor_k=temp_log_floor_k,
            )
        except Exception as exc:
            print(f"[warn] Failed to build Nyx phase-history series: {exc}")
            nyx_phase_history = []
        if nyx_phase_history:
            _write_phase_history_csv(output_dir / "phase_history_nyx.csv", nyx_phase_history)

    # ------------------------ plotting ------------------------
    save_overview_compare(
        plots_dir / "ic_overview_compare.png",
        dm_nyx=dm_dh_ic,
        dm_dh=dm_dh_ic,
        gas_nyx=gas_nyx_ic,
        gas_dh=gas_dh_ic,
        temp_nyx=temp_nyx_ic,
        temp_dh=temp_dh_ic,
        title_prefix=f"IC comparison (z={z_init:.3f})",
    )
    save_overview_compare(
        plots_dir / "final_overview_compare.png",
        dm_nyx=dm_nyx_final,
        dm_dh=dm_dh_final,
        gas_nyx=gas_nyx_final,
        gas_dh=gas_dh_final,
        temp_nyx=temp_nyx_final,
        temp_dh=temp_dh_final,
        title_prefix=f"Final comparison Nyx vs DiffHydro (z_target={z_target:.3f})",
    )

    # Per-field/time 3-panel slices
    save_3panel(gas_nyx_ic, plots_dir / "nyx_ic_gas_density_norm_3panel.png", f"Nyx IC Gas Density Norm (z={z_init:.3f})", log=True)
    save_3panel(gas_dh_ic, plots_dir / "diffhydro_ic_gas_density_norm_3panel.png", f"DiffHydro IC Gas Density Norm (z={z_init:.3f})", log=True)
    save_3panel(temp_nyx_ic, plots_dir / "nyx_ic_temp_k_3panel.png", f"Nyx IC Temperature [K] (z={z_init:.3f})", log=True, cmap="inferno")
    save_3panel(temp_dh_ic, plots_dir / "diffhydro_ic_temp_k_3panel.png", f"DiffHydro IC Temperature [K] (z={z_init:.3f})", log=True, cmap="inferno")
    save_3panel(dm_dh_ic, plots_dir / "ic_dm_density_norm_3panel.png", f"IC DM Density Norm (z={z_init:.3f})", log=True, cmap="magma")

    save_3panel(gas_nyx_final, plots_dir / "nyx_final_gas_density_norm_3panel.png", f"Nyx Final Gas Density Norm (z={z_target:.3f})", log=True)
    save_3panel(gas_dh_final, plots_dir / "diffhydro_final_gas_density_norm_3panel.png", f"DiffHydro Final Gas Density Norm (z={z_target:.3f})", log=True)
    save_3panel(gas_dh_final - gas_nyx_final, plots_dir / "final_residual_gas_density_norm_3panel.png", "Final Gas Density Residual (DiffHydro - Nyx)", log=False, diverging=True, cmap="coolwarm")

    save_3panel(temp_nyx_final, plots_dir / "nyx_final_temp_k_3panel.png", f"Nyx Final Temperature [K] (z={z_target:.3f})", log=True, cmap="inferno")
    save_3panel(temp_dh_final, plots_dir / "diffhydro_final_temp_k_3panel.png", f"DiffHydro Final Temperature [K] (z={z_target:.3f})", log=True, cmap="inferno")
    save_3panel(temp_dh_final - temp_nyx_final, plots_dir / "final_residual_temp_k_3panel.png", "Final Temperature Residual (DiffHydro - Nyx)", log=False, diverging=True, cmap="coolwarm")

    save_3panel(dm_nyx_final, plots_dir / "nyx_final_dm_density_norm_3panel.png", f"Nyx Final DM Density Norm (z={z_target:.3f})", log=True, cmap="magma")
    save_3panel(dm_dh_final, plots_dir / "diffhydro_final_dm_density_norm_3panel.png", f"DiffHydro Final DM Density Norm (z={z_target:.3f})", log=True, cmap="magma")
    save_3panel(dm_dh_final - dm_nyx_final, plots_dir / "final_residual_dm_density_norm_3panel.png", "Final DM Density Residual (DiffHydro - Nyx)", log=False, diverging=True, cmap="coolwarm")

    save_matched_2x3_compare(
        plots_dir / "final_matched_gas_density_norm_nyx_vs_diffhydro.png",
        gas_nyx_final,
        gas_dh_final,
        f"Final Gas Density Norm (Matched Scale, z={z_target:.3f})",
        log=True,
        cmap="viridis",
    )
    save_matched_2x3_compare(
        plots_dir / "final_matched_temp_k_nyx_vs_diffhydro.png",
        temp_nyx_final,
        temp_dh_final,
        f"Final Temperature [K] (Matched Scale, z={z_target:.3f})",
        log=True,
        cmap="inferno",
    )
    save_matched_2x3_compare(
        plots_dir / "final_matched_dm_density_norm_nyx_vs_diffhydro.png",
        dm_nyx_final,
        dm_dh_final,
        f"Final DM Density Norm (Matched Scale, z={z_target:.3f})",
        log=True,
        cmap="magma",
    )

    save_spectra_plot(
        plots_dir / "final_spectra_compare.png",
        dm_ref=dm_nyx_final,
        dm_dh=dm_dh_final,
        gas_ref=gas_nyx_final,
        gas_dh=gas_dh_final,
        temp_ref=temp_nyx_final,
        temp_dh=temp_dh_final,
        boxsize_3d=np.asarray([box_L, box_L, box_L], dtype=np.float64),
        temp_log_floor_k=float(args.temp_log_floor_k),
    )
    save_hist_plot(
        plots_dir / "final_hist_compare.png",
        dm_ref=dm_nyx_final,
        dm_dh=dm_dh_final,
        gas_ref=gas_nyx_final,
        gas_dh=gas_dh_final,
        temp_ref=temp_nyx_final,
        temp_dh=temp_dh_final,
        temp_log_floor_k=float(args.temp_log_floor_k),
    )
    save_temp_density_phase_plot(
        plots_dir / "final_phase_temp_density_compare.png",
        rho_ref=gas_nyx_final,
        temp_ref=temp_nyx_final,
        rho_dh=gas_dh_final,
        temp_dh=temp_dh_final,
        temp_log_floor_k=float(args.temp_log_floor_k),
    )
    if phase_history_every > 0:
        save_phase_history_compare_plot(
            plots_dir / "phase_history_compare.png",
            phase_history_dh,
            nyx_phase_history,
        )
    save_scatter_plot(
        plots_dir / "final_scatter_dm_density.png",
        dm_nyx_final,
        dm_dh_final,
        "Final DM Density Norm: Nyx vs DiffHydro",
        log10=True,
    )
    save_scatter_plot(
        plots_dir / "final_scatter_gas_density.png",
        gas_nyx_final,
        gas_dh_final,
        "Final Gas Density Norm: Nyx vs DiffHydro",
        log10=True,
    )
    save_scatter_plot(
        plots_dir / "final_scatter_temp_k.png",
        temp_nyx_final,
        temp_dh_final,
        "Final Temperature [K]: Nyx vs DiffHydro",
        log10=True,
    )

    print("[done] Wrote metrics:")
    print(f"  {output_dir / 'metrics.npz'}")
    print(f"  {output_dir / 'metrics.txt'}")
    print("[done] Wrote snapshot arrays:")
    print(f"  {snapshot_fields_dir / 'fields_ic_final.npz'}")
    print(f"  {snapshot_fields_dir / 'final_snapshot_3d.npz'}")
    if phase_history_every > 0:
        print("[done] Wrote phase-history diagnostics:")
        print(f"  {output_dir / 'phase_history_diffhydro.csv'}")
        if nyx_phase_history:
            print(f"  {output_dir / 'phase_history_nyx.csv'}")
        print(f"  {plots_dir / 'phase_history_compare.png'}")
    print("[done] Wrote plots under:")
    print(f"  {plots_dir}")
    print(
        "[test] pass/fail: "
        f"gas_rho_corr={metrics['gas_density_corr_final']:.4f} "
        f"(>= {float(args.min_gas_rho_corr):.3f}: {pass_gas_rho}), "
        f"gas_temp_corr={metrics['gas_temp_corr_final']:.4f} "
        f"(>= {float(args.min_gas_temp_corr):.3f}: {pass_gas_temp}), "
        f"dm_corr={metrics['dm_density_corr_final']:.4f} "
        f"(>= {float(args.min_dm_corr):.3f}: {pass_dm}), "
        f"pass_all={metrics['pass_all']}"
    )


if __name__ == "__main__":
    main()
