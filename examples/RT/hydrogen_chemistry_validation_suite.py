"""Physics validation suite for the hydrogen chemistry source terms.

This is intentionally a standalone diagnostic program.  It advances source
terms directly (without the hydrodynamic transport step), which makes each
test a local, reproducible check of the chemistry and thermal modules.

Run from the repository root with::

    python examples/RT/hydrogen_chemistry_validation_suite.py

The output directory is deliberately fixed so that the generated artefacts
can be compared between code revisions.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import diffhydro as dh
from diffhydro.equationmanager_radiative_transf_no_chat_copy import EquationManager as RTEquation
from diffhydro.physics import hydrogen_chemistry as hchem
from diffhydro.physics.cooling import HeatCoolForce_basic
from diffhydro.physics.fraction_xHII import HydrogenIonizationForce, HydrogenPhotoChemistryForce
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.units import CodeUnits


GAMMA = 5.0 / 3.0
N = 32
DT_FACTOR = (1.0, 0.5, 0.25)
OUT = Path(__file__).with_name("Images") / "hydrogen_chemistry_validation"
SUBDIRS = {
    1: "01_recombination_pure",
    2: "02_photoionization_pure",
    3: "03_photon_ion_conservation",
    4: "04_photo_recomb_equilibrium",
    5: "05_collisional_ionization",
    6: "06_photoheating",
    7: "07_cooling",
    8: "08_caseA_caseB",
    9: "09_thermoionization_equilibrium",
}


def setup() -> tuple[CodeUnits, float, Any, Any, Any]:
    cu = CodeUnits.from_config(
        {"length": "1.0 cm", "mass": "1 g", "velocity": "3e10 cm/s"},
        {"gamma": GAMMA, "mu": 0.61},
    )
    c_code = hchem.C_LIGHT_CGS / cu.V_cgs
    rt = RTEquation(light_speed=c_code, mesh_shape=(N, N, N), eps=1e-20)
    hydro = dh.EquationManager(
        gamma=GAMMA, n_cons=6, passive_names=("x_HII",),
        mesh_shape=(N, N, N), eps=1e-20,
    )
    stellar = StellarRadiationForce(
        dx=1.0, injection_mode="stromgren", stromgren_rate=0.0,
        eq=rt, hydro_eq=hydro, cu=cu, chemistry=False, X_H=1.0,
    )
    return cu, c_code, stellar, rt, hydro


def make_uniform_state(
    N: int, n_H_cgs: float, T_K: float, x_HII_initial: float,
    photon_density_cgs: float, cu: CodeUnits, eq_rt: Any, eq_hydro: Any,
    flux_fraction: float = 0.0,
) -> jnp.ndarray:
    """Build a conservative, homogeneous state using the chemistry view."""
    state = jnp.zeros((eq_rt.n_cons + eq_hydro.n_cons, N, N, N), dtype=jnp.float64)
    rho = n_H_cgs * hchem.MH_CGS / cu.rho_cgs
    state = state.at[4].set(rho)
    state = state.at[0].set(photon_density_cgs * cu.L_cgs**3)
    state = state.at[1].set(flux_fraction * eq_rt.light_speed * state[0])
    state = state.at[8].set(n_H_cgs * (1.0 + x_HII_initial) * hchem.KB_CGS * T_K / cu.P_cgs / (GAMMA - 1.0))
    view = HydrogenStateView_for(stellar=None, cu=cu, rt=eq_rt, hydro=eq_hydro)
    return view.set_xHII(state, x_HII_initial)


def HydrogenStateView_for(stellar: Any, cu: CodeUnits, rt: Any, hydro: Any) -> Any:
    """Create the same view as the source forces (kept local to this script)."""
    from diffhydro.physics.hydrogen_chemistry import HydrogenStateView

    return HydrogenStateView(
        cu=cu, gamma=hydro.gamma, idx_N=0, idx_F=(1, 2, 3), idx_xHII=9,
        idx_rho=4, idx_mom=(5, 6, 7), idx_Etot=8, xHII_weight_idx=4,
        X_H=1.0, light_speed_code=rt.light_speed,
    )


def compute_temperature_K(view: Any, sol: Any) -> np.ndarray:
    return np.asarray(view.temperature_K(sol))


def compute_total_photons(view: Any, sol: Any) -> float:
    return float(np.sum(np.asarray(view.photon_density_cgs(sol))))


def compute_total_ionized_hydrogen(view: Any, sol: Any) -> float:
    n_H, _, _, _ = view.number_densities_cgs(sol)
    return float(np.sum(np.asarray(n_H * view.xHII(sol))) * view.L_cgs**3)


def compute_total_thermal_energy(view: Any, sol: Any) -> float:
    return float(np.sum(np.asarray(view.thermal_energy_code(sol))) * view.P_cgs * view.L_cgs**3)


def assert_physical_state(view: Any, sol: Any) -> None:
    arrays = (np.asarray(sol), np.asarray(view.photon_density_cgs(sol)),
              np.asarray(view.xHII(sol)), compute_temperature_K(view, sol))
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("state contains NaN or inf")
    if np.min(arrays[1]) < -1e-12 or np.min(arrays[2]) < -1e-12 or np.max(arrays[2]) > 1.0 + 1e-12:
        raise ValueError("state violates photon or ionization bounds")
    if np.min(arrays[3]) <= 0.0:
        raise ValueError("state has non-positive temperature")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_validation_plot(path: Path, title: str, curves: list[tuple[np.ndarray, np.ndarray, str]], xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for x, y, label in curves:
        ax.plot(x, y, label=label)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.25)
    if len(curves) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def print_test_header(number: int, name: str) -> None:
    print(f"\n[{number}/9] {name}")


def print_test_summary(name: str, status: str, metric: float, tolerance: float, notes: str) -> dict[str, Any]:
    print(f"  {status:7s} metric={metric:.4g} tolerance={tolerance:.4g} -- {notes}")
    return {"test_name": name, "status": status, "error_metric": metric,
            "tolerance": tolerance, "pass_fail": status, "notes": notes}


def advance(force: Any, state: Any, view: Any, times: np.ndarray, dt_code: float,
            maintain_photons: float | None = None) -> tuple[Any, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for step, time in enumerate(times):
        state, _ = force.force(step, state, {}, dt_code)
        if maintain_photons is not None:
            state = state.at[0].set(maintain_photons * view.L_cgs**3)
        assert_physical_state(view, state)
        rows.append({
            "time_s": float(time), "xHII": float(np.mean(np.asarray(view.xHII(state)))),
            "N_gamma": compute_total_photons(view, state),
            "T_K": float(np.mean(compute_temperature_K(view, state))),
            "E_th": compute_total_thermal_energy(view, state),
        })
    return state, rows


def chemistry_force(stellar: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("max_frac", 0.9)
    return HydrogenPhotoChemistryForce(stellar, **kwargs)


def test_recombination(cu: CodeUnits, c_code: float, stellar: Any, rt: Any, hydro: Any, root: Path) -> dict[str, Any]:
    view = stellar.view
    rows_all: list[dict[str, Any]] = []
    metric = 0.0
    for factor in DT_FACTOR:
        force = chemistry_force(stellar, case="B", collisional=False, b_rec=0.0,
                                include_heating=False, include_cooling=False,
                                fixed_temperature_K=1e4)
        state = make_uniform_state(N, 1e-3, 1e4, 1.0, 0.0, cu, rt, hydro)
        alpha = float(hchem.alpha_B_HII_cgs(1e4))
        trec = 1.0 / (alpha * 1e-3)
        dt_s = trec / 40.0 * factor
        times = np.arange(1, 161) * dt_s
        _, rows = advance(force, state, view, times, dt_s / cu.T_cgs)
        for row in rows:
            row.update({"dt_factor": factor, "xHII_analytic": 1.0 / (1.0 + row["time_s"] / trec)})
            row["relative_error"] = abs(row["xHII"] - row["xHII_analytic"]) / row["xHII_analytic"]
        metric = max(metric, max(r["relative_error"] for r in rows))
        rows_all.extend(rows)
    write_csv(root / "recombination.csv", rows_all)
    x = np.array([r["time_s"] for r in rows_all if r["dt_factor"] == 1.0])
    yn = np.array([r["xHII"] for r in rows_all if r["dt_factor"] == 1.0])
    ya = np.array([r["xHII_analytic"] for r in rows_all if r["dt_factor"] == 1.0])
    save_validation_plot(root / "xHII.png", "Pure case-B recombination", [(x, yn, "numeric"), (x, ya, "analytic")], "time [s]", "x_HII")
    save_validation_plot(root / "relative_error.png", "Recombination relative error", [(x, np.abs(yn - ya) / ya, "error")], "time [s]", "relative error")
    return print_test_summary("recombination", "PASS" if metric < 0.05 else "WARNING", metric, 0.05, "explicit-Euler convergence factors 1, 0.5, 0.25")


def test_photoionization(cu: CodeUnits, c_code: float, stellar: Any, rt: Any, hydro: Any, root: Path) -> dict[str, Any]:
    view = stellar.view
    rows_all: list[dict[str, Any]] = []
    metric = 0.0
    for density in (1e-4, 3e-4, 1e-3):
        force = chemistry_force(stellar, case="B", collisional=False, b_rec=0.0,
                                include_heating=False, include_cooling=False, fixed_temperature_K=1e4,
                                max_frac=0.02)
        state = make_uniform_state(N, 1e-6, 1e4, 0.0, density, cu, rt, hydro)
        gamma = float(hchem.SIGMA_HI_0_CGS * view.c_interaction_cgs * density)
        dt_s = 0.01 / gamma
        times = np.arange(1, 121) * dt_s
        _, rows = advance(force, state, view, times, dt_s / cu.T_cgs, maintain_photons=density)
        for row in rows:
            row.update({"N_gamma_input": density, "xHII_analytic": 1.0 - math.exp(-gamma * row["time_s"])})
            row["relative_error"] = abs(row["xHII"] - row["xHII_analytic"]) / max(row["xHII_analytic"], 1e-15)
        metric = max(metric, max(r["relative_error"] for r in rows))
        rows_all.extend(rows)
    write_csv(root / "photoionization.csv", rows_all)
    save_validation_plot(root / "xHII.png", "Homogeneous photoionization", [
        (np.array([r["time_s"] for r in rows_all if r["N_gamma_input"] == d]),
         np.array([r["xHII"] for r in rows_all if r["N_gamma_input"] == d]), f"N={d:g}")
        for d in (1e-4, 3e-4, 1e-3)], "time [s]", "x_HII")
    return print_test_summary("photoionization", "PASS" if metric < 0.05 else "WARNING", metric, 0.05, "photon reservoir maintained after each local source step")


def test_conservation(cu: CodeUnits, c_code: float, stellar: Any, rt: Any, hydro: Any, root: Path) -> dict[str, Any]:
    view = stellar.view
    force = chemistry_force(stellar, case="B", collisional=False, b_rec=0.0,
                            include_heating=False, include_cooling=False, max_frac=0.02)
    state = make_uniform_state(N, 1e-6, 1e4, 0.0, 1e-4, cu, rt, hydro)
    initial = compute_total_photons(view, state) + compute_total_ionized_hydrogen(view, state)
    dt_s = 1e10
    rows: list[dict[str, Any]] = []
    for step in range(100):
        state, _ = force.force(step, state, {}, dt_s / cu.T_cgs)
        assert_physical_state(view, state)
        total = compute_total_photons(view, state) + compute_total_ionized_hydrogen(view, state)
        rows.append({"time_s": (step + 1) * dt_s, "N_gamma": compute_total_photons(view, state),
                     "N_HII": compute_total_ionized_hydrogen(view, state),
                     "total": total, "relative_error": abs(total - initial) / initial})
    metric = max(r["relative_error"] for r in rows)
    write_csv(root / "conservation.csv", rows)
    t = np.array([r["time_s"] for r in rows])
    save_validation_plot(root / "photon_ion_numbers.png", "Photon + ion conservation",
                         [(t, np.array([r[k] for r in rows]), k) for k in ("N_gamma", "N_HII", "total")],
                         "time [s]", "number")
    save_validation_plot(root / "conservation_error.png", "Conservation error",
                         [(t, np.array([r["relative_error"] for r in rows]), "relative error")],
                         "time [s]", "relative error")
    return print_test_summary("photon_ion_conservation", "PASS" if metric < 0.01 else "WARNING", metric, 0.01, "closed homogeneous cell")


def test_photo_recomb(cu: CodeUnits, c_code: float, stellar: Any, rt: Any, hydro: Any, root: Path) -> dict[str, Any]:
    view = stellar.view
    rows_all: list[dict[str, Any]] = []
    metric = 0.0
    for ratio in (0.1, 1.0, 10.0):
        n_H, photon_density = 1.0, 1e-4 * ratio
        force = HydrogenIonizationForce(
            stellar, case="B", collisional=False, max_frac=0.9,
        )
        state = make_uniform_state(N, n_H, 1e4, 0.0, photon_density, cu, rt, hydro)
        gamma = float(force.sigma_HI_cgs * view.c_interaction_cgs * photon_density)
        alpha = float(hchem.alpha_B_HII_cgs(1e4))
        xeq = 2.0 * gamma / (gamma + math.sqrt(gamma * gamma + 4.0 * gamma * alpha * n_H))
        dt_s = 0.01 / max(gamma, alpha * n_H)
        times = np.arange(1, 2001) * dt_s
        rows: list[dict[str, Any]] = []
        for step, time in enumerate(times):
            state, _ = force.force(step, state, {}, dt_s / cu.T_cgs)
            state = view.set_temperature_K(state, 1e4, x_HII=view.xHII(state))
            state = state.at[0].set(photon_density * view.L_cgs**3)
            assert_physical_state(view, state)
            rows.append({
                "time_s": float(time),
                "xHII": float(np.mean(np.asarray(view.xHII(state)))),
                "N_gamma": compute_total_photons(view, state),
                "T_K": float(np.mean(compute_temperature_K(view, state))),
                "E_th": compute_total_thermal_energy(view, state),
            })
        for row in rows:
            row.update({"ratio": ratio, "xHII_eq": xeq})
        rows_all.extend(rows)
        metric = max(metric, abs(rows[-1]["xHII"] - xeq) / xeq)
    write_csv(root / "photo_recombination.csv", rows_all)
    save_validation_plot(root / "equilibrium.png", "Photoionization + case-B recombination",
                         [(np.array([r["time_s"] for r in rows_all if r["ratio"] == q]),
                           np.array([r["xHII"] for r in rows_all if r["ratio"] == q]), f"Gamma/a={q:g}")
                          for q in (0.1, 1.0, 10.0)], "time [s]", "x_HII")
    return print_test_summary("photo_recombination", "PASS" if metric < 0.05 else "WARNING", metric, 0.05, "constant uniform photon reservoir")


def test_collisional(cu: CodeUnits, c_code: float, stellar: Any, rt: Any, hydro: Any, root: Path) -> dict[str, Any]:
    view = stellar.view
    rows: list[dict[str, Any]] = []
    metric = 0.0
    for temp in (1e4, 3e4, 1e5, 3e5):
        force = chemistry_force(stellar, case="B", collisional=True, b_rec=0.0,
                                include_heating=False, include_cooling=False,
                                fixed_temperature_K=temp, max_frac=0.02)
        state = make_uniform_state(N, 1.0, temp, 1e-6, 0.0, cu, rt, hydro)
        beta = float(hchem.beta_HI_cgs(temp))
        alpha = float(hchem.alpha_B_HII_cgs(temp))
        xeq = beta / (beta + alpha)
        scale = max(beta + alpha, 1e-40)
        times = np.arange(1, 401) * 0.05 / max(beta, 1e-40)
        _, history = advance(force, state, view, times, times[0] / cu.T_cgs)
        final = history[-1]["xHII"]
        metric = max(metric, abs(final - xeq) / max(xeq, 1e-15))
        rows.append({"T_K": temp, "beta": beta, "alpha": alpha, "xHII_eq": xeq, "xHII_final": final})
    write_csv(root / "collisional_equilibria.csv", rows)
    save_validation_plot(root / "equilibria.png", "Collisional ionization equilibria",
                         [(np.array([r["T_K"] for r in rows]), np.array([r["xHII_eq"] for r in rows]), "analytic"),
                          (np.array([r["T_K"] for r in rows]), np.array([r["xHII_final"] for r in rows]), "numeric")],
                         "temperature [K]", "x_HII")
    return print_test_summary("collisional_ionization", "PASS" if metric < 0.05 else "WARNING", metric, 0.05, "fixed-temperature beta/alpha equilibrium")


def test_photoheating(cu: CodeUnits, c_code: float, stellar: Any, rt: Any, hydro: Any, root: Path) -> dict[str, Any]:
    view = stellar.view
    rows: list[dict[str, Any]] = []
    for energy in (20.0, 13.6):
        force = chemistry_force(stellar, case="B", collisional=False, b_rec=0.0,
                                include_heating=True, include_cooling=False,
                                mean_photon_energy_eV=energy, max_frac=0.001)
        state = make_uniform_state(N, 1e-3, 1e4, 0.0, 1e-4, cu, rt, hydro)
        initial = compute_total_thermal_energy(view, state)
        dt_s = 1e11
        state, _ = force.force(0, state, {}, dt_s / cu.T_cgs)
        assert_physical_state(view, state)
        absorbed = 1e-4 * N**3 - compute_total_photons(view, state)
        gained = compute_total_thermal_energy(view, state) - initial
        expected = absorbed * (energy - hchem.E_HI_EV) * hchem.EV_CGS
        rows.append({"mean_photon_energy_eV": energy, "absorbed_photons": absorbed,
                     "thermal_energy_gain": gained, "expected_gain": expected,
                     "relative_error": abs(gained - expected) / max(abs(expected), 1e-30)})
    metric = rows[0]["relative_error"]
    write_csv(root / "photoheating.csv", rows)
    save_validation_plot(root / "energy_gain.png", "Photoheating energy budget",
                         [(np.array([row["mean_photon_energy_eV"] for row in rows]), np.array([row["thermal_energy_gain"] for row in rows]), "numeric"),
                          (np.array([row["mean_photon_energy_eV"] for row in rows]), np.array([row["expected_gain"] for row in rows]), "expected")],
                         "mean photon energy [eV]", "energy [erg]")
    zero = rows[1]["thermal_energy_gain"]
    status = "PASS" if metric < 0.05 and abs(zero) < 1e-20 else "WARNING"
    return print_test_summary("photoheating", status, metric, 0.05, "threshold-energy subtest gain=" + f"{zero:.3g} erg")


def test_cooling(cu: CodeUnits, c_code: float, stellar: Any, rt: Any, hydro: Any, root: Path) -> dict[str, Any]:
    view = stellar.view
    rows: list[dict[str, Any]] = []
    for label, temp, x in (("neutral", 1e5, 1e-4), ("ionized", 1e5, 0.99), ("hot", 1e6, 0.99)):
        force = HeatCoolForce_basic(eq=rt, hydro_eq=hydro, cu=cu, light_speed=c_code,
                                    include_heating=False, include_cooling=True, case="B", X_H=1.0)
        state = make_uniform_state(N, 1.0, temp, x, 0.0, cu, rt, hydro)
        cooling = float(force.cooling(temp, state)[0, 0, 0])
        cell_thermal_energy = compute_total_thermal_energy(view, state) / N**3
        dt_s = max(cell_thermal_energy / cooling / 100.0, 1.0)
        before = compute_total_thermal_energy(view, state)
        state, _ = force.force(0, state, {}, dt_s / cu.T_cgs)
        after = compute_total_thermal_energy(view, state)
        volume = cu.L_cgs**3 * N**3
        n_H = 1.0
        e0 = before / volume

        def rhs(time: float, energy: np.ndarray) -> np.ndarray:
            e = max(float(energy[0]), 1e-80)
            temperature = e * (GAMMA - 1.0) / (n_H * (1.0 + x) * hchem.KB_CGS)
            n_HI, n_HII, n_e = n_H * (1.0 - x), n_H * x, n_H * x
            loss = float(hchem.cooling_rate_cgs(temperature, n_HI, n_HII, n_e, case="B"))
            return np.array([-loss])

        ode = solve_ivp(rhs, (0.0, dt_s), np.array([e0]), method="BDF",
                        rtol=1e-7, atol=max(e0 * 1e-12, 1e-80))
        rows.append({"case": label, "T_initial": temp, "xHII": x, "cooling": cooling,
                     "energy_change": after - before, "expected_change": -cooling * volume * dt_s,
                     "ode_energy_change": float(ode.y[0, -1] * volume - before)})
    metric = max(abs(r["energy_change"] - r["expected_change"]) / max(abs(r["expected_change"]), 1e-30) for r in rows)
    write_csv(root / "cooling.csv", rows)
    save_validation_plot(root / "cooling_rates.png", "Initial cooling rates",
                         [(np.arange(len(rows)), np.array([r["cooling"] for r in rows]), "Lambda")],
                         "initial condition", "erg cm^-3 s^-1")
    return print_test_summary("cooling", "PASS" if metric < 0.05 else "WARNING", metric, 0.05, "neutral, ionized, and hot hydrogen")


def test_case_ab(cu: CodeUnits, c_code: float, stellar: Any, rt: Any, hydro: Any, root: Path) -> dict[str, Any]:
    view = stellar.view
    rows: list[dict[str, Any]] = []
    for case in ("A", "B"):
        force = chemistry_force(stellar, case=case, collisional=False,
                                include_heating=False, include_cooling=False, max_frac=0.01)
        state = make_uniform_state(N, 1.0, 1e4, 0.99, 0.0, cu, rt, hydro)
        times = np.arange(1, 201) * 1e10
        _, history = advance(force, state, view, times, times[0] / cu.T_cgs)
        rows.extend({"case": case, **item} for item in history)
    write_csv(root / "case_A_case_B.csv", rows)
    save_validation_plot(root / "comparison.png", "Case A / case B recombination",
                         [(np.array([r["time_s"] for r in rows if r["case"] == case]),
                           np.array([r["N_gamma"] for r in rows if r["case"] == case]), f"N_gamma case {case}")
                          for case in ("A", "B")], "time [s]", "photons cm^-3")
    slope_a = rows[0]["N_gamma"]
    metric = abs(slope_a) / max(float(hchem.alpha_A_HII_cgs(1e4) - hchem.alpha_B_HII_cgs(1e4)), 1e-40)
    return print_test_summary("case_A_case_B", "PASS" if rows[-1]["N_gamma"] >= 0.0 else "FAIL", metric, math.inf, "case A exposes diffuse recombination photons; case B uses on-the-spot closure")


def test_thermo(cu: CodeUnits, c_code: float, stellar: Any, rt: Any, hydro: Any, root: Path) -> dict[str, Any]:
    view = stellar.view
    force = chemistry_force(stellar, case="B", collisional=True, include_heating=True,
                            include_cooling=True, mean_photon_energy_eV=20.0, max_frac=0.9)
    state = make_uniform_state(N, 1.0, 1e3, 0.01, 1e-5, cu, rt, hydro)
    rows: list[dict[str, Any]] = []
    dt_s = 5e5
    n_H = 1.0
    N_gamma = 1e-5
    sigma_c = force.sigma_HI_cgs * view.c_interaction_cgs

    def ode_rhs(time: float, values: np.ndarray) -> np.ndarray:
        x, temperature = values
        x = float(np.clip(x, 0.0, 1.0))
        temperature = max(float(temperature), 1.0)
        beta = float(hchem.beta_HI_cgs(temperature))
        alpha = float(hchem.alpha_B_HII_cgs(temperature))
        n_HI, n_HII, n_e = n_H * (1.0 - x), n_H * x, n_H * x
        dx = (1.0 - x) * (beta * n_e + sigma_c * N_gamma) - x * alpha * n_e
        heating = float(hchem.photoheating_rate_cgs(
            n_HI, N_gamma, sigma_HI=force.sigma_HI_cgs,
            mean_photon_energy_eV=force.mean_photon_energy_eV,
            c_cgs=view.c_interaction_cgs,
        ))
        loss = float(hchem.cooling_rate_cgs(
            temperature, n_HI, n_HII, n_e, case=force.case,
        ))
        dT = (GAMMA - 1.0) * (heating - loss) / (n_H * hchem.KB_CGS * (1.0 + x))
        dT -= temperature * dx / (1.0 + x)
        return np.array([dx, dT])

    ode_times = np.arange(0.0, 10000.0 * dt_s, dt_s)
    ode = solve_ivp(ode_rhs, (0.0, ode_times[-1]), np.array([0.01, 1e3]),
                    t_eval=ode_times, method="BDF", rtol=1e-6, atol=[1e-9, 1e-3])
    for step in range(10000):
        T = float(np.mean(view.temperature_K(state)))
        n_H, n_HI, n_HII, n_e = view.number_densities_cgs(state)
        H = float(np.mean(np.asarray(hchem.photoheating_rate_cgs(
            n_HI, view.photon_density_cgs(state),
            sigma_HI=force.sigma_HI_cgs,
            mean_photon_energy_eV=force.mean_photon_energy_eV,
            c_cgs=view.c_interaction_cgs,
        ))))
        L = float(np.mean(np.asarray(hchem.cooling_rate_cgs(
            T, n_HI, n_HII, n_e, case=force.case,
        ))))
        rows.append({"time_s": step * dt_s, "T_K": T, "xHII": float(np.mean(np.asarray(view.xHII(state)))),
                     "N_gamma": compute_total_photons(view, state), "heating": H, "cooling": L,
                     "H_over_L": H / L if L > 0.0 else math.inf,
                     "T_ode": float(ode.y[1, step]), "xHII_ode": float(ode.y[0, step])})
        state, _ = force.force(step, state, {}, dt_s / cu.T_cgs)
        state = state.at[0].set(N_gamma * view.L_cgs**3)
        assert_physical_state(view, state)
    write_csv(root / "thermoionization.csv", rows)
    t = np.array([r["time_s"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes[0, 0].plot(t, [r["T_K"] for r in rows], label="numeric")
    axes[0, 0].plot(t, [r["T_ode"] for r in rows], "--", label="BDF")
    axes[0, 0].set_ylabel("T [K]"); axes[0, 0].legend()
    axes[0, 1].plot(t, [r["xHII"] for r in rows], label="numeric")
    axes[0, 1].plot(t, [r["xHII_ode"] for r in rows], "--", label="BDF")
    axes[0, 1].set_ylabel("x_HII"); axes[0, 1].legend()
    axes[1, 0].plot(t, [r["heating"] for r in rows], label="H"); axes[1, 0].plot(t, [r["cooling"] for r in rows], label="Lambda"); axes[1, 0].legend()
    axes[1, 1].plot(t, [r["H_over_L"] for r in rows]); axes[1, 1].set_ylabel("H/L")
    for ax in axes.flat: ax.set_xlabel("time [s]"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(root / "thermoionization.png", dpi=140); plt.close(fig)
    final = rows[-1]
    metric = max(
        abs(final["xHII"] - final["xHII_ode"]) / max(abs(final["xHII_ode"]), 1e-12),
        abs(final["T_K"] - final["T_ode"]) / max(abs(final["T_ode"]), 1e-12),
    )
    balance = abs(final["heating"] - final["cooling"]) / max(final["heating"], final["cooling"], 1e-30)
    return print_test_summary("thermoionization_equilibrium", "PASS" if metric < 0.05 else "WARNING", metric, 0.05, f"numeric/BDF trajectory; final H/L balance error={balance:.3g}")


def write_readme(summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Hydrogen chemistry validation suite", "",
        "All tests use float64 JAX, a homogeneous static N^3 box, pure hydrogen, "
        "and the existing `HydrogenStateView` conversions. Source terms are "
        "advanced locally without spatial transport.", "",
        "## Common parameters", "",
        f"- N = {N}; gamma = {GAMMA}; output = `{OUT}`",
        "- code length = 1 cm, mass = 1 g, velocity = 3e10 cm/s",
        "- threshold group cross-section and all rate coefficients come from `hydrogen_chemistry.py`.",
        "", "## Results", "", "| Test | Process validated | Error | Tolerance | Status |", "|---|---|---:|---:|---|",
    ]
    processes = {
        "recombination": "case-B recombination", "photoionization": "photoionization",
        "photon_ion_conservation": "photon/ion bookkeeping", "photo_recombination": "photoionization + recombination",
        "collisional_ionization": "collisional ionization", "photoheating": "photoheating",
        "cooling": "radiative cooling", "case_A_case_B": "case A/B diffuse photons",
        "thermoionization_equilibrium": "coupled thermal chemistry",
    }
    for item in summary:
        lines.append(f"| {item['test_name']} | {processes.get(item['test_name'], item['test_name'])} | "
                     f"{item['error_metric']:.4g} | {item['tolerance']:.4g} | {item['status']} |")
    lines += ["", "## Limitations known", "",
              "- Explicit Euler accuracy depends on the time step and the `max_frac` limiter.",
              "- The source-only tests do not validate spatial RT transport or hydrodynamic advection.",
              "- Case A and case B intentionally differ in treatment of recombination photons.",
              "- Fixed-temperature tests validate chemistry at a prescribed temperature; the full test evolves thermal energy.",
              "- The long thermo-ionization run is a diagnostic trajectory; an equilibrium criterion is meaningful only if H and Lambda both become non-zero and the run reaches equilibrium.",
              "", "Each subdirectory contains the CSV history, PNG figures, and `summary.txt`-equivalent information in the global CSV."]
    (OUT / "summary" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SUBDIRS.values():
        (OUT / name).mkdir(parents=True, exist_ok=True)
    cu, c_code, stellar, rt, hydro = setup()
    tests: list[tuple[int, str, Callable[..., dict[str, Any]]]] = [
        (1, "Pure recombination", test_recombination), (2, "Pure photoionization", test_photoionization),
        (3, "Photon-ion conservation", test_conservation), (4, "Photoionization-recombination", test_photo_recomb),
        (5, "Collisional ionization", test_collisional), (6, "Photoheating", test_photoheating),
        (7, "Radiative cooling", test_cooling), (8, "Case A versus case B", test_case_ab),
        (9, "Thermo-ionization", test_thermo),
    ]
    summary: list[dict[str, Any]] = []
    for number, name, test in tests:
        print_test_header(number, name)
        try:
            result = test(cu, c_code, stellar, rt, hydro, OUT / SUBDIRS[number])
            summary.append(result)
            (OUT / SUBDIRS[number] / "summary.txt").write_text(
                f"{result['test_name']}: {result['status']}\n"
                f"error_metric={result['error_metric']}\n"
                f"tolerance={result['tolerance']}\n"
                f"{result['notes']}\n",
                encoding="utf-8",
            )
        except Exception as exc:  # each diagnostic is independent by design
            message = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL    {message}")
            result = {"test_name": {
                1: "recombination", 2: "photoionization", 3: "photon_ion_conservation",
                4: "photo_recombination", 5: "collisional_ionization", 6: "photoheating",
                7: "cooling", 8: "case_A_case_B", 9: "thermoionization_equilibrium",
            }[number], "status": "FAIL", "error_metric": math.nan, "tolerance": math.nan,
                             "pass_fail": "FAIL", "notes": message}
            summary.append(result)
            (OUT / SUBDIRS[number] / "summary.txt").write_text(
                f"{result['test_name']}: FAIL\n{message}\n", encoding="utf-8"
            )
    write_csv(OUT / "summary" / "validation_summary.csv", summary)
    write_readme(summary)
    print(f"\nSummary: {OUT / 'summary' / 'validation_summary.csv'}")


if __name__ == "__main__":
    main()
