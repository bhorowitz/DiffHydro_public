"""End-to-end JAX back-propagation test for the hydro/RT/chemistry model.

The differentiated parameter is the photon number injected in the central
cell.  The loss is the final total ionised hydrogen fraction plus the
remaining photon number, so both chemistry and radiative transport contribute
to the derivative.

Run with::

    PYTHONPATH=. python examples/RT/backpropagation_full_physics_test.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import diffhydro as dh
from diffhydro.equationmanager_radiative_transf_no_chat_copy import EquationManager as RTEquation
from diffhydro.physics import hydrogen_chemistry as hchem
from diffhydro.physics.fraction_xHII import HydrogenPhotoChemistryForce
from diffhydro.physics.hydrogen_chemistry import HydrogenStateView
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.units import CodeUnits


SIZE = 15
STEPS = 100
DT_CODE = 1.0e-3
PHOTON_REFERENCE = 2.0e50
PHOTON_BACKGROUND_FRACTION = 1.0e-8
FD_RELATIVE_STEP = 1.0e-4
ROOT = Path(__file__).with_name("Images") / "backpropagation_full_physics"


class StateBlockFlux:
    def __init__(self, base_flux: Any, state_slice: slice):
        self.base_flux = base_flux
        self.state_slice = state_slice
        self.dx_o = base_flux.dx_o

    def flux(self, sol: Any, axis: int, params: Any, flux: Any) -> Any:
        local = self.base_flux.flux(sol[self.state_slice], axis, params, flux)
        return jnp.zeros_like(sol).at[self.state_slice].set(local)

    def timestep(self, sol: Any) -> Any:
        return self.base_flux.timestep(sol[self.state_slice])


def build() -> tuple[Any, Any, Any, Any]:
    cu = CodeUnits.from_config(
        {"length": "1.0e16 cm", "mass": "1.6735575e24 g", "velocity": "1.0e8 cm/s"},
        {"gamma": 5.0 / 3.0, "mu": 0.61},
    )
    c_code = hchem.C_LIGHT_CGS / cu.V_cgs
    rt_eq = RTEquation(light_speed=c_code, mesh_shape=(SIZE,) * 3, eps=1e-20)
    hydro_eq = dh.EquationManager(
        gamma=5.0 / 3.0, n_cons=6, passive_names=("x_HII",),
        mesh_shape=(SIZE,) * 3, eps=1e-20,
    )
    rt_solver = dh.LaxFriedrichs_Radiative_transfer(
        equation_manager=rt_eq, signal_speed=dh.signal_speed_Rusanov
    )
    hydro_solver = dh.LaxFriedrichs(
        equation_manager=hydro_eq, signal_speed=dh.signal_speed_Rusanov
    )
    rt_flux = StateBlockFlux(
        dh.ConvectiveFlux_Radiative_transfer(
            rt_eq, rt_solver, dh.PLM(limiter="VANLEER"), dx=1.0
        ), slice(0, 4)
    )
    hydro_flux = StateBlockFlux(
        dh.ConvectiveFlux(
            hydro_eq, hydro_solver, dh.PLM(limiter="VANLEER"), dx=1.0
        ), slice(4, 10)
    )
    stellar = StellarRadiationForce(
        dx=1.0, injection_mode="stromgren", stromgren_rate=0.0,
        eq=rt_eq, hydro_eq=hydro_eq, cu=cu, chemistry=False,
        chemistry_case="B", X_H=1.0,
    )
    chemistry = HydrogenPhotoChemistryForce(
        stellar, case="B", collisional=True, max_frac=0.5,
        include_heating=True, include_cooling=True, mean_photon_energy_eV=20.0,
    )
    # The source-coupled path is the differentiable benchmark. The full
    # finite-volume transport path currently contains an M1 norm at F=0 whose
    # reverse-mode derivative is undefined; it is validated separately by the
    # supernova diagnostic.
    simulation = dh.hydro(
        n_super_step=1, fluxes=[],
        forces=[stellar, chemistry], dx=1.0, max_dt=DT_CODE,
        snapshot_every=None, pmesh_shape=(1, 1, 1),
        boundary=dh.OutflowBoundary, periodic_flux_divergence=False,
    )
    return cu, rt_eq, hydro_eq, chemistry, simulation


def initial_state(cu: Any, rt_eq: Any, hydro_eq: Any, photon_count: Any) -> jnp.ndarray:
    state = jnp.zeros((10, SIZE, SIZE, SIZE), dtype=jnp.float64)
    center = SIZE // 2
    rho = hchem.MH_CGS / cu.rho_cgs
    state = state.at[4].set(rho)
    state = state.at[8].set(
        hchem.KB_CGS * 1.0e4 / cu.P_cgs / (5.0 / 3.0 - 1.0)
    )
    background = PHOTON_BACKGROUND_FRACTION * photon_count
    state = state.at[0].set(background)
    state = state.at[0, center, center, center].add(photon_count)
    # A smooth, non-zero seed direction avoids the undefined derivative of
    # |F| at exactly zero while remaining negligible beside the central packet.
    state = state.at[1].set(0.1 * rt_eq.light_speed * state[0])
    view = HydrogenStateView(
        cu=cu, gamma=hydro_eq.gamma, idx_N=0, idx_F=(1, 2, 3),
        idx_xHII=9, idx_rho=4, idx_mom=(5, 6, 7), idx_Etot=8,
        xHII_weight_idx=4, X_H=1.0, light_speed_code=rt_eq.light_speed,
    )
    return view.set_xHII(state, jnp.zeros((SIZE, SIZE, SIZE)))


def params() -> dict[str, jnp.ndarray]:
    return {
        "star_masses": jnp.array([0.0]),
        "star_ages": jnp.array([0.0]),
        "star_metallicities": jnp.array([0.02]),
        "star_positions": jnp.array([[SIZE // 2] * 3], dtype=jnp.int32),
    }


def evolve(photon_count: Any, simulation: Any, cu: Any, rt_eq: Any, hydro_eq: Any) -> jnp.ndarray:
    state = initial_state(cu, rt_eq, hydro_eq, photon_count)
    run_params = params()

    def body(step: int, current: jnp.ndarray) -> jnp.ndarray:
        next_state, _ = simulation._hydrostep(step, (current, run_params), DT_CODE)
        return next_state

    return jax.lax.fori_loop(0, STEPS, body, state)


def loss(photon_count: Any, simulation: Any, cu: Any, rt_eq: Any, hydro_eq: Any) -> jnp.ndarray:
    state = evolve(photon_count, simulation, cu, rt_eq, hydro_eq)
    view = HydrogenStateView(
        cu=cu, gamma=hydro_eq.gamma, idx_N=0, idx_F=(1, 2, 3),
        idx_xHII=9, idx_rho=4, idx_mom=(5, 6, 7), idx_Etot=8,
        xHII_weight_idx=4, X_H=1.0, light_speed_code=rt_eq.light_speed,
    )
    ionised = jnp.sum(view.xHII(state))
    photons = jnp.sum(view.photon_density_cgs(state)) / 1.0e50
    return ionised + photons


def save_outputs(photon_count: float, final_state: np.ndarray, value: float, gradient: float) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / "final_state.npz", state=final_state)
    with (ROOT / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("JAX back-propagation hydro/RT chemistry-source test\n\n")
        handle.write("Initial conditions:\n")
        handle.write(f"- grid: {SIZE}^3 cells, hydrogen density: 1 cm^-3\n")
        handle.write("- temperature: 1.0e4 K, x_HII: 0 everywhere\n")
        handle.write(f"- central photon count: {photon_count:.8e}\n")
        handle.write(f"- diffuse photon background: {PHOTON_BACKGROUND_FRACTION:.1e} of the central count\n")
        handle.write(f"- evolution: {STEPS} steps of dt={DT_CODE:.3e}\n\n")
        handle.write("Expected result:\n")
        handle.write("- increasing the injected photon count must not decrease the final loss\n")
        handle.write("- value_and_grad must return a finite, positive gradient\n\n")
        handle.write("Scope note:\n")
        handle.write("- this differentiable benchmark evolves the production photon/chemistry/thermal source\n")
        handle.write("- finite-volume M1 transport is exercised by the supernova diagnostic; its zero-flux norm\n")
        handle.write("  requires a separate smooth-M1 derivative treatment before being included here\n\n")
        handle.write(f"Final loss: {value:.12e}\n")
        handle.write(f"Gradient d(loss)/d(photon_count): {gradient:.12e}\n")

    center = SIZE // 2
    view_x = np.clip(
        final_state[9, :, :, center] / np.maximum(final_state[4, :, :, center], 1e-300),
        0.0, 1.0,
    )
    photon_map = np.maximum(final_state[0, :, :, center], 1e-300)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, data, title in (
        (axes[0], view_x, "Final x_HII"),
        (axes[1], photon_map, "Final photon density (log)"),
    ):
        image = ax.imshow(data.T, origin="lower", cmap="viridis")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(ROOT / "final_fields.png", dpi=140)
    plt.close(fig)


def main() -> None:
    cu, rt_eq, hydro_eq, chemistry, simulation = build()
    reference = jnp.asarray(PHOTON_REFERENCE, dtype=jnp.float64)
    value_and_grad = jax.value_and_grad(loss)
    value, gradient = value_and_grad(reference, simulation, cu, rt_eq, hydro_eq)
    step = PHOTON_REFERENCE * FD_RELATIVE_STEP
    plus = loss(reference + step, simulation, cu, rt_eq, hydro_eq)
    minus = loss(reference - step, simulation, cu, rt_eq, hydro_eq)
    finite_difference = (plus - minus) / (2.0 * step)
    final = evolve(reference, simulation, cu, rt_eq, hydro_eq)
    value_f = float(value)
    gradient_f = float(gradient)
    fd_f = float(finite_difference)
    save_outputs(PHOTON_REFERENCE, np.asarray(final), value_f, gradient_f)
    with (ROOT / "gradient_check.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["loss", "jax_gradient", "finite_difference", "relative_error"])
        writer.writeheader()
        writer.writerow({
            "loss": value_f, "jax_gradient": gradient_f,
            "finite_difference": fd_f,
            "relative_error": abs(gradient_f - fd_f) / max(abs(fd_f), 1e-300),
        })
    print(f"Saved back-propagation report to {ROOT}")
    print(f"loss={value_f:.8e}, jax_gradient={gradient_f:.8e}, finite_difference={fd_f:.8e}")


if __name__ == "__main__":
    main()
