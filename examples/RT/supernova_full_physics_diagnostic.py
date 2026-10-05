"""Small 3-D supernova-like hydro/RT/chemistry diagnostic.

The explosion is represented by thermal energy deposited in the central cell.
An initially localised photon packet exercises M1 transport, photoionisation,
photoheating, recombination and radiative cooling while the hydro solver
advects the blast.  The program writes conservative snapshots, field PNGs,
CSV diagnostics and one GIF per field.

Run with the project GPU interpreter::

    PYTHONPATH=. python examples/RT/supernova_full_physics_diagnostic.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from PIL import Image

import diffhydro as dh
from diffhydro.equationmanager_radiative_transf_no_chat_copy import EquationManager as RTEquation
from diffhydro.physics import hydrogen_chemistry as hchem
from diffhydro.physics.cooling import HeatCoolForce_basic
from diffhydro.physics.fraction_xHII import HydrogenPhotoChemistryForce
from diffhydro.physics.hydrogen_chemistry import HydrogenStateView
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.units import CodeUnits


SIZE = 24
DT_CODE = 1.0e-3
STEPS = 600
FRAME_EVERY = 5
EXPLOSION_ENERGY_ERG = 1.0e41
PHOTON_PACKET_ENERGY_FRACTION = 0.1
MEAN_PHOTON_ENERGY_EV = 20.0
ROOT = Path(__file__).with_name("Images") / "supernova_full_physics"
SNAPSHOTS = ROOT / "snapshots"
FIELDS = ("rho", "temperature", "pressure", "vx", "vy", "vz",
          "E_gamma", "F_gamma", "x_HII", "heating", "cooling")


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


def build() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
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
        include_heating=True, include_cooling=True,
        mean_photon_energy_eV=MEAN_PHOTON_ENERGY_EV,
    )
    return cu, rt_eq, hydro_eq, rt_flux, hydro_flux, stellar, chemistry


def initial_state(cu: CodeUnits, rt_eq: Any, hydro_eq: Any) -> jnp.ndarray:
    state = jnp.zeros((10, SIZE, SIZE, SIZE), dtype=jnp.float64)
    center = SIZE // 2
    n_H = 1.0
    rho = n_H * hchem.MH_CGS / cu.rho_cgs
    state = state.at[4].set(rho)
    state = state.at[8].set(
        n_H * hchem.KB_CGS * 1.0e4 / cu.P_cgs / (5.0 / 3.0 - 1.0)
    )
    # A compact supernova-like thermal explosion.  This scaled energy is
    # expressed in the deliberately chosen code-unit normalization below.
    # The conservative energy slot is an energy density, so divide the
    # explosion energy by the physical cell volume before converting units.
    cell_volume_cgs = cu.L_cgs**3
    state = state.at[8, center, center, center].add(
        EXPLOSION_ENERGY_ERG / cell_volume_cgs / cu.P_cgs
    )
    # The photon slot is a number density in code units. Convert a fraction
    # of the explosion energy into photons instead of using an arbitrary
    # dimensionless value (the latter was ~10^-52 photons/cm^3 here).
    photon_energy_erg = MEAN_PHOTON_ENERGY_EV * hchem.EV_CGS
    photon_packet_count = (
        PHOTON_PACKET_ENERGY_FRACTION * EXPLOSION_ENERGY_ERG
        / photon_energy_erg
    )
    state = state.at[0, center, center, center].set(photon_packet_count)
    view = HydrogenStateView(
        cu=cu, gamma=hydro_eq.gamma, idx_N=0, idx_F=(1, 2, 3),
        idx_xHII=9, idx_rho=4, idx_mom=(5, 6, 7), idx_Etot=8,
        xHII_weight_idx=4, X_H=1.0, light_speed_code=rt_eq.light_speed,
    )
    return view.set_xHII(state, jnp.zeros((SIZE, SIZE, SIZE)))


def diagnostics(state: np.ndarray, view: HydrogenStateView, chemistry: Any) -> dict[str, np.ndarray]:
    x = np.asarray(view.xHII(state))
    temperature = np.asarray(view.temperature_K(state))
    _, n_HI, n_HII, n_e = view.number_densities_cgs(state)
    heating = np.asarray(hchem.photoheating_rate_cgs(
        n_HI, view.photon_density_cgs(state), sigma_HI=chemistry.sigma_HI_cgs,
        mean_photon_energy_eV=chemistry.mean_photon_energy_eV,
        c_cgs=view.c_interaction_cgs,
    ))
    cooling = np.asarray(hchem.cooling_rate_cgs(
        temperature, n_HI, n_HII, n_e, case=chemistry.case,
    ))
    return {
        "rho": np.asarray(state[4]),
        "temperature": temperature,
        "pressure": np.asarray(view.pressure_code(state) * view.P_cgs),
        "vx": np.asarray(state[5] / jnp.maximum(state[4], 1e-300)),
        "vy": np.asarray(state[6] / jnp.maximum(state[4], 1e-300)),
        "vz": np.asarray(state[7] / jnp.maximum(state[4], 1e-300)),
        "E_gamma": np.asarray(view.photon_density_cgs(state)),
        "F_gamma": np.asarray(jnp.sqrt(state[1] ** 2 + state[2] ** 2 + state[3] ** 2)),
        "x_HII": x, "heating": heating, "cooling": cooling,
    }


def save_frame(fields: dict[str, np.ndarray], step: int, time_code: float) -> None:
    frame_dir = ROOT / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    mid = fields["rho"].shape[2] // 2
    for name in FIELDS:
        values = fields[name][:, :, mid]
        fig, ax = plt.subplots(figsize=(5, 4))
        finite = values[np.isfinite(values)]
        positive = finite[finite > 0.0]
        if name == "x_HII":
            # Ionisation fractions are bounded probabilities, not signed
            # fields. Keep the displayed range physical even when a frame is
            # entirely neutral or contains values below the log floor.
            log_floor = 1.0e-8
            image = ax.imshow(
                np.clip(values.T, log_floor, 1.0), origin="lower", cmap="viridis",
                norm=LogNorm(vmin=log_floor, vmax=1.0),
            )
        elif name in {"rho", "temperature", "pressure", "E_gamma", "F_gamma", "heating", "cooling"} and positive.size:
            lower = max(float(np.percentile(positive, 1.0)), np.finfo(float).tiny)
            upper = max(float(np.percentile(positive, 99.5)), lower * (1.0 + 1e-12))
            image = ax.imshow(
                np.maximum(values.T, lower), origin="lower", cmap="magma",
                norm=LogNorm(vmin=lower, vmax=upper),
            )
        else:
            scale = max(float(np.percentile(np.abs(finite), 99.5)), 1e-30) if finite.size else 1.0
            image = ax.imshow(
                values.T, origin="lower", cmap="coolwarm",
                norm=SymLogNorm(linthresh=scale * 1e-6, vmin=-scale, vmax=scale),
            )
        ax.set_title(f"{name}, step={step}, t_code={time_code:.4g}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(frame_dir / f"{name}_{step:05d}.png", dpi=100)
        plt.close(fig)


def make_gifs() -> None:
    frame_dir = ROOT / "frames"
    for name in FIELDS:
        paths = sorted(frame_dir.glob(f"{name}_*.png"))
        if not paths:
            continue
        images = [Image.open(path).convert("P") for path in paths]
        images[0].save(
            ROOT / f"{name}.gif", save_all=True, append_images=images[1:],
            duration=120, loop=0,
        )
        for image in images:
            image.close()


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    cu, rt_eq, hydro_eq, rt_flux, hydro_flux, stellar, chemistry = build()
    view = chemistry.view
    state = initial_state(cu, rt_eq, hydro_eq)
    simulation = dh.hydro(
        n_super_step=STEPS, fluxes=[rt_flux, hydro_flux],
        forces=[stellar, chemistry], dx=1.0, max_dt=DT_CODE,
        snapshot_every=None, pmesh_shape=(1, 1, 1),
        boundary=dh.OutflowBoundary, periodic_flux_divergence=False,
    )
    params = {
        "star_masses": jnp.array([0.0]), "star_ages": jnp.array([0.0]),
        "star_metallicities": jnp.array([0.02]),
        "star_positions": jnp.array([[SIZE // 2] * 3], dtype=jnp.int32),
    }
    rows: list[dict[str, float]] = []
    # Keep host-side frames while using the exact production hydro step.
    for step in range(STEPS):
        state, _ = simulation._hydrostep(step, (state, params), DT_CODE)
        host_state = np.asarray(state)
        fields = diagnostics(host_state, view, chemistry)
        if step % FRAME_EVERY == 0 or step == STEPS - 1:
            save_frame(fields, step, (step + 1) * DT_CODE)
        rows.append({
            "step": step, "time_code": (step + 1) * DT_CODE,
            "rho_min": float(fields["rho"].min()),
            "rho_max": float(fields["rho"].max()),
            "rho_center": float(fields["rho"][SIZE // 2, SIZE // 2, SIZE // 2]),
            "speed_max": float(np.sqrt(
                fields["vx"] ** 2 + fields["vy"] ** 2 + fields["vz"] ** 2
            ).max()),
            "temperature_max": float(fields["temperature"].max()),
            "E_gamma_total": float(fields["E_gamma"].sum()),
            "xHII_max": float(fields["x_HII"].max()),
            "heating_total": float(fields["heating"].sum()),
            "cooling_total": float(fields["cooling"].sum()),
        })
    with (ROOT / "diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_gifs()
    np.savez(ROOT / "final_state.npz", state=np.asarray(state))
    print(f"Saved full-physics diagnostic to {ROOT}")
    print(f"GIF fields: {', '.join(FIELDS)}")


if __name__ == "__main__":
    main()
