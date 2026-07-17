"""Gracklepy-backed tabulated cooling/heating support.

This mirrors the Nyx cooling table architecture:
    1. build/load a table offline
    2. save it to NPZ
    3. evaluate it in JAX at runtime with trilinear interpolation

The Grackle table is parameterized by:
    - redshift z
    - hydrogen number density n_H [cm^-3]
    - temperature T [K]

The stored fields are:
    - heat_prim_cgs: primordial/background heating contribution [erg s^-1 cm^-3]
    - cool_prim_cgs: primordial cooling contribution [erg s^-1 cm^-3]
    - cool_metal_solar_cgs: additional solar-metallicity cooling [erg s^-1 cm^-3]
    - mu: mean molecular weight from Grackle
"""

from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import numpy as np


_M_PROTON_CGS = 1.6726231e-24


@dataclass(frozen=True)
class GrackleCoolingTableData:
    z_nodes: np.ndarray
    lognH_nodes: np.ndarray
    logT_nodes: np.ndarray
    heat_prim_cgs: np.ndarray
    cool_prim_cgs: np.ndarray
    cool_metal_solar_cgs: np.ndarray
    mu: np.ndarray
    z_sun: float
    h_species: float
    uvb_model: str
    grackle_data_file: str
    gracklepy_version: str

    @classmethod
    def load(cls, path: Path) -> "GrackleCoolingTableData":
        with np.load(path) as d:
            return cls(
                z_nodes=np.asarray(d["z_nodes"], dtype=np.float64),
                lognH_nodes=np.asarray(d["lognH_nodes"], dtype=np.float64),
                logT_nodes=np.asarray(d["logT_nodes"], dtype=np.float64),
                heat_prim_cgs=np.asarray(d["heat_prim_cgs"], dtype=np.float32),
                cool_prim_cgs=np.asarray(d["cool_prim_cgs"], dtype=np.float32),
                cool_metal_solar_cgs=np.asarray(d["cool_metal_solar_cgs"], dtype=np.float32),
                mu=np.asarray(d["mu"], dtype=np.float32),
                z_sun=float(np.asarray(d["z_sun"])),
                h_species=float(np.asarray(d["h_species"])),
                uvb_model=str(np.asarray(d["uvb_model"]).item()),
                grackle_data_file=str(np.asarray(d["grackle_data_file"]).item()),
                gracklepy_version=str(np.asarray(d["gracklepy_version"]).item()),
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            z_nodes=np.asarray(self.z_nodes, dtype=np.float64),
            lognH_nodes=np.asarray(self.lognH_nodes, dtype=np.float64),
            logT_nodes=np.asarray(self.logT_nodes, dtype=np.float64),
            heat_prim_cgs=np.asarray(self.heat_prim_cgs, dtype=np.float32),
            cool_prim_cgs=np.asarray(self.cool_prim_cgs, dtype=np.float32),
            cool_metal_solar_cgs=np.asarray(self.cool_metal_solar_cgs, dtype=np.float32),
            mu=np.asarray(self.mu, dtype=np.float32),
            z_sun=np.asarray(self.z_sun, dtype=np.float64),
            h_species=np.asarray(self.h_species, dtype=np.float64),
            uvb_model=np.asarray(self.uvb_model),
            grackle_data_file=np.asarray(self.grackle_data_file),
            gracklepy_version=np.asarray(self.gracklepy_version),
        )

    def summary_dict(self) -> dict[str, object]:
        return {
            "z_min": float(np.min(self.z_nodes)),
            "z_max": float(np.max(self.z_nodes)),
            "n_z": int(np.size(self.z_nodes)),
            "lognH_min": float(np.min(self.lognH_nodes)),
            "lognH_max": float(np.max(self.lognH_nodes)),
            "n_lognH": int(np.size(self.lognH_nodes)),
            "logT_min": float(np.min(self.logT_nodes)),
            "logT_max": float(np.max(self.logT_nodes)),
            "n_logT": int(np.size(self.logT_nodes)),
            "uvb_model": str(self.uvb_model),
            "grackle_data_file": str(self.grackle_data_file),
            "gracklepy_version": str(self.gracklepy_version),
            "z_sun": float(self.z_sun),
            "h_species": float(self.h_species),
        }


def _import_gracklepy():
    try:
        import gracklepy  # type: ignore
        from gracklepy import chemistry_data, setup_fluid_container  # type: ignore
        from gracklepy.utilities.physical_constants import (  # type: ignore
            cm_per_mpc,
            mass_hydrogen_cgs,
            sec_per_Myr,
        )
    except Exception as exc:  # pragma: no cover - depends on external package
        raise ImportError(
            "gracklepy is required to build Grackle cooling tables. "
            "Install/import gracklepy or provide a prebuilt grackle_table_npz."
        ) from exc
    return gracklepy, chemistry_data, setup_fluid_container, mass_hydrogen_cgs, sec_per_Myr, cm_per_mpc


def _configure_grackle_chemistry(
    *,
    chemistry_data,
    grackle_data_file: Path,
    redshift: float,
    uvb_model: str,
) -> object:
    chem = chemistry_data()
    chem.use_grackle = 1
    chem.with_radiative_cooling = 0
    chem.primordial_chemistry = 0
    chem.metal_cooling = 1
    chem.UVbackground = 0 if str(uvb_model).lower() in {"", "none", "off"} else 1
    chem.self_shielding_method = 0
    chem.H2_self_shielding = 0
    chem.grackle_data_file = str(grackle_data_file.resolve())
    chem.use_specific_heating_rate = 1
    chem.use_volumetric_heating_rate = 1
    chem.comoving_coordinates = 0
    chem.a_units = 1.0
    chem.a_value = 1.0 / (1.0 + float(redshift))
    return chem


def _evaluate_grackle_scalar(
    *,
    setup_fluid_container,
    chem,
    density_cgs: float,
    temperature_k: float,
    metal_mass_fraction: float,
    mass_hydrogen_cgs: float,
    sec_per_Myr: float,
    cm_per_mpc: float,
    converge: bool = True,
) -> tuple[float, float]:
    chem.density_units = float(mass_hydrogen_cgs)
    chem.length_units = float(cm_per_mpc)
    chem.time_units = float(sec_per_Myr)
    chem.set_velocity_units()

    with redirect_stderr(io.StringIO()):
        fc = setup_fluid_container(
            chem,
            density=float(density_cgs),
            temperature=float(temperature_k),
            metal_mass_fraction=float(max(metal_mass_fraction, 0.0)),
            # converge=True iterates the chemistry to ionization equilibrium before
            # reading the rate. The old converge=False shortcut left gas in the
            # 'ionized' initial guess, which INFLATED the thermal-equilibrium locus
            # by ~0.2 dex (validated in tests/validate_grackle_flags.py) and drove a
            # too-hot IGM. Equilibrium tables must use converge=True.
            converge=bool(converge),
        )
    if getattr(chem, "use_specific_heating_rate", 0):
        fc["specific_heating_rate"][:] = 0.0
    if getattr(chem, "use_volumetric_heating_rate", 0):
        fc["volumetric_heating_rate"][:] = 0.0
    data = fc.finalize_data()
    cooling_rate = float(np.asarray(data["cooling_rate"]).reshape(-1)[0])
    mu = float(np.asarray(data["mean_molecular_weight"]).reshape(-1)[0])
    return cooling_rate, mu


def build_grackle_cooling_table(
    out_npz: Path,
    *,
    grackle_data_file: Path,
    z_nodes: Iterable[float],
    lognH_nodes: Iterable[float],
    logT_nodes: Iterable[float],
    uvb_model: str = "HM2012",
    z_sun: float = 0.01295,
    h_species: float = 0.76,
    converge: bool = True,
) -> GrackleCoolingTableData:
    gracklepy, chemistry_data, setup_fluid_container, mass_hydrogen_cgs, sec_per_Myr, cm_per_mpc = _import_gracklepy()

    grackle_data_file = Path(grackle_data_file).resolve()
    if not grackle_data_file.exists():
        raise FileNotFoundError(f"Grackle data file not found: {grackle_data_file}")

    z_arr = np.asarray(sorted(float(v) for v in z_nodes), dtype=np.float64)
    ln_arr = np.asarray(list(lognH_nodes), dtype=np.float64)
    lt_arr = np.asarray(list(logT_nodes), dtype=np.float64)
    if z_arr.size < 2:
        raise ValueError("Need at least two redshift nodes for interpolation.")
    if ln_arr.size < 2 or lt_arr.size < 2:
        raise ValueError("Need at least two lognH/logT nodes for interpolation.")

    heat_prim = np.zeros((z_arr.size, ln_arr.size, lt_arr.size), dtype=np.float32)
    cool_prim = np.zeros_like(heat_prim)
    cool_metal_solar = np.zeros_like(heat_prim)
    mu = np.zeros_like(heat_prim)

    for iz, z in enumerate(z_arr):
        chem = _configure_grackle_chemistry(
            chemistry_data=chemistry_data,
            grackle_data_file=grackle_data_file,
            redshift=float(z),
            uvb_model=str(uvb_model),
        )
        solar_mass_fraction = float(z_sun)
        if hasattr(chem, "SolarMetalFractionByMass") and float(z_sun) <= 0.0:
            solar_mass_fraction = float(chem.SolarMetalFractionByMass)

        for inh, lognH in enumerate(ln_arr):
            nH_cgs = 10.0 ** float(lognH)
            density_cgs = nH_cgs * _M_PROTON_CGS / max(float(h_species), 1.0e-30)
            for it, logT in enumerate(lt_arr):
                temp_k = 10.0 ** float(logT)
                coeff_prim, mu_prim = _evaluate_grackle_scalar(
                    setup_fluid_container=setup_fluid_container,
                    chem=chem,
                    density_cgs=density_cgs,
                    temperature_k=temp_k,
                    metal_mass_fraction=0.0,
                    mass_hydrogen_cgs=mass_hydrogen_cgs,
                    sec_per_Myr=sec_per_Myr,
                    cm_per_mpc=cm_per_mpc,
                    converge=bool(converge),
                )
                coeff_solar, _ = _evaluate_grackle_scalar(
                    setup_fluid_container=setup_fluid_container,
                    chem=chem,
                    density_cgs=density_cgs,
                    temperature_k=temp_k,
                    metal_mass_fraction=solar_mass_fraction,
                    mass_hydrogen_cgs=mass_hydrogen_cgs,
                    sec_per_Myr=sec_per_Myr,
                    cm_per_mpc=cm_per_mpc,
                    converge=bool(converge),
                )

                # Gracklepy's signed cooling_rate is a net coefficient:
                #   positive -> net heating
                #   negative -> net cooling
                # so the corresponding volumetric source term keeps the same sign.
                net_prim = (nH_cgs * nH_cgs) * coeff_prim
                net_solar = (nH_cgs * nH_cgs) * coeff_solar

                heat_prim[iz, inh, it] = np.float32(max(net_prim, 0.0))
                cool_prim[iz, inh, it] = np.float32(max(-net_prim, 0.0))
                cool_metal_solar[iz, inh, it] = np.float32(max(net_prim - net_solar, 0.0))
                mu[iz, inh, it] = np.float32(max(mu_prim, 1.0e-6))

    table = GrackleCoolingTableData(
        z_nodes=z_arr,
        lognH_nodes=ln_arr,
        logT_nodes=lt_arr,
        heat_prim_cgs=heat_prim,
        cool_prim_cgs=cool_prim,
        cool_metal_solar_cgs=cool_metal_solar,
        mu=mu,
        z_sun=float(z_sun),
        h_species=float(h_species),
        uvb_model=str(uvb_model),
        grackle_data_file=str(grackle_data_file),
        gracklepy_version=str(getattr(gracklepy, "__version__", "unknown")),
    )
    table.save(out_npz)
    return table


class GrackleCoolingRateInterpolator:
    """JAX runtime interpolator for Gracklepy cooling/heating tables."""

    def __init__(self, table: GrackleCoolingTableData):
        self.z_nodes = jnp.asarray(table.z_nodes, dtype=jnp.float32)
        self.lognH_nodes = jnp.asarray(table.lognH_nodes, dtype=jnp.float32)
        self.logT_nodes = jnp.asarray(table.logT_nodes, dtype=jnp.float32)
        self.heat_prim = jnp.asarray(table.heat_prim_cgs, dtype=jnp.float32)
        self.cool_prim = jnp.asarray(table.cool_prim_cgs, dtype=jnp.float32)
        self.cool_metal_solar = jnp.asarray(table.cool_metal_solar_cgs, dtype=jnp.float32)
        self.mu = jnp.asarray(table.mu, dtype=jnp.float32)
        self.z_sun = float(table.z_sun)
        self.h_species = float(table.h_species)

        if self.lognH_nodes.size < 2 or self.logT_nodes.size < 2:
            raise ValueError("GrackleCoolingRateInterpolator needs >=2 nodes along both lognH/logT axes.")

        self.ln_min = self.lognH_nodes[0]
        self.ln_max = self.lognH_nodes[-1]
        self.lt_min = self.logT_nodes[0]
        self.lt_max = self.logT_nodes[-1]
        self.n_ln = int(self.lognH_nodes.size)
        self.n_lt = int(self.logT_nodes.size)
        self._ln_scale = (self.n_ln - 1) / jnp.maximum(self.ln_max - self.ln_min, 1.0e-12)
        self._lt_scale = (self.n_lt - 1) / jnp.maximum(self.lt_max - self.lt_min, 1.0e-12)

    def _bilinear(self, grid2d: jnp.ndarray, lognH: jnp.ndarray, logt: jnp.ndarray) -> jnp.ndarray:
        x = (jnp.clip(lognH, self.ln_min, self.ln_max) - self.ln_min) * self._ln_scale
        y = (jnp.clip(logt, self.lt_min, self.lt_max) - self.lt_min) * self._lt_scale
        i0 = jnp.floor(x).astype(jnp.int32)
        j0 = jnp.floor(y).astype(jnp.int32)
        i0 = jnp.clip(i0, 0, self.n_ln - 2)
        j0 = jnp.clip(j0, 0, self.n_lt - 2)
        fx = x - i0.astype(jnp.float32)
        fy = y - j0.astype(jnp.float32)

        v00 = grid2d[i0, j0]
        v10 = grid2d[i0 + 1, j0]
        v01 = grid2d[i0, j0 + 1]
        v11 = grid2d[i0 + 1, j0 + 1]
        return (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11

    def evaluate(
        self,
        nH_cgs: jnp.ndarray,
        temp_k: jnp.ndarray,
        z: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_scalar = jnp.asarray(z, dtype=jnp.float32)
        lognH = jnp.log10(jnp.maximum(jnp.asarray(nH_cgs, dtype=jnp.float32), 1.0e-40))
        logt = jnp.log10(jnp.maximum(jnp.asarray(temp_k, dtype=jnp.float32), 1.0e-40))

        zc = jnp.clip(z_scalar, self.z_nodes[0], self.z_nodes[-1])
        hi = jnp.searchsorted(self.z_nodes, zc, side="right")
        hi = jnp.clip(hi, 1, self.z_nodes.shape[0] - 1)
        lo = hi - 1
        z0 = self.z_nodes[lo]
        z1 = self.z_nodes[hi]
        wz = (zc - z0) / jnp.maximum(z1 - z0, 1.0e-30)

        heat0 = self._bilinear(self.heat_prim[lo], lognH, logt)
        heat1 = self._bilinear(self.heat_prim[hi], lognH, logt)
        cool0 = self._bilinear(self.cool_prim[lo], lognH, logt)
        cool1 = self._bilinear(self.cool_prim[hi], lognH, logt)
        metal0 = self._bilinear(self.cool_metal_solar[lo], lognH, logt)
        metal1 = self._bilinear(self.cool_metal_solar[hi], lognH, logt)
        mu0 = self._bilinear(self.mu[lo], lognH, logt)
        mu1 = self._bilinear(self.mu[hi], lognH, logt)

        heat = (1.0 - wz) * heat0 + wz * heat1
        cool = (1.0 - wz) * cool0 + wz * cool1
        metal = (1.0 - wz) * metal0 + wz * metal1
        mu = (1.0 - wz) * mu0 + wz * mu1
        return heat, cool, metal, mu
