"""Metadata helpers for persisting CodeUnits with outputs."""

from __future__ import annotations

from typing import Any, Mapping

from .code_units import CodeUnits


def build_unit_metadata(cu: CodeUnits) -> dict[str, Any]:
    return {
        "base_cgs": {
            "length": cu.L_cgs,
            "mass": cu.M_cgs,
            "velocity": cu.V_cgs,
        },
        "derived_cgs": {
            "time": cu.T_cgs,
            "density": cu.rho_cgs,
            "pressure": cu.P_cgs,
            "energy_density": cu.Eden_cgs,
            "temperature": cu.Temp_cgs,
        },
        "thermo": {
            "gamma": cu.gamma,
            "mu": cu.mu,
            "kB_cgs": cu.kB_cgs,
            "mH_cgs": cu.mH_cgs,
        },
    }


def code_units_from_metadata(metadata: Mapping[str, Any]) -> CodeUnits:
    base = metadata["base_cgs"]
    thermo = metadata.get("thermo", {})
    return CodeUnits(
        L_cgs=float(base["length"]),
        M_cgs=float(base["mass"]),
        V_cgs=float(base["velocity"]),
        gamma=float(thermo.get("gamma", 5.0 / 3.0)),
        mu=float(thermo.get("mu", 0.6)),
        kB_cgs=float(thermo.get("kB_cgs", 1.380649e-16)),
        mH_cgs=float(thermo.get("mH_cgs", 1.6735575e-24)),
    )