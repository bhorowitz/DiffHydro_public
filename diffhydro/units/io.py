"""Metadata helpers for persisting CodeUnits with outputs."""

from __future__ import annotations

from typing import Any, Mapping

from .code_units import CodeUnits


def build_unit_metadata(cu: CodeUnits) -> dict[str, Any]:
    return {
        "schema_version": 2,
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
            "radiation_flux": cu.RadFlux_cgs,
            "light_speed_code": cu.light_speed_code,
            "temperature": cu.Temp_cgs,
        },
        "radiation": {
            "light_speed_cgs": cu.light_speed_cgs,
            "light_speed_code": cu.light_speed_code,
        },
        "thermo": {
            "gamma": cu.gamma,
            "mu": cu.mu,
            "kB_cgs": cu.kB_cgs,
            "mH_cgs": cu.mH_cgs,
        },
    }


def code_units_from_metadata(metadata: Mapping[str, Any]) -> CodeUnits:
    base = metadata.get("base_cgs", metadata)
    derived = metadata.get("derived_cgs", {})
    radiation = metadata.get("radiation", {})
    thermo = metadata.get("thermo", {})

    c_rt_cgs = radiation.get("light_speed_cgs")
    if c_rt_cgs is None:
        c_rt_cgs = radiation.get("c_rt_cgs")
    if c_rt_cgs is None:
        c_rt_cgs = derived.get("light_speed_cgs")
    if c_rt_cgs is None:
        c_rt_cgs = derived.get("light_speed")
    if c_rt_cgs is None:
        light_speed_code = radiation.get("light_speed_code", derived.get("light_speed_code"))
        if light_speed_code is not None:
            c_rt_cgs = float(light_speed_code) * float(base["velocity"])
    if c_rt_cgs is None:
        c_rt_cgs = 3.0e10

    return CodeUnits(
        L_cgs=float(base["length"]),
        M_cgs=float(base["mass"]),
        V_cgs=float(base["velocity"]),
        c_rt_cgs=float(c_rt_cgs),
        gamma=float(thermo.get("gamma", 5.0 / 3.0)),
        mu=float(thermo.get("mu", 0.6)),
        kB_cgs=float(thermo.get("kB_cgs", 1.380649e-16)),
        mH_cgs=float(thermo.get("mH_cgs", 1.6735575e-24)),
    )

