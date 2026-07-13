"""Field and dimension-level conversions between physical and code units."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from .code_units import CodeUnits
from .field_dims import FIELD_DIMS
from .registry import Quantity, UnitParser


def to_code(value: Any, dim: str, cu: CodeUnits, parser: UnitParser | None = None) -> Any:
    unit_parser = parser or UnitParser()
    if isinstance(value, str):
        if value.strip().lower().startswith("code:"):
            return float(value.split(":", 1)[1].strip())
        parsed = unit_parser.parse(value, expected_dim=dim)
        return parsed.cgs_value / cu.scale(dim)

    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr


def from_code(
    value: Any,
    dim: str,
    cu: CodeUnits,
    out_unit: str,
    parser: UnitParser | None = None,
) -> Quantity:
    unit_parser = parser or UnitParser()
    arr = np.asarray(value)

    if out_unit.strip().lower().startswith("code:"):
        unit = out_unit.split(":", 1)[1].strip() or "code"
        out_value = arr if arr.ndim > 0 else float(arr)
        return Quantity(value=out_value, unit=unit, dimension=dim)

    unit_factor = unit_parser.unit_factor_to_cgs(out_unit, expected_dim=dim)
    cgs_value = arr * cu.scale(dim)
    out = cgs_value / unit_factor
    out_value = out if out.ndim > 0 else float(out)
    return Quantity(value=out_value, unit=out_unit, dimension=dim)


def to_code_fields(
    fields: Mapping[str, Any],
    cu: CodeUnits,
    field_dims: Mapping[str, str] | None = None,
    parser: UnitParser | None = None,
) -> dict[str, Any]:
    dims = field_dims or FIELD_DIMS
    unit_parser = parser or UnitParser()
    out: dict[str, Any] = {}
    for name, value in fields.items():
        dim = dims.get(name)
        if dim is None:
            out[name] = value
            continue
        out[name] = to_code(value, dim, cu, parser=unit_parser)
    return out


def from_code_fields(
    fields: Mapping[str, Any],
    cu: CodeUnits,
    display_units: Mapping[str, str],
    field_dims: Mapping[str, str] | None = None,
    parser: UnitParser | None = None,
) -> dict[str, Quantity]:
    dims = field_dims or FIELD_DIMS
    unit_parser = parser or UnitParser()
    out: dict[str, Quantity] = {}
    for name, value in fields.items():
        dim = dims.get(name)
        if dim is None:
            continue
        out_unit = display_units.get(dim, unit_parser.default_cgs_unit(dim))
        out[name] = from_code(value, dim, cu, out_unit=out_unit, parser=unit_parser)
    return out


def temperature_from_Prho(p_code: Any, rho_code: Any, cu: CodeUnits) -> Any:
    p_arr = np.asarray(p_code)
    rho_arr = np.asarray(rho_code)
    p_cgs = p_arr * cu.P_cgs
    rho_cgs = rho_arr * cu.rho_cgs
    T_k = p_cgs * cu.mu * cu.mH_cgs / (rho_cgs * cu.kB_cgs)
    return T_k if T_k.ndim > 0 else float(T_k)


def pressure_from_Trho(
    T: Any,
    rho_code: Any,
    cu: CodeUnits,
    parser: UnitParser | None = None,
) -> Any:
    unit_parser = parser or UnitParser()
    if isinstance(T, str):
        if T.strip().lower().startswith("code:"):
            t_code = float(T.split(":", 1)[1].strip())
            T_k = t_code * cu.Temp_cgs
        else:
            parsed = unit_parser.parse(T, expected_dim="temperature")
            T_k = parsed.cgs_value
    else:
        T_k = np.asarray(T)

    rho_arr = np.asarray(rho_code)
    p_cgs = rho_arr * cu.rho_cgs * cu.kB_cgs * T_k / (cu.mu * cu.mH_cgs)
    p_code = p_cgs / cu.P_cgs
    return p_code if np.asarray(p_code).ndim > 0 else float(p_code)


def format_quantity(
    value: Any,
    dim: str,
    cu: CodeUnits,
    out_unit: str | None = None,
    parser: UnitParser | None = None,
    precision: int = 6,
) -> str:
    """Return a human-readable string for a code-unit quantity.

    Scalars are formatted as ``value unit`` and arrays are rendered with
    ``numpy.array2string`` followed by the unit. This is convenient in tests
    when inspecting profiles such as 1/r^2 beams or anisotropic momentum dumps.
    """
    unit_parser = parser or UnitParser()
    quantity = from_code(
        value,
        dim,
        cu,
        out_unit=out_unit or unit_parser.default_cgs_unit(dim),
        parser=unit_parser,
    )
    arr = np.asarray(quantity.value)
    if arr.ndim == 0:
        value_text = f"{float(arr):.{precision}e}"
    else:
        value_text = np.array2string(arr, precision=precision, separator=", ")
    return f"{value_text} {quantity.unit}"

