from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def parse_pmesh_shape(text: str) -> tuple[int, int, int]:
    raw = text.lower().replace(",", "x").strip()
    parts = [int(p) for p in raw.split("x") if p]
    if len(parts) != 3:
        raise ValueError(f"Expected pmesh shape like '4x1x1', got {text!r}")
    if any(p < 1 for p in parts):
        raise ValueError(f"Invalid pmesh shape {parts}")
    return tuple(parts)


def parse_pdims(text: str) -> tuple[int, int]:
    raw = text.lower().replace(",", "x").strip()
    parts = [int(p) for p in raw.split("x") if p]
    if len(parts) != 2:
        raise ValueError(f"Expected pdims like '4x1', got {text!r}")
    if any(p < 1 for p in parts):
        raise ValueError(f"Invalid pdims {parts}")
    return tuple(parts)


def integrate_tau_target(background, a_init: float, a_final: float, n_steps: int = 8192) -> float:
    if a_final <= a_init:
        return 0.0
    a_grid = np.linspace(float(a_init), float(a_final), int(max(8, n_steps)), dtype=np.float64)
    da_dtau = np.asarray(background.da_dtau(a_grid), dtype=np.float64)
    dtau_da = 1.0 / np.maximum(da_dtau, 1.0e-30)
    return float(np.trapezoid(dtau_da, a_grid))


def write_json(path: str | Path, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
