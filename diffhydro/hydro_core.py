"""Compatibility shim that exposes the AMR hydro solver from the new package layout."""
from __future__ import annotations

from .hydro.amr.core import (
    HydroAMR,
    assemble_from_tiles,
    build_level0_config,
    extract_tiles,
    hydro,
    prolong_PLM_minmod,
    refine_mask_from_indicator_hyst,
    restrict_avg,
    step_tiles_with_halo,
)

__all__ = [
    "HydroAMR",
    "assemble_from_tiles",
    "build_level0_config",
    "extract_tiles",
    "hydro",
    "prolong_PLM_minmod",
    "refine_mask_from_indicator_hyst",
    "restrict_avg",
    "step_tiles_with_halo",
]
