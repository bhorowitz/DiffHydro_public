"""Legacy AMR helpers kept for reference."""
from __future__ import annotations

from .debug import (
    check_total_energy,
    conservative_restriction_with_correction,
    create_smooth_transition_mask,
    debug_amr_step,
    diagnose_tile_continuity,
    make_halos_x,
    make_halos_y,
    solve_step_freeze_ghosts,
)

__all__ = [
    "check_total_energy",
    "conservative_restriction_with_correction",
    "create_smooth_transition_mask",
    "debug_amr_step",
    "diagnose_tile_continuity",
    "make_halos_x",
    "make_halos_y",
    "solve_step_freeze_ghosts",
]
