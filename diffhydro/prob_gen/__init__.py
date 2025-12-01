"""Problem generation utilities for DiffHydro."""

from .blobs import make_gaussian_blob, make_gaussian_blob_rho
from .initial_conditions import sedov, sedov_2d
from .polytropes import lane_emden_rhs, polytropic_density_3d, solve_lane_emden

__all__ = [
    "make_gaussian_blob",
    "sedov",
    "sedov_2d",
    "lane_emden_rhs",
    "polytropic_density_3d",
    "solve_lane_emden",
]

