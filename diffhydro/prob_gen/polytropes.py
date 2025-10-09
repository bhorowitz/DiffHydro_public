"""Polytropic profiles and Lane-Emden helpers for initial conditions."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def lane_emden_rhs(xi: float, y, n: float):
    """Right hand side of the Lane-Emden equation."""

    theta, dtheta_dxi = y
    if theta > 0:
        d2theta_dxi2 = -(2 / xi) * dtheta_dxi - theta**n
    else:
        # Stop integrating when the density turns negative.
        d2theta_dxi2 = 0.0
    return [dtheta_dxi, d2theta_dxi2]


def solve_lane_emden(
    n: float,
    xi_max: float = 10.0,
    num_points: int = 1000,
):
    """Solve the Lane-Emden equation for a given polytropic index."""

    xi_vals = np.linspace(1e-6, xi_max, num_points)
    y0 = [1.0, 0.0]
    sol = solve_ivp(
        lane_emden_rhs,
        [xi_vals[0], xi_vals[-1]],
        y0,
        t_eval=xi_vals,
        args=(n,),
        method="RK45",
    )
    return sol.t, sol.y[0]


def polytropic_density_3d(
    n: float,
    grid_size: int = 256,
    scale_radius: float = 1.0,
):
    """Return a 3-D density profile for a polytrope with index ``n``."""

    xi, theta = solve_lane_emden(n)
    max_radius_idx = np.argmax(theta < 0)
    max_radius = xi[max_radius_idx]

    x = np.linspace(-max_radius, max_radius, grid_size)
    y = np.linspace(-max_radius, max_radius, grid_size)
    z = np.linspace(-max_radius, max_radius, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2) * scale_radius

    density = np.interp(R, xi, theta, left=0.0, right=0.0)
    return density


__all__ = [
    "lane_emden_rhs",
    "solve_lane_emden",
    "polytropic_density_3d",
]

