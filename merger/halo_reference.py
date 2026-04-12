"""
halo_reference.py
=================
Shared reference-profile helpers for the merger validation ladder.

The Stage 0/1 scripts use ``cluster_generator.HydrostaticEquilibrium`` to
sample dark-matter particles from a halo model that includes gas and stars.
Those sampled particles therefore represent the DM component, not the full
total NFW mass profile.  The helpers below build consistent total/gas/star/DM
reference profiles so diagnostics compare against the right target.
"""

from __future__ import annotations

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as us

import cluster_generator as cg


def cumulative_trapezoid(y, x):
    """Simple cumulative trapezoid integral with ``initial=0`` semantics."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


class HaloReference:
    """Reference radial profiles for a single cluster_generator halo."""

    def __init__(
        self,
        *,
        z: float = 0.295,
        m200_msun: float = 5.0e14,
        conc: float = 3.5,
        star_fraction: float = 0.02,
        r_grid_max: float | None = None,
        n_r: int = 4096,
    ):
        self.z = float(z)
        self.m200_msun = float(m200_msun)
        self.conc = float(conc)
        self.star_fraction = float(star_fraction)

        cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
        self.rhoc = cosmo.critical_density(self.z).to(us.M_sun / us.kpc**3).value

        self.r200 = cg.find_overdensity_radius(self.m200_msun, 200.0, z=self.z)
        self.a_scale = self.r200 / self.conc
        self.rho_s = (
            200.0
            * self.rhoc
            * self.conc**3
            / ((np.log(1.0 + self.conc) - self.conc / (1.0 + self.conc)) * 3.0)
        )

        self.total_mass_profile = cg.nfw_mass_profile(self.rho_s, self.a_scale)
        self.total_density_profile = cg.nfw_density_profile(self.rho_s, self.a_scale)

        self.r500, self.m500 = cg.find_radius_mass(self.total_mass_profile, z=self.z, delta=500.0)
        self.r2500, self.m2500 = cg.find_radius_mass(self.total_mass_profile, z=self.z, delta=2500.0)

        self.f_gas_r500 = cg.f_gas(self.m500)
        gas_profile = cg.vikhlinin_density_profile(
            1.0, 0.4 * self.r2500, 0.8 * self.r200, 1.0, 0.8, 1.5
        )
        self.gas_density_profile = cg.rescale_profile_by_mass(
            gas_profile, self.f_gas_r500 * self.m500, self.r500
        )

        self.hse = cg.HydrostaticEquilibrium.from_dens_and_tden(
            0.1,
            20000.0,
            self.gas_density_profile,
            self.total_density_profile,
            stellar_density=self.star_fraction * self.total_density_profile,
        )

        r_max = float(r_grid_max) if r_grid_max is not None else 5.0 * self.r200
        r_min = max(1.0e-3, 1.0e-4 * self.r200)
        self.r_grid = np.geomspace(r_min, r_max, int(n_r))

        self.rho_total = np.asarray(self.total_density_profile(self.r_grid), dtype=np.float64)
        self.rho_gas = np.asarray(self.gas_density_profile(self.r_grid), dtype=np.float64)
        self.rho_star = self.star_fraction * self.rho_total
        self.rho_dm = np.clip(self.rho_total - self.rho_gas - self.rho_star, 0.0, None)

        prefac = 4.0 * np.pi * self.r_grid**2
        self.mass_total = cumulative_trapezoid(prefac * self.rho_total, self.r_grid)
        self.mass_gas = cumulative_trapezoid(prefac * self.rho_gas, self.r_grid)
        self.mass_star = cumulative_trapezoid(prefac * self.rho_star, self.r_grid)
        self.mass_dm = cumulative_trapezoid(prefac * self.rho_dm, self.r_grid)

    def _interp(self, values, r_eval):
        r_eval = np.asarray(r_eval, dtype=np.float64)
        return np.interp(r_eval, self.r_grid, values, left=values[0], right=values[-1])

    def rho_dm_at(self, r_eval):
        return self._interp(self.rho_dm, r_eval)

    def rho_total_at(self, r_eval):
        return self._interp(self.rho_total, r_eval)

    def mass_dm_at(self, r_eval):
        return self._interp(self.mass_dm, r_eval)

    def mass_total_at(self, r_eval):
        return self._interp(self.mass_total, r_eval)

