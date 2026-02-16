"""Equation-of-state projection operators."""

from __future__ import annotations

import jax.numpy as jnp


class EOSProjectionForce:
    """
    Force-like operator that projects conservative states back to the configured EOS.

    For isothermal mode, this enforces
      p = c_s^2 rho
    by resetting the conservative energy to:
      E = 0.5 * |m|^2 / rho + p / (gamma - 1)
    """

    def __init__(self, equation_manager, *, project_even_if_adiabatic: bool = False):
        self.eq = equation_manager
        self.project_even_if_adiabatic = bool(project_even_if_adiabatic)

        self.i_rho = self.eq.mass_ids
        self.i_mx, self.i_my, self.i_mz = self.eq.vel_ids
        self.i_E = self.eq.energy_ids

    def timestep(self, U):
        return 1.0e10

    def force(self, i_step, U, params, dt):
        if (not self.eq.isothermal) and (not self.project_even_if_adiabatic):
            return U, params

        rho = jnp.maximum(U[self.i_rho], self.eq.eps)
        mx = U[self.i_mx]
        my = U[self.i_my]
        mz = U[self.i_mz]

        kinetic = 0.5 * (mx * mx + my * my + mz * mz) / rho
        p_iso = self.eq.get_isothermal_pressure(rho)
        thermal = p_iso / (self.eq.gamma - 1.0)
        E_proj = kinetic + thermal

        U = U.at[self.i_E].set(E_proj)
        return U, params
