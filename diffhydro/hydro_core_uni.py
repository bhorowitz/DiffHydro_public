"""Static (non-AMR) hydrodynamics solver."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import jax.numpy as jnp

from .hydro.common import Array, positivity_fix


class HydroStatic:
    """Uniform-mesh hydrodynamics driver with Strang-like splitting."""

    def __init__(
        self,
        fluxes: Iterable,
        forces: Iterable = (),
        boundary=None,
        recon=None,
        splitting_schemes: Sequence[Sequence[int]] = ((1, 2, 2, 1), (2, 1, 1, 2)),
        max_dt: float = 1.0,
        dx: float = 1.0,
        maxjit: bool = False,
        snapshots: int | None = None,
        n_super_step: int = 25,
    ) -> None:
        self.fluxes = list(fluxes)
        self.forces = list(forces)
        self.boundary = boundary
        self.recon = recon
        self.splitting_schemes = tuple(tuple(s) for s in splitting_schemes)
        self.max_dt = float(max_dt)
        self.dx_o = float(dx)
        self.maxjit = bool(maxjit)
        self.snapshots = snapshots
        self.n_super_step = int(n_super_step)
        self._trace: Dict[str, Array] = {}

    # PDE wrappers -----------------------------------------------------
    def flux(self, sol: Array, ax: int, params: Dict) -> Array:
        total = jnp.zeros_like(sol)
        for flux in self.fluxes:
            total = total + flux.flux(sol, ax, params)
        return total

    def timestep(self, fields: Array) -> Array:
        dts = [flux.timestep(fields) for flux in self.fluxes]
        for force in self.forces:
            dts.append(force.timestep(fields))
        if not dts:
            return jnp.asarray(self.max_dt)
        return jnp.min(jnp.stack(dts))

    # Time stepping ----------------------------------------------------
    def _hydrostep_uniform(self, U: Array, params: Dict, dt: float) -> Array:
        for scheme in self.splitting_schemes:
            for ax in scheme:
                ra = 2 if ax == 1 else 1
                fu1 = self.flux(U, ax, params)
                rhs1 = fu1 - jnp.roll(fu1, 1, axis=ra)
                U = positivity_fix(self, U - (dt / (2.0 * self.dx_o)) * rhs1)
                fu2 = self.flux(U, ax, params)
                rhs2 = fu2 - jnp.roll(fu2, 1, axis=ra)
                U = positivity_fix(self, U - (dt / self.dx_o) * rhs2)
        return U

    # Public API -------------------------------------------------------
    def evolve(self, input_fields: Array, params: Dict):
        U = input_fields
        snapshots: list[Array] = []

        for step in range(self.n_super_step):
            dt = jnp.minimum(self.max_dt, self.timestep(U))
            U = self._hydrostep_uniform(U, params, float(dt))
            if self.snapshots and (step % self.snapshots == 0):
                snapshots.append(U)

        trace: Dict[str, Array] | Dict[str, list[Array]] = {}
        if snapshots:
            trace["snapshots"] = snapshots
        return U, trace


# Backwards compatible alias expected by callers ----------------------
hydro = HydroStatic

__all__ = ["HydroStatic", "hydro"]
