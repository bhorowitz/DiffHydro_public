
from __future__ import annotations

from jax import Array
from dataclasses import dataclass, field
import jax.numpy as jnp
import jax
import numpy as np


@dataclass
class EquationManager:
    """
    Generic M1 radiative-transfer equation manager with optional passive
    scalars.

    Active variables (always first, fixed order for compatibility):
        primitives active:    [E_gamma, F_gamma_x, F_gamma_y, F_gamma_z]
        conservatives active: [E_gamma, F_gamma_x, F_gamma_y, F_gamma_z]

    Passive variables: none by default.

    The ionization fraction x_HII now belongs to the gas equation manager
    (:mod:`diffhydro.equationmanager`), where it is advected with the gas
    density as a passive scalar. The radiative manager only carries the
    photon moments [E_gamma, F_gamma_x, F_gamma_y, F_gamma_z].

    The code below avoids hard-coded indices in the numerics by:
      - defining active indices once in __post_init__
      - using active_slice/passive_slice everywhere else.
    """
    gamma: float = 1.6
    n_cons: int = 4          # 4 active (E,Fx,Fy,Fz)
    eps: float = 1e-10
    isothermal: bool = False
    light_speed: float = 1  # reduced/physical speed of light, code units
    mesh_shape: tuple[int, int, int] = (100, 100, 100)

    # Active variable names/order (single source of truth)
    active_names: tuple[str, ...] = ("E_gamma", "F_gamma_x", "F_gamma_y", "F_gamma_z")

    passive_names: tuple[str, ...] = ()

    debug: bool = False

    # Derived index maps (filled in __post_init__)
    active_map: dict[str, int] = field(init=False, repr=False)
    passive_map: dict[str, int] = field(init=False, repr=False)  # NEW
    mass_ids: int = field(init=False)
    vel_ids: tuple[int, int, int] = field(init=False)
    n_active: int = field(init=False)
    normalization: bool = False

    # NEW: bounds applied to specific passive scalars after every
    # primitive reconstruction (name -> (lo, hi)). x_HII must stay in
    # [0, 1] by physical definition of an ionized fraction.
    passive_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # NEW: how the passive block is stored and transported.
    #
    # passive_weighted = True  -> conserved variable is E_gamma * s_k
    #                    False -> the slot stores s_k itself (raw)
    # passive_advected = True  -> s_k is upwinded with the PHOTON flux,
    #                             i.e. it travels at ~c
    #                    False -> zero flux, s_k only evolves through forces
    #
    # PHYSICS WARNING. x_HII is a property of the GAS, so the physically
    # correct conserved quantity is rho * x_HII advected with the gas
    # velocity (that is what RAMSES-RT does, and what
    # coupled_rhd.ChemBlockFlux implements). Keeping it in the RT block
    # with (True, True) advects the ionization state at the speed of light,
    # which makes the ionization front follow the photon front instead of
    # the chemistry. Use (False, False) for a static-gas Stromgren test:
    # x_HII then becomes a purely local variable, exact when v = 0 and a
    # good approximation while v << c.
    passive_weighted: bool = True
    passive_advected: bool = True

    def __post_init__(self):
        self.debug = bool(self.debug)
        self.n_active = len(self.active_names)
        n_passive_declared = len(self.passive_names)

        if self.n_active + n_passive_declared > self.n_cons:
            raise ValueError(
                f"active_names ({self.n_active}) + passive_names "
                f"({n_passive_declared}) exceeds n_cons ({self.n_cons})."
            )

        self.active_map = {name: i for i, name in enumerate(self.active_names)}
        # NEW: passive_map gives the position of each passive scalar
        # *within the passive block* (0-indexed), NOT the absolute
        # index in the full state array. Use passive_slice to get the
        # absolute slice, or self.get_passive_index(name) below.
        self.passive_map = {name: i for i, name in enumerate(self.passive_names)}

        self.mass_ids = self.active_map["E_gamma"]
        self.vel_ids = (
            self.active_map["F_gamma_x"],
            self.active_map["F_gamma_y"],
            self.active_map["F_gamma_z"],
        )

        self.passive_weighted = bool(self.passive_weighted)
        self.passive_advected = bool(self.passive_advected)

        self.velocity_minor_axes = ((2, 3), (3, 1), (1, 2))
        self.equation_type = "SINGLE-PHASE"
        self.thermal_conductivity_model = "SUTHERLAND"
        self.sutherland_parameters = [0.1, 1.0, 1.0]
        self.cfl = 0.4
        self.R = 1.0
        self.cp = self.gamma / (self.gamma - 1.0) * self.R
        self.light_speed = float(self.light_speed)
        self.normalization = bool(self.normalization)
        if self.light_speed <= 0.0:
            raise ValueError("light_speed must be > 0.")

    # ---------------------------
    # Convenience slices / indices
    # ---------------------------
    @property
    def active_slice(self):
        return slice(0, self.n_active)

    @property
    def passive_slice(self):
        return slice(self.n_active, self.n_cons)

    @property
    def n_passive(self) -> int:
        return max(0, self.n_cons - self.n_active)

    def get_passive_index(self, name: str) -> int:
        """
        Absolute index of a named passive scalar in the FULL state array
        (primitives or conservatives), i.e. usable directly as
        primitives[idx] / conservatives[idx].
        """
        if name not in self.passive_map:
            raise KeyError(
                f"Unknown passive scalar '{name}'. Known: {list(self.passive_map)}"
            )
        return self.n_active + self.passive_map[name]

    # ---------------------------
    # Primitive <-> Conservative
    # ---------------------------
    def get_conservatives_from_primitives(self, primitives):
        """
        primitives: (>= n_active, ...)
        returns conservatives: (>= n_active, ...)
        """
        prim_a = primitives[self.active_slice]

        E_gamma = jnp.maximum(prim_a[self.mass_ids], self.eps)
        fx, fy, fz = (prim_a[i] for i in self.vel_ids)
        cons_a = jnp.stack([E_gamma, fx, fy, fz], axis=0)

        if primitives.shape[0] > self.n_active:
            prim_p = primitives[self.passive_slice]           # (n_passive,...)
            if self.passive_weighted:
                cons_p = E_gamma[jnp.newaxis, ...] * prim_p    # E_gamma * s_k
            else:
                cons_p = prim_p                                # raw scalar
            out = jnp.concatenate([cons_a, cons_p], axis=0)
        else:
            out = cons_a

        if self.debug:
            jax.debug.print("any nan primitives = {}", jnp.any(jnp.isnan(primitives)))
            jax.debug.print("any inf primitives = {}", jnp.any(jnp.isinf(primitives)))
            jax.debug.print("min primitives = {}", jnp.nanmin(primitives))
            jax.debug.print("max primitives = {}", jnp.nanmax(primitives))

        return out

    def get_primitives_from_conservatives(self, conservatives):
        """
        conservatives: (>= n_active, ...)
        returns primitives: (>= n_active, ...)
        """
        cons_a = conservatives[self.active_slice]
        if self.debug:
            jax.debug.print("cons_a_0: {v}", v=cons_a[0, 0, 50, 50])

        E_gamma = jnp.maximum(cons_a[self.mass_ids], self.eps)
        F_gamma_x = cons_a[1]
        F_gamma_y = cons_a[2]
        F_gamma_z = cons_a[3]

        prim_a = jnp.stack([E_gamma, F_gamma_x, F_gamma_y, F_gamma_z], axis=0)

        if conservatives.shape[0] > self.n_active:
            cons_p = conservatives[self.passive_slice]
            if self.passive_weighted:
                prim_p = cons_p * (1.0 / E_gamma[jnp.newaxis, ...])
            else:
                prim_p = cons_p

            return jnp.concatenate([prim_a, prim_p], axis=0)

        return prim_a

    # ---------------------------
    # M1 radiation physics (unchanged)
    # ---------------------------
    def get_signal_speed(self, primitives, axis):
        E_gamma = primitives[self.mass_ids]
        return self.get_speed_of_sound(E_gamma)

    def get_speed_of_sound(self, E_gamma: Array) -> Array:
        E_gamma_safe = jnp.maximum(E_gamma, self.eps)
        return jnp.full_like(E_gamma_safe, self.light_speed)

    def _normalize_radiative_conservatives(self, conservatives):
        return self.limit_radiative_flux_cone(conservatives)

    def limit_radiative_flux_cone(self, conservatives, f_max=1.0):
        """Limit physical radiation flux to |F| <= f_max * c * E."""
        cons = conservatives
        cons_a = cons[self.active_slice]
        E_gamma = jnp.maximum(cons_a[self.mass_ids], self.eps)
        Fx, Fy, Fz = (cons_a[i] for i in self.vel_ids)
        F_norm = jnp.sqrt(Fx**2 + Fy**2 + Fz**2)
        F_limit = f_max * self.light_speed * E_gamma
        scale = jnp.minimum(1.0, F_limit / jnp.maximum(F_norm, self.eps))
        cons_a = cons_a.at[self.vel_ids[0]].set(Fx * scale)
        cons_a = cons_a.at[self.vel_ids[1]].set(Fy * scale)
        cons_a = cons_a.at[self.vel_ids[2]].set(Fz * scale)
        return cons.at[self.active_slice].set(cons_a)

    def _m1_chi_from_reduced_flux(self, fred: Array) -> Array:
        """
        M1 Eddington factor:
            chi(f) = (3 + 4 f^2) / (5 + 2 sqrt(4 - 3 f^2))
        fred = |F| / (c E), so fred in [0, 1].
        """
        fred = jnp.clip(fred, 0.0, 1.0 - 1e-12)
        radicand = jnp.maximum(4.0 - 3.0 * fred * fred, self.eps)
        return (3.0 + 4.0 * fred * fred) / (5.0 + 2.0 * jnp.sqrt(radicand))

    def _radiation_pressure_tensor(self, E_gamma: Array, F_vec: Array) -> Array:
        """
        Compute M1 radiation pressure tensor P_ij from:
            E_gamma : (...)
            F_vec   : (3, ...)
        Returns:
            P : (3, 3, ...)
        """
        E_safe = jnp.maximum(E_gamma, self.eps)
        F_norm = jnp.sqrt(jnp.sum(F_vec * F_vec, axis=0))
        fred = F_norm / (self.light_speed * E_safe)
        fred = jnp.clip(fred, 0.0, 1.0 - 1e-12)

        chi = self._m1_chi_from_reduced_flux(fred)

        F_norm_safe = jnp.maximum(F_norm, self.eps)
        n_vec = F_vec / F_norm_safe[jnp.newaxis, ...]

        I = jnp.eye(3, dtype=E_gamma.dtype).reshape(3, 3, *([1] * E_gamma.ndim))
        n_outer = jnp.einsum("i...,j...->ij...", n_vec, n_vec)

        D = (
            0.5 * (1.0 - chi)[jnp.newaxis, jnp.newaxis, ...] * I
            + 0.5 * (3.0 * chi - 1.0)[jnp.newaxis, jnp.newaxis, ...] * n_outer
        )

        D_iso = (1.0 / 3.0) * I
        mask_zero = (F_norm <= self.eps)[jnp.newaxis, jnp.newaxis, ...]
        D = jnp.where(mask_zero, D_iso, D)

        P = D * E_safe[jnp.newaxis, jnp.newaxis, ...]
        return P

    def get_radiative_pressure_tensor_from_primitives(self, primitives: Array) -> Array:
        prim_a = primitives[self.active_slice]
        E_gamma = jnp.maximum(prim_a[self.mass_ids], self.eps)
        F_vec = jnp.stack([prim_a[i] for i in self.vel_ids], axis=0)
        return self._radiation_pressure_tensor(E_gamma, F_vec)

    def get_fluxes_xi(self, primitives, conservatives, axis: int):
        """
        RT moments (M1 closure), direction axis in {0,1,2}.
        State:      U = [E_gamma, F_gamma_x, F_gamma_y, F_gamma_z]
        Flux:       F_i(U) = [F_i, c^2 P_{i,x}, c^2 P_{i,y}, c^2 P_{i,z}]

        No passive scalars are transported here; the gas equation manager
        carries x_HII as a gas-side passive scalar.
        """
        del primitives
        if self.normalization:
            conservatives = self._normalize_radiative_conservatives(conservatives)

        cons_a = conservatives[self.active_slice]
        E_gamma = jnp.maximum(cons_a[self.mass_ids], self.eps)

        F_gamma_x, F_gamma_y, F_gamma_z = (cons_a[i] for i in self.vel_ids)
        F_vec = jnp.stack([F_gamma_x, F_gamma_y, F_gamma_z], axis=0)
        P = self._radiation_pressure_tensor(E_gamma, F_vec)
        E_flux = F_vec[axis]
        c2 = self.light_speed * self.light_speed

        flux_a = jnp.stack(
            [E_flux, c2 * P[axis, 0], c2 * P[axis, 1], c2 * P[axis, 2]],
            axis=0,
        )

        return flux_a

    # ---------------------------
    # Diagnostics / accessors
    # ---------------------------
    def get_energy_grid(self, primitives: Array) -> Array:
        prim_a = primitives[self.active_slice]
        return jnp.maximum(prim_a[self.mass_ids], self.eps)

    def get_F_over_E_components_grid(self, primitives: Array) -> Array:
        prim_a = primitives[self.active_slice]
        E = jnp.maximum(prim_a[self.mass_ids], self.eps)
        Fx, Fy, Fz = (prim_a[i] for i in self.vel_ids)
        return jnp.stack([Fx, Fy, Fz], axis=0) / E[jnp.newaxis, ...]

    def get_F_components_grid(self, primitives: Array) -> Array:
        prim_a = primitives[self.active_slice]
        return jnp.stack([prim_a[i] for i in self.vel_ids], axis=0)

    def get_F_over_E_norm_grid(self, primitives: Array) -> Array:
        F_over_E = self.get_F_over_E_components_grid(primitives)
        return jnp.sqrt(jnp.sum(F_over_E * F_over_E, axis=0))

    def get_reduced_flux_grid(self, primitives: Array) -> Array:
        return self.get_F_over_E_norm_grid(primitives) / self.light_speed

    # NEW: direct grid accessor for the ionization fraction
