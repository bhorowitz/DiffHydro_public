from __future__ import annotations

from jax import Array
from dataclasses import dataclass, field
import jax.numpy as jnp
import jax
from matplotlib import axis
import numpy as np


@dataclass
class EquationManager:
    """
    Generic compressible Euler equation manager with optional passive scalars.

        Active variables (always first, fixed order for compatibility):
            primitives active:  [E_gamma, F_gamma_x, F_gamma_y, F_gamma_z]
            conservatives active: [E_gamma, E_gamma*F_gamma_x, E_gamma*F_gamma_y, E_gamma*F_gamma_z]

    Passive variables (any extra slots beyond active):
      primitives passive:  s_k
      conservatives passive: E_gamma * s_k

    The code below avoids hard-coded indices in the numerics by:
      - defining active indices once in __post_init__
      - using active_slice/passive_slice everywhere else.
    """
    gamma: float = 1.6
    n_cons: int = 4 #avant c etait 5
    eps: float = 1e-10
    isothermal: bool = False
    light_speed: float = 1#3e8 #en m.s-1
    # fraction_escape: float = 0.1
    # volume_cell: float = 0.1
    # star_mass: float = 1.0
    # star_age: float = 1.0
    # star_metallicity: float = 1.0
    mesh_shape: tuple[int, int, int] = (100, 100, 100)
    # Active variable names/order (single source of truth)
    active_names: tuple[str, ...] = ("E_gamma", "F_gamma_x", "F_gamma_y", "F_gamma_z")#,"p" 
    # passive_names: tuple[str, ...] = ("E_gamma",) #new
    debug: bool = False
    # Derived index maps (filled in __post_init__)
    active_map: dict[str, int] = field(init=False, repr=False)
    mass_ids: int = field(init=False)
    vel_ids: tuple[int, int, int] = field(init=False)
    n_active: int = field(init=False)
    normalization: bool = True
    # energy_ids: int = field(init=False)
    def __post_init__(self):
        self.debug = bool(self.debug)
        self.n_active = len(self.active_names)
        if self.n_active > self.n_cons:
            raise ValueError(
                f"Expected max {self.n_cons} total vars. Got {self.n_active} active."
            )

        self.active_map = {name: i for i, name in enumerate(self.active_names)}
        self.mass_ids = self.active_map["E_gamma"]
        
        self.vel_ids = (
            self.active_map["F_gamma_x"],
            self.active_map["F_gamma_y"],
            self.active_map["F_gamma_z"],
        )
        # self.energy_ids = self.active_map["p"]
        self.velocity_minor_axes = ((2, 3), (3, 1), (1, 2))
        self.equation_type = "SINGLE-PHASE"#equation_information.equation_type
        self.thermal_conductivity_model = "SUTHERLAND"
        self.sutherland_parameters = [0.1, 1.0, 1.0]
        self.cfl = 0.01 #original value 0.4
        # self.mesh_shape = list(self.mesh_shape)
        self.R = 1.0
        self.cp = self.gamma / (self.gamma - 1.0) * self.R
        self.light_speed = float(self.light_speed)
        self.normalization = True
        if self.light_speed <= 0.0:
            raise ValueError("light_speed must be > 0.")
        
    def get_conservatives_from_primitives(self, primitives):
        """
        primitives: (>= n_active, ...)
        returns conservatives: (>= n_active, ...)
        """
        prim_a = primitives[self.active_slice]

        E_gamma = jnp.maximum(prim_a[self.mass_ids], self.eps)
        # jax.debug.print("E_gamma: {v}", v=E_gamma[ :, :, :])
        u, v, w = (prim_a[i] for i in self.vel_ids)
        # # p = prim_a[self.energy_ids]
        # if self.isothermal:
        #     p = self.get_isothermal_pressure(E_gamma)
            
        # e = self.get_specific_energy(E_gamma)
        # kin = 0.5 * (u*u + v*v + w*w)
        # Etot = E_gamma * (kin + e) #a modifier
        cons_a = jnp.stack([E_gamma, E_gamma*u, E_gamma*v, E_gamma*w], axis=0) #Etot
        # cons_a = prim_a
        if primitives.shape[0] > self.n_active:
            prim_p = primitives[self.passive_slice]                 # (n_passive,...)
            cons_p = E_gamma[jnp.newaxis, ...] * prim_p             # E_gamma*s_k
            return jnp.concatenate([cons_a,cons_p], axis=0) #, cons_p
        if self.debug:
            jax.debug.print("any nan primitives = {}", jnp.any(jnp.isnan(primitives)))
            jax.debug.print("any inf primitives = {}", jnp.any(jnp.isinf(primitives)))
            jax.debug.print("min primitives = {}", jnp.nanmin(primitives))
            jax.debug.print("max primitives = {}", jnp.nanmax(primitives))
            # jax.debug.print("Value variables mass ids: {}", self.mass_ids)
            
        return cons_a

    def get_primitives_from_conservatives(self, conservatives):
        """
        conservatives: (>= n_active, ...)
        returns primitives: (>= n_active, ...)
        """
        # cons_a = conservatives[self.active_slice]

        # E_gamma = cons_a[self.mass_ids]
        # E_gamma_safe = jnp.maximum(E_gamma, self.eps)
        # inv_E_gamma = 1.0 / E_gamma_safe

        # u = cons_a[1] * inv_E_gamma
        # v = cons_a[2] * inv_E_gamma
        # w = cons_a[3] * inv_E_gamma

        # E = cons_a[4] * inv_E_gamma
        # kin = 0.5 * (u*u + v*v + w*w)
        # e = jnp.maximum(E - kin, self.eps) #a modifier

        # if self.isothermal:
        #     p = self.get_isothermal_pressure(E_gamma_safe)
        # else:
        #     p = self.get_pressure(e, E_gamma_safe)
            
        # prim_a = jnp.stack([E_gamma_safe, u, v, w], axis=0) #new ,p

        # if conservatives.shape[0] > self.n_active:
        #     cons_p = conservatives[self.passive_slice]              # (n_passive,...)
        #     prim_p = cons_p * inv_E_gamma[jnp.newaxis, ...]             # s_k
        #     return jnp.concatenate([prim_a, prim_p], axis=0)
        """
        For the RT moment system, active primitives and conservatives are identical.
        """
        cons_a = conservatives[self.active_slice]
        if self.debug:
            jax.debug.print("cons_a_0: {v}", v=cons_a[0,0,50,50])

        E_gamma = jnp.maximum(cons_a[self.mass_ids], self.eps) #ajouter garde fous division par 0 
        F_gamma_x = cons_a[1]/E_gamma
        F_gamma_y = cons_a[2]/E_gamma
        F_gamma_z = cons_a[3]/E_gamma 

        prim_a = jnp.stack([E_gamma, F_gamma_x, F_gamma_y, F_gamma_z], axis=0)

        if conservatives.shape[0] > self.n_active:
            cons_p = conservatives[self.passive_slice]
            prim_p = cons_p * 1/E_gamma[jnp.newaxis, ...] 
            return jnp.concatenate([prim_a, prim_p], axis=0)

        return prim_a

   # ---------------------------
    # Convenience slices
    # ---------------------------
    @property
    def active_slice(self):
        return slice(0, self.n_active) #modifier ici pour choisir quoi utiliser probablement 4 
    
    @property
    def passive_slice(self):
        return slice(self.n_active, self.n_cons)

    @property
    def n_passive(self) -> int:
        return max(0, self.n_cons - self.n_active)

    # def get_pressure(self, e, E_gamma):
    #     if self.isothermal:
    #         return self.get_isothermal_pressure(E_gamma)
    #     return (self.gamma - 1.0) * jnp.maximum(e, self.eps) * jnp.maximum(E_gamma, self.eps)
    
    def get_signal_speed(self, primitives, axis): #probablement a changer pour c of speed
        # p = primitives[self.energy_ids]
        E_gamma = primitives[self.mass_ids]
        return self.get_speed_of_sound(E_gamma)
    
    def get_speed_of_sound(self, E_gamma: Array) -> Array: # self, p: Array,probablement a changer pour retourne speed of light
        """See base class. """
        E_gamma_safe = jnp.maximum(E_gamma, self.eps)
        return jnp.full_like(E_gamma_safe, self.light_speed)
    
    def _normalize_radiative_conservatives(self, conservatives):
        cons = conservatives

        cons_a = cons[self.active_slice]

        E_gamma = jnp.maximum(cons_a[self.mass_ids], self.eps)
        E_gamma_safe = jnp.maximum(E_gamma, self.eps)

        mom_x, mom_y, mom_z = (cons_a[i] for i in self.vel_ids)

        # Ici les variables conservatives stockent déjà les composantes "flux-like"
        # que tu veux renormaliser.
        F_gamma_x = mom_x
        F_gamma_y = mom_y
        F_gamma_z = mom_z

        F_norm = jnp.sqrt(F_gamma_x**2 + F_gamma_y**2 + F_gamma_z**2)
        F_norm_safe = jnp.maximum(F_norm, self.eps)

        # Norme cible imposée par ta convention interne
        F_target = (self.light_speed ** 2) * (E_gamma_safe ** 2)

        F_gamma_x_norm = F_gamma_x / F_norm_safe * F_target
        F_gamma_y_norm = F_gamma_y / F_norm_safe * F_target
        F_gamma_z_norm = F_gamma_z / F_norm_safe * F_target

        cons_a = cons_a.at[self.vel_ids[0]].set(F_gamma_x_norm)
        cons_a = cons_a.at[self.vel_ids[1]].set(F_gamma_y_norm)
        cons_a = cons_a.at[self.vel_ids[2]].set(F_gamma_z_norm)

        cons = cons.at[self.active_slice].set(cons_a)
        return cons
    # def _eddington_factor(self, reduced_flux: Array) -> Array:
    #     f = jnp.clip(reduced_flux, 0.0, 1.0 - 1e-6)
    #     return (3.0 + 4.0 * f * f) / (5.0 + 2.0 * jnp.sqrt(4.0 - 3.0 * f * f))


    # # def _radiation_pressure_tensor(self, E_gamma: Array, F_vec: Array) -> Array:
    # #     """
    # #     Returns P_ij with M1 closure:
    # #     P = D * E
    # #     D = (1-chi)/2 I + (3chi-1)/2 n⊗n
    # #     """
        
    # #     c = self.light_speed
    # #     F_norm = jnp.sqrt(jnp.sum(F_vec * F_vec, axis=0))
    # #     reduced_flux = F_norm / (c * jnp.maximum(E_gamma, self.eps))
    # #     chi = self._eddington_factor(reduced_flux)

    # #     n = F_vec / jnp.maximum(F_norm, self.eps)

    # #     ndim = E_gamma.ndim
    # #     eye = jnp.eye(3).reshape(3, 3, *([1] * ndim))
    # #     outer = n[:, None, ...] * n[None, :, ...]

    # #     D = (
    # #         ((1.0 - chi) / 2.0)[None, None, ...] * eye
    # #         + ((3.0 * chi - 1.0) / 2.0)[None, None, ...] * outer
    # #     )

    # #     return D * E_gamma[None, None, ...]


    # def _radiation_pressure_tensor(self, E_gamma, F_vec):
    #     c = self.light_speed

    #     E_safe = jnp.where(jnp.isfinite(E_gamma) & (E_gamma > self.eps), E_gamma, self.eps)
    #     F_vec_safe = jnp.where(jnp.isfinite(F_vec), F_vec, 0.0)

    #     F_norm = jnp.sqrt(jnp.sum(F_vec_safe * F_vec_safe, axis=0))
    #     F_norm_safe = jnp.maximum(F_norm, self.eps)

    #     reduced_flux = F_norm / (c * E_safe)
    #     reduced_flux = jnp.where(jnp.isfinite(reduced_flux), reduced_flux, 0.0)

    #     chi = self._eddington_factor(reduced_flux)
    #     n = F_vec_safe / F_norm_safe

    #     ndim = E_gamma.ndim
    #     eye = jnp.eye(3).reshape(3, 3, *([1] * ndim))
    #     outer = n[:, None, ...] * n[None, :, ...]

    #     D = (
    #         ((1.0 - chi) / 2.0)[None, None, ...] * eye
    #         + ((3.0 * chi - 1.0) / 2.0)[None, None, ...] * outer
    #     )
    #     if self.debug:
    #         jax.debug.print("E_gamma has nan? {}", jnp.any(jnp.isnan(E_gamma)))
    #         jax.debug.print("F_vec has nan? {}", jnp.any(jnp.isnan(F_vec)))
    #         jax.debug.print("E_gamma min/max: {} {}", jnp.min(E_gamma), jnp.max(E_gamma))
    #         jax.debug.print("F_norm min/max: {} {}", jnp.min(F_norm), jnp.max(F_norm))

    #     return D * E_safe[None, None, ...]


    def _m1_chi_from_reduced_flux(self, fred: Array) -> Array:
        """
        M1 Eddington factor:
            chi(f) = (3 + 4 f^2) / (5 + 2 sqrt(4 - 3 f^2))

        Here fred = |F| / (c E), so fred in [0, 1].
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

        # identity tensor with broadcast shape
        I = jnp.eye(3, dtype=E_gamma.dtype).reshape(
            3, 3, *([1] * E_gamma.ndim)
        )

        n_outer = jnp.einsum("i...,j...->ij...", n_vec, n_vec)

        D = (
            0.5 * (1.0 - chi)[jnp.newaxis, jnp.newaxis, ...] * I
            + 0.5 * (3.0 * chi - 1.0)[jnp.newaxis, jnp.newaxis, ...] * n_outer
        )

        # isotropic limit when F -> 0
        D_iso = (1.0 / 3.0) * I
        mask_zero = (F_norm <= self.eps)[jnp.newaxis, jnp.newaxis, ...]
        D = jnp.where(mask_zero, D_iso, D)

        P = D * E_safe[jnp.newaxis, jnp.newaxis, ...]
        return P

    def get_radiative_pressure_tensor_from_primitives(self, primitives: Array) -> Array:
        """
        Convenience wrapper from primitive state [E, f_x, f_y, f_z].
        Returns P_ij shape: (3, 3, ...)
        """
        prim_a = primitives[self.active_slice]
        E_gamma = jnp.maximum(prim_a[self.mass_ids], self.eps)
        fx, fy, fz = (prim_a[i] for i in self.vel_ids)
        F_vec = E_gamma[jnp.newaxis, ...] * jnp.stack([fx, fy, fz], axis=0)
        return self._radiation_pressure_tensor(E_gamma, F_vec)

    # ---------------------------
    # def get_specific_energy(self, E_gamma: Array):
    #     E_gamma_safe = jnp.maximum(E_gamma, self.eps)
    #     # p_safe = jnp.maximum(p, self.eps)
    #     energy_thermique = 0.0
    #     if self.isothermal:
    #         # p_safe = self.get_isothermal_pressure(E_gamma_safe)
    #     return energy_thermique + #p_safe / ((self.gamma - 1.0) * E_gamma_safe + self.eps)

    def get_fluxes_xi(self, primitives, conservatives, axis: int): ##ATTENTION AU UNITE POUR ETRE CONSISTANT AVEC GET PRIMITIVE AND CONSERVATIVE 
        """
        RT moments (M1 closure), direction axis in {0,1,2}.
        State:
        U = [E_gamma, F_gamma_x, F_gamma_y, F_gamma_z]
        Flux in direction i:
        F_i(U) = [F_i, c^2 P_{i,x}, c^2 P_{i,y}, c^2 P_{i,z}]
        """
        del primitives
        if self.normalization:
            conservatives = self._normalize_radiative_conservatives(conservatives)
        # Cohérence d'unités: on part des variables conservatives.
        # Hypothèse conservative active: [E_gamma, E_gamma*F_x, E_gamma*F_y, E_gamma*F_z].
        cons_a = conservatives[self.active_slice]
        E_gamma = jnp.maximum(cons_a[self.mass_ids], self.eps)
        E_gamma_safe = jnp.maximum(E_gamma, self.eps)

        mom_x, mom_y, mom_z = (cons_a[i] for i in self.vel_ids)
        # mom_x = jnp.maximum(mom_x, self.eps)
        # mom_y = jnp.maximum(mom_y, self.eps)
        # mom_z = jnp.maximum(mom_z, self.eps)
        F_gamma_x = mom_x #/ E_gamma_safe
        F_gamma_y = mom_y #/ E_gamma_safe
        F_gamma_z = mom_z #/ E_gamma_safe

        F_vec = jnp.stack([F_gamma_x, F_gamma_y, F_gamma_z], axis=0)
        P = self._radiation_pressure_tensor(E_gamma, F_vec)  
        E_flux = F_vec[axis]
        c2 = self.light_speed * self.light_speed

        flux_a = jnp.stack(
            [E_flux, c2 * P[axis, 0], c2 * P[axis, 1], c2 * P[axis, 2]],
            axis=0,
        )
        if self.debug:
            jax.debug.print("flux_a: {v}", v=flux_a[0, :, :, :])
            nonzero_coords = jnp.argwhere(flux_a[0, :, :, :] != 0, size=flux_a[0].size, fill_value=-1)
            jax.debug.print("flux_a non-zero coords (i,j,k): {v}", v=nonzero_coords)
            nx, ny, nz = flux_a.shape[1], flux_a.shape[2], flux_a.shape[3]
            coords = jnp.argwhere(flux_a[0, :, :, :] != 0, size=nx * ny * nz, fill_value=-1)
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            jax.debug.print("flux_a non-zero x (padding=-1): {v}", v=x)
            jax.debug.print("flux_a non-zero y (padding=-1): {v}", v=y)
            jax.debug.print("flux_a non-zero z (padding=-1): {v}", v=z)

            mask = flux_a[0] != 0
            n_nonzero = jnp.sum(mask)

            E_gamma_safe =  E_gamma_safe > 1e-12
            ratio_FE = jnp.sqrt(F_gamma_x**2 + F_gamma_y**2 + F_gamma_z**2) / E_gamma_safe**2
            jax.debug.print("F_gamma / E_gamma**2 dans eqmana no cond min: {}", jnp.min(ratio_FE))
            jax.debug.print("F_gamma / E_gamma**2 dans eqmana no cond max: {}", jnp.max(ratio_FE))
            # if jnp.max(ratio_FE) != 4 :
            #     jax.debug.print("flux_a[0] min/max/mean: {} {} {}", jnp.min(flux_a[0]), jnp.max(flux_a[0]), jnp.mean(flux_a[0]))
            #     jax.debug.print("flux_a[0] nonzero count: {}", n_nonzero)
            #     jax.debug.print("E_gamma min/max/mean: {} {} {}", jnp.min(E_gamma), jnp.max(E_gamma), jnp.mean(E_gamma))
            #     jax.debug.print("F_gamma_x min/max/mean: {} {} {}", jnp.min(F_gamma_x), jnp.max(F_gamma_x), jnp.mean(F_gamma_x))
            #     jax.debug.print("F_gamma_y min/max/mean: {} {} {}", jnp.min(F_gamma_y), jnp.max(F_gamma_y), jnp.mean(F_gamma_y))
            #     jax.debug.print("F_gamma_z min/max/mean: {} {} {}", jnp.min(F_gamma_z), jnp.max(F_gamma_z), jnp.mean(F_gamma_z))
            #     jax.debug.print("c2*P[axis,0] min/max/mean: {} {} {}", 
            #                     jnp.min(c2 * P[axis, 0]), jnp.max(c2 * P[axis, 0]), jnp.mean(c2 * P[axis, 0]))
            #     jax.debug.print("c2*P[axis,1] min/max/mean: {} {} {}", 
            #                     jnp.min(c2 * P[axis, 1]), jnp.max(c2 * P[axis, 1]), jnp.mean(c2 * P[axis, 1]))
            #     jax.debug.print("c2*P[axis,2] min/max/mean: {} {} {}", 
            #                     jnp.min(c2 * P[axis, 2]), jnp.max(c2 * P[axis, 2]), jnp.mean(c2 * P[axis, 2]))
            #     jax.debug.print("F_gamma / E_gamma dans eqmana min: {}", jnp.min(ratio_FE))
            #     jax.debug.print("F_gamma / E_gamma dans eqmana max: {}", jnp.max(ratio_FE))
            ratio_FE = jnp.sqrt(F_gamma_x**2 + F_gamma_y**2 + F_gamma_z**2) / E_gamma_safe**2
            pred = jnp.max(ratio_FE) != 4

            def true_branch(_):
                jax.debug.print("flux_a[0] min/max/mean: {} {} {}", jnp.min(flux_a[0]), jnp.max(flux_a[0]), jnp.mean(flux_a[0]))
                jax.debug.print("flux_a[0] nonzero count: {}", n_nonzero)
                jax.debug.print("E_gamma min/max/mean: {} {} {}", jnp.min(E_gamma), jnp.max(E_gamma), jnp.mean(E_gamma))
                jax.debug.print("F_gamma_x min/max/mean: {} {} {}", jnp.min(F_gamma_x), jnp.max(F_gamma_x), jnp.mean(F_gamma_x))
                jax.debug.print("F_gamma_y min/max/mean: {} {} {}", jnp.min(F_gamma_y), jnp.max(F_gamma_y), jnp.mean(F_gamma_y))
                jax.debug.print("F_gamma_z min/max/mean: {} {} {}", jnp.min(F_gamma_z), jnp.max(F_gamma_z), jnp.mean(F_gamma_z))
                jax.debug.print("c2*P[axis,0] min/max/mean: {} {} {}", 
                                jnp.min(c2 * P[axis, 0]), jnp.max(c2 * P[axis, 0]), jnp.mean(c2 * P[axis, 0]))
                jax.debug.print("c2*P[axis,1] min/max/mean: {} {} {}", 
                                jnp.min(c2 * P[axis, 1]), jnp.max(c2 * P[axis, 1]), jnp.mean(c2 * P[axis, 1]))
                jax.debug.print("c2*P[axis,2] min/max/mean: {} {} {}", 
                                jnp.min(c2 * P[axis, 2]), jnp.max(c2 * P[axis, 2]), jnp.mean(c2 * P[axis, 2]))
                jax.debug.print("F_gamma / E_gamma**2 dans eqmana min: {}", jnp.min(ratio_FE))
                jax.debug.print("F_gamma / E_gamma**2 dans eqmana max: {}", jnp.max(ratio_FE))
                return 0

            def false_branch(_):
                return 0

            jax.lax.cond(pred, true_branch, false_branch, operand=None)
        if conservatives.shape[0] > self.n_active:
            # Passive block transporté de manière upwind dans ConvectiveFlux
            return jnp.concatenate([flux_a, conservatives[self.passive_slice]], axis=0)
        if self.normalization:
            conservatives = self._normalize_radiative_conservatives(conservatives)
        return flux_a
   

    def get_energy_grid(self, primitives: Array) -> Array:
        """
        Retourne E sur toute la grille.
        primitives shape: (nvar, Nx, Ny, Nz)
        return shape: (Nx, Ny, Nz)
        """
        prim_a = primitives[self.active_slice]
        return jnp.maximum(prim_a[self.mass_ids], self.eps)

    def get_F_over_E_components_grid(self, primitives: Array) -> Array:
        """
        Retourne F/E composante par composante sur toute la grille.
        Dans ta convention actuelle, c'est directement (f_x, f_y, f_z).

        return shape: (3, Nx, Ny, Nz)
        """
        prim_a = primitives[self.active_slice]
        fx, fy, fz = (prim_a[i] for i in self.vel_ids)
        return jnp.stack([fx, fy, fz], axis=0)

    def get_F_components_grid(self, primitives: Array) -> Array:
        """
        Retourne F = E * (F/E) sur toute la grille.

        return shape: (3, Nx, Ny, Nz)
        """
        E = self.get_energy_grid(primitives)
        F_over_E = self.get_F_over_E_components_grid(primitives)
        return E[jnp.newaxis, ...] * F_over_E

    def get_F_over_E_norm_grid(self, primitives: Array) -> Array:
        """
        Retourne |F|/E sur toute la grille.

        return shape: (Nx, Ny, Nz)
        """
        F_over_E = self.get_F_over_E_components_grid(primitives)
        return jnp.sqrt(jnp.sum(F_over_E * F_over_E, axis=0))

    def get_reduced_flux_grid(self, primitives: Array) -> Array:
        """
        Retourne |F|/(cE) sur toute la grille.

        return shape: (Nx, Ny, Nz)
        """
        return self.get_F_over_E_norm_grid(primitives) / self.light_speed

    # def get_fluxes_xi(self, primitives, conservatives, axis: int): #version chat
    #     prim_a = primitives[self.active_slice]
    #     # cons_a = conservatives[self.active_slice]

    #     E_gamma = jnp.maximum(prim_a[self.mass_ids], self.eps)

    #     f_x, f_y, f_z = (prim_a[i] for i in self.vel_ids)
    #     vel = (f_x,f_y,f_z)
    #     ui = vel[axis]
    #     E_gamma_ui = E_gamma * ui
    #     F_gamma_x = E_gamma_ui * f_x
    #     F_gamma_y = E_gamma_ui * f_y
    #     F_gamma_z = E_gamma_ui * f_z
        

    #     F_vec = jnp.stack([F_gamma_x, F_gamma_y, F_gamma_z], axis=0)

    #     P = self._radiation_pressure_tensor(E_gamma, F_vec)

    #     c2 = self.light_speed * self.light_speed
    #     flux_a = jnp.stack(
    #         [F_vec[axis], c2 * P[axis, 0], c2 * P[axis, 1], c2 * P[axis, 2]],
    #         axis=0,
    #     )

    #     if conservatives.shape[0] > self.n_active:
    #         return jnp.concatenate([flux_a, conservatives[self.passive_slice]], axis=0)

    #     return flux_a

#     def get_isothermal_pressure(self, E_gamma):
#         E_gamma_safe = jnp.maximum(E_gamma, self.eps)
#         cs2 = self.light_speed * self.light_speed
#         return cs2 * E_gamma_safe

#     # ---------------------------
#     # Thermodynamics




#     def get_sound_speed(self, p, E_gamma):
#         E_gamma_safe = jnp.maximum(E_gamma, self.eps)
#         if self.isothermal:
#             return jnp.full_like(E_gamma_safe, self.light_speed)
#         p_safe = jnp.maximum(p, self.eps)
#         return jnp.sqrt(self.gamma * p_safe / E_gamma_safe)

#     # ---------------------------
#     # Primitive <-> Conservative
#     # ---------------------------
    
#     # ---------------------------
#     # Physical fluxes
#     # ---------------------------

#     # ---------------------------
#     # Wavespeeds (for CFL / solvers)
#     # ---------------------------
#     def get_wavespeeds_xi(self, primitives, axis: int):
#         prim_a = primitives[self.active_slice]
#         E_gamma = prim_a[self.mass_ids]
#         p = prim_a[self.energy_ids]
#         ui = prim_a[self.vel_ids[axis]]
#         a = self.get_sound_speed(p, E_gamma)
#         return ui - a, ui + a

#     def get_specific_heat_capacity(self, T: Array): #-> Union[float, Array]:
#         """Calculates the specific heat coefficient per unit mass.
#         [c_p] = J / kg / K

#         :param T: _description_
#         :type T: Array
#         :raises NotImplementedError: _description_
#         :return: _description_
#         :rtype: Array
#         """
#         return self.cp

#     def get_specific_heat_ratio(self, T: Array): #-> Union[float, Array]:
#         """Calculates the specific heat ratio.

#         :param T: _description_
#         :type T: Array
#         :raises NotImplementedError: _description_
#         :return: _description_
#         :rtype: Array
#         """
#         return self.gamma

#     def get_E_gamma(self, p: Array, E_gamma: Array) -> Array:
#         """See base class. """
#         return p / E_gamma

#     def get_grueneisen(self, E_gamma: Array, T: Array = None) -> Array:
#         """See base class. """
#         return self.gamma - 1


#     def get_pressure(self, e, E_gamma):
#         if self.isothermal:
#             return self.get_isothermal_pressure(E_gamma)
#         return (self.gamma - 1.0) * jnp.maximum(e, self.eps) * jnp.maximum(E_gamma, self.eps)

#     def get_temperature(self, p, E_gamma):
#         p_safe   = jnp.maximum(p,   self.eps)
#         E_gamma_safe = jnp.maximum(E_gamma, self.eps)
#         T = p_safe / (E_gamma_safe * self.R + self.eps)
#         # cap used temp to keep any downstream log/exp/table sane
#         return jnp.clip(T, 0.10, 10.0**10.5) 

#     def get_specific_energy(self, p, E_gamma):
#         if self.isothermal:
#             p_safe = self.get_isothermal_pressure(E_gamma)
#         else:
#             p_safe = jnp.maximum(p, self.eps)
#         E_gamma_safe = jnp.maximum(E_gamma, self.eps)
#         return p_safe / (E_gamma_safe * (self.gamma - 1.0) + self.eps)

#     def project_conservatives_to_eos(self, conservatives):
#         if not self.isothermal:
#             return conservatives
#         primitives = self.get_primitives_from_conservatives(conservatives)
#         return self.get_conservatives_from_primitives(primitives)

#     def get_total_energy(
#             self,
#             p:Array,
#             E_gamma:Array,
#             u:Array,
#             v:Array,
#             w:Array
#             ) -> Array:
#         """See base class. """
#         # Total energy per unit volume
#         # (sensible, i.e., without heat of formation)
#         return p / (self.gamma - 1) + 0.5 * E_gamma * ( (u * u + v * v + w * w) )

#     def get_total_enthalpy(
#             self,
#             p:Array,
#             E_gamma:Array,
#             u:Array,
#             v:Array,
#             w:Array
#             ) -> Array:
#         """See base class. """
#         # Total specific enthalpy
#         # (sensible, i.e., without heat of formation)
#         return (self.get_total_energy(p, E_gamma, u, v, w) + p) / E_gamma

#     def get_stagnation_temperature(
#             self,
#             p:Array,
#             E_gamma:Array,
#             u:Array,
#             v:Array,
#             w:Array
#         ) -> Array:
#         T = self.get_temperature(p, E_gamma)
#         cp = self.get_specific_heat_capacity(T)
#         return T + 0.5 * (u * u + v * v + w * w) / cp


    
#     def _set_transport_properties(self,func) -> None:

#         if self.thermal_conductivity_model is not None:
#             if self.thermal_conductivity_model == "CUSTOM":
#                 self.thermal_conductivity_fun = func

#             elif self.thermal_conductivity_model == "SUTHERLAND":
#                 sutherland_parameters = self.sutherland_parameters
#                 self.kappa_ref = sutherland_parameters[0]
#                 self.T_ref_kappa = sutherland_parameters[1]
#                 self.C_kappa = sutherland_parameters[2]
#             else:
#                 raise
#         else:
#             raise
    
#     def get_thermal_conductivity_old(
#             self,
#             temperature: Array,
#             primitives: Array,
#             density: Array = None,
#             partial_densities: Array = None,
#             volume_fractions: Array = None,
#         ) -> Array:
#         """Computes the thermal conductivity

#         :param T: _description_
#         :type T: Array
#         :raises NotImplementedError: _description_
#         :return: _description_
#         :rtype: Array
#         """
        
#         T = temperature
#  #       checkify.check(jax.numpy.all(T>0), "temperature must be non-negative, got {i}", i=T.min())

            
#         if self.thermal_conductivity_model == "CUSTOM":
#             thermal_conductivity = self.thermal_conductivity_fun(T)
    
#         elif self.thermal_conductivity_model == "SUTHERLAND":
#             t_1 = ((self.T_ref_kappa + self.C_kappa)/(T + self.C_kappa))
#             t_2 = (T/self.T_ref_kappa)**1.5
#             thermal_conductivity = \
#                 self.kappa_ref * t_1 * t_2
#         elif self.thermal_conductivity_model == "ELBADRY":
#             ALPHA= 813.142554365 
#             # 1e-20/(1.4mH)^2 erg*s^{-1}*cm^3 in simulation unit
#             # (cooling, Sutherland & Dopita 1993)
#             BETA =406571277.182611837
#             #1e-20/(1.4mH)^2 erg*s^{-1}*cm^3 in simulation unit
#             #heating, Kim & Ostriker 2015)
#             GAMMA= 406.571277183
#             # 1.4mH/cm^3 in simulation unit
#             #(hydrogen number density, Kim & Ostriker 2015)
#             DELTA=0.03459841649374997
#             # 1.2/(1.4mH)*1e2 cm^3 in simulation unit
#             #(electron number density, El-Badry 2019)
#             EPSILON=3468.366826027353
#             # erg*s^{-1}*cm^{-1} in simulation unit
#             #thermal conductivity, El-Badry 2019)
#             ZETA=5.111496271545331E-12
#             #hydrogen mass over Boltzmann constant in simulation unit
#             MHKB=115.98518596699539

#             E_gamma = primitives[self.mass_ids]
#             p   = primitives[self.energy_ids]
            
#             # Physical temperature like Athena (Kelvin)
#             # numeric factor mirrors their 1.272727*MHKB*p/E_gamma
#             T_phys = 1.272727 * MHKB * p / (E_gamma + self.eps)
            
#             # Hot vs. cool branches (Spitzer / Parker)
#             temp7 = T_phys / 1.0e7
#             ne2   = jnp.maximum(EPSILON * E_gamma, self.eps)  # guard
            
#             kappa_hot  = (1.7e11 * temp7**2.5) / (1.0 + 0.029 * jnp.log(temp7 / jnp.sqrt(ne2)))  # NATURAL log
#             kappa_cool = 2.5e5 * jnp.sqrt(jnp.maximum(T_phys / 1.0e4, 0.0))
            
#             kappa = jnp.where(T_phys > 6.6e4, kappa_hot, kappa_cool)
            
#             # Athena "adjust for units": multiply by 1.4*MHKB/E_gamma
#             kappa = kappa * 1.4 * MHKB / (E_gamma + self.eps)
            
#             # Apply El-Badry ceiling here (saturation comes later)
#             k_ceiling = 1.8e12/ DELTA * 1.4 * MHKB #1.8e12 / DELTA * 1.4 * MHKB
#             thermal_conductivity = jnp.minimum(kappa, k_ceiling)
#         else:
#             raise NotImplementedError

#         return thermal_conductivity
    
#     def get_thermal_conductivity(self, temperature, primitives, 
#                                            density=None, partial_densities=None, 
#                                            volume_fractions=None):
#         """
#         Corrected thermal conductivity matching Athena++ implementation.
#         This returns κ in "code units" that already includes unit adjustments.
#         """
#         T = temperature

#         if self.thermal_conductivity_model == "ELBADRY":
#             # Constants (same as Athena++)
#             MHKB = 115.98518596699539
#             EPSILON = 3468.366826027353

#             E_gamma = primitives[self.mass_ids]
#             p   = primitives[self.energy_ids]

#             # Temperature in Kelvin (matching Athena++ line 416)
#             T_phys = 1.272727 * MHKB * p / (E_gamma + self.eps)

#             # Hot vs. cool branches (matching Athena++ lines 418-427)
#             temp7 = T_phys / 1.0e7
#             temp4 = T_phys / 1.0e4
#             ne2   = EPSILON * E_gamma

#             # Spitzer conductivity for hot gas (T > 6.6e4 K)
#             kappa_hot = (1.7e11 * temp7**2.5) / (1.0 + 0.029 * jnp.log(temp7 / jnp.sqrt(ne2)))

#             # Parker conductivity for cool gas (T < 6.6e4 K)
#             kappa_cool = 2.5e5 * jnp.sqrt(temp4)

#             # Branch selection
#             kappa = jnp.where(T_phys > 6.6e4, kappa_hot, kappa_cool)

#             # Adjust for Athena++ units (line 430)
#             # This converts from physical units to code units
#             kappa = kappa * 1.4 * MHKB / (E_gamma + self.eps)

#             return kappa

#         elif self.thermal_conductivity_model == "SUTHERLAND":
#             t_1 = ((self.T_ref_kappa + self.C_kappa)/(T + self.C_kappa))
#             t_2 = (T/self.T_ref_kappa)**1.5
#             thermal_conductivity = self.kappa_ref * t_1 * t_2
#             return thermal_conductivity

#         else:
#             raise NotImplementedError