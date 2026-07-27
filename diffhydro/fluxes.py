from jax import Array
from functools import partial
from typing import List
from .solver import recon
from .solver.stencils import *
from .physics import mhd
from .utils.debug_checks import _check_finite
import jax.numpy as jnp
import jax
from jax.experimental import checkify


class ConvectiveFlux:
    def __init__(self,
                 EquationManager,
                 Solver,
                 Recon,
                 positivity=True,
                 dx=1,
                 ):
        self.eq_manage = EquationManager
        self.solver = Solver
        self.recon = Recon

        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        # Taille de cellule en unites code : doit etre la meme que hydro(dx=...).
        self.dx_o = dx

        try:  # 3d
            self.flux_shapes = (
                EquationManager.n_cons,
                EquationManager.mesh_shape[0],
                EquationManager.mesh_shape[1],
                EquationManager.mesh_shape[2]
            )
        except:  # 2d
            self.flux_shapes = (
                EquationManager.n_cons,
                EquationManager.mesh_shape[0],
                EquationManager.mesh_shape[1]
            )

    def flux(self, sol, ax, params, flux):
        eq = self.eq_manage

        # cell primitives
        primitives = eq.get_primitives_from_conservatives(sol)
        
        # quick finite check on primitives before reconstruction
        _check_finite("primitives", primitives)
        primitives_xi_L = self.recon.reconstruct_xi(
            primitives,
            axis=ax,
            j=0)

        primitives_xi_R = self.recon.reconstruct_xi(
            primitives,
            axis=ax,
            j=1)

        # convert reconstructed primitives -> conservatives (full state)
        conservative_xi_L = eq.get_conservatives_from_primitives(primitives_xi_L)
        conservative_xi_R = eq.get_conservatives_from_primitives(primitives_xi_R)

        # positivity fix (full state)
        if self.positivity:
            conservative_xi_L, primitives_xi_L, count_L = self.compute_positivity_preserving_interpolation(
                primitives=primitives,
                primitives_xi_j=primitives_xi_L,
                j=0,
                axis=ax)
            conservative_xi_R, primitives_xi_R, count_R = self.compute_positivity_preserving_interpolation(
                primitives=primitives,
                primitives_xi_j=primitives_xi_R,
                j=1,
                axis=ax)

        # -------------------------------
        # Active block for the solver
        # -------------------------------
        prim_L_act = primitives_xi_L[eq.active_slice]
        prim_R_act = primitives_xi_R[eq.active_slice]
        cons_L_act = conservative_xi_L[eq.active_slice]
        cons_R_act = conservative_xi_R[eq.active_slice]
        
        # # CHAT GPT RAJOUTE CA BIZZARE
        # # Defensive alignment: RT manager variants may temporarily build 
        # # inconsistent active sizes during experimentation. Keep L/R identical.
        # n_act = min(
        #     prim_L_act.shape[0],
        #     prim_R_act.shape[0],
        #     cons_L_act.shape[0],
        #     cons_R_act.shape[0],
        # )
        # prim_L_act = prim_L_act[:n_act]
        # prim_R_act = prim_R_act[:n_act]
        # cons_L_act = cons_L_act[:n_act]
        # cons_R_act = cons_R_act[:n_act]

        # Existing GLM-MHD special case:
        # n_cons == 9 means active MHD (8 vars) + ψ scalar.
        glm_active = (getattr(eq, "n_cons", conservative_xi_L.shape[0]) == 9) #see article 

        # Solve Riemann on active-only
        F_act, _, _ = self.solver.solve_riemann_problem_xi(
            prim_L_act, prim_R_act,
            cons_L_act, cons_R_act, ax-1
        ) # _ means ignore the returned values, and ax-1 changes the indexing base we calculate here the problems at the interface with godunov

        F = F_act

        # -------------------------------
        # GLM ψ handling (unchanged logic,
        # just no hard-coded solver slicing)
        # -------------------------------
        if glm_active:
            c_h = getattr(eq, "glm_ch", 1.0)
            F_Bn, F_psi = mhd.glm_face_flux(primitives_xi_L, primitives_xi_R, ax-1, c_h)

            # normal B index in the *active* MHD block
            # (this is still MHD-physics-specific)
            Bn_idx = (4, 5, 6)[ax-1]
            F = F.at[Bn_idx].set(F_Bn)

            # append ψ flux -> makes 9 rows total at this stage
            F = jnp.vstack([F, F_psi[jnp.newaxis, :]])

        # -------------------------------
        # Generic passive scalars
        # Anything beyond the active block
        # (and beyond ψ if GLM) is passive.
        # -------------------------------
        if glm_active:
            # ψ is the first scalar beyond the active MHD block
            passive_start = eq.n_active + 1
        else:
            passive_start = eq.n_active

        if conservative_xi_L.shape[0] > passive_start: # calculation of passive fluxes that is to say quantities that have no effect on dynamics and are transported by mass flux
            # mass flux from active solver (row 0 always rho*u_n)
            mass_flux = F_act[eq.mass_ids]  # shape (...)

            passive_L = primitives_xi_L[passive_start:]
            passive_R = primitives_xi_R[passive_start:]

            passive_face = jnp.where(
                mass_flux[jnp.newaxis, ...] >= 0.0,
                passive_L, passive_R
            )

            F_passive = mass_flux[jnp.newaxis, ...] * passive_face
            F = jnp.concatenate([F, F_passive], axis=0)

        return F

    def timestep(self, sol):
        """
        Multi-D CFL timestep for unsplit FV:
          dt = CFL / max_x,y,z ( a_x/dx + a_y/dy + a_z/dz )
        with a_d = |v_d| + signal_speed(prims, axis=d)
        """
        eq = self.eq_manage
        prim = eq.get_primitives_from_conservatives(sol)

        # velocities from active block (no hard-coded 1,2,3)
        u = prim[eq.vel_ids[0]]
        v = prim[eq.vel_ids[1]] if len(eq.vel_ids) > 1 else 0.0
        w = prim[eq.vel_ids[2]] if len(eq.vel_ids) > 2 else 0.0

        a_x = jnp.abs(u) + eq.get_signal_speed(prim, axis=0)
        a_y = jnp.abs(v) + eq.get_signal_speed(prim, axis=1) if sol.ndim >= 3 else 0.0
        a_z = jnp.abs(w) + eq.get_signal_speed(prim, axis=2) if sol.ndim >= 4 else 0.0

        dx = float(self.dx_o)
        dy = float(self.dx_o)
        dz = float(self.dx_o)

        inv_dt_local = a_x / dx
        if sol.ndim >= 3:
            inv_dt_local = inv_dt_local + a_y / dy
        if sol.ndim >= 4:
            inv_dt_local = inv_dt_local + a_z / dz

        inv_dt_max = jnp.max(inv_dt_local)

        dt = eq.cfl / (inv_dt_max + eq.eps)
        return dt

    def compute_positivity_preserving_interpolation(self,
                                                    primitives: Array,
                                                    primitives_xi_j: Array,
                                                    j: int,
                                                    axis: int):

        cell_state_xi_safe_j = self.positivity_stencil.reconstruct_xi(
            primitives, axis, j)

        rho_j = primitives_xi_j[self.eq_manage.mass_ids]

        mask = jnp.where(rho_j < self.eq_manage.eps, 0, 1)
        counter = jnp.sum(1 - mask)

        primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

        # check pressure on active p slot
        p_j = primitives_xi_j[self.eq_manage.energy_ids]
        mask = jnp.where(p_j < self.eq_manage.eps, 0, 1)

        counter += jnp.sum(1 - mask)
        primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

        conservative_xi_j = self.eq_manage.get_conservatives_from_primitives(primitives_xi_j)

        return conservative_xi_j, primitives_xi_j, counter


class ConductiveFlux:
    def __init__(self,
                 EquationManager,
                 Solver,
                 Recon,
                 positivity=False,
                 zeta=0,
                 dx=1,
                 ):
        self.eq_manage = EquationManager
        self.solver = Solver
        self.recon_heat = CentralSecondOrderReconstruction()
        self.zeta = zeta
        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        # Taille de cellule en unites code : doit etre la meme que hydro(dx=...).
        self.dx_o = dx

        try:  # 3d
            self.flux_shapes = (
                EquationManager.n_cons,
                EquationManager.mesh_shape[0],
                EquationManager.mesh_shape[1],
                EquationManager.mesh_shape[2]
            )
        except:  # 2d
            self.flux_shapes = (
                EquationManager.n_cons,
                EquationManager.mesh_shape[0],
                EquationManager.mesh_shape[1]
            )

    def flux(self, sol, axis, params, flux):
        """
        Physically consistent conductive flux for finite-volume hydro.
        Conduction only affects active energy slot; passives are untouched.
        """
        eq = self.eq_manage
        prim = eq.get_primitives_from_conservatives(sol)

        rho = prim[eq.mass_ids]
        p = prim[eq.energy_ids]
        T = eq.get_temperature(p, rho)
        ax0 = axis - 1  # 0-based axis

        try:
            k_center = eq.get_thermal_conductivity(T, prim, None, None, None)
        except TypeError:
            k_center = eq.get_thermal_conductivity(T, prim)

        kL = k_center
        kR = jnp.roll(k_center, -1, axis=ax0)
        k_face = 2.0 * kL * kR / (kL + kR + 1e-30)

        dT_dxi_face = (jnp.roll(T, -1, axis=ax0) - T) / self.dx_o

        model = getattr(eq, "thermal_conductivity_model", "")
        if model == "ELBADRY":
            rho_face = 0.5 * (rho + jnp.roll(rho, -1, axis=ax0))
            p_face = 0.5 * (p + jnp.roll(p, -1, axis=ax0))
            T_face = 0.5 * (T + jnp.roll(T, -1, axis=ax0))

            dTdx = (jnp.roll(T, -1, axis=0) - T) / self.dx_o if (sol.ndim - 1) >= 1 else 0.0
            dTdy = (jnp.roll(T, -1, axis=1) - T) / self.dx_o if (sol.ndim - 1) >= 2 else 0.0
            dTdz = (jnp.roll(T, -1, axis=2) - T) / self.dx_o if (sol.ndim - 1) >= 3 else 0.0
            gradT_mag_face = jnp.sqrt(dTdx**2 + dTdy**2 + dTdz**2) + 1e-30

            cs_face = eq.get_speed_of_sound(p_face, rho_face)

            kappa_max = 1.5 * cs_face**3 / (gradT_mag_face * self.zeta)

            k_face_used = 1.0 / (1.0/(k_face + 1e-30) + 1.0/(kappa_max + 1e-30))
        else:
            k_face_used = -k_face

        heat_flux_xi = k_face_used * dT_dxi_face

        conductive_flux = jnp.zeros_like(sol)
        conductive_flux = conductive_flux.at[eq.energy_ids].set(heat_flux_xi)
        return conductive_flux

    def timestep(self, sol):
        const = 0.1
        dx2 = float(self.dx_o) ** 2

        prim = self.eq_manage.get_primitives_from_conservatives(sol)
        rho = prim[self.eq_manage.mass_ids]
        p = prim[self.eq_manage.energy_ids]
        T = self.eq_manage.get_temperature(p, rho)
        cp_m = self.eq_manage.get_specific_heat_capacity(T)

        kappa_unsat = self.eq_manage.get_thermal_conductivity(T, prim)

        model = getattr(self.eq_manage, "thermal_conductivity_model", "")

        if model == "ELBADRY":
            dTdx = self.recon_heat.derivative_xi(T, 0) if (sol.ndim - 1) >= 1 else 0.0
            dTdy = self.recon_heat.derivative_xi(T, 1) if (sol.ndim - 1) >= 2 else 0.0
            dTdz = self.recon_heat.derivative_xi(T, 2) if (sol.ndim - 1) >= 3 else 0.0
            gradT = jnp.sqrt(dTdx*dTdx + dTdy*dTdy + dTdz*dTdz) + 1e-30

            cs = self.eq_manage.get_speed_of_sound(p, rho)
            kappa_max = 1.5 * cs**3 / (gradT * self.zeta)

            kappa_used = 1.0 / (1.0/(kappa_unsat + 1e-30) + 1.0/(kappa_max + 1e-30)) * self.zeta * 75
            denom = jnp.maximum(cp_m, 1e-30)
        else:
            kappa_used = kappa_unsat
            rho_floor = jnp.maximum(rho, 1e-6)
            denom = jnp.maximum(rho_floor * cp_m, 1e-30)

        chi = kappa_used / denom
        chi_max = jnp.max(chi)

        return const * dx2 / (chi_max + 1e-30)
    

    from jax import Array

# class ConvectiveFlux_Radiative_transfer:
#     def __init__(self,
#                  EquationManager,
#                  Solver,
#                  Recon,
#                  positivity=True
#                  ):
#         self.eq_manage = EquationManager
#         self.solver = Solver
#         self.recon = Recon

#         self.positivity = positivity
#         self.positivity_stencil = recon.WENO1()
#         self.dx_o = 1

#         try:  # 3d
#             self.flux_shapes = (
#                 EquationManager.n_cons,
#                 EquationManager.mesh_shape[0],
#                 EquationManager.mesh_shape[1],
#                 EquationManager.mesh_shape[2]
#             )
#         except:  # 2d
#             self.flux_shapes = (
#                 EquationManager.n_cons,
#                 EquationManager.mesh_shape[0],
#                 EquationManager.mesh_shape[1]
#             )

#     def flux(self, sol, ax, params, flux):
#         eq = self.eq_manage

#         # cell primitives
#         primitives = eq.get_primitives_from_conservatives(sol)

#         # quick finite check on primitives before reconstruction
#         _check_finite("primitives", primitives)
#         primitives_xi_L = self.recon.reconstruct_xi(
#             primitives,
#             axis=ax,
#             j=0)

#         primitives_xi_R = self.recon.reconstruct_xi(
#             primitives,
#             axis=ax,
#             j=1)

#         # convert reconstructed primitives -> conservatives (full state)
#         conservative_xi_L = eq.get_conservatives_from_primitives(primitives_xi_L)
#         conservative_xi_R = eq.get_conservatives_from_primitives(primitives_xi_R)

#         # positivity fix (full state)
#         if self.positivity:
#             conservative_xi_L, primitives_xi_L, count_L = self.compute_positivity_preserving_interpolation(
#                 primitives=primitives,
#                 primitives_xi_j=primitives_xi_L,
#                 j=0,
#                 axis=ax)
#             conservative_xi_R, primitives_xi_R, count_R = self.compute_positivity_preserving_interpolation(
#                 primitives=primitives,
#                 primitives_xi_j=primitives_xi_R,
#                 j=1,
#                 axis=ax)

#         # -------------------------------
#         # Active block for the solver
#         # -------------------------------
#         prim_L_act = primitives_xi_L[eq.active_slice]
#         prim_R_act = primitives_xi_R[eq.active_slice]
#         cons_L_act = conservative_xi_L[eq.active_slice]
#         cons_R_act = conservative_xi_R[eq.active_slice]

#         # Existing GLM-MHD special case:
#         # n_cons == 9 means active MHD (8 vars) + ψ scalar.
#         glm_active = (getattr(eq, "n_cons", conservative_xi_L.shape[0]) == 9) #see article 

#         # Solve Riemann on active-only
#         F_act, _, _ = self.solver.solve_riemann_problem_xi(
#             prim_L_act, prim_R_act,
#             cons_L_act, cons_R_act, ax-1
#         ) # _ signifie ignore les veleurs retourner, et ax-1 change l indexation de la base on calcule ici les problemes a l interface avec godunov

#         F = F_act

#         # -------------------------------
#         # GLM ψ handling (unchanged logic,
#         # just no hard-coded solver slicing)
#         # -------------------------------
#         if glm_active:
#             c_h = getattr(eq, "glm_ch", 1.0)
#             F_Bn, F_psi = mhd.glm_face_flux(primitives_xi_L, primitives_xi_R, ax-1, c_h)

#             # normal B index in the *active* MHD block
#             # (this is still MHD-physics-specific)
#             Bn_idx = (4, 5, 6)[ax-1]
#             F = F.at[Bn_idx].set(F_Bn)

#             # append ψ flux -> makes 9 rows total at this stage
#             F = jnp.vstack([F, F_psi[jnp.newaxis, :]])

#         # -------------------------------
#         # Generic passive scalars
#         # Anything beyond the active block
#         # (and beyond ψ if GLM) is passive.
#         # -------------------------------
#         if glm_active:
#             # ψ is the first scalar beyond the active MHD block
#             passive_start = eq.n_active + 1
#         else:
#             passive_start = eq.n_active

#         if conservative_xi_L.shape[0] > passive_start: #calcule des flux passif c est a dire des grandeurs qui n ont pas d influence sur la dynamique et qui sont transportees par le flux de masse
#             # mass flux from active solver (row 0 always rho*u_n)
#             mass_flux = F_act[eq.mass_ids]  # shape (...)

#             passive_L = primitives_xi_L[passive_start:]
#             passive_R = primitives_xi_R[passive_start:]

#             passive_face = jnp.where(
#                 mass_flux[jnp.newaxis, ...] >= 0.0,
#                 passive_L, passive_R
#             )

#             F_passive = mass_flux[jnp.newaxis, ...] * passive_face
#             F = jnp.concatenate([F, F_passive], axis=0)

#         return F

#     # def timestep(self, sol):
#     #     """
#     #     Multi-D CFL timestep for unsplit FV:
#     #       dt = CFL / max_x,y,z ( a_x/dx + a_y/dy + a_z/dz )
#     #     with a_d = |v_d| + signal_speed(prims, axis=d)
#     #     """
#     #     eq = self.eq_manage
#     #     prim = eq.get_primitives_from_conservatives(sol)

#     #     # velocities from active block (no hard-coded 1,2,3)
#     #     u = prim[eq.vel_ids[0]]
#     #     v = prim[eq.vel_ids[1]] if len(eq.vel_ids) > 1 else 0.0
#     #     w = prim[eq.vel_ids[2]] if len(eq.vel_ids) > 2 else 0.0

#     #     a_x = jnp.abs(u) + eq.get_signal_speed(prim, axis=0)
#     #     a_y = jnp.abs(v) + eq.get_signal_speed(prim, axis=1) if sol.ndim >= 3 else 0.0
#     #     a_z = jnp.abs(w) + eq.get_signal_speed(prim, axis=2) if sol.ndim >= 4 else 0.0

#     #     dx = float(self.dx_o)
#     #     dy = float(self.dx_o)
#     #     dz = float(self.dx_o)

#     #     inv_dt_local = a_x / dx
#     #     if sol.ndim >= 3:
#     #         inv_dt_local = inv_dt_local + a_y / dy
#     #     if sol.ndim >= 4:
#     #         inv_dt_local = inv_dt_local + a_z / dz

#     #     inv_dt_max = jnp.max(inv_dt_local)

#     #     dt = eq.cfl / (inv_dt_max + eq.eps)
#     #     return dt
   
#     def timestep(self, sol): #new by the chat not bad
#         """
#         CFL RT: dt = cfl / sum_d(c/dx_d)
#         """
#         eq = self.eq_manage
#         dim = sol.ndim - 1
#         dx = float(self.dx_o)
#         c = float(eq.light_speed)
    
#         inv_dt = dim * c / dx
#         print(eq.cfl / (inv_dt + eq.eps))
    
#         return eq.cfl / (inv_dt + eq.eps) 
    
#     def compute_positivity_preserving_interpolation(self,
#                                                 primitives: Array,
#                                                 primitives_xi_j: Array,
#                                                 j: int,
#                                                 axis: int):

#         cell_state_xi_safe_j = self.positivity_stencil.reconstruct_xi(
#         primitives, axis, j
#         )

#         E_j = primitives_xi_j[self.eq_manage.mass_ids]
#         mask = jnp.where(E_j < self.eq_manage.eps, 0, 1)
#         counter = jnp.sum(1 - mask)

#         primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)
#         conservative_xi_j = self.eq_manage.get_conservatives_from_primitives(primitives_xi_j)

#         return conservative_xi_j, primitives_xi_j, counter


class ConvectiveFlux_Radiative_transfer:
    def __init__(self,
                EquationManager,
                Solver,
                Recon,
                positivity=True,
                dx=1,
                ):
        self.eq_manage = EquationManager
        self.solver = Solver
        self.recon = Recon

        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        self.dx_o = dx

        try:  # 3d
            self.flux_shapes = (
                EquationManager.n_cons,
                EquationManager.mesh_shape[0],
                EquationManager.mesh_shape[1],
                EquationManager.mesh_shape[2]
            )
        except:  # 2d
            self.flux_shapes = (
                EquationManager.n_cons,
                EquationManager.mesh_shape[0],
                EquationManager.mesh_shape[1]
            )
    # def _debug_grid_stats(self, sol, eq, label, ax):
    #     E = jnp.maximum(sol[0], eq.eps)
    #     Fx = sol[1]
    #     Fy = sol[2]
    #     Fz = sol[3]

    #     Fmag = jnp.sqrt(Fx**2 + Fy**2 + Fz**2)
    #     ratio = Fmag / jnp.maximum(E**2, 1e-30)
    #     if jnp.any(sol[0] > eq.eps):
    #         jax.debug.print(
    #         """
    #         [{label}] ax={ax}
    #         shape = {shape}

    #         E:
    #         {label} mean={E_mean} {label} E min={E_min} max={E_max}
    #         any nan? {E_nan}
    #         any inf? {E_inf}
    #         any nonfinite? {E_bad}

    #         Fx:
    #         {label} mean={Fx_mean} {label} Fx min={Fx_min} max={Fx_max}
    #         Fy:
    #         {label} mean={Fy_mean} {label} Fy min={Fy_min} max={Fy_max}
    #         Fz:
    #         {label} mean={Fz_mean} {label} Fz min={Fz_min} max={Fz_max}

    #         |F|:
    #         mean={F_mean} min={F_min} max={F_max}
    #         any nan? {F_nan}
    #         any inf? {F_inf}
    #         any nonfinite? {F_bad}

    #         |F|/(E^2):
    #         mean={r_mean} min={r_min} max={r_max}
    #         any > 1 ? {r_gt1}
    #         tableau {ratio}
    #         """,
    #         label=label,
    #         ax=ax,
    #         shape=sol.shape,

    #         E_mean=jnp.mean(E),
    #         E_min=jnp.min(E),
    #         E_max=jnp.max(E),
    #         E_nan=jnp.any(jnp.isnan(E)),
    #         E_inf=jnp.any(jnp.isinf(E)),
    #         E_bad=jnp.any(~jnp.isfinite(E)),

    #         Fx_mean=jnp.mean(Fx),
    #         Fx_min=jnp.min(Fx),
    #         Fx_max=jnp.max(Fx),

    #         Fy_mean=jnp.mean(Fy),
    #         Fy_min=jnp.min(Fy),
    #         Fy_max=jnp.max(Fy),

    #         Fz_mean=jnp.mean(Fz),
    #         Fz_min=jnp.min(Fz),
    #         Fz_max=jnp.max(Fz),

    #         F_mean=jnp.mean(Fmag),
    #         F_min=jnp.min(Fmag),
    #         F_max=jnp.max(Fmag),
    #         F_nan=jnp.any(jnp.isnan(Fmag)),
    #         F_inf=jnp.any(jnp.isinf(Fmag)),
    #         F_bad=jnp.any(~jnp.isfinite(Fmag)),

    #         r_mean=jnp.mean(ratio),
    #         r_min=jnp.min(ratio),
    #         r_max=jnp.max(ratio),
    #         r_gt1=jnp.any(ratio > 1.0 + 1e-6),
    #         ratio=ratio[0:20,40:60,50],
    #         ordered=True,
    #     )
    def _debug_grid_stats(self, sol, eq, label, ax):
        E = sol[0]#jnp.maximum(sol[0], eq.eps)
        Fx = sol[1]
        Fy = sol[2]
        Fz = sol[3]

        Fmag = jnp.sqrt(Fx**2 + Fy**2 + Fz**2)

        mask = sol[0] > eq.eps
        mask_ratio = E > eq.eps

        ratio = jnp.where(
            mask_ratio,
            Fmag / jnp.maximum(eq.light_speed * E, 1e-30),
            0.0,
        )

        n_active = jnp.sum(mask)
        n_ratio_active = jnp.sum(mask_ratio)

        E_masked     = jnp.where(mask, E, 0.0)
        Fx_masked    = jnp.where(mask, Fx, 0.0)
        Fy_masked    = jnp.where(mask, Fy, 0.0)
        Fz_masked    = jnp.where(mask, Fz, 0.0)
        Fmag_masked  = jnp.where(mask, Fmag, 0.0)
        ratio_masked = jnp.where(mask_ratio, ratio, 0.0)

        E_min_vis     = jnp.min(jnp.where(mask, E, jnp.inf))
        Fx_min_vis    = jnp.min(jnp.where(mask, Fx, jnp.inf))
        Fy_min_vis    = jnp.min(jnp.where(mask, Fy, jnp.inf))
        Fz_min_vis    = jnp.min(jnp.where(mask, Fz, jnp.inf))
        Fmag_min_vis  = jnp.min(jnp.where(mask, Fmag, jnp.inf))
        ratio_min_vis = jnp.min(jnp.where(mask_ratio, ratio, jnp.inf))

        E_max_vis     = jnp.max(jnp.where(mask, E, -jnp.inf))
        Fx_max_vis    = jnp.max(jnp.where(mask, Fx, -jnp.inf))
        Fy_max_vis    = jnp.max(jnp.where(mask, Fy, -jnp.inf))
        Fz_max_vis    = jnp.max(jnp.where(mask, Fz, -jnp.inf))
        Fmag_max_vis  = jnp.max(jnp.where(mask, Fmag, -jnp.inf))
        ratio_max_vis = jnp.max(jnp.where(mask_ratio, ratio, -jnp.inf))

        denom = jnp.maximum(n_active, 1)
        denom_ratio = jnp.maximum(n_ratio_active, 1)

        E_mean_vis     = jnp.sum(E_masked) / denom
        Fx_mean_vis    = jnp.sum(Fx_masked) / denom
        Fy_mean_vis    = jnp.sum(Fy_masked) / denom
        Fz_mean_vis    = jnp.sum(Fz_masked) / denom
        Fmag_mean_vis  = jnp.sum(Fmag_masked) / denom
        ratio_mean_vis = jnp.sum(ratio_masked) / denom_ratio

        any_active = jnp.any(mask)
        any_ratio_active = jnp.any(mask_ratio)

    #     jax.debug.print(
    #     """
    #     [{label}] ax={ax}
    #     shape = {shape}
    #     n_active = {n_active}
    #     n_ratio_active = {n_ratio_active}
    #     any_active = {any_active}
    #     any_ratio_active = {any_ratio_active}

    #     E on active cells:
    #     mean={E_mean} min={E_min} max={E_max}
    #     any nan? {E_nan}
    #     any inf? {E_inf}
    #     any nonfinite? {E_bad}

    #     Fx on active cells:
    #     mean={Fx_mean} min={Fx_min} max={Fx_max}

    #     Fy on active cells:
    #     mean={Fy_mean} min={Fy_min} max={Fy_max}

    #     Fz on active cells:
    #     mean={Fz_mean} min={Fz_min} max={Fz_max}

    #     |F| on active cells:
    #     mean={F_mean} min={F_min} max={F_max}
    #     any nan? {F_nan}
    #     any inf? {F_inf}
    #     any nonfinite? {F_bad}

    #     |F|/(c E) on cells where E > eps:
    #     mean={r_mean} min={r_min} max={r_max}
    #     any > 1 ? {r_gt1}

    #     ratio slice:
    #     {ratio_slice}
    #     """,
    #     label=label,
    #     ax=ax,
    #     shape=sol.shape,
    #     n_active=n_active,
    #     n_ratio_active=n_ratio_active,
    #     any_active=any_active,
    #     any_ratio_active=any_ratio_active,

    #     E_mean=E_mean_vis,
    #     E_min=jnp.where(any_active, E_min_vis, 0.0),
    #     E_max=jnp.where(any_active, E_max_vis, 0.0),
    #     E_nan=jnp.any(jnp.isnan(jnp.where(mask, E, jnp.nan))),
    #     E_inf=jnp.any(jnp.isinf(jnp.where(mask, E, 0.0))),
    #     E_bad=jnp.any(~jnp.isfinite(jnp.where(mask, E, 0.0))),

    #     Fx_mean=Fx_mean_vis,
    #     Fx_min=jnp.where(any_active, Fx_min_vis, 0.0),
    #     Fx_max=jnp.where(any_active, Fx_max_vis, 0.0),

    #     Fy_mean=Fy_mean_vis,
    #     Fy_min=jnp.where(any_active, Fy_min_vis, 0.0),
    #     Fy_max=jnp.where(any_active, Fy_max_vis, 0.0),

    #     Fz_mean=Fz_mean_vis,
    #     Fz_min=jnp.where(any_active, Fz_min_vis, 0.0),
    #     Fz_max=jnp.where(any_active, Fz_max_vis, 0.0),

    #     F_mean=Fmag_mean_vis,
    #     F_min=jnp.where(any_active, Fmag_min_vis, 0.0),
    #     F_max=jnp.where(any_active, Fmag_max_vis, 0.0),
    #     F_nan=jnp.any(jnp.isnan(jnp.where(mask, Fmag, jnp.nan))),
    #     F_inf=jnp.any(jnp.isinf(jnp.where(mask, Fmag, 0.0))),
    #     F_bad=jnp.any(~jnp.isfinite(jnp.where(mask, Fmag, 0.0))),

    #     r_mean=ratio_mean_vis,
    #     r_min=jnp.where(any_ratio_active, ratio_min_vis, 0.0),
    #     r_max=jnp.where(any_ratio_active, ratio_max_vis, 0.0),
    #     r_gt1=jnp.any(jnp.where(mask_ratio, ratio > 1.0 + 1e-6, False)),

    #     ratio_slice=jnp.where(mask_ratio[0:20, 40:60, 50], ratio[0:20, 40:60, 50], 0.0),
    #     ordered=True,
    # )
    def flux(self, sol, ax, params, flux):
        eq = self.eq_manage
        
        # DEBUG: full grid before reconstruction / Riemann
        
        self._debug_grid_stats(sol, eq, "GRID BEFORE RIEMANN", ax)

        # cell primitives
        primitives = eq.get_primitives_from_conservatives(sol)
        # jax.debug.print('min dans flux de E {}', jnp.min(primitives[0]))
        # quick finite check on primitives before reconstruction
        _check_finite("primitives", primitives)

        primitives_xi_L = self.recon.reconstruct_xi(
            primitives,
            axis=ax,
            j=0)

        primitives_xi_R = self.recon.reconstruct_xi(
            primitives,
            axis=ax,
            j=1)
        
        # convert reconstructed primitives -> conservatives (full state)
        conservative_xi_L = eq.get_conservatives_from_primitives(primitives_xi_L)
        conservative_xi_R = eq.get_conservatives_from_primitives(primitives_xi_R)
        # primitives_xi_L = jnp.roll(primitives, shift=1, axis=ax)
        # primitives_xi_R = primitives

        # conservative_xi_L = jnp.roll(sol, shift=1, axis=ax)
        # conservative_xi_R = sol
        # F_norme = jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2)
        # F_x_normalize = sol[1]/F_norme*eq.light_speed**2*sol[0]**2
        # F_y_normalize = sol[2]/F_norme*eq.light_speed**2*sol[0]**2
        # F_z_normalize = sol[3]/F_norme*eq.light_speed**2*sol[0]**2
        # sol = sol.at[1].set(F_x_normalize)
        # sol = sol.at[2].set(F_y_normalize)
        # sol = sol.at[3].set(F_z_normalize)
        # positivity fix (full state)
        # self._debug_grid_stats(sol, eq, "GRID AFTER NORMALISATION", ax)
        if self.positivity:
            conservative_xi_L, primitives_xi_L, count_L = self.compute_positivity_preserving_interpolation(
                primitives=primitives,
                primitives_xi_j=primitives_xi_L,
                j=0,
                axis=ax)
            conservative_xi_R, primitives_xi_R, count_R = self.compute_positivity_preserving_interpolation(
                primitives=primitives,
                primitives_xi_j=primitives_xi_R,
                j=1,
                axis=ax)
        
        # self._debug_grid_stats(sol, eq, "PRIMITIVES LEFT AFTER RECON", ax)

        ####### mettre un check soit enlever la reconstruction et aussi mettre un check ici 
        # -------------------------------
        # Active block for the solver
        # -------------------------------
        prim_L_act = primitives_xi_L[eq.active_slice]
        prim_R_act = primitives_xi_R[eq.active_slice]
        cons_L_act = conservative_xi_L[eq.active_slice]
        cons_R_act = conservative_xi_R[eq.active_slice]

        # Existing GLM-MHD special case:
        # n_cons == 9 means active MHD (8 vars) + ψ scalar.
        glm_active = (getattr(eq, "n_cons", conservative_xi_L.shape[0]) == 9)  # see article

        # Solve Riemann on active-only
        F_act, _, _ = self.solver.solve_riemann_problem_xi(
            prim_L_act, prim_R_act,
            cons_L_act, cons_R_act, ax-1
        )

        F = F_act

        # -------------------------------
        # GLM ψ handling (unchanged logic,
        # just no hard-coded solver slicing)
        # -------------------------------
        if glm_active:
            c_h = getattr(eq, "glm_ch", 1.0)
            F_Bn, F_psi = mhd.glm_face_flux(primitives_xi_L, primitives_xi_R, ax-1, c_h)

            # normal B index in the *active* MHD block
            # (this is still MHD-physics-specific)
            Bn_idx = (4, 5, 6)[ax-1]
            F = F.at[Bn_idx].set(F_Bn)

            # append ψ flux -> makes 9 rows total at this stage
            F = jnp.vstack([F, F_psi[jnp.newaxis, :]])
            # print("caca boucle glm")
        # -------------------------------
        # Generic passive scalars
        # Anything beyond the active block
        # (and beyond ψ if GLM) is passive.
        # -------------------------------
        if glm_active:
            # ψ is the first scalar beyond the active MHD block
            passive_start = eq.n_active + 1
            # print("caca boucle passive start glm")
        else:
            passive_start = eq.n_active
            # print("caca boucle passive start no glm")

        if conservative_xi_L.shape[0] > passive_start:  # calculation of passive fluxes
            # mass flux from active solver (row 0 always rho*u_n)
            mass_flux = F_act[eq.mass_ids]  # shape (...)

            passive_L = primitives_xi_L[passive_start:]
            passive_R = primitives_xi_R[passive_start:]

            passive_face = jnp.where(
                mass_flux[jnp.newaxis, ...] >= 0.0,
                passive_L, passive_R
            )

            F_passive = mass_flux[jnp.newaxis, ...] * passive_face
            F = jnp.concatenate([F, F_passive], axis=0)
            # print("caca conservative")
        # DEBUG: flux returned on the entire grid after Riemann
        self._debug_grid_stats(F, eq, "GRID AFTER RIEMANN", ax)
        # jax.debug.print('min dans flux de E apres{}', jnp.min(primitives[0]))
        return F

    def timestep(self, sol):  # new by the chat not bad
        """
        CFL RT: dt = cfl / sum_d(c/dx_d)
        """
        eq = self.eq_manage
        dim = sol.ndim - 1
        dx = float(self.dx_o)
        c = float(eq.light_speed)
        # jax.debug.print('Light speed: {}', c)

        inv_dt = dim * c / dx
        # print(eq.cfl / (inv_dt + eq.eps))

        return eq.cfl / (inv_dt + eq.eps)

    def compute_positivity_preserving_interpolation(self,
                                                    primitives: Array,
                                                    primitives_xi_j: Array,
                                                    j: int,
                                                    axis: int):

        cell_state_xi_safe_j = self.positivity_stencil.reconstruct_xi(
            primitives, axis, j
        )
        # jax.debug.print('min dans flux de E recon avant{}', jnp.min(primitives[0]))
        E_j = primitives_xi_j[self.eq_manage.mass_ids]
        mask = jnp.where(E_j < self.eq_manage.eps, 0, 1)
        counter = jnp.sum(1 - mask)

        primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)
        conservative_xi_j = self.eq_manage.get_conservatives_from_primitives(primitives_xi_j)
        # jax.debug.print('min dans flux de E recon apres{}', jnp.min(primitives[0]))
        return conservative_xi_j, primitives_xi_j, counter
