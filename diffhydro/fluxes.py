from jax import Array
from functools import partial
from typing import List
from .solver import recon
from .solver.stencils import *
from .physics import mhd
import jax.numpy as jnp
import jax
from jax.experimental import checkify


class ConvectiveFlux:
    def __init__(self,
                 EquationManager,
                 Solver,
                 Recon,
                 positivity=True
                 ):
        self.eq_manage = EquationManager
        self.solver = Solver
        self.recon = Recon

        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        self.dx_o = 1

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

        # reconstructed face primitives (full state, incl passives)
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
        ) # _ signifie ignore les veleurs retourner, et ax-1 change l indexation de la base on calcule ici les problemes a l interface avec godunov

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

        if conservative_xi_L.shape[0] > passive_start: #calcule des flux passif c est a dire des grandeurs qui n ont pas d influence sur la dynamique et qui sont transportees par le flux de masse
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
                 zeta=0
                 ):
        self.eq_manage = EquationManager
        self.solver = Solver
        self.recon_heat = CentralSecondOrderReconstruction()
        self.zeta = zeta
        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        self.dx_o = 1

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

class ConvectiveFlux_Radiative_transfer:
    def __init__(self,
                 EquationManager,
                 Solver,
                 Recon,
                 positivity=True
                 ):
        self.eq_manage = EquationManager
        self.solver = Solver
        self.recon = Recon

        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        self.dx_o = 1

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

        # reconstructed face primitives (full state, incl passives)
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

        # Existing GLM-MHD special case:
        # n_cons == 9 means active MHD (8 vars) + ψ scalar.
        glm_active = (getattr(eq, "n_cons", conservative_xi_L.shape[0]) == 9) #see article 

        # Solve Riemann on active-only
        F_act, _, _ = self.solver.solve_riemann_problem_xi(
            prim_L_act, prim_R_act,
            cons_L_act, cons_R_act, ax-1
        ) # _ signifie ignore les veleurs retourner, et ax-1 change l indexation de la base on calcule ici les problemes a l interface avec godunov

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

        if conservative_xi_L.shape[0] > passive_start: #calcule des flux passif c est a dire des grandeurs qui n ont pas d influence sur la dynamique et qui sont transportees par le flux de masse
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

    # def timestep(self, sol):
    #     """
    #     Multi-D CFL timestep for unsplit FV:
    #       dt = CFL / max_x,y,z ( a_x/dx + a_y/dy + a_z/dz )
    #     with a_d = |v_d| + signal_speed(prims, axis=d)
    #     """
    #     eq = self.eq_manage
    #     prim = eq.get_primitives_from_conservatives(sol)

    #     # velocities from active block (no hard-coded 1,2,3)
    #     u = prim[eq.vel_ids[0]]
    #     v = prim[eq.vel_ids[1]] if len(eq.vel_ids) > 1 else 0.0
    #     w = prim[eq.vel_ids[2]] if len(eq.vel_ids) > 2 else 0.0

    #     a_x = jnp.abs(u) + eq.get_signal_speed(prim, axis=0)
    #     a_y = jnp.abs(v) + eq.get_signal_speed(prim, axis=1) if sol.ndim >= 3 else 0.0
    #     a_z = jnp.abs(w) + eq.get_signal_speed(prim, axis=2) if sol.ndim >= 4 else 0.0

    #     dx = float(self.dx_o)
    #     dy = float(self.dx_o)
    #     dz = float(self.dx_o)

    #     inv_dt_local = a_x / dx
    #     if sol.ndim >= 3:
    #         inv_dt_local = inv_dt_local + a_y / dy
    #     if sol.ndim >= 4:
    #         inv_dt_local = inv_dt_local + a_z / dz

    #     inv_dt_max = jnp.max(inv_dt_local)

    #     dt = eq.cfl / (inv_dt_max + eq.eps)
    #     return dt
   
    def timestep(self, sol): #new by the chat not bad
        """
        CFL RT: dt = cfl / sum_d(c/dx_d)
        """
        eq = self.eq_manage
        dim = sol.ndim - 1
        dx = float(self.dx_o)
        c = float(eq.light_speed)
    
        inv_dt = dim * c / dx
        return eq.cfl / (inv_dt + eq.eps)
    


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
        # p_j = primitives_xi_j[self.eq_manage.energy_ids]
        # mask = jnp.where(p_j < self.eq_manage.eps, 0, 1)

        # counter += jnp.sum(1 - mask)
        # primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)
        counter += jnp.sum(1 )
        primitives_xi_j = primitives_xi_j  + cell_state_xi_safe_j 

        conservative_xi_j = self.eq_manage.get_conservatives_from_primitives(primitives_xi_j)

        return conservative_xi_j, primitives_xi_j, counter


