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
                 positivity = True
                ):
        self.eq_manage = EquationManager
        self.solver = Solver
        self.recon = Recon
        
        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        self.dx_o = 1
        
        try: #3d
            self.flux_shapes = (EquationManager.n_cons,EquationManager.mesh_shape[0],EquationManager.mesh_shape[1],EquationManager.mesh_shape[2])
        except: #2d, z velocity axis usually just constant...
            self.flux_shapes = (EquationManager.n_cons,EquationManager.mesh_shape[0],EquationManager.mesh_shape[1])

        
    def flux(self,sol,ax,params,flux):
        primitives = self.eq_manage.get_primitives_from_conservatives(sol)
        
        primitives_xi_L = self.recon.reconstruct_xi(
            primitives,
            axis=ax,
            j=0)
        
        primitives_xi_R = self.recon.reconstruct_xi(
            primitives,
            axis=ax,
            j=1)


        conservative_xi_L = self.eq_manage.get_conservatives_from_primitives(primitives_xi_L)
        conservative_xi_R = self.eq_manage.get_conservatives_from_primitives(primitives_xi_R)

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
        cons_L_solver = conservative_xi_L
        cons_R_solver = conservative_xi_R
        
        glm_active = (getattr(self.eq_manage, "n_cons", conservative_xi_L.shape[0]) == 9)
        
        if glm_active:
            # Pass only the MHD rows (ρ, ρu, ρv, ρw, B1, B2, B3, E) to HLL/HLLD
            cons_L_solver = conservative_xi_L[:8]
            cons_R_solver = conservative_xi_R[:8]
        
        # final axis is 0-based for solver
        F, _, _ = self.solver.solve_riemann_problem_xi(
            primitives_xi_L, primitives_xi_R,
            cons_L_solver, cons_R_solver, ax-1)        # HLL/HLLD returns 8 rows (MHD) 
        
        # ---- NEW: GLM face update (2×2 subsystem) and ψ row append ----
        if glm_active:
            c_h = getattr(self.eq_manage, "glm_ch", 1.0) 
            
            F_Bn, F_psi = mhd.glm_face_flux(primitives_xi_L,primitives_xi_R,ax-1,c_h)

            # overwrite the normal-B row in the 8-row MHD flux and then append ψ to make 9 rows
            Bn_idx = (4, 5, 6)[ax-1]    
            F = F.at[Bn_idx].set(F_Bn)
            F = jnp.vstack([F, F_psi[jnp.newaxis,:]])    # return 9 rows when GLM is active

        return F

        
    def timestep(self, sol):
        """
        Multi-D CFL timestep for unsplit FV:
          dt = CFL / max_x,y,z ( a_x/dx + a_y/dy + a_z/dz )
        with a_d = |v_d| + signal_speed(prims, axis=d)
        """
        eq = self.eq_manage
        prim = eq.get_primitives_from_conservatives(sol)

        # velocities
        u = prim[eq.vel_ids[0]]
        v = prim[eq.vel_ids[1]] if len(eq.vel_ids) > 1 else 0.0
        w = prim[eq.vel_ids[2]] if len(eq.vel_ids) > 2 else 0.0

        # per-axis characteristic speeds (no velocity inside get_signal_speed!)
        a_x = jnp.abs(u) + eq.get_signal_speed(prim, axis=0)
        a_y = jnp.abs(v) + eq.get_signal_speed(prim, axis=1) if sol.ndim >= 3 else 0.0
        a_z = jnp.abs(w) + eq.get_signal_speed(prim, axis=2) if sol.ndim >= 4 else 0.0

        # grid spacing per axis; you’re using dx_o=1 for now
        dx = float(self.dx_o)
        dy = float(self.dx_o)
        dz = float(self.dx_o)

        # in general you want:
        #   inv_dt_local = a_x/dx + a_y/dy (+ a_z/dz)
        # then dt = CFL / max(inv_dt_local)
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
        counter = jnp.sum(1 - mask)    # TODO check in parallel

        primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

        # CHECK ENERGY / PRESSURE
        p_j = primitives_xi_j[self.eq_manage.energy_ids]
        
        # OPTION 1 - CHECK PRESSURE DIRECTLY
        mask = jnp.where(p_j < self.eq_manage.eps, 0, 1)

        # OPTION 2 - CHECK VIA INTERNAL ENERGY, seems overkill though
        # rhoe_j = rho_j * self.material_manager.get_specific_energy(p_j, rho=rho_j)
        # mask_j = jnp.where(rhoe_j - pb < self.eps.pressure, 0, 1)

        counter += jnp.sum(1 - mask)    # TODO check in parallel
        primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

        conservative_xi_j = self.eq_manage.get_conservatives_from_primitives(primitives_xi_j)

        return conservative_xi_j,primitives_xi_j,counter
    
class ConductiveFlux:
    def __init__(self,
                EquationManager,
                 Solver,
                 Recon,
                 positivity = False,
                 zeta = 0
                ):
        self.eq_manage = EquationManager
        self.solver = Solver
        self.recon_heat = CentralSecondOrderReconstruction()
        self.zeta = zeta #5.111496271545331e-12
        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        self.dx_o = 1
        
        try: #3d
            self.flux_shapes = (EquationManager.n_cons,EquationManager.mesh_shape[0],EquationManager.mesh_shape[1],EquationManager.mesh_shape[2])
        except: #2d, z velocity axis usually just constant...
            self.flux_shapes = (EquationManager.n_cons,EquationManager.mesh_shape[0],EquationManager.mesh_shape[1])
    def flux_old_old(
            self,
            sol,axis,params,flux
        ) -> Array:
        """Computes the heat flux in axis direction.
        
        q = - \lambda \nabla T

        :param temperature: Buffer with temperature.
        :type temperature: Array
        :param axis: Spatial direction along which the heat flux is calculated.
        :type axis: int
        :return: Heat flux in axis direction.
        :rtype: Array
        """
        primitives = self.eq_manage.get_primitives_from_conservatives(sol)
        
        temperature = self.eq_manage.get_temperature(primitives[self.eq_manage.energy_ids],primitives[self.eq_manage.mass_ids])
        

        temperature_at_cf = self.recon_heat.reconstruct_xi(temperature, axis-1)

        thermal_conductivity = self.eq_manage.get_thermal_conductivity(
            temperature_at_cf,
            primitives,
            None,
            None,
            None)

        temperature_grad = self.recon_heat.derivative_xi(
            temperature, axis-1)
        heat_flux_xi = thermal_conductivity * temperature_grad #negative sign? not sure where I got my sign flip...

        conductive_flux = jnp.zeros_like(sol)
        conductive_flux = conductive_flux.at[self.eq_manage.energy_ids].set(heat_flux_xi)

        return conductive_flux
    def flux_b(self, sol, axis, params, flux):
        """
        Conductive energy flux along `axis`.
        Uses κ_eff (Balbus–McKee saturation) only for ELBADRY; otherwise plain κ.
        q = - κ_used * ∂T/∂x_axis
        """
        # --- primitives & temperature (cell-centered) ---
        prim = self.eq_manage.get_primitives_from_conservatives(sol)
        rho  = prim[self.eq_manage.mass_ids]
        p    = prim[self.eq_manage.energy_ids]
        T    = self.eq_manage.get_temperature(p, rho)
    
        # --- face-centered T for evaluating κ(T) ---
        ax0 = axis - 1  # your code uses 1-based axis at call sites
        T_face = self.recon_heat.reconstruct_xi(T, ax0)
        
        # Get unsaturated conductivity from the manager
        # (keep signature matching your manager; extra args are ignored if not present)
        try:
            kappa_unsat = self.eq_manage.get_thermal_conductivity(T_face, prim, None, None, None)
        except TypeError:
            kappa_unsat = self.eq_manage.get_thermal_conductivity(T_face, prim)
    
        # --- temperature gradients (cell-centered) ---
        dTdx = self.recon_heat.derivative_xi(T, 0) if (sol.ndim - 1) >= 1 else 0.0
        dTdy = self.recon_heat.derivative_xi(T, 1) if (sol.ndim - 1) >= 2 else 0.0
        dTdz = self.recon_heat.derivative_xi(T, 2) if (sol.ndim - 1) >= 3 else 0.0
        gradT_mag = jnp.sqrt(dTdx*dTdx + dTdy*dTdy + dTdz*dTdz) + 1e-30
    
        # directional gradient for this flux component
        #dT_dxi = dTdx if ax0 == 0 else (dTdy if ax0 == 1 else dTdz)
        dT_dxi = self.recon_heat.derivative_xi(
            T, axis-1)
        
        # --- choose κ_used ---
        model = getattr(self.eq_manage, "thermal_conductivity_model", "")
        if model == "ELBADRY":
            # saturation only for El-Badry
            cs = self.eq_manage.get_speed_of_sound(p, rho)
            #ZETA = 5.111496271545331e-12  # same constant used in your ELBADRY block
            kappa_max = 1.5 * cs**3 / (gradT_mag * self.zeta)
            # harmonic combine (κ_eff ≤ κ_unsat, κ_eff ≤ κ_max)
            kappa_used = 1.0 / (1.0/(kappa_unsat + 1e-30) + 1.0/(kappa_max + 1e-30)) * self.zeta
        else:
            # plain, unsaturated conductivity for non-ElBadry models
            kappa_used = kappa_unsat
    
        # --- flux (correct sign: heat flows hot → cold) ---
        heat_flux_xi = kappa_used * dT_dxi
    
        conductive_flux = jnp.zeros_like(sol)
        conductive_flux = conductive_flux.at[self.eq_manage.energy_ids].set(heat_flux_xi)
        return conductive_flux
    def flux_c(self, sol, axis, params, flux):
        # primitives & temperature (cell-centered)
        prim = self.eq_manage.get_primitives_from_conservatives(sol)
        rho  = prim[self.eq_manage.mass_ids]
        p    = prim[self.eq_manage.energy_ids]
        T    = self.eq_manage.get_temperature(p, rho)

        ax0 = axis - 1  # 0-based spatial axis

        # --- face-centered temperature & primitives for kappa(T, prim) ---
        T_face   = self.recon_heat.reconstruct_xi(T, ax0)
        rho_face = self.recon_heat.reconstruct_xi(rho, ax0)
        p_face   = self.recon_heat.reconstruct_xi(p,   ax0)
        prim_face = prim.at[self.eq_manage.mass_ids].set(rho_face)\
                       .at[self.eq_manage.energy_ids].set(p_face)

        # unsaturated kappa at faces
        try:
            kappa_unsat = self.eq_manage.get_thermal_conductivity(T_face, prim_face, None, None, None)
        except TypeError:
            kappa_unsat = self.eq_manage.get_thermal_conductivity(T_face, prim_face)

        # --- temperature gradient at faces (collocated!) ---
        dT_dxi_face = (jnp.roll(T, -1, axis=ax0) - T) / self.dx_o  # lives at i+1/2 "slot"

        # --- saturation (El-Badry) uses |∇T| at faces, also collocated ---
        model = getattr(self.eq_manage, "thermal_conductivity_model", "")
        if model == "ELBADRY":
            # build face-aligned gradients in each dir
            dTdx_f = (jnp.roll(T, -1, axis=0) - T) / self.dx_o if (sol.ndim - 1) >= 1 else 0.0
            dTdy_f = (jnp.roll(T, -1, axis=1) - T) / self.dx_o if (sol.ndim - 1) >= 2 else 0.0
            dTdz_f = (jnp.roll(T, -1, axis=2) - T) / self.dx_o if (sol.ndim - 1) >= 3 else 0.0
            gradT_mag_face = jnp.sqrt(dTdx_f**2 + dTdy_f**2 + dTdz_f**2) + 1e-30

            cs_face = self.eq_manage.get_speed_of_sound(p_face, rho_face)
            kappa_max = 1.5 * cs_face**3 / (gradT_mag_face * self.zeta)
            kappa_used = 1.0 / (1.0/(kappa_unsat + 1e-30) + 1.0/(kappa_max + 1e-30)) * self.zeta
        else:
            kappa_used = kappa_unsat

        # --- face flux with correct sign ---
        heat_flux_xi = - kappa_used * dT_dxi_face

        # pack into full flux array (only energy row nonzero)
        conductive_flux = jnp.zeros_like(sol)
        conductive_flux = conductive_flux.at[self.eq_manage.energy_ids].set(heat_flux_xi)
        return conductive_flux
    def flux(self, sol, axis, params, flux):
        """
        Physically consistent conductive flux for finite-volume hydro.
        - κ is cell-centered, harmonic-mean to faces
        - Temperature gradient is staggered, face-collocated
        - Heat flux sign convention: q = -κ ∇T
        - Compatible with El-Badry saturation (computed at faces)
        """
        eq = self.eq_manage
        prim = eq.get_primitives_from_conservatives(sol)

        rho = prim[eq.mass_ids]
        p   = prim[eq.energy_ids]
        T   = eq.get_temperature(p, rho)
        ax0 = axis - 1  # 0-based axis

        # -----------------------------------------------------
        # 1) Cell-centered thermal conductivity
        # -----------------------------------------------------
        try:
            k_center = eq.get_thermal_conductivity(T, prim, None, None, None)
        except TypeError:
            k_center = eq.get_thermal_conductivity(T, prim)

        # -----------------------------------------------------
        # 2) Face-centered harmonic mean of κ
        # -----------------------------------------------------
        kL = k_center
        kR = jnp.roll(k_center, -1, axis=ax0)
        k_face = 2.0 * kL * kR / (kL + kR + 1e-30)

        # -----------------------------------------------------
        # 3) Staggered temperature gradient at faces
        # -----------------------------------------------------
        dT_dxi_face = (jnp.roll(T, -1, axis=ax0) - T) / self.dx_o

        # -----------------------------------------------------
        # 4) El-Badry saturation (optional)
        # -----------------------------------------------------
        model = getattr(eq, "thermal_conductivity_model", "")
        if model == "ELBADRY":
            # Face-centered primitive quantities for saturation
            rho_face = 0.5 * (rho + jnp.roll(rho, -1, axis=ax0))
            p_face   = 0.5 * (p   + jnp.roll(p,   -1, axis=ax0))
            T_face   = 0.5 * (T   + jnp.roll(T,   -1, axis=ax0))

            # |∇T| magnitude at faces (use forward diffs on each axis)
            dTdx = (jnp.roll(T, -1, axis=0) - T) / self.dx_o if (sol.ndim - 1) >= 1 else 0.0
            dTdy = (jnp.roll(T, -1, axis=1) - T) / self.dx_o if (sol.ndim - 1) >= 2 else 0.0
            dTdz = (jnp.roll(T, -1, axis=2) - T) / self.dx_o if (sol.ndim - 1) >= 3 else 0.0
            gradT_mag_face = jnp.sqrt(dTdx**2 + dTdy**2 + dTdz**2) + 1e-30

            # Sound speed at faces
            cs_face = eq.get_speed_of_sound(p_face, rho_face)

            # Saturation limiter
            kappa_max = 1.5 * cs_face**3 / (gradT_mag_face * self.zeta)

            # Combine harmonically
            k_face_used = 1.0 / (1.0/(k_face + 1e-30) + 1.0/(kappa_max + 1e-30))
        else:
            k_face_used = k_face

        # -----------------------------------------------------
        # 5) Compute face flux and pack into array
        # -----------------------------------------------------
        heat_flux_xi = k_face_used * dT_dxi_face  # minus sign = energy flows down T gradient

        conductive_flux = jnp.zeros_like(sol)
        conductive_flux = conductive_flux.at[eq.energy_ids].set(heat_flux_xi)
        return conductive_flux
    def timestep(self, sol):
        # Explicit diffusion CFL (safe in 1–3D)
        const = 0.1
        dx2 = float(self.dx_o) ** 2
    
        prim = self.eq_manage.get_primitives_from_conservatives(sol)
        rho  = prim[self.eq_manage.mass_ids]
        p    = prim[self.eq_manage.energy_ids]
        T    = self.eq_manage.get_temperature(p, rho)
        cp_m = self.eq_manage.get_specific_heat_capacity(T)  # per-unit-mass c_p
    
        # Unsaturated conductivity from the manager (face eval not required for dt)
        kappa_unsat = self.eq_manage.get_thermal_conductivity(T, prim)
    
        model = getattr(self.eq_manage, "thermal_conductivity_model", "")
    
        if model == "ELBADRY":
            # --- Use κ_eff only for El-Badry (saturation limiter) ---
            # Build |∇T| and c_s for κ_max
            dTdx = self.recon_heat.derivative_xi(T, 0) if (sol.ndim - 1) >= 1 else 0.0
            dTdy = self.recon_heat.derivative_xi(T, 1) if (sol.ndim - 1) >= 2 else 0.0
            dTdz = self.recon_heat.derivative_xi(T, 2) if (sol.ndim - 1) >= 3 else 0.0
            gradT = jnp.sqrt(dTdx*dTdx + dTdy*dTdy + dTdz*dTdz) + 1e-30
    
            cs = self.eq_manage.get_speed_of_sound(p, rho)
            ZETA = self.zeta  # same constant used in flux
            kappa_max = 1.5 * cs**3 / (gradT * self.zeta)
    
            # Harmonic combine for effective κ
            kappa_used = 1.0 / (1.0/(kappa_unsat + 1e-30) + 1.0/(kappa_max + 1e-30))* self.zeta * 75
    
            # El-Badry κ from your manager already contains a 1/ρ factor; do NOT divide by ρ again
            denom = jnp.maximum(cp_m, 1e-30)
    
        else:
            # --- Standard diffusion: use unsaturated κ and χ = κ/(ρ c_p) ---
            kappa_used = kappa_unsat
            rho_floor = jnp.maximum(rho, 1e-6)  # avoid ultra-low density blowups in dt
            denom = jnp.maximum(rho_floor * cp_m, 1e-30)
    
        chi = kappa_used / denom
        chi_max = jnp.max(chi)
    
        return const * dx2 / (chi_max + 1e-30)
