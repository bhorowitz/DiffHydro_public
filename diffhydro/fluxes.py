from jax import Array 
from functools import partial
from typing import List
from .solver import recon
from .solver.stencils import CentralSixthOrderReconstruction
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
        # Primitives once, then derive speeds axis-by-axis
        primitives = self.eq_manage.get_primitives_from_conservatives(sol)
    
        # Pull velocities via indices (don’t slice 1:-1; that would include B in MHD)
        u = primitives[self.eq_manage.vel_ids[0]]
        v = primitives[self.eq_manage.vel_ids[1]]
        # Handle 2D/3D transparently
        w = primitives[self.eq_manage.vel_ids[2]] if len(self.eq_manage.vel_ids) > 2 else 0.0
    
        # Max wave speed per axis = |v_d| + signal_speed(prims, axis)
        # Euler managers: signal_speed = c_s;  MHD managers: signal_speed = c_f (fast magnetosonic)
        # axis numbering here matches flux() call convention (ax=1,2,3 are spatial axes)
        speed_x = jnp.abs(u) + self.eq_manage.get_signal_speed(primitives, axis=0)
        speed_y = jnp.abs(v) + self.eq_manage.get_signal_speed(primitives, axis=1)
        speed_z = jnp.abs(w) + (self.eq_manage.get_signal_speed(primitives, axis=2) if sol.ndim >= 4 else 0.0)
    
        # Global max over the grid (and axes)
        cmax = jnp.max(jnp.stack([
            jnp.max(speed_x),
            jnp.max(speed_y),
            jnp.max(speed_z),
        ]))
    
        # CFL timestep
        dt = self.eq_manage.cfl * self.dx_o / (cmax + self.eq_manage.eps)
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
        self.recon_heat = CentralSixthOrderReconstruction()
        self.zeta = zeta #5.111496271545331e-12
        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        self.dx_o = 1
        
        try: #3d
            self.flux_shapes = (EquationManager.n_cons,EquationManager.mesh_shape[0],EquationManager.mesh_shape[1],EquationManager.mesh_shape[2])
        except: #2d, z velocity axis usually just constant...
            self.flux_shapes = (EquationManager.n_cons,EquationManager.mesh_shape[0],EquationManager.mesh_shape[1])
    def flux_old(
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
    def flux(self, sol, axis, params, flux):
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
            kappa_used = 1.0 / (1.0/(kappa_unsat + 1e-30) + 1.0/(kappa_max + 1e-30))
        else:
            # plain, unsaturated conductivity for non-ElBadry models
            kappa_used = kappa_unsat
    
        # --- flux (correct sign: heat flows hot → cold) ---
        heat_flux_xi = kappa_used * dT_dxi
    
        conductive_flux = jnp.zeros_like(sol)
        conductive_flux = conductive_flux.at[self.eq_manage.energy_ids].set(heat_flux_xi)
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
            ZETA = 5.111496271545331e-12  # same constant used in flux
            kappa_max = 1.5 * cs**3 / (gradT * ZETA)
    
            # Harmonic combine for effective κ
            kappa_used = 1.0 / (1.0/(kappa_unsat + 1e-30) + 1.0/(kappa_max + 1e-30))
    
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
