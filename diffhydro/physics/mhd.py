from jax import Array
import jax.numpy as jnp
from diffhydro.solver import recon
import jax

## some slightly disorganized routines related to MHD stability/divergence-constraining
##  GLM Forcing, Constrianted Transport and 

def glm_face_flux(WL, WR, axis0, c_h):
    # WL, WR: primitive states at the face (must contain psi if you store it in W;
    # if psi lives separately, pass it in instead)
    # axis0: 0,1,2 for x,y,z
    # Returns: (F_Bn_glm, F_psi)
    # pull Bn and psi (if psi not in W, set psi_L/R = 0 here and evolve psi separately)
    B = (WL[4], WL[5], WL[6]); Bn_L = B[axis0]
    B = (WR[4], WR[5], WR[6]); Bn_R = B[axis0]
    psi_L = WL[-1]  # if you store psi at the end of primitives; else pass explicitly
    psi_R = WR[-1]

    # Rusanov upwinding for the GLM 2x2 block (eigs ± c_h)
    F_Bn = 0.5 * (psi_L + psi_R) - 0.5 * c_h * (Bn_R - Bn_L)
    F_psi = 0.5 * (c_h**2) * (Bn_L + Bn_R) - 0.5 * c_h * (psi_R - psi_L)
    return F_Bn, F_psi

class GLMForcing:
    def __init__(self, eq, tau):
        self.eq, self.tau = eq, tau
    def force(self, i, U, params, dt):
        if not getattr(self.eq, "use_glm", False): 
            return jnp.zeros_like(U)
        dU = jnp.zeros_like(U)
        psi = U[self.eq.psi_id]
        dU = dU.at[self.eq.psi_id].set(-psi / self.tau)   # damping
        return U+ dU * dt
    def timestep(self, U): 
        return 1e9  # not limiting

## 

class ConstrainedTransportFlux:
    """
    CT-style correction flux:
      - Reconstruct L/R face primitives along axis `ax`.
      - Build a single face state by averaging L/R (cheap, robust).
      - Compute induction (magnetic) flux at that face using F_B = v_d B - B_d v.
      - Return ONLY the correction on magnetic components by subtracting whatever
        flux is already accumulated (prev_flux) at those components.
      - Energy/momentum left at zero here to avoid double-counting; the baseline
        convective flux already carries consistent terms.

    Assumes EquationManagerMHD-style indexing:
      U = [rho, rho*u, rho*v, rho*w, B1, B2, B3, E]
      W = [rho, u, v, w, B1, B2, B3, p]
    """

    def __init__(self, EquationManager, Solver, Recon, positivity=False):
        self.eq_manage = EquationManager
        self.solver = Solver
        self.recon = Recon  # e.g., MUSCL3/WENO

        self.positivity = positivity
        self.positivity_stencil = recon.WENO1()
        self.dx_o = 1

        # shape bookkeeping like other fluxes
        try:
            self.flux_shapes = (
                8,  # MHD conservatives
                EquationManager.mesh_shape[0],
                EquationManager.mesh_shape[1],
                EquationManager.mesh_shape[2],
            )
        except Exception:
            self.flux_shapes = (
                8,
                EquationManager.mesh_shape[0],
                EquationManager.mesh_shape[1],
            )

        # indices (match EquationManagerMHD)
        self.i_rho = self.eq_manage.mass_ids
        self.i_u, self.i_v, self.i_w = self.eq_manage.vel_ids  # in primitives
        # magnetic components (prims and cons share indices 4..6 in MHD manager)
        self.i_B = getattr(self.eq_manage, "mag_ids", (4, 5, 6))
        self.i_E = self.eq_manage.energy_ids

    # ---- helper: average L/R face primitives (cheap, stable) ----
    def _avg_face_state(self, WL: Array, WR: Array) -> Array:
        # simple arithmetic average for upwinded EMFs,
        # maybe need to  bias this using signs of vd or reuse Riemann star states.
        return 0.5 * (WL + WR)

    # ---- helper: build induction flux from a single face primitive state ----
    def _induction_flux_from_face_prims(self, W_face: Array, ax: int) -> Array:
        # unpack
        rho = W_face[self.i_rho]
        u = W_face[self.i_u]
        v = W_face[self.i_v]
        w = W_face[self.i_w]
        B1 = W_face[self.i_B[0]]
        B2 = W_face[self.i_B[1]]
        B3 = W_face[self.i_B[2]]

        # pick normal component index and assemble vectors
        v_vec = (u, v, w)
        B_vec = (B1, B2, B3)
        vd = v_vec[ax - 1]   # note: hydro_core passes ax in {1,2,3}
        Bd = B_vec[ax - 1]

        # magnetic fluxes (induction):  v_d B - B_d v
        F_B1 = vd * B1 - Bd * u
        F_B2 = vd * B2 - Bd * v
        F_B3 = vd * B3 - Bd * w

        # assemble a full-sized flux array (zeros elsewhere)
        # conservatives layout is 8 vars for MHD
        F = jnp.zeros_like(W_face).reshape((-1,) + W_face.shape[1:])
        F = F.at[self.i_B[0]].set(F_B1)
        F = F.at[self.i_B[1]].set(F_B2)
        F = F.at[self.i_B[2]].set(F_B3)

        # Energy correction can be added?
        # F_E_face = ((E + p + 0.5*B^2) * vd - (v·B) * Bd)  [needs conservative E]
        # For now, keep energy = 0.0 to avoid double counting with ConvectiveFlux.
        return F

    # ---- public API used by hydro_core (accepts prev_flux / total_flux) ----
    def flux(self, sol, ax, params, prev_flux=None):
        """
        CT–HLL correction flux.
    
        - Reconstruct WL/WR on faces along axis `ax` (hydro uses 1-based ax for space).
        - Build HLL face flux F_HLL using EquationManagerMHD.get_fluxes_xi and
          fast magnetosonic Davis bounds.
        - Return a correction that replaces ONLY magnetic rows (and optionally energy)
          relative to the already-accumulated flux `prev_flux`.
        """
        # 1) primitives on cells
        W = self.eq_manage.get_primitives_from_conservatives(sol)
    
        # 2) L/R recon to face along axis ax
        WL = self.recon.reconstruct_xi(W, axis=ax, j=0)  # left  face state
        WR = self.recon.reconstruct_xi(W, axis=ax, j=1)  # right face state
    
        # (optional) minimal positivity fix on density/pressure, seems to be bad for blast waves though
        if getattr(self, "positivity", False):
            WL = self._positivity_fix(W, WL, j=0, axis=ax)
            WR = self._positivity_fix(W, WR, j=1, axis=ax)
    
        # 3) convert to conservatives (needed by get_fluxes_xi and HLL formula)
        UL = self.eq_manage.get_conservatives_from_primitives(WL)
        UR = self.eq_manage.get_conservatives_from_primitives(WR)
    
        # 4) physical fluxes from the equation manager at this axis (0-based there)
        axis0 = ax - 1
        F_L = self.eq_manage.get_fluxes_xi(WL, UL, axis0)
        F_R = self.eq_manage.get_fluxes_xi(WR, UR, axis0)  # includes induction, energy terms
        # (see EquationManagerMHD: induction F_B = v_d B - B_d v; energy flux consistent) 
    
        # 5) Davis wave-speed bounds S_L, S_R from fast magnetosonic speeds
        uL = WL[self.eq_manage.vel_ids[axis0]]
        uR = WR[self.eq_manage.vel_ids[axis0]]
        c_fL = self.eq_manage.get_fast_magnetosonic_speed(WL, axis0)
        c_fR = self.eq_manage.get_fast_magnetosonic_speed(WR, axis0)
        S_L = jnp.minimum(uL - c_fL, uR - c_fR)
        S_R = jnp.maximum(uL + c_fL, uR + c_fR)  # :contentReference[oaicite:5]{index=5}
    
        # 6) HLL face flux (LLF fallback when S_R≈S_L)
        denom = jnp.where(jnp.abs(S_R - S_L) < 1e-12, 1.0, S_R - S_L)
        F_HLL = (S_R * F_L - S_L * F_R + S_L * S_R * (UR - UL)) / denom
    
        # 7) Return a correction that *replaces* magnetic rows (and optionally energy)
        #    relative to previously accumulated flux.
        if prev_flux is None:
            # If called first, just return the upwinded magnetic/energy part;
            # the hydro_core will sum it directly.
            out = jnp.zeros_like(F_HLL)
            for k in self.eq_manage.mag_ids:  # [4,5,6] in MHD manager
                out = out.at[k].set(F_HLL[k])
            if getattr(self, "correct_energy", False):
                out = out.at[self.eq_manage.energy_ids].set(F_HLL[self.eq_manage.energy_ids])
            return out
    
        # Normal case: compute a *correction* so (prev_flux + corr) has the HLL magnetic (and energy) rows
        corr = jnp.zeros_like(prev_flux)
        for k in self.eq_manage.mag_ids:
            corr = corr.at[k].set(F_HLL[k] - prev_flux[k])
        if getattr(self, "correct_energy", False):
            e_i = self.eq_manage.energy_ids
            corr = corr.at[e_i].set(F_HLL[e_i] - prev_flux[e_i])
    
        return corr

    def timestep(self, sol: Array):
        """
        CFL based on fast magnetosonic speed (if available), falling back
        to Euler-type estimate. Mirrors ConvectiveFlux behavior.
        """
        # cheap, robust safety — reuse ConvectiveFlux logic for now
        v = jnp.abs(sol[1:-1] / jnp.maximum(sol[0], self.eq_manage.eps))
        temp_quant = (self.eq_manage.gamma - 1) * (
            sol[-1] - sol[0] * jnp.sum(v ** 2.0, axis=0) / 2.0
        )
        P = jnp.maximum(jnp.where(temp_quant > 0, temp_quant, 0.0), 0.0)
        cs = jnp.sqrt(self.eq_manage.gamma * P / jnp.maximum(sol[0], self.eq_manage.eps))
        cmax = jnp.max(jnp.max(v) + cs)
        dt = self.eq_manage.cfl * self.dx_o / (cmax + self.eq_manage.eps)
        return dt

    # ------- minimal positivity patch like ConvectiveFlux, but probably bad for blasts-------
    def _positivity_fix(self, W: Array, Wx: Array, j: int, axis: int) -> Array:
        Wsafe = self.positivity_stencil.reconstruct_xi(W, axis, j)
        rho = Wx[self.eq_manage.mass_ids]
        mask = jnp.where(rho < self.eq_manage.eps, 0, 1)
        Wx = Wx * mask + Wsafe * (1 - mask)

        p = Wx[self.eq_manage.energy_ids]
        mask_p = jnp.where(p < self.eq_manage.eps, 0, 1)
        Wx = Wx * mask_p + Wsafe * (1 - mask_p)
        return Wx

###


class PPCTForce: #Provably Positivity-Preserving Constrained Transport (PPCT)
    """
    Picard/PPCT-style implicit micro-iteration on (B, v) over a half-step dt:
       B_{k+1} = B_k - dt * curl( - v_k x B_k )
       v_{k+1} = v_k - dt * ( B_k x curl(B_k) ) / rho
    After convergence, recompute total energy E to stay thermodynamically consistent
    with the new v and the existing pressure p (primitive).

    See https://arxiv.org/abs/2410.05173

    Assumes EquationManagerMHD-like indexing:
      U = [rho, rho*u, rho*v, rho*w, B1, B2, B3, E]
      W = [rho,     u,     v,     w, B1, B2, B3, p]
    """

    def __init__(
        self,
        equation_manager,                 # e.g., EquationManagerMHD()
        boundary=None,                    # optional: pass the same boundary as hydro.boundary
        n_iter_max: int = 100,            # Picard cap
        tol: float = 1e-10,               # L2 relative tol on (B,v)
        curl_scheme: str = "central",     # or "upwind" later
        safety_eps: float = 1e-20,
    ):
        self.eq = equation_manager
        self.boundary = boundary
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.eps = safety_eps
        self.curl_scheme = curl_scheme

        # indices
        self.i_rho = self.eq.mass_ids
        self.i_u, self.i_v, self.i_w = self.eq.vel_ids
        self.iBx, self.iBy, self.iBz = getattr(self.eq, "mag_ids", (4, 5, 6))
        self.iE = self.eq.energy_ids

    # ---------------- public API expected by hydro.forcing(...) ----------------
    def timestep(self, sol: Array):
        # Let CFL be governed by fluxes; choose a permissive large dt here.
        return 1e10

    def force(self, i: int, sol: Array, params: dict, dt: float) -> Array:
        """
        Apply a PPCT half-step (dt supplied by hydro_core) as an *updated state*.
        This matches the current forcing contract where the returned array replaces fields
        (see NoForcing + hydro_core.forcing)  .
        """
        # Enforce dt >= 0
        dt = jnp.asarray(dt)
        dt = jnp.maximum(dt, 0.0)

        # Optionally impose boundary before the micro-iteration
        if self.boundary is not None:
            for ax in range(1, sol.ndim):
                sol = self.boundary.impose(sol, ax)

        # Primitives
        W = self.eq.get_primitives_from_conservatives(sol)

        rho = jnp.maximum(W[self.i_rho], self.eps)
        u = W[self.i_u]; v = W[self.i_v]; w = W[self.i_w]
        Bx = W[self.iBx]; By = W[self.iBy]; Bz = W[self.iBz]
        p  = jnp.maximum(W[self.iE], self.eps)  # in primitives, last is pressure for MHD manager

        # Pack iterates
        V = jnp.stack([u, v, w], axis=0)
        B = jnp.stack([Bx, By, Bz], axis=0)

        def curl(Fx, Fy, Fz):
            """Discrete curl using periodic rolls.
            Fx, Fy, Fz are scalar fields with shapes:
              2D: (nx, ny)
              3D: (nx, ny, nz)
            Returns a stack [cx, cy, cz] with the same spatial shape.
            """
            def dd(A, ax):
                return 0.5 * (jnp.roll(A, -1, axis=ax) - jnp.roll(A, 1, axis=ax))
        
            if Fx.ndim == 2:
                # axes: x=0, y=1
                dFz_dy = dd(Fz, 1)
                dFz_dx = dd(Fz, 0)
                dFy_dx = dd(Fy, 0)
                dFx_dy = dd(Fx, 1)
                cx = dFz_dy               # ∂Fz/∂y - ∂Fy/∂z ; ∂/∂z=0 in 2D
                cy = -dFz_dx              # ∂Fx/∂z - ∂Fz/∂x ; ∂/∂z=0
                cz = dFy_dx - dFx_dy      # ∂Fy/∂x - ∂Fx/∂y
                return jnp.stack([cx, cy, cz], axis=0)
        
            elif Fx.ndim == 3:
                # axes: x=0, y=1, z=2
                dFz_dy = dd(Fz, 1); dFy_dz = dd(Fy, 2)
                dFx_dz = dd(Fx, 2); dFz_dx = dd(Fz, 0)
                dFy_dx = dd(Fy, 0); dFx_dy = dd(Fx, 1)
                cx = dFz_dy - dFy_dz
                cy = dFx_dz - dFz_dx
                cz = dFy_dx - dFx_dy
                return jnp.stack([cx, cy, cz], axis=0)
        
            else:
                # 1D fallback: curl is zero
                shape = Fx.shape
                zeros = jnp.zeros(shape, dtype=Fx.dtype)
                return jnp.stack([zeros, zeros, zeros], axis=0)

        def cross(A, B):
            Ax, Ay, Az = A[0], A[1], A[2]
            Bx, By, Bz = B[0], B[1], B[2]
            return jnp.stack([Ay*Bz - Az*By, Az*Bx - Ax*Bz, Ax*By - Ay*Bx], axis=0)

        def l2_norm(*arrays):
            tot = 0.0
            for arr in arrays:
                tot = tot + jnp.sum(arr * arr)
            return jnp.sqrt(tot)

        # Fixed-point / Picard loop (checkpointed)
        def body(state):
            k, V_k, B_k, res0 = state

            # E = - v x B
            E = -cross(V_k, B_k)
            curlE = curl(E[0], E[1], E[2])  # = - curl(v x B)

            # Lorentz "forcing": (B x curl B) / rho
            curlB = curl(B_k[0], B_k[1], B_k[2])
            Fv = cross(B_k, curlB) / rho

            B_next = B_k - dt * curlE
            V_next = V_k - dt * Fv

            # Boundary every iter (if provided)
            if self.boundary is not None:
                # Rebuild a temporary primitives to apply boundary neatly on variables
                Wtmp = W.at[self.i_u].set(V_next[0]).at[self.i_v].set(V_next[1]).at[self.i_w].set(V_next[2]) \
                       .at[self.iBx].set(B_next[0]).at[self.iBy].set(B_next[1]).at[self.iBz].set(B_next[2])
                Utmp = self.eq.get_conservatives_from_primitives(Wtmp)
                for ax in range(1, Utmp.ndim):
                    Utmp = self.boundary.impose(Utmp, ax)
                Wb = self.eq.get_primitives_from_conservatives(Utmp)
                V_next = jnp.stack([Wb[self.i_u], Wb[self.i_v], Wb[self.i_w]], axis=0)
                B_next = jnp.stack([Wb[self.iBx], Wb[self.iBy], Wb[self.iBz]], axis=0)

            # residual
            res = l2_norm(V_next - V_k, B_next - B_k)
            res0 = jnp.where(k == 0, jnp.maximum(res, self.eps), res0)

            return (k + 1, V_next, B_next, res0)

        def cond(state):
            k, V_k, B_k, res0 = state
            res = l2_norm(V_k - V_init, B_k - B_init)
            rel = res / (res0 + self.eps)
            return jnp.logical_and(k < self.n_iter_max, rel > self.tol)

        V_init = V; B_init = B
        init_state = (0, V_init, B_init, 0.0)
        # Use a checkpointed while_loop so reverse-mode is memory-light
        kf, Vf, Bf, res0 = jax.lax.while_loop(cond, jax.remat(body), init_state)

        # Compute total energy of old state (conservative)
        E_old = sol[self.iE]
        
        # Compute kinetic and magnetic contributions of new fields
        kin_new = 0.5 * rho * (Vf[0]**2 + Vf[1]**2 + Vf[2]**2)
        mag_new = 0.5 * (Bf[0]**2 + Bf[1]**2 + Bf[2]**2)
        
        # Recover internal energy from total conservation
        E_int_new = E_old - kin_new - mag_new
        E_int_new = jnp.maximum(E_int_new, self.eps)
        
        # Convert back to primitive pressure
        p_new = (self.eq.gamma - 1.0) * E_int_new
        
        # Rebuild primitives with updated v, B, and new pressure
        W_upd = (W.at[self.i_u].set(Vf[0])
                   .at[self.i_v].set(Vf[1])
                   .at[self.i_w].set(Vf[2])
                   .at[self.iBx].set(Bf[0])
                   .at[self.iBy].set(Bf[1])
                   .at[self.iBz].set(Bf[2])
                   .at[self.iE].set(p_new))
        
        U_upd = self.eq.get_conservatives_from_primitives(W_upd)
        
        # Optional final boundary
        if self.boundary is not None:
            for ax in range(1, U_upd.ndim):
                U_upd = self.boundary.impose(U_upd, ax)

        return U_upd