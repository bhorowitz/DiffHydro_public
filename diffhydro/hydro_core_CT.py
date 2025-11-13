
from jax import Array 
from functools import partial
from typing import List, Optional, Tuple
import jax
from jax import jit, lax
import jax.numpy as jnp

from diffhydro import NoBoundary, NoForcing
from .solver.integrator import INTEGRATOR_DICT


@jax.tree_util.register_pytree_node_class
class hydro:
    """
    Hydro core with Method-of-Lines (MOL) path and
    Constrained Transport (CT) applied on the updated state.
    """
    #TO DO, pretty up this area...
    def __init__(self,
                 n_super_step = 600,
                 max_dt = 0.5, 
                 boundary = NoBoundary,
                 snapshots = False,
                 splitting_schemes=[[3,1,2,2,1,3],[1,2,3,3,2,1],[2,3,1,1,3,2]], #cyclic permutations
                 fluxes = None, #convection, conduction
                 forces = [NoForcing()], #gravity, etc.
                 maxjit=True,
                 use_mol=True,
                 use_ct=True,
                 integrator="RK2"):
        #parameters that are held constant per run (i.e. probably don't want to take derivatives with respect to...)
        self.splitting_schemes = splitting_schemes #strang splitting for x,y,z sweeps
        self.max_dt = max_dt
        self.boundary = boundary
        #supersteps, each superstep has len(splitting_schemes) time steps
        self.n_super_step = n_super_step
        self.snapshots = snapshots #poorly named
        self.outputs = []
        self.fluxes = fluxes or []
        self.forces = forces or []
        self.maxjit = maxjit
        self.dx_o = 1.0
        self.timescale = jnp.zeros(self.n_super_step)
        self.use_mol = use_mol
        self.integrator = INTEGRATOR_DICT[integrator]  # callable
        self._integrator_name = integrator
        self.use_ct = use_ct
        print("using CT?", use_ct)

        # indices expected by EquationManagerMHD; leave generic
        self.iBx, self.iBy, self.iBz = 4, 5, 6  # if Euler run, these rows may not exist

        #square for now...
        self.dx = self.dx_o
        self.dy = self.dx_o
        self.dz = self.dx_o

    # ---------------- timing/flux/forcing ----------------

    def timestep(self, fields):
        dt_list = []
        for flux in self.fluxes:
            dt_list.append(flux.timestep(fields))
        for force in self.forces:
            dt_list.append(force.timestep(fields))
        # print("dt", dt_list)  # noisy, keep optional
        return jnp.min(jnp.array(dt_list))

    def flux(self, sol, ax, params):
        total_flux = jnp.zeros(sol.shape)
        for flux in self.fluxes: 
            # ordered: allows a flux_correction depending on calculated fluxes
            total_flux = total_flux + flux.flux(sol, ax, params, total_flux)
        return total_flux

    def forcing(self, i, sol, params, dt): # axis independent
        for force in self.forces:
            sol = force.force(i, sol, params, dt)  # each returns UPDATED fields
        return sol

    # ---------------- split-sweeps (unchanged) ----------------

    def split_solve_step(self, sol, dt, ax, params):
        ##RK2 method
        fu1 = self.flux(sol, ax, params)
        rhs_cons = (fu1 - jnp.roll(fu1, 1, axis=ax)) 
        u1 = sol - rhs_cons * dt / (2.0 * self.dx_o)
        fu = self.flux(sol, ax, params)  
        rhs_cons = (fu - jnp.roll(fu, 1, axis=ax))
        sol = sol - (rhs_cons) * dt / self.dx_o
        return sol

    @jax.checkpoint
    def sweep_stack(self, state, dt, i):
        sol, params = state
        for scheme in self.splitting_schemes:
            for nn, ax in enumerate(scheme):
                sol = self.boundary.impose(sol, ax)
                sol = self.split_solve_step(sol, dt/(len(scheme)), int(ax), params)
                # experimental sign fixes (kept as in original, optional)
                sol = sol.at[0].set(jnp.abs(sol[0]))
                sol = sol.at[-1].set(jnp.abs(sol[-1]))
        return sol

    # ---------------- evolve loop ----------------

    def evolve(self, input_fields, params):
        self.outputs=[]
        state = (input_fields, params, jnp.array(0.0))

        if self.maxjit:
            state  = jax.lax.fori_loop(0, self.n_super_step, self.hydrostep_adapt, state)
        else:
            for i in range(0, self.n_super_step):
                state = self.hydrostep_adapt(i, state)
                if self.snapshots:
                    if i % self.snapshots == 0:
                        self.outputs.append(state)
        return state

    @partial(jit, static_argnums=0)
    def hydrostep_adapt(self, i, state):
        fields, params, t = state
        ttt = self.timestep(fields)
        ttt = jnp.minimum(self.max_dt, ttt)
        dt = ttt
        fields2, params2 = self._hydrostep(i, (fields, params), dt)  # unchanged _hydrostep
        return (fields2, params2, t + dt)

    @jax.jit
    def _hydrostep(self, i, state, dt):
        # split forcing outside of core hydro loop
        fields, params = state
        fields = self.forcing(i, fields, params, dt/2)

        if self.use_mol and self.use_ct:
           # jax.debug.print("use ct")
            fields = self.mol_solve_step_ct(fields, dt, params)  # <<< unsplit (MOL + CT-on-state)
        elif self.use_mol:
            fields = self.mol_solve_step(fields, dt, params)  # <<< unsplit (MOL)
        else:
            fields = self.sweep_stack(state, dt, i)

        fields = self.forcing(i, fields, params, dt/2)
        return (fields, params)

    # ---------------- RHS (MOL) ----------------

    def rhs_unsplit(self, sol, params):
        """
        Standard flux divergence RHS for all variables EXCEPT magnetic components,
        which will be advanced via CT on the updated state (to preserve ∇·B).
        """
        rhs = jnp.zeros_like(sol)
        # accumulate as usual
        for ax in range(1, sol.ndim):
            sol_b = sol
            fu = self.flux(sol_b, ax, params)
            rhs = rhs - (fu - jnp.roll(fu, 1, axis=ax)) / self.dx_o

        # Zero out magnetic rows; CT will advance them after the integrator stage
        if True:
            rhs = rhs.at[self.iBx].set(0.0)
            rhs = rhs.at[self.iBy].set(0.0)
            rhs = rhs.at[self.iBz].set(0.0)
        return rhs

    # ---------------- MOL + CT-on-updated-state ----------------

    def mol_solve_step_ct(self, sol, dt, params):
        """
        MOL with CT applied on the UPDATED state, using this step's fluxes.
        For SSPRK3 / RK2 integrators we inline the stages to insert CT after each stage.
        For other integrators we apply CT once after the full step (fallback).
        """
        name = self._integrator_name.upper()

        if name in ("SSPRK3", "RK3", "SSP3"):
            # --- SSPRK(3,3) ---
            # stage 1
            k1 = self.rhs_unsplit(sol, params); u1 = sol + dt * k1
            u1 = self._apply_ct_on_state(u1, params, dt)

            # stage 2
            k2 = self.rhs_unsplit(u1, params); u2 = 0.75 * sol + 0.25 * (u1 + dt * k2)
            # Effective substep on convex combo -> 0.25*dt contributes to new part; use 0.25*dt for CT
            u2 = self._apply_ct_on_state(u2, params, 0.25 * dt)

            # stage 3
            k3 = self.rhs_unsplit(u2, params); u3 = (1.0/3.0) * sol + (2.0/3.0) * (u2 + dt * k3)
            # Effective increment is (2/3)*dt on the last convex part; apply CT with that weight
            u3 = self._apply_ct_on_state(u3, params, (2.0/3.0) * dt)
            return u3

        elif name in ("RK2", "HEUN", "MIDPOINT"):
            # --- RK2 (Heun) ---
            k1 = self.rhs_unsplit(sol, params); u1 = sol + dt * k1
            u1 = self._apply_ct_on_state(u1, params, dt)

            k2 = self.rhs_unsplit(u1, params)
            u2_pred = sol + 0.5 * dt * (k1 + k2)
            # Apply CT for the second half contribution (0.5*dt)
            u2 = self._apply_ct_on_state(u2_pred, params, 0.5*dt) #0.5 dt?
            return u2

        else:
            # Unknown integrator: apply CT once
            u = self.integrator(self.rhs_unsplit, sol, dt, params)
            u = self._apply_ct_on_state(u, params, dt)
            return u

    def _apply_ct_on_state_old(self, sol, params, dt):
        """
        Constrained Transport applied to the *updated* state (MOL path).
       
        Works in 2D and 3D (if the state has 3 spatial dims).
        """
        # If no magnetic rows_present, nothing to do
        if sol.shape[0] <= self.iBy:
            return sol

        # 1) Per-axis fluxes on the UPDATED state
        Fx = self.flux(sol, 1, params)  # (vars, x, y[, z])
        Fy = self.flux(sol, 2, params)
        Fz = self.flux(sol, 3, params) if sol.ndim >= 4 else None

        # 2) EMF mapping from magnetic flux rows (face-centered)
        #   Fx[By] = -E_z,  Fx[Bz] = +E_y
        #   Fy[Bx] = +E_z,  Fy[Bz] = -E_x
        #   Fz[Bx] = -E_y,  Fz[By] = +E_x
        Ez_face = 0.5 * (-Fx[self.iBy] + Fy[self.iBx])

        if sol.ndim == 3:
            # ---------- 2D (vars, x, y) ----------
            # corners (i+1/2, j+1/2) from faces: average over x(0) and y(1)
            Ez_corner = 0.25 * (
                Ez_face
                + jnp.roll(Ez_face, -1, axis=0)
                + jnp.roll(Ez_face, -1, axis=1)
                + jnp.roll(jnp.roll(Ez_face, -1, axis=0), -1, axis=1)
            )

            # curl(-E_z k̂):
            # dBx/dt on x-faces =  ∂(-Ez)/∂y  ;  dBy/dt on y-faces = -∂(-Ez)/∂x
            dbx_face = ( -Ez_corner + jnp.roll(-Ez_corner, 1, axis=1) ) / self.dx_o   # derivative in y
            dby_face = (  Ez_corner - jnp.roll( Ez_corner, 1, axis=0) ) / self.dx_o   # derivative in x

            # face → cell-center averages along the normal axis
            dBx = 0.5 * (dbx_face + jnp.roll(dbx_face, 1, axis=0))  # average along x
            dBy = 0.5 * (dby_face + jnp.roll(dby_face, 1, axis=1))  # average along y

            sol = sol.at[self.iBx].add(dt * dBx)
            sol = sol.at[self.iBy].add(dt * dBy)
            return sol

        # ---------- 3D (vars, x, y, z) ----------
        # Additional EMFs from other flux rows
        Ex_face = 0.5 * ((-Fy[self.iBz]) + Fz[self.iBy])
        Ey_face = 0.5 * (( Fx[self.iBz]) - Fz[self.iBx])

        def avg4(A, ax_a, ax_b):
            """Average A with neighbors shifted by -1 along (ax_a, ax_b) in A's own axes."""
            A1 = jnp.roll(A, -1, axis=ax_a)
            A2 = jnp.roll(A, -1, axis=ax_b)
            A3 = jnp.roll(A1, -1, axis=ax_b)
            return 0.25 * (A + A1 + A2 + A3)

        # After slicing var, EMFs are (x,y,z) with axes (0,1,2)
        Ez_corner = avg4(Ez_face, 0, 1)  # x–y corners
        Ex_corner = avg4(Ex_face, 1, 2)  # y–z corners
        Ey_corner = avg4(Ey_face, 0, 2)  # x–z corners

        # dB/dt = -curl(E) using corner EMFs
        dBx_face = (-(Ez_corner - jnp.roll(Ez_corner, 1, axis=1)) / self.dx_o   # -∂Ez/∂y
                    + ( Ey_corner - jnp.roll( Ey_corner, 1, axis=2)) / self.dx_o)  # +∂Ey/∂z
        dBy_face = (-( Ex_corner - jnp.roll( Ex_corner, 1, axis=2)) / self.dx_o   # -∂Ex/∂z
                    + ( Ez_corner - jnp.roll(Ez_corner, 1, axis=0)) / self.dx_o)  # +∂Ez/∂x
        dBz_face = (-( Ey_corner - jnp.roll( Ey_corner, 1, axis=0)) / self.dx_o   # -∂Ey/∂x
                    + ( Ex_corner - jnp.roll( Ex_corner, 1, axis=1)) / self.dx_o)  # +∂Ex/∂y

        # face → cell-center averages along normal axes
        dBx = 0.5 * (dBx_face + jnp.roll(dBx_face, 1, axis=0))  # along x
        dBy = 0.5 * (dBy_face + jnp.roll(dBy_face, 1, axis=1))  # along y
        dBz = 0.5 * (dBz_face + jnp.roll(dBz_face, 1, axis=2))  # along z

        sol = sol.at[self.iBx].add(dt * dBx)
        sol = sol.at[self.iBy].add(dt * dBy)
        sol = sol.at[self.iBz].add(dt * dBz)
        return sol

    def _apply_ct_on_state(self, sol, params, dt):
        """
        This one actually keeps divergence down, but seems to introduce more wiggles sooner...
        Constrained Transport applied to the *updated* state (MOL path).
 
        Works in 2D and 3D (if the state has 3 spatial dims).
        """
        # If no magnetic rows present, nothing to do
        if sol.shape[0] <= self.iBy:
            return sol

        # 1) Per-axis fluxes on the UPDATED state
        Fx = self.flux(sol, 1, params)  # (vars, x, y[, z])
        Fy = self.flux(sol, 2, params)
        Fz = self.flux(sol, 3, params) if sol.ndim >= 4 else None

        if sol.ndim == 3:
            # ---------- 2D (vars, x, y) ----------
            # EMF mapping: Fx[By] = -Ez, Fy[Bx] = +Ez
            Ez_face = 0.5 * (-Fx[self.iBy] + Fy[self.iBx])

            # Corner average (4-pt avg to edge centers)
            Ez_corner = 0.25 * (
                Ez_face
              + jnp.roll(Ez_face, -1, axis=0)
              + jnp.roll(Ez_face, -1, axis=1)
              + jnp.roll(jnp.roll(Ez_face, -1, axis=0), -1, axis=1)
            )

            # Induction: ∂Bx/∂t = -∂Ez/∂y, ∂By/∂t = +∂Ez/∂x
            # Using backward differences: (f[i] - f[i-1]) / dx
            dBx = -(Ez_corner - jnp.roll(Ez_corner, 1, axis=1)) / self.dy
            dBy =  (Ez_corner - jnp.roll(Ez_corner, 1, axis=0)) / self.dx

            # Update B fields (Ez_corner is already at cell centers)
            sol = sol.at[self.iBx].add(dt * dBx)
            sol = sol.at[self.iBy].add(dt * dBy)
            return sol

        # ---------- 3D (vars, x, y, z) ----------
        # EMF mapping from magnetic flux rows:
        #   Fx[By] = -Ez,  Fx[Bz] = +Ey
        #   Fy[Bx] = +Ez,  Fy[Bz] = -Ex
        #   Fz[Bx] = -Ey,  Fz[By] = +Ex
        Ez_face = 0.5 * (-Fx[self.iBy] + Fy[self.iBx])
        Ex_face = 0.5 * (-Fy[self.iBz] + Fz[self.iBy])
        Ey_face = 0.5 * ( Fx[self.iBz] - Fz[self.iBx])

        def avg4(A, ax_a, ax_b):
            """Average A with neighbors shifted by -1 along (ax_a, ax_b)."""
            A1 = jnp.roll(A, -1, axis=ax_a)
            A2 = jnp.roll(A, -1, axis=ax_b)
            A3 = jnp.roll(A1, -1, axis=ax_b)
            return 0.25 * (A + A1 + A2 + A3)

        # Average EMFs to edge centers
        # Ez lives on x-y edges (average over x,y)
        # Ex lives on y-z edges (average over y,z)
        # Ey lives on x-z edges (average over x,z)
        Ez_corner = avg4(Ez_face, 0, 1)  # x–y edges
        Ex_corner = avg4(Ex_face, 1, 2)  # y–z edges
        Ey_corner = avg4(Ey_face, 0, 2)  # x–z edges

        # Compute dB/dt = -curl(E) using edge-centered EMFs
        # ∂Bx/∂t = -∂Ez/∂y + ∂Ey/∂z
        # ∂By/∂t = -∂Ex/∂z + ∂Ez/∂x
        # ∂Bz/∂t = -∂Ey/∂x + ∂Ex/∂y
        # Using backward differences: (f[i] - f[i-1]) / d_axis
        dBx = (-(Ez_corner - jnp.roll(Ez_corner, 1, axis=1)) / self.dy
               +(Ey_corner - jnp.roll(Ey_corner, 1, axis=2)) / self.dz)

        dBy = (-(Ex_corner - jnp.roll(Ex_corner, 1, axis=2)) / self.dz
               +(Ez_corner - jnp.roll(Ez_corner, 1, axis=0)) / self.dx)

        dBz = (-(Ey_corner - jnp.roll(Ey_corner, 1, axis=0)) / self.dx
               +(Ex_corner - jnp.roll(Ex_corner, 1, axis=1)) / self.dy)

        # Update B fields (EMFs are already at cell centers after averaging)
        sol = sol.at[self.iBx].add( dt * dBx)
        sol = sol.at[self.iBy].add( dt * dBy)
        sol = sol.at[self.iBz].add( dt * dBz)

        return sol

    # ---------------- JAX pytree plumbing ----------------

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def tree_flatten(self):
        # this method is needed for JAX control flow; keep arrays out of aux_data
        children = ()  # arrays / dynamic values
        aux_data = {
            "boundary": self.boundary,
            "snapshots": self.snapshots,
            "splitting_schemes": self.splitting_schemes,
            "fluxes": self.fluxes,
            "forces": self.forces,
            "maxjit": self.maxjit,
            "use_mol": self.use_mol,
            "use_ct": self.use_ct,
            "integrator": self._integrator_name,
            "n_super_step": self.n_super_step,
            "max_dt": self.max_dt,
        }  # static values
        return (children, aux_data)
