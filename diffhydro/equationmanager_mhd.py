from jax import Array
import jax.numpy as jnp

class EquationManagerMHD:
    """
    Ideal MHD Equation Manager (semi-discrete / MOL friendly).
    Conservative state U = [rho, rho*u, rho*v, rho*w, B1, B2, B3, E]
    Primitive state   W = [rho, u, v, w, B1, B2, B3, p]
    Matches the API of existing EquationManager.
    """
    def __init__(self):
        # indices & meta to match your style
        self.mass_ids = 0
        self.vel_ids = (1, 2, 3)          # velocity components in primitives
        self.velocity_minor_axes = ((2, 3), (3, 1), (1, 2))

        self.mag_ids = (4, 5, 6)          # magnetic field components in primitives
        self.energy_ids = -1              # last entry is p (prims) / E (cons)
        self.equation_type = "MHD"
        self.gamma = 1.4
        self.R = 1.0
        self.cp = self.gamma / (self.gamma - 1.0) * self.R
        self.eps = 1e-20
        self.cfl = 0.3
        self.mesh_shape = [100, 100, 100]  # will be overwritten by caller
        self.n_cons = 8                    # 8 vars (9 if you add GLM-psi)

    # ---------- primitive <-> conservative ----------
    def get_conservatives_from_primitives(self, primitives: Array) -> Array:
        rho = primitives[self.mass_ids]
        u = primitives[self.vel_ids[0]]
        v = primitives[self.vel_ids[1]]
        w = primitives[self.vel_ids[2]]
        B1 = primitives[self.mag_ids[0]]
        B2 = primitives[self.mag_ids[1]]
        B3 = primitives[self.mag_ids[2]]
        p  = primitives[self.energy_ids]

        rhou, rhov, rhow = rho*u, rho*v, rho*w
        ke = 0.5 * rho * (u*u + v*v + w*w)
        me = 0.5 * (B1*B1 + B2*B2 + B3*B3)
        e_int = self.get_specific_energy(p, rho) * rho  # = p/(gamma-1)

        E = ke + me + e_int
        return jnp.stack([rho, rhou, rhov, rhow, B1, B2, B3, E], axis=0)


    def get_primitives_from_conservatives(self, U):
        # Local aliases
        rho = U[self.mass_ids]
        E   = U[self.energy_ids]
    
        # Safe density and inverse
        rho_safe = jnp.maximum(rho, self.eps)
        inv_rho  = 1.0 / rho_safe
    
        # Safe velocities via guarded divide (no where-branching that still divides by 0 under JIT)
        u_i, v_i, w_i = self.vel_ids
        u = jnp.divide(U[u_i], rho_safe)
        v = jnp.divide(U[v_i], rho_safe) if len(self.vel_ids) > 1 else 0.0
        w = jnp.divide(U[w_i], rho_safe) if len(self.vel_ids) > 2 else 0.0
    
        # Magnetic energy from B in conservatives (cell-centered B stored directly)
        Bsq = 0.0
        if hasattr(self, "mag_ids"):
            for bi in self.mag_ids:
                Bsq = Bsq + U[bi] * U[bi]
        me = 0.5 * Bsq
    
        # Kinetic energy
        ke = 0.5 * rho_safe * (u*u + v*v + w*w)
    
        # Safe internal energy and pressure
        E_safe   = jnp.nan_to_num(E, nan=self.eps)     # do not inject huge numbers
        e_int    = E_safe - ke - me
        e_int    = jnp.where(jnp.isfinite(e_int), e_int, self.eps)
        e_int    = jnp.maximum(e_int, self.eps)
        p        = (self.gamma - 1.0) * e_int
    
        # Assemble primitives
        if hasattr(self, "mag_ids"):
            B = [U[bi] for bi in self.mag_ids]
            W = jnp.stack([rho_safe, u, v, w, *B, p], axis=0)
        else:
            W = jnp.stack([rho_safe, u, v, w, p], axis=0)
    
        # Final finite check (cheap)
        W = jnp.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        return W
    
    def get_primitives_from_conservatives_old(self, conservatives: Array) -> Array:
#        conservatives = self._sanitize_cons(conservatives)
        rho = conservatives[self.mass_ids]
        inv_rho = 1.0 / (rho+ self.eps)
        rhou = conservatives[self.vel_ids[0]]
        rhov = conservatives[self.vel_ids[1]]
        rhow = conservatives[self.vel_ids[2]]
        u, v, w = rhou * inv_rho, rhov * inv_rho, rhow * inv_rho

        B1 = conservatives[4]; B2 = conservatives[5]; B3 = conservatives[6]
        E  = conservatives[self.energy_ids]

        ke = 0.5 * rho * (u*u + v*v + w*w)
        me = 0.5 * (B1*B1 + B2*B2 + B3*B3)
        e_int = jnp.maximum(E - ke - me, self.eps) * inv_rho
        p = self.get_pressure(e_int, rho)

        return jnp.stack([rho, u, v, w, B1, B2, B3, p], axis=0)

    # ---------- physical flux in direction 'axis' (0,1,2) ----------
    def get_fluxes_xi(self, primitives: Array, conservatives: Array, axis: int) -> Array:
        """
        Ideal MHD flux (finite-volume form), consistent with Euler get_fluxes_xi
        signature and axis handling (pressure added to the proper momentum component)
        """
        rho = primitives[self.mass_ids]
        u = primitives[self.vel_ids[0]]
        v = primitives[self.vel_ids[1]]
        w = primitives[self.vel_ids[2]]
        B1 = primitives[4]; B2 = primitives[5]; B3 = primitives[6]
        p  = primitives[self.energy_ids]

        v_vec = (u, v, w)
        B_vec = (B1, B2, B3)
        vd = v_vec[axis]
        Bd = B_vec[axis]

        vdotB = u*B1 + v*B2 + w*B3
        B2sum = B1*B1 + B2*B2 + B3*B3
        ptot = p + 0.5 * B2sum

        # mass flux
        F_rho = rho * vd

        # momentum flux: rho v v_d + (p + B^2/2) e_d - B B_d
        F_m1 = rho * u * vd
        F_m2 = rho * v * vd
        F_m3 = rho * w * vd
        if axis == 0:
            F_m1 = F_m1 + ptot - B1 * Bd
            F_m2 = F_m2 - B2 * Bd
            F_m3 = F_m3 - B3 * Bd
        elif axis == 1:
            F_m1 = F_m1 - B1 * Bd
            F_m2 = F_m2 + ptot - B2 * Bd
            F_m3 = F_m3 - B3 * Bd
        else:
            F_m1 = F_m1 - B1 * Bd
            F_m2 = F_m2 - B2 * Bd
            F_m3 = F_m3 + ptot - B3 * Bd

        # induction (magnetic field) flux: v_d B - B_d v
        F_B1 = vd * B1 - Bd * u
        F_B2 = vd * B2 - Bd * v
        F_B3 = vd * B3 - Bd * w

        # energy flux: (E + p + B^2/2) v_d - (v·B) B_d
        E = conservatives[self.energy_ids]
        F_E = (E + ptot) * vd - vdotB * Bd

        return jnp.stack([F_rho, F_m1, F_m2, F_m3, F_B1, F_B2, F_B3, F_E], axis=0)

    # ---------- thermodynamics helpers (same names as Euler) ----------
    def get_specific_heat_capacity(self, T: Array):
        return self.cp

    def get_specific_heat_ratio(self, T: Array):
        return self.gamma

    def get_pressure(self, e: Array, rho: Array) -> Array:
        # e = specific internal energy
        return (self.gamma - 1.0) * jnp.maximum(e, self.eps) * jnp.maximum(rho, self.eps)

    def get_temperature(self, p: Array, rho: Array) -> Array:
        return p / (jnp.maximum(rho, self.eps) * self.R + self.eps)

    def get_specific_energy(self, p: Array, rho: Array) -> Array:
        return p / (jnp.maximum(rho, self.eps) * (self.gamma - 1.0))

    def get_fast_magnetosonic_speed(self, primitives: Array, axis: int) -> Array:
        rho = jnp.maximum(primitives[self.mass_ids], self.eps)
        u = primitives[self.vel_ids[0]]
        v = primitives[self.vel_ids[1]]
        w = primitives[self.vel_ids[2]]
        B1, B2, B3 = primitives[4], primitives[5], primitives[6]
        p  = jnp.maximum(primitives[self.energy_ids], self.eps)

        cs2 = self.gamma * p / rho
        B2sum = B1*B1 + B2*B2 + B3*B3
        va2 = B2sum / jnp.maximum(rho, self.eps)

        # component of B along axis
        Bd = (B1, B2, B3)[axis]
        va_d2 = (Bd*Bd) / jnp.maximum(rho, self.eps)

        # fast magnetosonic speed squared (1D projection)
        term = jnp.sqrt(jnp.maximum((cs2 + va2)*(cs2 + va2) - 4.0*cs2*va_d2, 0.0))
        cf2 = 0.5 * ((cs2 + va2) + term)
        cf = jnp.sqrt(cf2)

        vabs_d = jnp.abs((u, v, w)[axis])
        return vabs_d + cf
        
    def get_signal_speed(self, primitives, axis: int):
    # Euler managers can implement this too (return get_speed_of_sound from p,rho).
        return self.get_fast_magnetosonic_speed(primitives, axis)
