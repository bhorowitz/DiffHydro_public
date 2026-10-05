
"""
coupled_rhd.py

Implementation of "Strategy A": a single combined conservative state tensor

    sol.shape == (n_cons_total, Nx, Ny, Nz)

    hydro_slice = slice(0, 5)     # rho, vx, vy, vz, p            (hydro EOS)
    rt_slice    = slice(5, 9)     # E_gamma, Fx, Fy, Fz            (M1 RT)
    chem_slice  = slice(9, 10)    # x_HII (ionization fraction)    (passive,
                                    advected with the GAS velocity, weighted
                                    by rho, i.e. the conserved quantity is
                                    rho * x_HII)

Design principles
------------------
1. NO cross-block numerical flux: each block keeps its OWN Riemann solver /
   reconstruction / CFL. This is critical because c (radiative signal speed)
   is typically >> sound speed; mixing them into a single Lax-Friedrichs
   dissipation would massively over-diffuse the hydro block.
2. All PHYSICAL coupling (photoionization, photo-heating, recombination
   cooling) happens exclusively in `forces`, exactly like HeatCoolForce
   already does for radiative cooling. This mirrors the RAMSES-RT
   Strang-splitting philosophy already present in hydro_core (`_hydrostep`
   does: forcing(dt/2) -> hydro transport -> forcing(dt/2)).
3. EquationManagerCoupled is a thin *indexing* helper used by forces to find
   "where is rho", "where is E_gamma", "where is x_HII" in the combined
   tensor -- it does NOT itself perform Riemann solves. Flux dispatch is
   done by lightweight `BlockFlux` / `ChemBlockFlux` wrappers around your
   EXISTING hydro and RT ConvectiveFlux objects.

IMPORTANT -- names to double check against your actual codebase
-----------------------------------------------------------------
This file assumes the hydro EquationManager exposes the same conventions as
the RT one you shared: `.mass_ids`, `.vel_ids` (3-tuple), `.energy_ids`,
`.active_slice`, `.n_active`, `.n_cons`, `.cfl`, `.eps`,
`.get_conservatives_from_primitives`, `.get_primitives_from_conservatives`,
`.get_fluxes_xi(primitives, conservatives, axis)`. If your hydro
EquationManager names anything differently, adjust the small number of
`getattr(...)` / attribute accesses flagged with "# ADAPT IF NEEDED" below.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import jax
import jax.numpy as jnp
from jax import Array


# ============================================================================
# 1) EquationManagerCoupled -- pure indexing / conversion helper
# ============================================================================

@dataclass
class EquationManagerCoupled:
    hydro_eq: object   # instance of your hydro EquationManager (rho,vx,vy,vz,p)
    rt_eq: object       # instance of your radiative EquationManager (E,Fx,Fy,Fz)
    eps: float = 1e-10
    n_chem: int = 1     # number of chemistry passive scalars (x_HII only, for now)

    n_hydro: int = field(init=False)
    n_rt: int = field(init=False)
    n_cons: int = field(init=False)
    hydro_slice: slice = field(init=False)
    rt_slice: slice = field(init=False)
    chem_slice: slice = field(init=False)

    def __post_init__(self):
        # ADAPT IF NEEDED: hydro_eq.n_cons / rt_eq.n_cons must be the ACTIVE
        # counts only here (5 and 4), i.e. do NOT put x_HII as a passive
        # scalar inside hydro_eq or rt_eq -- it lives in its own chem_slice.
        self.n_hydro = int(getattr(self.hydro_eq, "n_active", self.hydro_eq.n_cons))
        self.n_rt = int(getattr(self.rt_eq, "n_active", self.rt_eq.n_cons))
        self.n_cons = self.n_hydro + self.n_rt + self.n_chem

        self.hydro_slice = slice(0, self.n_hydro)
        self.rt_slice = slice(self.n_hydro, self.n_hydro + self.n_rt)
        self.chem_slice = slice(self.n_hydro + self.n_rt, self.n_cons)

    # -- convenience absolute indices --------------------------------------
    @property
    def i_rho(self) -> int:
        return self.hydro_slice.start + self.hydro_eq.mass_ids

    @property
    def i_vel(self) -> tuple[int, int, int]:
        return tuple(self.hydro_slice.start + v for v in self.hydro_eq.vel_ids)

    @property
    def i_p(self) -> int:
        return self.hydro_slice.start + self.hydro_eq.energy_ids  # ADAPT IF NEEDED

    @property
    def i_Egamma(self) -> int:
        return self.rt_slice.start + self.rt_eq.mass_ids

    @property
    def i_Fgamma(self) -> tuple[int, int, int]:
        return tuple(self.rt_slice.start + v for v in self.rt_eq.vel_ids)

    @property
    def i_xHII(self) -> int:
        return self.chem_slice.start

    # -- full-state primitive <-> conservative conversion ------------------
    # Used ONLY for building/reading the combined state (e.g. initial
    # conditions, diagnostics) -- NOT by the flux/transport machinery.
    def get_conservatives_from_primitives(self, primitives: Array) -> Array:
        prim_hydro = primitives[self.hydro_slice]
        prim_rt = primitives[self.rt_slice]
        x_HII = primitives[self.chem_slice][0]

        cons_hydro = self.hydro_eq.get_conservatives_from_primitives(prim_hydro)
        cons_rt = self.rt_eq.get_conservatives_from_primitives(prim_rt)

        rho = jnp.maximum(prim_hydro[self.hydro_eq.mass_ids], self.eps)
        cons_chem = (rho * jnp.clip(x_HII, 0.0, 1.0))[jnp.newaxis, ...]

        return jnp.concatenate([cons_rt, cons_hydro, cons_chem], axis=0)

    def get_primitives_from_conservatives(self, conservatives: Array) -> Array:
        cons_hydro = conservatives[self.hydro_slice]
        cons_rt = conservatives[self.rt_slice]
        cons_chem = conservatives[self.chem_slice]

        prim_hydro = self.hydro_eq.get_primitives_from_conservatives(cons_hydro)
        prim_rt = self.rt_eq.get_primitives_from_conservatives(cons_rt)

        rho = jnp.maximum(prim_hydro[self.hydro_eq.mass_ids], self.eps)
        x_HII = jnp.clip(cons_chem[0] / rho, 0.0, 1.0)
        prim_chem = x_HII[jnp.newaxis, ...]

        return jnp.concatenate([ prim_rt,prim_hydro ,prim_chem], axis=0)


# ============================================================================
# 2) Flux wrappers: each block keeps ITS OWN Riemann solver / reconstruction
# ============================================================================

@dataclass
class BlockFlux:
    """
    Wraps an EXISTING ConvectiveFlux object (hydro's or the RT's, already
    configured with its own solver/reconstruction/dx) so it only touches its
    own slice of the combined tensor `sol`, and returns a full-shape array
    (zero outside its slice). This lets `hydro.flux()` -- which just sums
    `flux.flux(sol, ax, params, total_flux)` over `self.fluxes` -- transport
    several physically independent blocks without their numerical
    dissipation/CFL mixing together.
    """
    inner_flux: object     # e.g. your existing cf_hydro or cf_rt
    var_slice: slice
    n_cons_total: int

    def flux(self, sol, ax, params, total_flux_so_far):
        sol_block = sol[self.var_slice]
        zeros_block = jnp.zeros_like(sol_block)
        f_block = self.inner_flux.flux(sol_block, ax, params, zeros_block)

        if f_block.shape[0] != sol_block.shape[0]:
            # Some inner flux implementations return a reduced active-only shape.
            # Pad it to the block width before placing it back into the full state.
            if f_block.shape[0] < sol_block.shape[0]:
                pad_needed = sol_block.shape[0] - f_block.shape[0]
                f_block = jnp.pad(f_block, ((0, pad_needed),) + ((0, 0),) * (f_block.ndim - 1))

        pad_before = self.var_slice.start
        pad_after = self.n_cons_total - self.var_slice.stop
        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (sol.ndim - 1)
        return jnp.pad(f_block, pad_width)

    def timestep(self, sol):
        sol_block = sol[self.var_slice]
        return self.inner_flux.timestep(sol_block)


@dataclass
class ChemBlockFlux:
    """
    First-order upwind advection of the passive chemistry scalar
    (rho * x_HII), transported with the GAS velocity read from the hydro
    block. No independent CFL constraint: the hydro block's own BlockFlux
    already limits dt for this velocity field, so `timestep()` returns +inf.

    NOTE: first-order upwind is intentionally simple/robust for a scalar
    that must stay in [0, 1]; a higher-order scheme (PLM/TENO) would need an
    extra flux limiter to preserve boundedness. Swap in your own
    reconstruction here if you need better accuracy on ionization fronts.
    """
    hydro_eq: object
    hydro_slice: slice
    chem_slice: slice
    n_cons_total: int
    eps: float = 1e-10

    def flux(self, sol, ax, params, total_flux_so_far):
        cons_hydro = sol[self.hydro_slice]
        prim_hydro = self.hydro_eq.get_primitives_from_conservatives(cons_hydro)

        u_ax = prim_hydro[self.hydro_eq.vel_ids[ax-1]]          # (Nx,Ny,Nz)
        rho_xHII = sol[self.chem_slice][0]                     # (Nx,Ny,Nz)

        # simple upwind flux at the +1/2 face, matching the sign convention
        # already used by hydro's own flux()/roll_with_halo divergence
        u_pos = jnp.maximum(u_ax, 0.0)
        u_neg = jnp.minimum(u_ax, 0.0)
        flux_chem = u_pos * rho_xHII + u_neg * jnp.roll(rho_xHII, -1, axis=ax - 1)

        pad_before = self.chem_slice.start
        pad_after = self.n_cons_total - self.chem_slice.stop
        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (sol.ndim - 1)
        return jnp.pad(flux_chem[jnp.newaxis, ...], pad_width)

    def timestep(self, sol):
        return jnp.array(1e30)  # constrained by the hydro block already


# ============================================================================
# 3) Force: enforce x_HII in [0, 1] after every hydro transport sub-step
# ============================================================================

@dataclass
class ChemistryBoundsForce:
    """
    Zero-physics "force": just re-projects the chemistry block back onto
    [0, 1] after transport. Needed because the RK/MOL integrator advances
    the conservative array directly (rho * x_HII) without necessarily
    passing through get_primitives_from_conservatives at every stage, so
    numerical overshoot near sharp ionization fronts is NOT auto-clipped
    unless you add this force to the `forces=[...]` list (put it LAST, right
    after any chemistry/heating force).
    """
    coupled_eq: EquationManagerCoupled

    def timestep(self, sol):
        return jnp.array(1e30)

    def force(self, i_step, U, params, dt):
        rho = jnp.maximum(U[self.coupled_eq.i_rho], self.coupled_eq.eps)
        rho_xHII = U[self.coupled_eq.i_xHII]
        x_HII = jnp.clip(rho_xHII / rho, 0.0, 1.0)
        U = U.at[self.coupled_eq.i_xHII].set(rho * x_HII)
        return U, params


# ============================================================================
# 4) Force: photoionization + recombination + photo-heating coupling
# ============================================================================

@dataclass
class IonizationForce:
    """
    Couples the three blocks:
        E_gamma (rt_slice)  --> photoionization rate Gamma_HI
        x_HII   (chem_slice) <-> Gamma_HI, alpha_B(T) recombination
        p       (hydro_slice) <-- photo-heating from each ionization event

    ODE solved per cell, per substep:
        dx_HII/dt = Gamma_HI * (1 - x_HII) - alpha_B(T) * n_e * x_HII
        dE_th/dt  = heat_per_ionization * n_H * Gamma_HI * (1 - x_HII)
                    - (recombination cooling, optional, can be folded into
                       HeatCoolForce instead to avoid double-counting)

    Units: everything here is written in your CODE units. sigma_HI, alpha_B,
    heat_per_ionization, m_H must be pre-converted to code units exactly like
    ALPHA/BETA/GAMMA/MHKB are in HeatCoolForce -- do NOT mix cgs and code
    values.
    """
    coupled_eq: EquationManagerCoupled
    sigma_HI: float = 6.3e-18          # cm^2 (convert to code units before use)
    alpha_B: float = 2.59e-13          # cm^3/s at 1e4 K (convert to code units)
    heat_per_ionization: float = 6.4e-13   # erg per ionization (~13.6 eV, convert)
    m_H: float = 1.6726e-24            # g (convert to code units)
    subcycles: int = 10
    eps: float = 1e-30

    def timestep(self, U):
        # Ionization/recombination timescale limiter, same spirit as
        # HeatCoolForce.timestep(): dt_chem = ctime * min(x_HII / |dx_HII/dt|)
        rho = jnp.maximum(U[self.coupled_eq.i_rho], self.eps)
        n_H = rho / self.m_H
        E_gamma = jnp.maximum(U[self.coupled_eq.i_Egamma], self.eps)
        rho_xHII = U[self.coupled_eq.i_xHII]
        x_HII = jnp.clip(rho_xHII / rho, 0.0, 1.0)

        c = self.coupled_eq.rt_eq.light_speed
        Gamma_HI = self.sigma_HI * c * E_gamma
        n_e = n_H * x_HII
        dxdt = Gamma_HI * (1.0 - x_HII) - self.alpha_B * n_e * x_HII
        tchem = jnp.abs(jnp.maximum(x_HII, 1e-6) / jnp.maximum(jnp.abs(dxdt), self.eps))
        return jnp.min(tchem)

    def force(self, i_step, U, params, dt):
        rho = jnp.maximum(U[self.coupled_eq.i_rho], self.eps)
        n_H = rho / self.m_H
        p = U[self.coupled_eq.i_p]           # ADAPT IF NEEDED: pressure slot
        E_gamma = jnp.maximum(U[self.coupled_eq.i_Egamma], self.eps)
        rho_xHII0 = U[self.coupled_eq.i_xHII]
        x0 = jnp.clip(rho_xHII0 / rho, 0.0, 1.0)

        c = self.coupled_eq.rt_eq.light_speed
        dt_loc = dt / float(self.subcycles)

        def body(_, carry):
            x_cur, p_cur = carry
            Gamma_HI = self.sigma_HI * c * E_gamma
            n_e = n_H * x_cur

            dxdt = Gamma_HI * (1.0 - x_cur) - self.alpha_B * n_e * x_cur
            x_new = jnp.clip(x_cur + dxdt * dt_loc, 0.0, 1.0)

            # photo-heating: energy deposited per newly-ionized atom
            n_ionized_rate = n_H * Gamma_HI * (1.0 - x_cur)
            dE_heat = self.heat_per_ionization * n_ionized_rate * dt_loc
            p_new = p_cur + (self.coupled_eq.hydro_eq.gamma - 1.0) * dE_heat

            return (x_new, p_new)

        x_final, p_final = jax.lax.fori_loop(0, int(self.subcycles), body, (x0, p))

        U = U.at[self.coupled_eq.i_xHII].set(rho * x_final)
        U = U.at[self.coupled_eq.i_p].set(p_final)

        # Attenuate the local radiation field by absorption (optical-depth
        # style sink on E_gamma), consistent with photons actually being
        # consumed by ionization events over this step.
        n_ionized_total = n_H * (x_final - x0)
        n_photons_consumed_density = jnp.maximum(n_ionized_total, 0.0)
        E_gamma_new = jnp.maximum(E_gamma - n_photons_consumed_density, self.eps)
        U = U.at[self.coupled_eq.i_Egamma].set(E_gamma_new)

        return U, params



"""
PhotonAbsorptionForce -- implements eq. (25') and (26') of the reduced
(hydrogen-only, monochromatic) M1 RT system:

    dN/dt = - n_HI * c * sigma_HI^N * N
            + b_rec * [alpha_HII^A - alpha_HII^B] * n_HII * n_e        (25')

    dF/dt = - n_HI * c * sigma_HI^N * F                                 (26')

This is the ABSORPTION sink term of the radiation field: photons are
destroyed by HI photoionization (first term of 25'), and case-A
recombinations partially re-emit ionizing photons back into the field
(second term of 25', the "on-the-spot" correction; b_rec is 1 if you track
diffuse/recombination photons explicitly, 0 if you use the pure case-B
approximation i.e. all recombination photons are assumed reabsorbed locally
and simply dropped -- see Rosdahl+2013 sec. 2). The photon FLUX F only has
the absorption sink (no recombination source), consistent with (26').

Discretization, following the user's convention dN/dt ~ Delta_N/Delta_t:
we use a simple explicit (or optionally implicit) Euler update per sub-cycle,
matching exactly how IonizationForce sub-cycles (28').

This force acts ONLY on the RT block (coupled_eq.rt_slice) but READS number
densities from the hydro block (rho -> n_H) and the ionization state from
the chem block (x_HII -> n_HI, n_HII, n_e). It must be placed in `forces=[...]`
so that its `dt` argument matches the dt used by `IonizationForce`, since
both integrate the SAME physical sink/source over the same interval -- if
you subcycle one and not the other, N and x_HII will desynchronize.

Units: sigma_HI, alpha_HII_A, alpha_HII_B, m_H must be pre-converted to code
units (same story as IonizationForce / HeatCoolForce) -- do NOT pass cgs
values directly.
"""

@dataclass
class PhotonAbsorptionForce:
    coupled_eq: EquationManagerCoupled

    sigma_HI: float = 6.3e-18          # cm^2 -> CONVERT TO CODE UNITS before use
    alpha_HII_A: float = 4.18e-13      # cm^3/s, case A, T=1e4K -> CONVERT
    alpha_HII_B: float = 2.59e-13      # cm^3/s, case B, T=1e4K -> CONVERT
    m_H: float = 1.6726e-24            # g -> CONVERT TO CODE UNITS

    case: str = "B"                    # "A": b_rec=1 (recombination photons
                                        #      re-injected into N, on-the-spot)
                                        # "B": b_rec=0 (case-B, no re-injection;
                                        #      alpha_HII_B alone is used for the
                                        #      chemistry in IonizationForce)
    subcycles: int = 10
    implicit: bool = True              # implicit Euler on the sink term:
                                        # unconditionally stable for large
                                        # n_HI*c*sigma_HI*dt (usually stiff)
    eps: float = 1e-30

    def _b_rec_and_alpha(self):
        if self.case == "A":
            b_rec = 1.0
            d_alpha = self.alpha_HII_A - self.alpha_HII_B
        else:
            b_rec = 0.0
            d_alpha = 0.0
        return b_rec, d_alpha

    def timestep(self, sol):
            # print("timestepblast")
            return 1e30

    def force(self, i_step, U, params, dt):
        rho = jnp.maximum(U[self.coupled_eq.i_rho], self.eps)
        n_H = rho / self.m_H
        x_HII0 = jnp.clip(U[self.coupled_eq.i_xHII] / rho, 0.0, 1.0)
        n_HI = n_H * (1.0 - x_HII0)
        n_HII = n_H * x_HII0
        n_e = n_HII  # pure-H, fully ionized-only electron source

        N = jnp.maximum(U[self.coupled_eq.i_Egamma], self.eps)
        Fx, Fy, Fz = (U[i] for i in self.coupled_eq.i_Fgamma)

        c = self.coupled_eq.rt_eq.light_speed
        b_rec, d_alpha = self._b_rec_and_alpha()

        absorption_rate = n_HI * c * self.sigma_HI          # [1/time]
        recomb_source = b_rec * d_alpha * n_HII * n_e         # [photons/(vol*time)]

        dt_loc = dt / float(self.subcycles)

        def body(_, carry):
            N_cur, Fx_cur, Fy_cur, Fz_cur = carry

            if self.implicit:
                # implicit Euler on the sink: N_new = (N_cur + source*dt) / (1+rate*dt)
                # -- unconditionally stable & keeps N >= 0 even if rate*dt >> 1
                N_new = (N_cur + recomb_source * dt_loc) / (1.0 + absorption_rate * dt_loc)
                Fx_new = Fx_cur / (1.0 + absorption_rate * dt_loc)
                Fy_new = Fy_cur / (1.0 + absorption_rate * dt_loc)
                Fz_new = Fz_cur / (1.0 + absorption_rate * dt_loc)
            else:
                # explicit Euler: dN/dt ~ Delta_N/Delta_t, as requested
                dNdt = -absorption_rate * N_cur + recomb_source
                dFdt_common = -absorption_rate
                N_new = jnp.maximum(N_cur + dNdt * dt_loc, self.eps)
                Fx_new = Fx_cur + dFdt_common * Fx_cur * dt_loc
                Fy_new = Fy_cur + dFdt_common * Fy_cur * dt_loc
                Fz_new = Fz_cur + dFdt_common * Fz_cur * dt_loc

            return (N_new, Fx_new, Fy_new, Fz_new)

        N_final, Fx_final, Fy_final, Fz_final = jax.lax.fori_loop(
            0, int(self.subcycles), body, (N, Fx, Fy, Fz)
        )

        U = U.at[self.coupled_eq.i_Egamma].set(N_final)
        i_Fx, i_Fy, i_Fz = self.coupled_eq.i_Fgamma
        U = U.at[i_Fx].set(Fx_final)
        U = U.at[i_Fy].set(Fy_final)
        U = U.at[i_Fz].set(Fz_final)

        return U, params


# ============================================================================
# 5) Example wiring (adapt object construction to your actual constructors)
# ============================================================================

def build_coupled_hydro_example(hydro_eq, rt_eq, cf_hydro, cf_rt, dx_code, size_shape,
                                 stellar_force, heatcool_force, hydro_module):
    """
    hydro_eq        : your existing hydro EquationManager instance (n_active=5)
    rt_eq           : your existing radiative EquationManager instance (n_active=4)
    cf_hydro        : your existing hydro ConvectiveFlux (already built w/ solver+recon+dx)
    cf_rt           : your existing ConvectiveFlux_Radiative_transfer (dx=dx_code)
    stellar_force   : your existing StellarRadiationForce instance
    heatcool_force  : your existing HeatCoolForce instance (built against hydro_eq)
    hydro_module    : the module exposing `hydro` (i.e. `import diffhydro as dh`, pass dh)
    """
    coupled_eq = EquationManagerCoupled(hydro_eq=hydro_eq, rt_eq=rt_eq)

    n_cons_total = coupled_eq.n_cons  # 5 + 4 + 1 = 10

    hydro_flux = BlockFlux(inner_flux=cf_hydro, var_slice=coupled_eq.hydro_slice,
                            n_cons_total=n_cons_total)
    rt_flux = BlockFlux(inner_flux=cf_rt, var_slice=coupled_eq.rt_slice,
                         n_cons_total=n_cons_total)
    chem_flux = ChemBlockFlux(hydro_eq=hydro_eq, hydro_slice=coupled_eq.hydro_slice,
                               chem_slice=coupled_eq.chem_slice, n_cons_total=n_cons_total)

    ionization_force = IonizationForce(coupled_eq=coupled_eq)
    bounds_force = ChemistryBoundsForce(coupled_eq=coupled_eq)

    # ORDER MATTERS inside forces=[...]: stellar injection -> ionization/heating
    # -> cooling -> bounds clip LAST, so nothing downstream re-violates [0,1].
    hydrosim = hydro_module.hydro(
        n_super_step=600,
        fluxes=[hydro_flux, rt_flux, chem_flux],
        forces=[stellar_force, ionization_force, heatcool_force, bounds_force],
        dx=dx_code,
        max_dt=0.5,
    )

    sol0 = jnp.zeros((n_cons_total, size_shape, size_shape, size_shape))
    # ADAPT IF NEEDED: initialize rho, p (hydro_slice) to your background ISM
    # state; E_gamma, F (rt_slice) to 0; x_HII (chem_slice) to 0 (neutral gas)

    return coupled_eq, hydrosim, sol0
