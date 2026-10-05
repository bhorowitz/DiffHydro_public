
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

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from coupled_rhd import EquationManagerCoupled  # adjust import path if needed


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

    def timestep(self, U):
        rho = jnp.maximum(U[self.coupled_eq.i_rho], self.eps)
        n_H = rho / self.m_H
        x_HII = jnp.clip(U[self.coupled_eq.i_xHII] / rho, 0.0, 1.0)
        n_HI = n_H * (1.0 - x_HII)

        c = self.coupled_eq.rt_eq.light_speed
        absorption_rate = n_HI * c * self.sigma_HI       # [1/time], sink rate on N,F

        # limiter dt so that the absorption sink doesn't remove more photons
        # than are present within one (unsplit) step -- same spirit as
        # HeatCoolForce.timestep()
        tabs = 1.0 / jnp.maximum(absorption_rate, self.eps)
        return jnp.min(tabs)

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
