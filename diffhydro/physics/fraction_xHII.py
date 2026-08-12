"""Non-equilibrium ionization fraction of hydrogen, x_HII.

Solves eq. (28') of the reduced RAMSES-RT system (hydrogen only, one photon
group at the ionization threshold):

    n_H dx_HII/dt = n_HI (beta_HI n_e + sigma_HI c N_gamma)
                    - n_HII alpha^A_HII n_e

Dividing by n_H and using n_HI/n_H = 1 - x, n_HII/n_H = x:

    dx/dt = (1 - x) (beta_HI n_e + sigma_HI c N_gamma) - x alpha^A_HII n_e

Time integration is explicit Euler (Delta x = xdot Delta t), with the
positivity/boundedness limiter requested for the chemistry: a single step
can never move x by more than ``max_frac`` of the distance to the bound it
is heading for, so x stays in [0, 1] for any stiffness.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax
from . import hydrogen_chemistry as hchem
from .hydrogen_chemistry import HydrogenStateView


class HydrogenIonizationForce:
    """Force term updating the x_HII slot of the combined state.

    Parameters
    ----------
    stellar_force :
        An already-configured :class:`StellarRadiationForce`. Only its index
        layout (``idx_rho``, ``idx_mom``, ``idx_xHII``, ...), its
        :class:`CodeUnits` and its hydro gamma are reused, so both forces
        always agree on where the fields live.
    case :
        ``"A"`` (default, consistent with eq. 28' and with re-injecting
        recombination photons via ``StellarRadiationForce.get_N_chemistry``)
        or ``"B"`` (on-the-spot approximation: then ``b_rec`` must be 0 in
        the photon equation, otherwise recombinations are counted twice).
    collisional :
        Include the Cen (1992) collisional ionization term beta_HI n_e.
    max_frac :
        Fractional limiter on |Delta x| per step (0.9 = at most 90 % of the
        way to the bound).
    """

    def __init__(
        self,
        stellar_force,
        case: str = "A",
        collisional: bool = True,
        max_frac: float = 0.9,
        frequency: float = hchem.NU_HI_CGS,
        eps: float = 1e-30,
        view: HydrogenStateView | None = None,
    ):
        self.sf = stellar_force
        self.case = case.upper()
        self.collisional = bool(collisional)
        self.max_frac = float(max_frac)
        self.frequency = float(frequency)
        self.eps = eps
        self.view = view if view is not None else _view_from_stellar_force(stellar_force)
        # cgs cross-section for the single group actually transported
        self.sigma_HI_cgs = float(
            hchem.sigma_HI_powerlaw_cgs(nu=self.frequency, nu0=hchem.NU_HI_CGS)
        )

    # ------------------------------------------------------------------
    # Rate coefficients (Kelvin in, cgs out) -- thin wrappers kept so the
    # class stays self-documenting; the formulas live in hydrogen_chemistry.
    # ------------------------------------------------------------------
    def beta_HI(self, T_K):
        """Collisional ionization rate coefficient [cm^3 s^-1], Cen (1992)."""
        return hchem.beta_HI_cgs(T_K)

    def alpha_HII(self, T_K):
        """Recombination rate coefficient [cm^3 s^-1], Hui & Gnedin (1997)."""
        return (hchem.alpha_A_HII_cgs(T_K) if self.case == "A"
                else hchem.alpha_B_HII_cgs(T_K))

    # ------------------------------------------------------------------
    def dxdt_cgs(self, sol, x=None):
        """dx_HII/dt in s^-1, from the CONSERVATIVE state ``sol``."""
        view = self.view
        x = view.xHII(sol) if x is None else x
        T_K = view.temperature_K(sol, x)
        _, _, _, n_e = view.number_densities_cgs(sol, x)
        N_cgs = view.photon_density_cgs(sol)

        photo = self.sigma_HI_cgs * view.c_interaction_cgs * N_cgs
        colli = self.beta_HI(T_K) * n_e if self.collisional else 0.0
        recomb = self.alpha_HII(T_K) * n_e
        return (1.0 - x) * (photo + colli) - x * recomb

    def timestep(self, sol):
        # Explicit Euler + fractional limiter -> no extra CFL constraint.
        return jnp.inf

    def force(self, i, sol, params, dt):
        view = self.view
        x = view.xHII(sol)
        dt_s = view.dt_cgs(dt)

        dx = self.dxdt_cgs(sol, x) * dt_s

        # two-sided fractional limiter: never overshoot 0 or 1
        dx = jnp.where(
            dx > 0.0,
            jnp.minimum(dx, self.max_frac * (1.0 - x)),
            jnp.maximum(dx, -self.max_frac * x),
        )
        # dx = jnp.where(dx > 1.0, 1.0, jnp.where(dx < 0.0, 0.0, dx)) #check que faire plutart
        # jax.debug.print("max dx: {dx_max},min_dx: {dx_min},  dt: {dt}",dx_max=jnp.max(dx), dx_min=jnp.min(dx), dt=dt)#x: {x},x=x,
        x_new = jnp.clip(x + dx, 0.0, 1.0)

        return view.set_xHII(sol, x_new), params


class HydrogenPhotoChemistryForce:
    """Photon-CONSERVING coupled update of (N_gamma, F_gamma, x_HII).

    Why this exists
    ---------------
    Splitting the photon sink (in ``StellarRadiationForce.force``) from the
    ionization update (in :class:`HydrogenIonizationForce`) makes the result
    depend on the order of ``forces=[...]``: the sink runs first, so the
    ionization then reads an already-depleted N and produces fewer
    ionizations than the number of photons that were destroyed. The error is
    O(n_HI sigma c dt) and does NOT vanish with grid refinement, because
    refining the grid also refinexHII_weighted=True,s dt at fixed CFL only linearly while the
    absorption per cell stays optically thick. Measured on the Iliev+2006
    Test 1 setup it costs ~10 % on the ionization front radius.

    This force instead enforces, per step and per cell, the bookkeeping
    identity "one absorbed photon = one ionization":

        N_abs   = N [1 - exp(-n_HI sigma c dt)]     photons actually absorbed
        dx      = N_abs / n_H                        + collisional - recombination
        N      -= N_abs (capped so dx cannot exceed 1 - x; the photons that
                  would have over-ionized the cell are NOT destroyed)
        F      *= exp(-n_HI sigma c dt)              eq. (26')

    Use it INSTEAD of the (StellarRadiationForce chemistry sink +
    HydrogenIonizationForce) pair, i.e. build the stellar force with
    ``chemistry=False`` so it only injects photons.
    """

    def __init__(
        self,
        stellar_force,
        case: str = "B",
        collisional: bool = True,
        b_rec: float | None = None,
        max_frac: float = 0.9,
        frequency: float = hchem.NU_HI_CGS,
        f_max: float | None = None,
        view: HydrogenStateView | None = None,
    ):
        self.sf = stellar_force
        self.case = case.upper()
        self.collisional = bool(collisional)
        self.b_rec = (1.0 if self.case == "A" else 0.0) if b_rec is None else float(b_rec)
        self.max_frac = float(max_frac)
        self.view = view if view is not None else _view_from_stellar_force(stellar_force)
        self.sigma_HI_cgs = float(
            hchem.sigma_HI_powerlaw_cgs(nu=frequency, nu0=hchem.NU_HI_CGS)
        )
        self.f_max = getattr(stellar_force, "beam_reduced_flux", 0.95) if f_max is None else f_max

    def alpha_HII(self, T_K):
        return (hchem.alpha_A_HII_cgs(T_K) if self.case == "A"
                else hchem.alpha_B_HII_cgs(T_K))

    def timestep(self, sol):
        return jnp.inf

    def force(self, i, sol, params, dt):
        view = self.view
        dt_s = view.dt_cgs(dt)
        c_cgs = view.c_interaction_cgs

        x = view.xHII(sol)
        T_K = view.temperature_K(sol, x)
        n_H, n_HI, n_HII, n_e = view.number_densities_cgs(sol, x)
        N_cgs = view.photon_density_cgs(sol)

        # --- absorbed photons over dt (exact for a frozen n_HI) -----------
        # The requested "at most 90 % of N per step" limiter is applied HERE,
        # to the absorption itself, so that the photon budget removed from N
        # and the ionizations produced from it stay exactly equal (applying
        # it afterwards to dN would destroy that identity).
        k = n_HI * self.sigma_HI_cgs * c_cgs                  # s^-1
        absorbed_frac = jnp.minimum(-jnp.expm1(-k * dt_s), self.max_frac)
        N_abs = N_cgs * absorbed_frac

        # --- ionizations that these photons can actually produce ----------
        n_H_safe = jnp.maximum(n_H, hchem.tiny_like(sol))
        dx_photo_wanted = N_abs / n_H_safe
        headroom = self.max_frac * (1.0 - x)
        dx_photo = jnp.minimum(dx_photo_wanted, headroom)
        # photons in excess of what the cell can absorb are NOT destroyed
        N_abs_eff = dx_photo * n_H

        # --- collisional ionization and recombination ---------------------
        dx_coll = ((1.0 - x) * hchem.beta_HI_cgs(T_K) * n_e * dt_s
                   if self.collisional else 0.0)
        dx_rec = -x * self.alpha_HII(T_K) * n_e * dt_s
        dx_other = dx_coll + dx_rec
        dx_other = jnp.where(
            dx_other > 0.0,
            jnp.minimum(dx_other, self.max_frac * (1.0 - x - dx_photo)),
            jnp.maximum(dx_other, -self.max_frac * x),
        )

        x_new = jnp.clip(x + dx_photo + dx_other, 0.0, 1.0)

        # --- radiation field ----------------------------------------------
        # diffuse recombination photons (case A only)
        S = self.b_rec * (hchem.alpha_A_HII_cgs(T_K)
                          - hchem.alpha_B_HII_cgs(T_K)) * n_HII * n_e
        # N_abs_eff <= max_frac * N by construction, so N stays > 0 without
        # any further clamping -- which is what keeps the scheme exactly
        # photon conserving.
        dN_cgs = -N_abs_eff + S * dt_s

        sol = view.add_photons_cgs(sol, dN_cgs)
        sol = view.set_xHII(sol, x_new)

        # flux sink, eq. (26'), then M1 cone
        decay = jnp.exp(-k * dt_s)
        for j in view.idx_F:
            sol = sol.at[j].multiply(decay)
        Fx, Fy, Fz = hchem.limit_m1_flux_cone(
            sol[view.idx_N], *[sol[j] for j in view.idx_F],
            c=self.sf.light_speed, f_max=self.f_max,
        )
        for j, F in zip(view.idx_F, (Fx, Fy, Fz)):
            sol = sol.at[j].set(F)
        return sol, params


def _view_from_stellar_force(sf) -> HydrogenStateView:
    """Build a :class:`HydrogenStateView` matching a StellarRadiationForce."""
    hydro_eq = getattr(sf, "hydro_eq", None)
    gamma = getattr(hydro_eq, "gamma", 5.0 / 3.0)
    idx_mom = getattr(sf, "idx_mom")
    if isinstance(idx_mom, slice):
        idx_mom = tuple(range(idx_mom.start, idx_mom.stop))
    return HydrogenStateView(
        cu=sf.cu,
        gamma=gamma,
        idx_N=0,
        idx_F=(1, 2, 3),
        idx_xHII=sf.idx_xHII,
        idx_rho=sf.idx_rho,
        idx_mom=tuple(idx_mom),
        idx_Etot=sf.idx_energy,
        xHII_weight_idx=sf.idx_rho,
        X_H=getattr(sf, "X_H", 1.0),
        light_speed_code=getattr(sf, "light_speed", None),
    )
