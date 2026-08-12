"""
Hydrogen-only non-equilibrium chemistry and cooling: single source of truth.

Everything in this module is written in **cgs** (T in Kelvin, densities in
cm^-3, cross-sections in cm^2, rates in cm^3 s^-1, cooling in erg cm^3 s^-1).
Conversions to/from code units happen in exactly one place --
:class:`HydrogenStateView` -- so that no rate coefficient ever has to be
"pre-converted", which is where the previous implementation drifted.

Reference set (RAMSES-RT, Rosdahl et al. 2013, appendix A):

  * photoionization cross-section        Verner et al. (1996) fit
  * collisional ionization  beta_HI      Cen (1992)
  * collisional ionization cooling zeta  Cen (1992)
  * collisional excitation cooling psi   Cen (1992)
  * recombination  alpha_HII^{A,B}       Hui & Gnedin (1997)
  * recombination cooling eta^{A,B}      Hui & Gnedin (1997)
  * bremsstrahlung theta_HII             Cen (1992)
  * Compton  varpi(T, a)                 (a = expansion factor, a = 1 today)

Governing equations actually solved (hydrogen only, single photon group at
the ionization threshold nu_0 = 13.6 eV / h, so the sum over groups reduces
to one term):

    dN/dt      = -n_HI  sigma_HI c N + b_rec (alpha_A - alpha_B) n_HII n_e   (25')
    dF/dt      = -n_HI  sigma_HI c F                                          (26')
    n_H dx/dt  =  n_HI (beta_HI n_e + sigma_HI c N) - n_HII alpha_A n_e       (28')
    dE_th/dt   =  H - L,  L given by :func:`cooling_rate_cgs` (eq. A17)

All time integration is explicit Euler (Delta X = Xdot * Delta t) as
requested, with positivity limiters (see :func:`limited_explicit_update`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Physical constants (cgs)
# ---------------------------------------------------------------------------
H_PLANCK_CGS = 6.62607015e-27        # erg s
EV_CGS = 1.602176634e-12             # erg / eV
KB_CGS = 1.380649e-16                # erg / K
MH_CGS = 1.6735575e-24               # g, mass of the hydrogen ATOM
C_LIGHT_CGS = 2.99792458e10          # cm / s

E_HI_EV = 13.6                       # HI ionization threshold [eV]
E_HI_CGS = E_HI_EV * EV_CGS          # erg
NU_HI_CGS = E_HI_CGS / H_PLANCK_CGS  # ~3.2884e15 Hz

T_CMB0_K = 2.727                     # CMB temperature today [K]

# Verner, Ferland, Korista & Yakovlev (1996) photoionization fit, HI row
# (this is the table reproduced in the RAMSES-RT appendix):
#   sigma(E) = sigma_0 * [(x - 1)^2 + y_w^2] * y^(0.5 P - 5.5)
#                       * (1 + sqrt(y / y_a))^(-P)
#   x = E / E_0 - y_0 ,  y = sqrt(x^2 + y_1^2)
VERNER_HI = dict(E0_eV=0.4298, sigma0_cm2=5.475e-14, P=2.963, ya=32.88,
                 yw=0.0, y0=0.0, y1=0.0)


# ---------------------------------------------------------------------------
# Photoionization cross-section
# ---------------------------------------------------------------------------
def sigma_HI_verner_cgs(energy_eV=E_HI_EV):
    """HI photoionization cross-section [cm^2], Verner et al. (1996).

    At the threshold (13.6 eV) this returns 6.30e-18 cm^2, the textbook
    value. NOTE: ``VERNER_HI['sigma0_cm2'] = 5.475e-14`` is a *fit
    parameter*, NOT the threshold cross-section -- using it directly
    overestimates sigma_HI by a factor ~8700.
    """
    p = VERNER_HI
    E = jnp.asarray(energy_eV)
    x = E / p["E0_eV"] - p["y0"]
    y = jnp.sqrt(x * x + p["y1"] ** 2)
    F = (
        ((x - 1.0) ** 2 + p["yw"] ** 2)
        * y ** (0.5 * p["P"] - 5.5)
        * (1.0 + jnp.sqrt(y / p["ya"])) ** (-p["P"])
    )
    sigma = p["sigma0_cm2"] * F
    # below threshold there is no photoionization
    return jnp.where(E >= E_HI_EV, sigma, 0.0)


#: threshold cross-section, evaluated once (6.30e-18 cm^2)
SIGMA_HI_0_CGS = float(sigma_HI_verner_cgs(E_HI_EV))


def sigma_HI_powerlaw_cgs(nu=NU_HI_CGS, nu0=NU_HI_CGS, sigma0=None, index=-3.0):
    """sigma_HI(nu) = sigma_HI(nu_0) * (nu / nu_0)^index, clipped below nu_0.

    This is the approximation requested for the current single-group setup:
    with nu = nu_0 it simply returns sigma_HI(nu_0) = 6.30e-18 cm^2.
    """
    sigma0 = SIGMA_HI_0_CGS if sigma0 is None else sigma0
    ratio = jnp.asarray(nu) / nu0
    return jnp.where(ratio >= 1.0, sigma0 * ratio ** index, 0.0)


# ---------------------------------------------------------------------------
# Temperature-dependent coefficients (T in Kelvin)
# ---------------------------------------------------------------------------
def _safeT(T, eps=1.0):
    """Clamp T away from 0 so that exp(-C/T) and 1/T stay finite."""
    return jnp.maximum(jnp.asarray(T), eps)


def lambda_HI(T):
    """lambda_HI(T) = 315614 K / T."""
    return 315614.0 / _safeT(T)


def beta_HI_cgs(T):
    """Collisional ionization rate coefficient [cm^3 s^-1], Cen (1992)."""
    Ts = _safeT(T)
    return (
        5.85e-11 * jnp.sqrt(Ts)
        / (1.0 + jnp.sqrt(Ts / 1.0e5))
        * jnp.exp(-157809.1 / Ts)
    )


def zeta_HI_cgs(T):
    """Collisional ionization COOLING [erg cm^3 s^-1], Cen (1992)."""
    Ts = _safeT(T)
    return (
        1.27e-21 * jnp.sqrt(Ts)
        / (1.0 + jnp.sqrt(Ts / 1.0e5))
        * jnp.exp(-157809.1 / Ts)
    )


def psi_HI_cgs(T):
    """Collisional excitation COOLING [erg cm^3 s^-1], Cen (1992)."""
    Ts = _safeT(T)
    return 7.5e-19 / (1.0 + jnp.sqrt(Ts / 1.0e5)) * jnp.exp(-118348.0 / Ts)


def alpha_A_HII_cgs(T):
    """Case A recombination rate [cm^3 s^-1], Hui & Gnedin (1997)."""
    lam = lambda_HI(T)
    return 1.269e-13 * lam ** 1.503 / (1.0 + (lam / 0.522) ** 0.47) ** 1.923


def alpha_B_HII_cgs(T):
    """Case B recombination rate [cm^3 s^-1], Hui & Gnedin (1997)."""
    lam = lambda_HI(T)
    return 2.753e-14 * lam ** 1.5 / (1.0 + (lam / 2.74) ** 0.407) ** 2.242


def eta_A_HII_cgs(T):
    """Case A recombination COOLING [erg cm^3 s^-1], Hui & Gnedin (1997)."""
    lam = lambda_HI(T)
    Ts = _safeT(T)
    return 1.778e-29 * Ts * lam ** 1.965 / (1.0 + (lam / 0.541) ** 0.502) ** 2.697


def eta_B_HII_cgs(T):
    """Case B recombination COOLING [erg cm^3 s^-1], Hui & Gnedin (1997)."""
    lam = lambda_HI(T)
    Ts = _safeT(T)
    return 3.435e-30 * Ts * lam ** 1.97 / (1.0 + (lam / 2.25) ** 0.376) ** 3.72


def theta_HII_cgs(T):
    """Bremsstrahlung (free-free) COOLING [erg cm^3 s^-1]."""
    return 1.42e-27 * jnp.sqrt(_safeT(T))


def varpi_cgs(T, a=1.0):
    """Compton cooling/heating off the CMB, per electron [erg s^-1].

    a is the cosmological expansion factor (a = 1 -> today).
    """
    T_gamma = T_CMB0_K / a
    return 1.017e-37 * (T_gamma) ** 4 * (jnp.asarray(T) - T_gamma)


def cooling_rate_cgs(T, n_HI, n_HII, n_e, a=1.0, case="A"):
    """Eq. (A17) restricted to hydrogen, in erg cm^-3 s^-1 (positive = losses).

        L = [zeta_HI + psi_HI] n_e n_HI
          + eta^{A|B}_HII      n_e n_HII
          + theta_HII          n_e n_HII        (bremsstrahlung)
          + varpi(T, a)        n_e              (Compton)

    ``case`` selects the recombination cooling branch and MUST match the
    recombination coefficient used in the ionization equation (case A if
    recombination photons are re-injected into the radiation field, case B
    if the on-the-spot approximation is used).
    """
    eta = eta_A_HII_cgs(T) if case.upper() == "A" else eta_B_HII_cgs(T)
    return (
        (zeta_HI_cgs(T) + psi_HI_cgs(T)) * n_e * n_HI
        + eta * n_e * n_HII
        + theta_HII_cgs(T) * n_e * n_HII
        + varpi_cgs(T, a) * n_e
    )


def photoheating_rate_cgs(n_HI, N_gamma, sigma_HI=None, mean_photon_energy_eV=E_HI_EV,
                          c_cgs=C_LIGHT_CGS):
    """Photoheating H = n_HI c sigma_HI N (<h nu> - h nu_0)  [erg cm^-3 s^-1].

    WARNING (physics, not a bug): with a single photon group *at* the
    threshold (<h nu> = h nu_0 = 13.6 eV) this is IDENTICALLY ZERO. A
    monochromatic threshold spectrum ionizes but cannot heat, so no
    10^4 K Stromgren sphere will ever form. Give the group a mean energy
    above 13.6 eV (e.g. 18-20 eV for a blackbody-weighted average) to get
    physical heating.
    """
    sigma_HI = SIGMA_HI_0_CGS if sigma_HI is None else sigma_HI
    excess_erg = (mean_photon_energy_eV - E_HI_EV) * EV_CGS
    return n_HI * c_cgs * sigma_HI * N_gamma * excess_erg


# ---------------------------------------------------------------------------
# Positivity limiters (explicit Euler, as requested)
# ---------------------------------------------------------------------------
def limited_explicit_update(X, rate, dt, max_frac=0.9, floor=0.0):
    """Delta X = rate * dt, clipped so a *negative* update never removes
    more than ``max_frac`` of X.

        dX = rate * dt
        if dX < 0:  dX = max(dX, -max_frac * (X - floor))

    This is the "Delta N > 0 -> Delta N = min(rate dt, 90% N)" rule: the
    field can lose at most 90 % of its value per step, so it stays strictly
    positive whatever the stiffness of the sink.
    """
    dX = rate * dt
    max_loss = max_frac * jnp.maximum(X - floor, 0.0)
    return jnp.where(dX < 0.0, jnp.maximum(dX, -max_loss), dX)


def tiny_like(x):
    """Smallest positive normal of x's dtype.

    Used as a 0/0 guard instead of a hard-coded 1e-30: code-unit amplitudes
    are arbitrary (they scale with L_cgs, M_cgs, V_cgs), and an absolute
    floor larger than the physical value silently replaces the field by the
    floor. With n_H = 1 cm^-3, T = 100 K and L = 1 cm, the thermal energy is
    ~2.5e-33 in code units, i.e. 400x BELOW a 1e-30 floor.
    """
    arr = jnp.asarray(x)
    try:
        return float(jnp.finfo(arr.dtype).tiny)
    except ValueError:      # integer dtype
        return 1e-38


def limit_m1_flux_cone(N, Fx, Fy, Fz, c, f_max=1.0, eps=1e-30):
    """Rescale (Fx, Fy, Fz) so that |F| <= f_max * c * N (M1 causality cone).

    Returns the rescaled components. The reduced flux f = |F| / (c N) must
    stay in [0, 1] for the M1 Eddington factor to be defined; whenever an
    absorption sink or an injection pushes |F| above c N, the direction is
    kept and only the magnitude is clipped.
    """
    # Avoid both an intermediate overflow in Fx**2 and the indeterminate
    # inf/inf division that would otherwise poison the whole M1 state.
    Fx = jnp.where(jnp.isfinite(Fx), Fx, 0.0)
    Fy = jnp.where(jnp.isfinite(Fy), Fy, 0.0)
    Fz = jnp.where(jnp.isfinite(Fz), Fz, 0.0)
    N = jnp.where(jnp.isfinite(N), N, 0.0)

    component_scale = jnp.maximum(jnp.maximum(jnp.abs(Fx), jnp.abs(Fy)), jnp.abs(Fz))
    component_scale_safe = jnp.maximum(component_scale, eps)
    Fnorm = component_scale * jnp.sqrt(
        (Fx / component_scale_safe) ** 2
        + (Fy / component_scale_safe) ** 2
        + (Fz / component_scale_safe) ** 2
    )
    Fmax = f_max * c * jnp.maximum(N, 0.0)
    scale = jnp.where(
        component_scale > 0.0,
        jnp.minimum(1.0, Fmax / jnp.maximum(Fnorm, eps)),
        1.0,
    )
    return Fx * scale, Fy * scale, Fz * scale


# ---------------------------------------------------------------------------
# State view: the ONLY place where code units meet cgs
# ---------------------------------------------------------------------------
@dataclass
class HydrogenStateView:
    """Read/write hydrogen-chemistry quantities from the combined state.

    IMPORTANT -- the array handed to a ``force`` by
    :meth:`diffhydro.hydro_core.hydro._hydrostep` is the **conservative**
    state, not the primitive one::

        sol[idx_N]      = N_gamma                      (photon number density)
        sol[idx_F]      = F_gamma                      (photon number flux)
        sol[idx_xHII]   = x_HII  or  N_gamma * x_HII   (see ``xHII_weighted``)
        sol[idx_rho]    = rho
        sol[idx_mom]    = rho * v
        sol[idx_Etot]   = E_tot = rho e + 0.5 rho v^2  (NOT the pressure!)

    Reading ``sol[idx_Etot]`` as if it were the pressure is the single most
    damaging bug this class exists to prevent.

    ``xHII_weighted`` must match the passive-scalar convention of the RT
    EquationManager: it stores ``E_gamma * s_k`` as the conserved passive
    variable unless ``passive_weighted=False`` was requested.
    """

    cu: object                                  # CodeUnits
    gamma: float = 5.0 / 3.0
    idx_N: int = 0
    idx_F: Sequence[int] = (1, 2, 3)
    idx_xHII: int = 4
    idx_rho: int = 5
    idx_mom: Sequence[int] = (6, 7, 8)
    idx_Etot: int = 9
    #: index of the field the conserved passive scalar is weighted by, i.e.
    #: ``sol[idx_xHII] == sol[xHII_weight_idx] * x_HII``. For x_HII living in
    #: the HYDRO block (the physical choice) this is ``idx_rho``, because the
    #: conserved quantity is rho * x_HII advected with the gas. ``None``
    #: means the slot stores x_HII raw.
    xHII_weight_idx: int | None = None
    X_H: float = 1.0                            # hydrogen mass fraction
    eps: float = 1e-30
    T_floor_K: float = 1.0
    T_ceil_K: float = 1.0e9
    #: light speed used by the SOLVER, in code velocity units. Leave None to
    #: use the true c. Setting it makes the reduced-speed-of-light
    #: approximation (RSLA) self-consistent: with c_red the transported N is
    #: inflated by c/c_red, so the interaction terms must use c_red as well
    #: for n_HI sigma c N to keep its physical value (Rosdahl+ 2013, sec 3.3).
    light_speed_code: float | None = None

    # cached cgs scales
    L_cgs: float = field(init=False)
    T_s_cgs: float = field(init=False)
    P_cgs: float = field(init=False)
    rho_cgs: float = field(init=False)
    V_cgs: float = field(init=False)

    def __post_init__(self):
        self.L_cgs = float(self.cu.L_cgs)
        self.T_s_cgs = float(self.cu.T_cgs)
        self.P_cgs = float(self.cu.P_cgs)
        self.rho_cgs = float(self.cu.rho_cgs)
        self.V_cgs = float(self.cu.V_cgs)

    # -- conversion factors ------------------------------------------------
    @property
    def c_interaction_cgs(self) -> float:
        """Light speed [cm/s] to use in the photon-matter interaction terms."""
        if self.light_speed_code is None:
            return C_LIGHT_CGS
        return float(self.light_speed_code) * self.V_cgs

    @property
    def number_density_cgs_per_code(self) -> float:
        """n[cm^-3] = n_code / L_cgs^3."""
        return 1.0 / self.L_cgs ** 3

    @property
    def energy_density_cgs_per_code(self) -> float:
        """e[erg cm^-3] = e_code * P_cgs."""
        return self.P_cgs

    # -- readers -----------------------------------------------------------
    def photon_density_code(self, sol):
        return jnp.maximum(sol[self.idx_N], 0.0)

    def photon_density_cgs(self, sol):
        return self.photon_density_code(sol) * self.number_density_cgs_per_code

    def xHII(self, sol):
        """Ionized fraction in [0, 1], whatever the storage convention."""
        raw = sol[self.idx_xHII]
        if self.xHII_weight_idx is not None:
            raw = raw / jnp.maximum(sol[self.xHII_weight_idx], tiny_like(sol))
        return jnp.clip(raw, 0.0, 1.0)

    def set_xHII(self, sol, x):
        x = jnp.clip(x, 0.0, 1.0)
        if self.xHII_weight_idx is not None:
            x = x * jnp.maximum(sol[self.xHII_weight_idx], tiny_like(sol))
        return sol.at[self.idx_xHII].set(x)

    def rho_code(self, sol):
        # tiny_like, NOT a hard 1e-30: rho in code units can legitimately be
        # ~1e-24 (n_H = 1 cm^-3 with M_cgs = 1 g, L_cgs = 1 cm).
        return jnp.maximum(sol[self.idx_rho], tiny_like(sol))

    def thermal_energy_floor_code(self, sol):
        """Thermal energy density corresponding to ``T_floor_K``, code units.

        A *physical* floor, so it automatically has the right magnitude in
        any unit system, unlike an absolute 1e-30.
        """
        n_H = self.rho_code(sol) * self.rho_cgs * self.X_H / MH_CGS
        e_cgs = n_H * KB_CGS * self.T_floor_K / (self.gamma - 1.0)
        return e_cgs / self.P_cgs

    def thermal_energy_code(self, sol):
        rho = self.rho_code(sol)
        mom2 = sum(sol[i] ** 2 for i in self.idx_mom)
        e_th = sol[self.idx_Etot] - 0.5 * mom2 / rho
        return jnp.maximum(e_th, self.thermal_energy_floor_code(sol))

    def pressure_code(self, sol):
        """p = (gamma - 1) (E_tot - 0.5 rho v^2), from CONSERVATIVE state."""
        return (self.gamma - 1.0) * self.thermal_energy_code(sol)

    def number_densities_cgs(self, sol, x_HII=None):
        """(n_H, n_HI, n_HII, n_e) in cm^-3, pure hydrogen (n_e = n_HII)."""
        x = self.xHII(sol) if x_HII is None else jnp.clip(x_HII, 0.0, 1.0)
        n_H = self.rho_code(sol) * self.rho_cgs * self.X_H / MH_CGS
        n_HI = n_H * (1.0 - x)
        n_HII = n_H * x
        n_e = n_HII
        return n_H, n_HI, n_HII, n_e

    def temperature_K(self, sol, x_HII=None):
        """T [K] from the ideal gas law for a pure-hydrogen plasma.

        n_tot = n_H (1 + x_HII)  (protons/atoms + electrons), so
        T = p / (n_tot k_B). This avoids the fixed ``cu.mu = 0.61``, which
        is a *primordial fully ionized* value and is wrong by a factor 1.64
        for the cold neutral gas of a Stromgren test.
        """
        x = self.xHII(sol) if x_HII is None else jnp.clip(x_HII, 0.0, 1.0)
        n_H = self.rho_code(sol) * self.rho_cgs * self.X_H / MH_CGS
        n_tot = jnp.maximum(n_H * (1.0 + x), tiny_like(sol))
        p_cgs = self.pressure_code(sol) * self.P_cgs
        return jnp.clip(p_cgs / (n_tot * KB_CGS), self.T_floor_K, self.T_ceil_K)

    # -- writers -----------------------------------------------------------
    def add_thermal_energy_cgs(self, sol, dE_cgs):
        """Add an energy DENSITY given in erg cm^-3 to the total energy slot."""
        return sol.at[self.idx_Etot].add(dE_cgs / self.P_cgs)

    def add_photons_cgs(self, sol, dN_cgs):
        """Add a photon number DENSITY given in cm^-3 to the N_gamma slot."""
        return sol.at[self.idx_N].add(dN_cgs * self.L_cgs ** 3)

    # -- misc --------------------------------------------------------------
    def dt_cgs(self, dt_code):
        return dt_code * self.T_s_cgs

    def light_speed_code(self):
        return C_LIGHT_CGS / self.V_cgs
