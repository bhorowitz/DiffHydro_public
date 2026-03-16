"""Cosmological background helpers for supercomoving integrations."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

try:
    import jax_cosmo as jc

    _HAS_JAX_COSMO = True
except Exception:  # pragma: no cover - optional dependency
    jc = None
    _HAS_JAX_COSMO = False


@dataclass
class LCDMBackground:
    """Flat/curved LCDM background with optional JaxCosmo backend."""

    h: float = 0.6774
    Omega_m: float = 0.3075
    Omega_lambda: float = 0.6925
    Omega_r: float = 0.0
    Omega_k: float = 0.0
    Omega_b: float = 0.0486
    w0: float = -1.0
    wa: float = 0.0
    sigma8: float = 0.8159
    n_s: float = 0.9667
    use_jax_cosmo: bool = True

    def __post_init__(self) -> None:
        self.H0 = 100.0 * float(self.h)
        self._cosmo = None
        if self.use_jax_cosmo and _HAS_JAX_COSMO:
            omega_c = self.Omega_m - self.Omega_b
            self._cosmo = jc.Cosmology(
                Omega_c=omega_c,
                Omega_b=self.Omega_b,
                h=self.h,
                n_s=self.n_s,
                sigma8=self.sigma8,
                Omega_k=self.Omega_k,
                w0=self.w0,
                wa=self.wa,
            )

    def E2(self, a):
        """Dimensionless Hubble factor squared, E(a)^2 = H(a)^2/H0^2."""
        a = jnp.maximum(jnp.asarray(a), 1.0e-8)

        if self._cosmo is not None:
            return jc.background.Esqr(self._cosmo, a)

        de_term = self.Omega_lambda * a ** (-3.0 * (1.0 + self.w0 + self.wa))
        de_term = de_term * jnp.exp(-3.0 * self.wa * (1.0 - a))
        return (
            self.Omega_r / a**4
            + self.Omega_m / a**3
            + self.Omega_k / a**2
            + de_term
        )

    def H(self, a):
        """Hubble rate H(a) in the same units as H0."""
        if self._cosmo is not None:
            # jax_cosmo.background.H is normalized to 100 km/s/Mpc at a=1,
            # while this class tracks H0 = 100*h. Keep units consistent.
            aa = jnp.maximum(jnp.asarray(a), 1.0e-8)
            return self.H0 * jnp.sqrt(jc.background.Esqr(self._cosmo, aa))
        return self.H0 * jnp.sqrt(self.E2(a))

    def da_dtau(self, a):
        """Supercomoving background ODE: da/dtau = a^3 H(a)."""
        a = jnp.maximum(jnp.asarray(a), 1.0e-8)
        return a**3 * self.H(a)


def integrate_scale_factor_euler(background: LCDMBackground, a_init, dtau, n_steps: int):
    """Forward-Euler integration of a(tau)."""
    dtau = jnp.asarray(dtau)
    a = jnp.asarray(a_init)
    taus = [jnp.asarray(0.0, dtype=a.dtype)]
    scale_factors = [a]
    for _ in range(int(n_steps)):
        a = a + background.da_dtau(a) * dtau
        scale_factors.append(a)
        taus.append(taus[-1] + dtau)
    return jnp.asarray(taus), jnp.asarray(scale_factors)


def integrate_scale_factor_rk4(background: LCDMBackground, a_init, dtau, n_steps: int):
    """Reference RK4 integration of a(tau)."""
    dtau = jnp.asarray(dtau)
    a = jnp.asarray(a_init)
    taus = [jnp.asarray(0.0, dtype=a.dtype)]
    scale_factors = [a]
    for _ in range(int(n_steps)):
        k1 = background.da_dtau(a)
        k2 = background.da_dtau(a + 0.5 * dtau * k1)
        k3 = background.da_dtau(a + 0.5 * dtau * k2)
        k4 = background.da_dtau(a + dtau * k3)
        a = a + (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        scale_factors.append(a)
        taus.append(taus[-1] + dtau)
    return jnp.asarray(taus), jnp.asarray(scale_factors)
