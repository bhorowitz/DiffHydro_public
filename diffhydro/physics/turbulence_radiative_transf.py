"""
Multi-density turbulence initialization.

Supports arbitrary number of density fields and is compatible
with equationmanager_radiative_transf.EquationManager.

Storage layout:
  Conservatives: [rho_total, rho_total*vx, rho_total*vy, rho_total*vz, E_tot, rho_1, ..., rho_N]
"""

import jax
import jax.numpy as jnp


def init_turbulent_velocity(eq, Lbox, rho0, p0,
                            kmin=1, kmax=3, solenoidal_frac=1.0,
                            pslope=-2.0, target_M=1.0, seed=123):
    """
    Initialize turbulent velocity field for multi-density Euler.

    Returns conservative array of shape (5 + n_dens, nx, ny, nz):
      [rho_total, rho_total*vx, rho_total*vy, rho_total*vz, E_tot, rho_1, ..., rho_N]
    """
    nx, ny, nz = eq.mesh_shape

    # k-grid
    k0 = 2.0 * jnp.pi / Lbox
    kx = k0 * jnp.fft.fftfreq(nx) * nx
    ky = k0 * jnp.fft.fftfreq(ny) * ny
    kz = k0 * jnp.fft.fftfreq(nz) * nz
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")

    K2 = KX**2 + KY**2 + KZ**2
    K = jnp.sqrt(jnp.maximum(K2, 1e-30))

    # Band-pass mask
    kmag = jnp.sqrt((KX / k0)**2 + (KY / k0)**2 + (KZ / k0)**2)
    band = (kmag >= kmin) & (kmag <= kmax)

    # Random complex field
    key = jax.random.PRNGKey(seed)
    def rand_complex(k):
        a = jax.random.normal(k, (nx, ny, nz))
        b = jax.random.normal(jax.random.split(k)[0], (nx, ny, nz))
        return a + 1j * b
    g1 = rand_complex(key); key = jax.random.split(key)[0]
    g2 = rand_complex(key); key = jax.random.split(key)[0]
    g3 = rand_complex(key)
    G = jnp.stack([g1, g2, g3], axis=0)

    # Amplitude spectrum ~ k^{pslope}
    Amp = (K**(0.5 * pslope)) * band
    Amp = Amp / jnp.sqrt(jnp.mean(Amp**2) + 1e-30)

    # Solenoidal/compressive projection
    kk_over_k2 = jnp.stack([KX, KY, KZ], 0) / jnp.maximum(K, 1e-30)
    C = jnp.einsum("i...,j...->ij...", kk_over_k2, kk_over_k2)
    I = jnp.eye(3)[:, :, None, None, None]
    P = I - C
    Proj = solenoidal_frac * P + (1.0 - solenoidal_frac) * C

    Uhat = jnp.einsum("ij...,j...->i...", Proj, G) * Amp
    u = jnp.fft.ifftn(Uhat, axes=(1, 2, 3)).real  # (3, nx, ny, nz)

    # Normalize to target Mach
    # rho0 = physical density (active Euler variable)
    # Each passive density rho_i is also initialized to rho0
    rho = rho0 * jnp.ones((nx, ny, nz))
    p = p0 * jnp.ones_like(rho)

    # Passive densities: each one = rho0 (independent tracers)
    rhos_passive = jnp.stack([rho] * eq.n_dens, axis=0)  # (n_dens, nx, ny, nz) 
    # Only thing that differs from turbulence.py is rhos_passive, the rest is identical (except rho0 is the physical density not the total density)

    # Sound speed uses the physical density rho (= rho0)
    cs = jnp.sqrt(eq.gamma * p / rho)
    urms = jnp.sqrt(jnp.mean(jnp.sum(u**2, axis=0)))
    alpha = (target_M * jnp.mean(cs)) / (urms + 1e-30)
    v = alpha * u

    E_th = p / (eq.gamma - 1.0)
    E_kin = 0.5 * rho * (v**2).sum(axis=0)
    E_total = E_th + E_kin

    U = jnp.concatenate([
        rho[jnp.newaxis],              # active: physical density
        (rho * v[0])[jnp.newaxis],     # active: momentum
        (rho * v[1])[jnp.newaxis],
        (rho * v[2])[jnp.newaxis],
        E_total[jnp.newaxis],          # active: total energy
        rhos_passive,                  # passive: individual densities (each = rho0)
    ], axis=0)

    return U


def init_turbulent_velocity_cpu(eq, Lbox, rho0, p0,
                                kmin=1, kmax=3, solenoidal_frac=1.0,
                                pslope=-2.0, target_M=1.0, seed=123):
    """
    CPU version of multi-density turbulence initialization (uses NumPy).
    """
    import numpy as np

    nx, ny, nz = eq.mesh_shape
    rng = np.random.default_rng(seed)

    k0 = 2.0 * np.pi / Lbox
    kx = k0 * np.fft.fftfreq(nx) * nx
    ky = k0 * np.fft.fftfreq(ny) * ny
    kz = k0 * np.fft.fftfreq(nz) * nz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K = np.sqrt(np.maximum(K2, 1e-30))

    kmag = np.sqrt((KX / k0)**2 + (KY / k0)**2 + (KZ / k0)**2)
    band = (kmag >= kmin) & (kmag <= kmax)

    def rand_complex():
        a = rng.normal(size=(nx, ny, nz))
        b = rng.normal(size=(nx, ny, nz))
        return a + 1j * b

    G = np.stack([rand_complex(), rand_complex(), rand_complex()], axis=0)

    Amp = (K**(0.5 * pslope)) * band
    Amp = Amp / np.sqrt(np.mean(Amp**2) + 1e-30)

    kk_over_k = np.stack([KX, KY, KZ], 0) / np.maximum(K, 1e-30)
    C = np.einsum("i...,j...->ij...", kk_over_k, kk_over_k)
    I = np.eye(3)[:, :, None, None, None]
    P = I - C
    Proj = solenoidal_frac * P + (1.0 - solenoidal_frac) * C

    Uhat = np.einsum("ij...,j...->i...", Proj, G) * Amp
    u = np.fft.ifftn(Uhat, axes=(1, 2, 3)).real

    rho = rho0 * np.ones((nx, ny, nz))
    p = p0 * np.ones_like(rho)

    # Passive densities: each one = rho0 (independent tracers)
    rhos_passive = np.stack([rho.astype(np.float64)] * eq.n_dens, axis=0)

    # Sound speed uses the physical density rho (= rho0)
    cs = np.sqrt(eq.gamma * p / rho)
    urms = np.sqrt(np.mean(np.sum(u**2, axis=0)))
    alpha = (target_M * np.mean(cs)) / (urms + 1e-30)
    v = alpha * u

    E_th = p / (eq.gamma - 1.0)
    E_kin = 0.5 * rho * (v**2).sum(axis=0)
    E_total = E_th + E_kin

    U_np = np.concatenate([
        rho[np.newaxis],
        (rho * v[0])[np.newaxis],
        (rho * v[1])[np.newaxis],
        (rho * v[2])[np.newaxis],
        E_total[np.newaxis],
        rhos_passive,
    ], axis=0)

    return jnp.array(U_np)
