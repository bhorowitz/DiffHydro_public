
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

import eos_t  # your existing Nyx EOS interface


# ----------------------------
# Shared cosmology container (same as thalas2)
# ----------------------------
class CosmoParams(tuple):
    """
    Minimal container: (h, Omega_m, Omega_b, Omega_L)
    Kept tuple-like for easy use / serialization.
    """
    __slots__ = ()
    def __new__(cls, h, Omega_m, Omega_b, Omega_L):
        return tuple.__new__(cls, (float(h), float(Omega_m), float(Omega_b), float(Omega_L)))

    @property
    def h(self): return self[0]
    @property
    def Omega_m(self): return self[1]
    @property
    def Omega_b(self): return self[2]
    @property
    def Omega_L(self): return self[3]


# ----------------------------
# Physical constants (CGS)
# ----------------------------
G_cgs = 6.6743e-8  # dyne cm^2 / g^2
MPC_CM = 3.085677581e24


def H0_cgs(cp: CosmoParams) -> float:
    # H0 = 100 h km/s/Mpc -> s^-1
    return (100.0 * cp.h) * 1e5 / MPC_CM


def rho_crit_cgs(cp: CosmoParams) -> float:
    # rho_crit = 3 H0^2 / (8 pi G)
    H0 = H0_cgs(cp)
    return 3.0 * H0 * H0 / (8.0 * np.pi * G_cgs)


def rho_b_mean_cgs(cp: CosmoParams, z: float) -> float:
    # mean baryon density at redshift z (physical)
    return rho_crit_cgs(cp) * cp.Omega_b * (1.0 + float(z))**3


# ----------------------------
# JAX-friendly interpolation
# (reuse your CartesianGrid if you prefer; here’s a minimal local version)
# ----------------------------
from jax.scipy.ndimage import map_coordinates

# Small floor for log-space blending to guard log10(<=0).
_LOG_FLOOR = 1.0e-60


def _smoothstep(f):
    # C1 Hermite smoothstep: s(0)=0, s(1)=1, s'(0)=s'(1)=0.
    return f * f * (3.0 - 2.0 * f)


class CartesianGrid:
    def __init__(self, limits, values, mode="nearest", cval=jnp.nan,
                 smooth=False, log_space=False):
        self.limits = limits
        self.values = jnp.asarray(values)
        self.mode = mode
        self.cval = cval
        # smooth=True selects a C1 smoothstep-weighted multilinear gather instead
        # of jax bilinear map_coordinates (whose gradient jumps at every grid
        # line). log_space=True blends the corner values in log10 space (used
        # when the table stores raw nHI). Both default off to preserve exact
        # original behavior.
        self.smooth = bool(smooth)
        self.log_space = bool(log_space)

    def _pix(self, coords):
        coords = jnp.broadcast_arrays(*coords)
        return [
            (c - lo) * (n - 1) / (hi - lo)
            for (lo, hi), c, n in zip(self.limits, coords, self.values.shape)
        ]

    def _smooth_gather(self, pix):
        # Manual C1 multilinear interpolation: floor indices + clamp + gather the
        # 2^ndim corners + smoothstep-weighted blend. Index clamping reproduces
        # mode="nearest" edge handling (in-range fields never hit the clamp).
        ndim = len(pix)
        shape = self.values.shape
        base = [jnp.floor(p) for p in pix]
        frac = [p - b for p, b in zip(pix, base)]
        w = [_smoothstep(f) for f in frac]
        i0 = [jnp.clip(b.astype(jnp.int32), 0, s - 1) for b, s in zip(base, shape)]
        i1 = [jnp.clip(b.astype(jnp.int32) + 1, 0, s - 1) for b, s in zip(base, shape)]

        result = jnp.zeros_like(w[0])
        for corner in range(2 ** ndim):
            idx = []
            weight = jnp.ones_like(w[0])
            for d in range(ndim):
                if (corner >> d) & 1:
                    idx.append(i1[d])
                    weight = weight * w[d]
                else:
                    idx.append(i0[d])
                    weight = weight * (1.0 - w[d])
            v = self.values[tuple(idx)]
            if self.log_space:
                v = jnp.log10(jnp.maximum(v, _LOG_FLOOR))
            result = result + weight * v
        if self.log_space:
            result = 10.0 ** result
        return result

    def __call__(self, *coords):
        pix = self._pix(coords)
        if self.smooth:
            return self._smooth_gather(pix)
        return map_coordinates(self.values, pix, mode=self.mode, cval=self.cval, order=1)


# ----------------------------
# Refactored EOS class
# ----------------------------
@dataclass
class EOSInterpolator:
    """
    Interpolates log10(nHI) or nHI as a function of:
      - log10(delta_b) where delta_b is baryon overdensity relative to mean
      - log10(T/K)
    Internally uses eos_t.eos.nyx_eos(z, rhob_cgs, temp_K) to populate a table.
    """
    z: float
    cosmo: CosmoParams
    logdelta_limits: Tuple[float, float] = (-3.0, 4.0)
    logT_limits: Tuple[float, float] = (2.0, 8.0)
    grid_size: int = 256  # << configurable; avoids hard-coded 1000
    store_log_nhi: bool = False  # optionally store log10(nHI) for stability
    smooth_eos: bool = False  # C1 smoothstep interpolation (default off = bilinear)

    def __post_init__(self):
        self.z = float(self.z)
        self.grid_size = int(self.grid_size)

        # Compute mean baryon density in cgs from cosmology
        self._rhob_mean_cgs = rho_b_mean_cgs(self.cosmo, self.z)

        # Build grids in (log10 delta_b, log10 T)
        ld0, ld1 = self.logdelta_limits
        lt0, lt1 = self.logT_limits
        N = self.grid_size

        self.logdelta_grid = np.linspace(ld0, ld1, N)
        self.logT_grid = np.linspace(lt0, lt1, N)

        delta_grid = 10.0 ** self.logdelta_grid
        T_grid = 10.0 ** self.logT_grid

        # Convert overdensity -> physical baryon density [g/cm^3]
        rhob_grid_cgs = self._rhob_mean_cgs * delta_grid

        # Populate table (still Python-level loops because eos_t is not JAX)
        vals = np.zeros((N, N), dtype=np.float32)

        # You can micro-opt later (chunking, multiprocessing, caching), but this is robust.
        for i in range(N):
            rhob_i = rhob_grid_cgs[i]
            for j in range(N):
                nhi = eos_t.eos.nyx_eos(self.z, rhob_i, T_grid[j])  # existing backend
                if self.store_log_nhi:
                    vals[i, j] = np.log10(max(nhi, 1e-60))
                else:
                    vals[i, j] = nhi

        # When the table stores raw nHI, blend in log space for the smooth path.
        self._grid = CartesianGrid(
            [self.logdelta_limits, self.logT_limits],
            vals,
            mode="nearest",
            smooth=bool(self.smooth_eos),
            log_space=bool(self.smooth_eos and not self.store_log_nhi),
        )

    @property
    def rhob_mean_cgs(self) -> float:
        return self._rhob_mean_cgs

    def compute_nhi(self, log10_delta_b, log10_temp_K):
        """
        Vectorized interpolation.
        Inputs can be scalars or arrays (JAX arrays OK).
        Returns nHI (or log10(nHI) if store_log_nhi=True).
        """
        return self._grid(log10_delta_b, log10_temp_K)

    def nhi_from_delta_T(self, delta_b, temp_K):
        """
        Convenience wrapper taking (delta_b, T) in linear space.
        """
        logd = jnp.log10(jnp.asarray(delta_b))
        logT = jnp.log10(jnp.asarray(temp_K))
        out = self.compute_nhi(logd, logT)
        if self.store_log_nhi:
            return 10.0 ** out
        return out