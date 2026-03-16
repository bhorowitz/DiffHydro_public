"""JAX post-processing for NHI and LyA optical depth.

Core LyA optical-depth kernel is adapted from local thalas utilities
(`~/thalas/thalas2.py`) and refactored here for reusable DiffHydro workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import jax.numpy as jnp
from jax import checkpoint, lax, vmap
from jax.scipy.special import erf


# Physical constants (CGS)
K_B_CGS = jnp.float32(1.380649e-16)  # erg/K
C_CGS = jnp.float32(2.99792458e10)  # cm/s
G_CGS = jnp.float32(6.6743e-8)  # dyne cm^2 / g^2
M_E_CGS = jnp.float32(9.10938356e-28)  # g
E_CGS = jnp.float32(4.80320425e-10)  # statC
M_H_CGS = jnp.float32(1.6735575e-24)  # g
MPC_CM = jnp.float32(3.085677581e24)

# LyA constants
F_ALPHA = jnp.float32(0.4164)
LAMBDA_ALPHA_CGS = jnp.float32(1215.67e-8)  # cm


@dataclass(frozen=True)
class CosmologyParams:
    h: float
    omega_m: float
    omega_b: float
    omega_l: float | None = None

    @property
    def omega_lambda(self) -> float:
        if self.omega_l is not None:
            return float(self.omega_l)
        return 1.0 - float(self.omega_m)


def _import_eos2(extra_paths: Iterable[Path] | None = None):
    # `eos2.py` imports compiled `eos_t` as a top-level module.
    # Ensure nyx_eos directory is on sys.path before importing.
    candidates: list[Path] = [Path(__file__).resolve().parents[1] / "nyx_eos"]
    if extra_paths is not None:
        candidates.extend(Path(p) for p in extra_paths)

    for c in candidates:
        c = c.resolve()
        if c.is_file():
            c = c.parent
        if c.is_dir():
            s = str(c)
            if s not in sys.path:
                sys.path.insert(0, s)

    import eos2  # type: ignore

    return eos2


def build_nyx_eos_interpolator(
    *,
    z: float,
    cosmo: CosmologyParams,
    grid_size: int = 192,
    logdelta_limits: tuple[float, float] = (-3.0, 4.0),
    logt_limits: tuple[float, float] = (0.0, 8.0),
    eos_search_paths: Iterable[Path] | None = None,
):
    eos2 = _import_eos2(eos_search_paths)
    cp = eos2.CosmoParams(cosmo.h, cosmo.omega_m, cosmo.omega_b, cosmo.omega_lambda)
    return eos2.EOSInterpolator(
        z=float(z),
        cosmo=cp,
        logdelta_limits=tuple(logdelta_limits),
        logT_limits=tuple(logt_limits),
        grid_size=int(grid_size),
        store_log_nhi=False,
    )


def compute_nhi_number_density(delta_b, temp_k, eos_interp):
    """Compute neutral hydrogen number density [cm^-3] from (delta_b, T[K])."""
    delta_b = jnp.maximum(jnp.asarray(delta_b, dtype=jnp.float32), 1.0e-8)
    temp_k = jnp.maximum(jnp.asarray(temp_k, dtype=jnp.float32), 1.0)
    return jnp.asarray(eos_interp.nhi_from_delta_T(delta_b, temp_k), dtype=jnp.float32)


def _e_z(cosmo: CosmologyParams, z):
    z = jnp.asarray(z, dtype=jnp.float32)
    return jnp.sqrt(
        jnp.asarray(cosmo.omega_m, dtype=jnp.float32) * (1.0 + z) ** 3
        + jnp.asarray(cosmo.omega_lambda, dtype=jnp.float32)
    )


def _h0_kmps_mpc(cosmo: CosmologyParams):
    return jnp.float32(100.0) * jnp.asarray(cosmo.h, dtype=jnp.float32)


def _h_z_cgs(cosmo: CosmologyParams, z):
    h_kmps_mpc = _h0_kmps_mpc(cosmo) * _e_z(cosmo, z)
    return h_kmps_mpc * jnp.float32(1.0e5) / MPC_CM


def lya_tau_prefactor(cosmo: CosmologyParams, z):
    """Prefactor with dimensions from Gunn-Peterson style LyA optical depth kernel."""
    return (
        (jnp.pi * E_CGS**2 / (M_E_CGS * C_CGS))
        * F_ALPHA
        * LAMBDA_ALPHA_CGS
        / _h_z_cgs(cosmo, z)
    )


def velocity_domain_cmps(cosmo: CosmologyParams, z, boxsize_comoving_mpc: float):
    """LOS Hubble velocity span [cm/s] over the full comoving box."""
    z = jnp.asarray(z, dtype=jnp.float32)
    h_kmps_mpc = _h0_kmps_mpc(cosmo) * _e_z(cosmo, z)
    v_kmps = h_kmps_mpc * (jnp.asarray(boxsize_comoving_mpc, dtype=jnp.float32) / (1.0 + z))
    return v_kmps * jnp.float32(1.0e5)


def _reshape_2d(arr):
    x, y, z = arr.shape
    return jnp.reshape(arr, (x * y, z))


def _move_axis_to_last(arr, axis_los: int):
    axis_los = int(axis_los)
    arr = jnp.asarray(arr, dtype=jnp.float32)
    return jnp.moveaxis(arr, axis_los, -1)


def _tau_kernel_scatter(
    od_factor,
    absorber_mass_cgs,
    velocity_domain_cms,
    num_elements,
    n_array,
    vpara_array,
    temp_array,
    num_pixels,
    num_integ_pixels: int = 20,
):
    """Piecewise-constant optical-depth assignment with Doppler broadening."""
    od_factor = jnp.asarray(od_factor, jnp.float32)
    absorber_mass_cgs = jnp.asarray(absorber_mass_cgs, jnp.float32)
    velocity_domain_cms = jnp.asarray(velocity_domain_cms, jnp.float32)
    n_array = jnp.asarray(n_array, jnp.float32)
    vpara_array = jnp.asarray(vpara_array, jnp.float32)
    temp_array = jnp.asarray(temp_array, jnp.float32)

    t_min = jnp.float32(1.0)
    element_dv = velocity_domain_cms / jnp.float32(num_elements)
    pixel_dv = velocity_domain_cms / jnp.float32(num_pixels)

    vth_factor = jnp.float32(2.0) * K_B_CGS / absorber_mass_cgs
    v_doppler = jnp.sqrt(vth_factor * (temp_array + t_min))

    all_inds = jnp.arange(num_elements, dtype=jnp.int32)
    all_inds_f = all_inds.astype(jnp.float32)

    vlc_l = element_dv * all_inds_f + vpara_array
    vlc_h = element_dv * (all_inds_f + jnp.float32(1.0)) + vpara_array
    vlc_m = element_dv * (all_inds_f + jnp.float32(0.5)) + vpara_array
    pix_lc = (vlc_m / pixel_dv).astype(jnp.int32)

    offsets = jnp.arange(-num_integ_pixels - 1, num_integ_pixels + 2, dtype=jnp.int32)
    ipixes = pix_lc[:, None] + offsets[None, :]

    v_pixel = pixel_dv * (ipixes.astype(jnp.float32) + jnp.float32(0.5))
    dxl = (v_pixel - vlc_l[:, None]) / v_doppler[:, None]
    dxh = (v_pixel - vlc_h[:, None]) / v_doppler[:, None]
    tau_local = n_array[:, None] * (erf(dxl[:, :-1]) - erf(dxh[:, :-1]))

    tgt = (ipixes[:, :-1] % num_pixels).reshape(-1)
    vals = tau_local.reshape(-1) * (jnp.float32(0.5) * od_factor)
    tau = jnp.zeros((num_pixels,), dtype=vals.dtype).at[tgt].add(vals)
    return tau


def tau_cube_from_nhi(
    n_hi_cgs,
    temp_k,
    v_para_cms,
    *,
    z: float,
    cosmo: CosmologyParams,
    boxsize_comoving_mpc: float,
    axis_los: int = 2,
    num_pixels: int | None = None,
    num_integ_pixels: int = 20,
    skewer_batch_size: int | None = None,
    absorber_mass_cgs: float = float(M_H_CGS),
):
    """Return tau cube with the same axis order as input arrays."""
    n_los = _move_axis_to_last(n_hi_cgs, axis_los)
    t_los = _move_axis_to_last(temp_k, axis_los)
    v_los = _move_axis_to_last(v_para_cms, axis_los)

    n_perp_x, n_perp_y, n_elements = n_los.shape
    if num_pixels is None:
        num_pixels = int(n_elements)

    od_factor = lya_tau_prefactor(cosmo, z)
    v_domain = velocity_domain_cmps(cosmo, z, boxsize_comoving_mpc)

    skew_n = _reshape_2d(n_los)
    skew_t = _reshape_2d(t_los)
    skew_v = _reshape_2d(v_los)

    def one_skewer(n_row, t_row, v_row):
        return _tau_kernel_scatter(
            od_factor,
            absorber_mass_cgs,
            v_domain,
            n_elements,
            n_row,
            v_row,
            t_row,
            num_pixels,
            num_integ_pixels=num_integ_pixels,
        )

    n_skewers = int(n_perp_x * n_perp_y)
    batch = int(n_skewers if skewer_batch_size is None else max(1, int(skewer_batch_size)))

    if batch >= n_skewers:
        tau_flat = vmap(one_skewer)(skew_n, skew_t, skew_v)
    else:
        n_batches = (n_skewers + batch - 1) // batch
        n_pad = n_batches * batch - n_skewers
        pad_shape = ((0, n_pad), (0, 0))
        skew_n_pad = jnp.pad(skew_n, pad_shape)
        skew_t_pad = jnp.pad(skew_t, pad_shape)
        skew_v_pad = jnp.pad(skew_v, pad_shape)

        def _eval_batch(n_chunk, t_chunk, v_chunk):
            return vmap(one_skewer)(n_chunk, t_chunk, v_chunk)

        eval_batch = checkpoint(_eval_batch)

        def batch_step(_, i_batch):
            i0 = i_batch * batch
            n_chunk = lax.dynamic_slice(skew_n_pad, (i0, 0), (batch, n_elements))
            t_chunk = lax.dynamic_slice(skew_t_pad, (i0, 0), (batch, n_elements))
            v_chunk = lax.dynamic_slice(skew_v_pad, (i0, 0), (batch, n_elements))
            tau_chunk = eval_batch(n_chunk, t_chunk, v_chunk)
            return None, tau_chunk

        _, tau_chunks = lax.scan(batch_step, None, xs=jnp.arange(n_batches, dtype=jnp.int32))
        tau_flat = tau_chunks.reshape((n_batches * batch, num_pixels))[:n_skewers, :]

    tau_los = tau_flat.reshape((n_perp_x, n_perp_y, num_pixels))
    return jnp.moveaxis(tau_los, -1, int(axis_los))


def flux_from_tau(tau):
    return jnp.exp(-jnp.asarray(tau, dtype=jnp.float32))


def column_density_map_nhi(n_hi_cgs, *, boxsize_comoving_mpc: float, z: float, axis_los: int = 2):
    """Return NHI column-density map [cm^-2] by integrating along LOS proper length."""
    n_hi_cgs = jnp.asarray(n_hi_cgs, dtype=jnp.float32)
    n_los = int(n_hi_cgs.shape[int(axis_los)])
    dl_proper_cm = (jnp.asarray(boxsize_comoving_mpc, dtype=jnp.float32) * MPC_CM) / (
        (1.0 + jnp.asarray(z, dtype=jnp.float32)) * float(n_los)
    )
    return jnp.sum(n_hi_cgs, axis=int(axis_los)) * dl_proper_cm


def mean_map_along_los(field, axis_los: int = 2):
    return jnp.mean(jnp.asarray(field, dtype=jnp.float32), axis=int(axis_los))
