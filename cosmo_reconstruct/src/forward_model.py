from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import jax_cosmo as jc
import jaxpm.pm as jpm
from jaxpm.growth import _growth_factor_ODE, dGfa, growth_factor, growth_rate
from jaxpm.painting import cic_paint


@dataclass(frozen=True)
class ForwardModelConfig:
    mesh_n: int = 128
    box_size_mpc_h: float = 25.0
    z_init: float = 127.0
    z_target: float = 2.0
    kdk_steps: int = 64
    lpt_order: int = 1
    omega_m: float = 0.3
    omega_b: float = 0.045
    h: float = 0.6711
    n_s: float = 0.9624
    sigma8: float = 0.8
    checkpoint: bool = True
    checkpoint_every: int = 1


@dataclass(frozen=True)
class GasModelParams:
    smooth_sigma_cells: float = 1.0
    bias_linear: float = 1.0
    bias_quadratic: float = 0.0
    gas_mean: float = 1.0
    temp_init_k: float = 344.0774
    temp_heat_gain: float = 80.0
    temp_slope: float = 1.0
    temp_quadratic: float = 0.0


def a_from_z(z: float) -> float:
    return 1.0 / (1.0 + float(z))


def build_cosmology(cfg: ForwardModelConfig) -> jc.Cosmology:
    return jc.Planck15(
        Omega_c=cfg.omega_m - cfg.omega_b,
        Omega_b=cfg.omega_b,
        h=cfg.h,
        n_s=cfg.n_s,
        sigma8=cfg.sigma8,
    )


def prime_growth_cache(cosmo: jc.Cosmology, a: float) -> None:
    # jaxpm.lpt calls growth_factor/growth_rate/dGfa, which lazily fill
    # cosmo._workspace. Prime with concrete values before jitted optimization
    # (especially with L-BFGS line-search) to avoid tracer-leak issues.
    #
    # jax_cosmo's background._growth_factor_ODE populates the same cache key
    # ("background.growth_factor") but only stores {"a", "g", "f"}, omitting the
    # "h" key that jaxpm's dGfa requires.  LCDMBackground(use_jax_cosmo=True)
    # triggers that path before we get here, so we must evict the stale entry.
    stale = cosmo._workspace.get("background.growth_factor", {})
    if "h" not in stale:
        cosmo._workspace.pop("background.growth_factor", None)
    a_scalar = float(a)
    import numpy as _np
    try:
        _growth_factor_ODE(cosmo, _np.atleast_1d(a_scalar))
    except Exception:
        pass
    a_arr = jnp.asarray([a_scalar], dtype=jnp.float32)
    try:
        _ = growth_factor(cosmo, a_scalar)
    except Exception:
        pass
    try:
        _ = growth_rate(cosmo, a_scalar)
    except Exception:
        pass
    try:
        _ = dGfa(cosmo, a_arr)
    except Exception:
        pass


def make_lattice_positions(n: int) -> jnp.ndarray:
    grid = jnp.arange(n, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(grid, grid, grid, indexing="ij")
    return jnp.stack([gx, gy, gz], axis=-1).reshape((-1, 3))


def _a_to_E(cosmo: jc.Cosmology, a: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jc.background.Esqr(cosmo, a))


def _wrap_positions(pos: jnp.ndarray, mesh_n: int) -> jnp.ndarray:
    return jnp.mod(pos, jnp.asarray(float(mesh_n), dtype=jnp.float32))


def _fftk(shape, symmetric: bool = True, dtype=np.float32):
    """Env-agnostic copy of jaxpm.kernels.fftk (the non-distributed convention).

    The jaxdecomp build of jaxpm replaces fftk with a distributed variant that
    expects a complex array, so we inline the real-shape version here to keep IC
    generation identical across the jax-gpu and jaxdecomp envs.
    """
    k = []
    for d in range(len(shape)):
        kd = np.fft.fftfreq(shape[d]) * 2 * np.pi
        kdshape = np.ones(len(shape), dtype="int")
        if symmetric and d == len(shape) - 1:
            kd = kd[: shape[d] // 2 + 1]
        kdshape[d] = len(kd)
        k.append(kd.reshape(kdshape).astype(dtype))
    return k


def make_pk_sqrt(cosmo: jc.Cosmology, cfg: ForwardModelConfig, k_points: int = 256) -> jnp.ndarray:
    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    box_size = [cfg.box_size_mpc_h] * 3

    k = jnp.logspace(-4, 1, k_points, dtype=jnp.float32)
    pk = jc.power.linear_matter_power(cosmo, k)

    def pk_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    kvec = _fftk(mesh_shape)
    kmesh = sum((kk / box_size[i] * mesh_shape[i]) ** 2 for i, kk in enumerate(kvec)) ** 0.5
    pk_mesh = pk_fn(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
        box_size[0] * box_size[1] * box_size[2]
    )
    return jnp.sqrt(jnp.asarray(pk_mesh, dtype=jnp.float32))


def white_noise_to_init_mesh(white_noise: jnp.ndarray, pk_sqrt: jnp.ndarray) -> jnp.ndarray:
    wn = jnp.asarray(white_noise, dtype=jnp.float32)
    mesh = jnp.fft.irfftn(jnp.fft.rfftn(wn) * pk_sqrt, s=wn.shape)
    return jnp.asarray(mesh, dtype=jnp.float32)


def make_lya_kaiser_scale(
    cosmo: jc.Cosmology,
    cfg: ForwardModelConfig,
    *,
    b_flux: float,
    beta_flux: float,
    n_eff: float,
    los_axis: int = 2,
    k_points: int = 256,
) -> jnp.ndarray:
    """Linear-LyA Kaiser posterior-preconditioning scale, in rfftn layout.

    Returns ``scale(k)`` of shape ``(mesh_n, mesh_n, mesh_n // 2 + 1)`` with

        scale(k) = sqrt(1 + n_eff * (D(a_obs) * b_flux * (1 + beta_flux * mu^2))^2 * P_mesh(k))

    where ``P_mesh`` uses the same normalization as :func:`make_pk_sqrt` (so that
    ``make_pk_sqrt(...)**2 == P_mesh``), ``mu`` is the cosine of the angle to the
    line-of-sight ``los_axis``, and ``D`` is the linear growth factor at the target
    scale factor. This is the LyA-flux analogue of FLBench's galaxy Kaiser boost
    ``D * (bE + f * mu^2)``; the posterior transfer is ``make_pk_sqrt / scale``.
    """
    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    box_size = [cfg.box_size_mpc_h] * 3

    k = jnp.logspace(-4, 1, k_points, dtype=jnp.float32)
    pk = jc.power.linear_matter_power(cosmo, k)

    def pk_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    kvec = _fftk(mesh_shape)
    kphys = [kk / box_size[i] * mesh_shape[i] for i, kk in enumerate(kvec)]
    kmesh = sum(kp**2 for kp in kphys) ** 0.5
    pk_mesh = pk_fn(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
        box_size[0] * box_size[1] * box_size[2]
    )

    # mu^2 = (k_los / |k|)^2 along the line-of-sight axis (0 at k=0).
    klos = kphys[int(los_axis)]
    kmesh_safe = jnp.where(kmesh > 0, kmesh, jnp.asarray(1.0, dtype=jnp.float32))
    mu2 = jnp.where(kmesh > 0, (klos / kmesh_safe) ** 2, jnp.asarray(0.0, dtype=jnp.float32))

    a_obs = jnp.atleast_1d(jnp.asarray(a_from_z(cfg.z_target), dtype=jnp.float32))
    growth = jnp.asarray(growth_factor(cosmo, a_obs), dtype=jnp.float32).reshape(())
    boost = growth * jnp.asarray(b_flux, dtype=jnp.float32) * (
        1.0 + jnp.asarray(beta_flux, dtype=jnp.float32) * mu2
    )
    scale = jnp.sqrt(1.0 + jnp.asarray(n_eff, dtype=jnp.float32) * boost**2 * pk_mesh)
    return jnp.asarray(scale, dtype=jnp.float32)


def make_pk_sqrt_post(pk_sqrt: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Posterior transfer ``P(k)^1/2 / scale(k)`` (rfftn layout).

    Replaces ``pk_sqrt`` in :func:`white_noise_to_init_mesh` so that the sampling
    coordinate has an approximately unit-Gaussian *posterior* under a linear LyA
    response model.
    """
    return jnp.asarray(pk_sqrt / scale, dtype=jnp.float32)


def make_lowpass_filter(cfg: ForwardModelConfig, k_cut_frac: float) -> jnp.ndarray:
    """Gaussian low-pass window in rfftn layout, for smoothing the REVERSE-pass
    gradient of the full-hydro forward (a surrogate force for adjusted samplers).

    The full-hydro/cooling gradient is rough (slope-limiter kinks, bilinear
    cooling-table corners, float32 jaggedness), which pins HMC/NUTS leapfrog to
    tiny steps. Low-passing the white-noise gradient removes that high-k roughness
    so the Hamiltonian flow is smooth and large steps are stable; with a
    Metropolis-adjusted sampler the accept/reject uses the EXACT forward
    log-density, so the filtered force only changes efficiency, not the target.

    k_cut_frac is the cutoff as a fraction of the per-axis Nyquist (pi, cell units).
    W(k) = exp(-0.5 (|k| / (k_cut_frac*pi))^2); W(0)=1. Shape (N, N, N//2+1).
    """
    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    kvec = _fftk(mesh_shape)
    kmesh = sum(kk**2 for kk in kvec) ** 0.5  # |k| in cell units (rfft layout)
    k_cut = max(float(k_cut_frac) * float(np.pi), 1.0e-6)
    W = jnp.exp(-0.5 * (kmesh / k_cut) ** 2)
    return jnp.asarray(W, dtype=jnp.float32)


def rfft_hermitian_weights(mesh_shape, dtype=np.float32) -> jnp.ndarray:
    """Per-mode multiplicities for the rfftn half-spectrum (Parseval weights).

    For a real field ``f`` with ``N`` total points,
    ``sum_x f(x)^2 == (1/N) * sum_k w_k * |rfftn(f)_k|^2`` where ``w_k`` is 1 for
    the self-conjugate planes (last-axis index 0 and, for even length, Nyquist)
    and 2 otherwise. Shape matches ``rfftn(f)``: ``(*mesh_shape[:-1], n_last//2+1)``.
    """
    n_last = int(mesh_shape[-1])
    w_last = np.full(n_last // 2 + 1, 2.0, dtype=dtype)
    w_last[0] = 1.0
    if n_last % 2 == 0:
        w_last[-1] = 1.0
    bshape = [1] * len(mesh_shape)
    bshape[-1] = w_last.size
    full = np.broadcast_to(
        w_last.reshape(bshape), (*mesh_shape[:-1], n_last // 2 + 1)
    ).astype(dtype)
    return jnp.asarray(full, dtype=dtype)


def _pm_acceleration(
    positions: jnp.ndarray,
    a: jnp.ndarray,
    cosmo: jc.Cosmology,
    cfg: ForwardModelConfig,
) -> jnp.ndarray:
    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    force = jpm.pm_forces(positions, mesh_shape=mesh_shape, r_split=0.0) * (1.5 * cosmo.Omega_m)
    return force / (a**2 * _a_to_E(cosmo, a))


def integrate_kdk(
    pos0: jnp.ndarray,
    vel0: jnp.ndarray,
    cosmo: jc.Cosmology,
    cfg: ForwardModelConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    steps = int(cfg.kdk_steps)
    if steps < 1:
        raise ValueError("kdk_steps must be >= 1")

    a0 = jnp.asarray(a_from_z(cfg.z_init), dtype=jnp.float32)
    a1 = jnp.asarray(a_from_z(cfg.z_target), dtype=jnp.float32)
    da = (a1 - a0) / jnp.asarray(steps, dtype=jnp.float32)

    def one_step(carry, i):
        pos, vel, acc = carry
        i_f = jnp.asarray(i, dtype=jnp.float32)
        a_mid = a0 + (i_f + 0.5) * da
        a_next = a0 + (i_f + 1.0) * da
        drift = 1.0 / (a_mid**3 * _a_to_E(cosmo, a_mid))

        vel_half = vel + 0.5 * da * acc
        pos_next = _wrap_positions(pos + da * drift * vel_half, cfg.mesh_n)
        acc_next = _pm_acceleration(pos_next, a_next, cosmo, cfg)
        vel_next = vel_half + 0.5 * da * acc_next
        return (pos_next, vel_next, acc_next)

    acc0 = _pm_acceleration(pos0, a0, cosmo, cfg)
    carry = (pos0, vel0, acc0)

    if cfg.checkpoint_every <= 1:
        def scan_step(c, i):
            return one_step(c, i), None

        if cfg.checkpoint:
            scan_step = jax.checkpoint(scan_step)

        (posf, velf, _), _ = jax.lax.scan(scan_step, carry, xs=jnp.arange(steps, dtype=jnp.int32))
        return posf, velf

    block = int(cfg.checkpoint_every)
    n_blocks = steps // block
    remainder = steps % block

    def inner_step(carry_in, _):
        state, idx = carry_in
        state = one_step(state, idx)
        return (state, idx + jnp.asarray(1, dtype=jnp.int32)), None

    def block_step(carry_in, _):
        carry_out, _ = jax.lax.scan(inner_step, carry_in, xs=None, length=block)
        return carry_out, None

    if cfg.checkpoint:
        block_step = jax.checkpoint(block_step)

    carry_with_idx = (carry, jnp.asarray(0, dtype=jnp.int32))

    if n_blocks > 0:
        carry_with_idx, _ = jax.lax.scan(block_step, carry_with_idx, xs=None, length=n_blocks)

    if remainder > 0:
        carry_with_idx, _ = jax.lax.scan(inner_step, carry_with_idx, xs=None, length=remainder)

    carry, _ = carry_with_idx
    posf, velf, _ = carry
    return posf, velf


def paint_density(positions: jnp.ndarray, mesh_n: int) -> jnp.ndarray:
    return cic_paint(jnp.zeros((mesh_n, mesh_n, mesh_n), dtype=jnp.float32), _wrap_positions(positions, mesh_n))


def gaussian_smooth_fft(field: jnp.ndarray, sigma_cells: float) -> jnp.ndarray:
    sigma = jnp.asarray(float(sigma_cells), dtype=jnp.float32)
    if float(sigma_cells) <= 0.0:
        return jnp.asarray(field, dtype=jnp.float32)

    nx, ny, nz = field.shape
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny)
    kz = 2.0 * jnp.pi * jnp.fft.rfftfreq(nz)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    kernel = jnp.exp(-0.5 * sigma**2 * k2)

    field_hat = jnp.fft.rfftn(field)
    smooth = jnp.fft.irfftn(field_hat * kernel, s=field.shape)
    return jnp.asarray(smooth, dtype=jnp.float32)


def _log_mean_exp(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable log(mean(exp(x))) via the log-sum-exp trick.

    Prevents float32 overflow (exp overflows above ~88) that occurs when the
    DM density field has extreme fluctuations (e.g. during MCLMC/HMC warmup).
    """
    x_max = jnp.max(x)
    return x_max + jnp.log(jnp.mean(jnp.exp(x - x_max)) + 1.0e-30)


def dm_to_gas_density(rho_dm: jnp.ndarray, params: GasModelParams) -> jnp.ndarray:
    log_dm = jnp.log(jnp.clip(rho_dm, 1.0e-6, None))
    log_dm_s = gaussian_smooth_fft(log_dm, params.smooth_sigma_cells)

    x = log_dm_s - jnp.mean(log_dm_s)
    q = x * x - jnp.mean(x * x)
    log_rho_gas = params.bias_linear * x + params.bias_quadratic * q

    # Clamp before exp to avoid float32 overflow (max safe value ~80).
    # Values above this would produce unphysically huge densities; the prior
    # gradient will push the chain back to reasonable amplitudes.
    log_rho_gas = jnp.clip(log_rho_gas, -80.0, 80.0)
    rho_gas = jnp.exp(log_rho_gas)
    rho_gas = rho_gas / (jnp.mean(rho_gas) + 1.0e-8) * params.gas_mean
    return jnp.asarray(rho_gas, dtype=jnp.float32)


def gas_density_to_temperature(rho_gas: jnp.ndarray, params: GasModelParams) -> jnp.ndarray:
    rho_n = rho_gas / (jnp.mean(rho_gas) + 1.0e-8)
    x = jnp.log(jnp.clip(rho_n, 1.0e-6, None))
    x = x - jnp.mean(x)
    q = x * x - jnp.mean(x * x)

    fluct = params.temp_slope * x + params.temp_quadratic * q
    # Keep the mean temperature anchored to temp_init_k * temp_heat_gain
    # irrespective of fluctuation variance.
    # Use the log-sum-exp trick (via _log_mean_exp) to avoid float32 overflow
    # when fluct has extreme values during sampler warmup/exploration.
    fluct = fluct - _log_mean_exp(fluct)
    base = jnp.log(jnp.asarray(params.temp_init_k * params.temp_heat_gain, dtype=jnp.float32) + 1.0e-12)
    log_t = base + fluct
    return jnp.exp(log_t).astype(jnp.float32)


def forward_fields(
    white_noise: jnp.ndarray,
    pk_sqrt: jnp.ndarray,
    grid_pos: jnp.ndarray,
    cosmo: jc.Cosmology,
    cfg: ForwardModelConfig,
    gas_params: GasModelParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    init_mesh = white_noise_to_init_mesh(white_noise, pk_sqrt)
    a_init = a_from_z(cfg.z_init)
    dx_mesh, p_mesh, _ = jpm.lpt(cosmo, init_mesh, grid_pos, a_init, order=cfg.lpt_order)
    pos0 = _wrap_positions(grid_pos + dx_mesh, cfg.mesh_n)
    vel0 = p_mesh

    posf, _ = integrate_kdk(pos0, vel0, cosmo, cfg)
    rho_dm = paint_density(posf, cfg.mesh_n)
    rho_gas = dm_to_gas_density(rho_dm, gas_params)
    temp_gas = gas_density_to_temperature(rho_gas, gas_params)
    return rho_dm, rho_gas, temp_gas, posf, init_mesh


def make_density_nlogposterior(
    target_rho: jnp.ndarray,
    pk_sqrt: jnp.ndarray,
    grid_pos: jnp.ndarray,
    cosmo: jc.Cosmology,
    cfg: ForwardModelConfig,
    gas_params: GasModelParams,
    noise_sigma: float = 0.25,
    prior_weight: float = 1.0,
    compare_space: str = "log",
):
    if compare_space not in {"log", "linear"}:
        raise ValueError(f"Unknown compare_space={compare_space}")

    target_rho = jnp.asarray(target_rho, dtype=jnp.float32)
    sigma = jnp.asarray(float(noise_sigma), dtype=jnp.float32)
    prior_w = jnp.asarray(float(prior_weight), dtype=jnp.float32)

    def nlogpost(white_noise: jnp.ndarray):
        _, rho_gas, _, _, _ = forward_fields(white_noise, pk_sqrt, grid_pos, cosmo, cfg, gas_params)

        pred = rho_gas / (jnp.mean(rho_gas) + 1.0e-8)
        targ = target_rho / (jnp.mean(target_rho) + 1.0e-8)

        if compare_space == "log":
            pred_c = jnp.log(jnp.clip(pred, 1.0e-6, None))
            targ_c = jnp.log(jnp.clip(targ, 1.0e-6, None))
        else:
            pred_c = pred
            targ_c = targ

        resid = pred_c - targ_c
        data_nll = 0.5 * jnp.mean((resid / sigma) ** 2 + 2.0 * jnp.log(sigma))
        prior_nll = 0.5 * prior_w * jnp.mean(white_noise**2)
        loss = data_nll + prior_nll
        return loss, (data_nll, prior_nll, rho_gas)

    return nlogpost
