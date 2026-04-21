from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
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


def make_pk_sqrt(cosmo: jc.Cosmology, cfg: ForwardModelConfig, k_points: int = 256) -> jnp.ndarray:
    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    box_size = [cfg.box_size_mpc_h] * 3

    k = jnp.logspace(-4, 1, k_points, dtype=jnp.float32)
    pk = jc.power.linear_matter_power(cosmo, k)

    def pk_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    kvec = jpm.fftk(mesh_shape)
    kmesh = sum((kk / box_size[i] * mesh_shape[i]) ** 2 for i, kk in enumerate(kvec)) ** 0.5
    pk_mesh = pk_fn(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
        box_size[0] * box_size[1] * box_size[2]
    )
    return jnp.sqrt(jnp.asarray(pk_mesh, dtype=jnp.float32))


def white_noise_to_init_mesh(white_noise: jnp.ndarray, pk_sqrt: jnp.ndarray) -> jnp.ndarray:
    wn = jnp.asarray(white_noise, dtype=jnp.float32)
    mesh = jnp.fft.irfftn(jnp.fft.rfftn(wn) * pk_sqrt, s=wn.shape)
    return jnp.asarray(mesh, dtype=jnp.float32)


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
