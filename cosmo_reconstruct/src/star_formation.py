from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxpm.painting import cic_paint
from diffhydro.cosmology import conversions as cosmo_conv

try:
    from jaxpm.painting import cic_read as _jaxpm_cic_read
except ImportError:  # pragma: no cover - depends on installed jaxpm version
    _jaxpm_cic_read = None

# Distributed mode: jaxpm's painting differs by env (the jaxdecomp build wants
# grid-shaped fields, not flat positions). When this is set, the SF painting uses
# the pure-JAX flat-position kernels below (env-agnostic, differentiable) instead
# of jaxpm. Toggled by the distributed driver; default off keeps single-GPU intact.
_USE_LOCAL_CIC = False


def use_local_cic(flag: bool = True) -> None:
    global _USE_LOCAL_CIC
    _USE_LOCAL_CIC = bool(flag)


_CIC_OFFSETS = jnp.array(
    [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
     [1., 1., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]], dtype=jnp.float32)[None, :, :]


def _cic_paint_local(particle_scalar, particle_positions, mesh_shape):
    """Pure-JAX trilinear CIC scatter (flat positions); matches jaxpm cic_paint math."""
    mesh = jnp.zeros(tuple(int(s) for s in mesh_shape), dtype=jnp.float32)
    pos = jnp.expand_dims(jnp.asarray(particle_positions, jnp.float32), 1)
    neigh = jnp.floor(pos) + _CIC_OFFSETS
    kernel = 1.0 - jnp.abs(pos - neigh)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]
    kernel = kernel * jnp.expand_dims(jnp.asarray(particle_scalar, jnp.float32), -1)
    neigh = jnp.mod(neigh.astype(jnp.int32), jnp.asarray(mesh.shape, jnp.int32))
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2))
    return jax.lax.scatter_add(mesh, neigh, kernel, dnums)


def particle_to_grid_cic_paint(
    particle_scalar: jnp.ndarray,
    particle_positions: jnp.ndarray,
    mesh_shape: tuple[int, int, int],
) -> jnp.ndarray:
    if _USE_LOCAL_CIC:
        return _cic_paint_local(particle_scalar, particle_positions, mesh_shape)
    mesh = jnp.zeros(mesh_shape, dtype=jnp.float32)
    return cic_paint(mesh, particle_positions, weight=jnp.asarray(particle_scalar, dtype=jnp.float32))


def apply_particle_sigma_threshold(
    particle_scalar: jnp.ndarray,
    sigma_threshold: float | None,
    min_mass: float | None = None,
) -> jnp.ndarray:
    scalar = jnp.maximum(jnp.asarray(particle_scalar, dtype=jnp.float32), 0.0)
    if min_mass is not None:
        min_mass_value = float(min_mass)
        if math.isfinite(min_mass_value):
            scalar = jnp.where(scalar >= jnp.asarray(min_mass_value, dtype=jnp.float32), scalar, 0.0)
    if sigma_threshold is None:
        return scalar
    sigma_threshold_value = float(sigma_threshold)
    if not math.isfinite(sigma_threshold_value):
        return scalar

    positive = scalar > 0.0
    count = jnp.sum(positive.astype(jnp.float32))

    def _apply_threshold(_: None) -> jnp.ndarray:
        mean = jnp.sum(jnp.where(positive, scalar, 0.0)) / jnp.maximum(count, 1.0)
        var = jnp.sum(jnp.where(positive, (scalar - mean) ** 2, 0.0)) / jnp.maximum(count, 1.0)
        std = jnp.sqrt(jnp.maximum(var, 0.0))
        cutoff = jnp.maximum(mean + jnp.asarray(sigma_threshold_value, dtype=jnp.float32) * std, 0.0)
        return jnp.where(scalar >= cutoff, scalar, 0.0)

    return jax.lax.cond(count > 0.0, _apply_threshold, lambda _: scalar, operand=None)


def _fallback_cic_read(grid_field: jnp.ndarray, particle_positions: jnp.ndarray) -> jnp.ndarray:
    grid_field = jnp.asarray(grid_field, dtype=jnp.float32)
    pos = jnp.asarray(particle_positions, dtype=jnp.float32)
    nx, ny, nz = (int(s) for s in grid_field.shape)

    x = jnp.mod(pos[:, 0], nx)
    y = jnp.mod(pos[:, 1], ny)
    z = jnp.mod(pos[:, 2], nz)

    i0 = jnp.floor(x).astype(jnp.int32)
    j0 = jnp.floor(y).astype(jnp.int32)
    k0 = jnp.floor(z).astype(jnp.int32)
    fx = x - i0.astype(jnp.float32)
    fy = y - j0.astype(jnp.float32)
    fz = z - k0.astype(jnp.float32)

    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    k1 = (k0 + 1) % nz

    def gather(ii, jj, kk):
        return grid_field[ii, jj, kk]

    w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
    w001 = (1.0 - fx) * (1.0 - fy) * fz
    w010 = (1.0 - fx) * fy * (1.0 - fz)
    w011 = (1.0 - fx) * fy * fz
    w100 = fx * (1.0 - fy) * (1.0 - fz)
    w101 = fx * (1.0 - fy) * fz
    w110 = fx * fy * (1.0 - fz)
    w111 = fx * fy * fz

    return (
        w000 * gather(i0, j0, k0)
        + w001 * gather(i0, j0, k1)
        + w010 * gather(i0, j1, k0)
        + w011 * gather(i0, j1, k1)
        + w100 * gather(i1, j0, k0)
        + w101 * gather(i1, j0, k1)
        + w110 * gather(i1, j1, k0)
        + w111 * gather(i1, j1, k1)
    )


def grid_to_particle_cic_readout(grid_field: jnp.ndarray, particle_positions: jnp.ndarray) -> jnp.ndarray:
    if _USE_LOCAL_CIC or _jaxpm_cic_read is None:
        return _fallback_cic_read(grid_field, particle_positions)
    return jnp.asarray(_jaxpm_cic_read(jnp.asarray(grid_field, dtype=jnp.float32), particle_positions), dtype=jnp.float32)


def grid_to_particle_mass_increment(
    grid_field: jnp.ndarray,
    particle_positions: jnp.ndarray,
    *,
    eps: float = 1.0e-8,
) -> jnp.ndarray:
    sampled = jnp.maximum(grid_to_particle_cic_readout(grid_field, particle_positions), 0.0)
    total_grid = jnp.sum(jnp.maximum(grid_field, 0.0))
    total_sampled = jnp.sum(sampled)
    scale = jnp.where(total_sampled > eps, total_grid / total_sampled, 0.0)
    return sampled * scale


def make_feedback_kernel(
    *,
    width_cells: float = 1.0,
    kind: str = "3x3x3",
) -> jnp.ndarray:
    kind_norm = str(kind).lower()
    if kind_norm not in {"3x3x3", "gaussian", "gaussian3x3x3"}:
        raise ValueError(f"Unsupported feedback kernel kind: {kind}")
    coords = jnp.arange(-1, 2, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(coords, coords, coords, indexing="ij")
    r2 = gx * gx + gy * gy + gz * gz
    sigma2 = jnp.maximum(jnp.asarray(float(width_cells), dtype=jnp.float32) ** 2, 1.0e-6)
    kernel = jnp.exp(-0.5 * r2 / sigma2)
    kernel = kernel / jnp.maximum(jnp.sum(kernel), 1.0e-30)
    return kernel.astype(jnp.float32)


def convolve_scalar_periodic(
    grid_field: jnp.ndarray,
    kernel: jnp.ndarray,
) -> jnp.ndarray:
    field = jnp.asarray(grid_field, dtype=jnp.float32)
    kern = jnp.asarray(kernel, dtype=jnp.float32)
    out = jnp.zeros_like(field)
    for ix, dx in enumerate((-1, 0, 1)):
        for iy, dy in enumerate((-1, 0, 1)):
            for iz, dz in enumerate((-1, 0, 1)):
                weight = kern[ix, iy, iz]
                out = out + weight * jnp.roll(field, shift=(dx, dy, dz), axis=(0, 1, 2))
    return out


def compute_momentum_feedback_fields(
    source_field: jnp.ndarray,
    *,
    momentum_scale: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    src = jnp.asarray(source_field, dtype=jnp.float32)
    coeff = jnp.asarray(momentum_scale, dtype=jnp.float32)
    dmx = coeff * (jnp.roll(src, -1, axis=0) - jnp.roll(src, 1, axis=0))
    dmy = coeff * (jnp.roll(src, -1, axis=1) - jnp.roll(src, 1, axis=1))
    dmz = coeff * (jnp.roll(src, -1, axis=2) - jnp.roll(src, 1, axis=2))
    return dmx, dmy, dmz


@jax.custom_vjp
def positive_source_product(scale: jnp.ndarray, source: jnp.ndarray) -> jnp.ndarray:
    source_pos = jnp.maximum(jnp.asarray(source, dtype=jnp.float32), 0.0)
    return jnp.asarray(scale, dtype=jnp.float32) * source_pos


def _positive_source_product_fwd(scale: jnp.ndarray, source: jnp.ndarray):
    scale_arr = jnp.asarray(scale, dtype=jnp.float32)
    source_pos = jnp.maximum(jnp.asarray(source, dtype=jnp.float32), 0.0)
    return scale_arr * source_pos, (scale_arr, source_pos)


def _positive_source_product_bwd(res, g):
    scale_arr, source_pos = res
    active = source_pos > 0.0
    g_finite = jnp.nan_to_num(jnp.asarray(g, dtype=jnp.float32), nan=0.0, posinf=0.0, neginf=0.0)
    g_active = jnp.where(active, g_finite, 0.0)
    dscale = jnp.sum(g_active * source_pos)
    dsource = g_active * scale_arr
    return dscale, dsource


positive_source_product.defvjp(_positive_source_product_fwd, _positive_source_product_bwd)


@jax.custom_vjp
def finite_cotangent_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(jnp.asarray(x, dtype=jnp.float32))


def _finite_cotangent_sigmoid_fwd(x: jnp.ndarray):
    y = jax.nn.sigmoid(jnp.asarray(x, dtype=jnp.float32))
    return y, y


def _finite_cotangent_sigmoid_bwd(y, g):
    g_finite = jnp.nan_to_num(jnp.asarray(g, dtype=jnp.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return (g_finite * y * (1.0 - y),)


finite_cotangent_sigmoid.defvjp(_finite_cotangent_sigmoid_fwd, _finite_cotangent_sigmoid_bwd)


_COT_CLIP = 1.0e6


@jax.custom_vjp
def finite_cotangent_passthrough(x):
    """Identity forward; backward sanitizes the cotangent (nan->0, clamp to +-1e6).

    The hydro adjoint w.r.t. an injected per-cell energy field is singular in many
    cells (NaN/huge), which poisons per-cell trainable weights (AGN net, CNN
    feedback head). This is a no-op for finite, in-range cotangents (so the working
    scalar-feedback path is unchanged) but zeros/bounds the pathological ones."""
    return x


def _fcp_fwd(x):
    return x, None


def _fcp_bwd(_, g):
    g = jnp.nan_to_num(jnp.asarray(g, dtype=jnp.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return (jnp.clip(g, -_COT_CLIP, _COT_CLIP),)


finite_cotangent_passthrough.defvjp(_fcp_fwd, _fcp_bwd)


def smooth_sigmoid(x: jnp.ndarray, center: float | jnp.ndarray, width: float | jnp.ndarray) -> jnp.ndarray:
    width_safe = max(float(width), 1.0e-10)
    return finite_cotangent_sigmoid((x - jnp.asarray(center, dtype=jnp.float32)) / width_safe)


def gumbel_relaxed_bernoulli(
    key: jax.Array,
    probability: jnp.ndarray,
    tau: float,
    clip_val: float = 1.0e-20,
) -> tuple[jax.Array, jnp.ndarray]:
    p = jnp.clip(jnp.asarray(probability, dtype=jnp.float32), clip_val, 1.0 - clip_val)
    k1, k2, kout = jr.split(key, 3)
    u1 = jr.uniform(k1, p.shape, minval=clip_val, maxval=1.0 - clip_val, dtype=jnp.float32)
    u2 = jr.uniform(k2, p.shape, minval=clip_val, maxval=1.0 - clip_val, dtype=jnp.float32)
    g1 = -jnp.log(-jnp.log(u1))
    g2 = -jnp.log(-jnp.log(u2))
    logits = (jnp.log(p) - jnp.log1p(-p) + g1 - g2) / max(float(tau), 1.0e-4)
    return kout, jax.nn.sigmoid(logits)


def initialize_star_particle_fields(
    particle_positions: jnp.ndarray,
    *,
    amplitude: float = 0.0,
    kind: str = "none",
    seed: int = 0,
    n_age_bins: int = 0,
) -> dict[str, jnp.ndarray]:
    n_particles = int(particle_positions.shape[0])
    star_mass = jnp.zeros((n_particles,), dtype=jnp.float32)
    kind_norm = str(kind).lower()

    if float(amplitude) > 0.0 and kind_norm != "none":
        if kind_norm == "random":
            key = jr.PRNGKey(int(seed))
            star_mass = float(amplitude) * jr.uniform(key, (n_particles,), dtype=jnp.float32)
        else:
            mesh_shape = jnp.max(particle_positions, axis=0) + 1.0
            center = 0.5 * mesh_shape
            disp = particle_positions - center[None, :]
            radius2 = jnp.sum(disp * disp, axis=1)
            sigma2 = jnp.maximum(0.1 * jnp.mean(mesh_shape) ** 2, 1.0)
            star_mass = float(amplitude) * jnp.exp(-0.5 * radius2 / sigma2)

    out = {"star_mass": jnp.maximum(star_mass, 0.0)}
    if int(n_age_bins) > 0:
        age_bins = jnp.zeros((n_particles, int(n_age_bins)), dtype=jnp.float32)
        age_bins = age_bins.at[:, 0].set(out["star_mass"])
        out["star_age_bins"] = age_bins
    return out


@dataclass(frozen=True)
class StarFormationDiagnostics:
    overdensity: jnp.ndarray
    nH_cgs: jnp.ndarray
    temperature_k: jnp.ndarray
    s_rho: jnp.ndarray
    s_temp: jnp.ndarray
    pi0: jnp.ndarray
    z_hat: jnp.ndarray
    delta_star_grid: jnp.ndarray
    stellar_density_grid: jnp.ndarray
    feedback_source_grid: jnp.ndarray
    metal_deposition_grid: jnp.ndarray
    thermal_deposition_grid: jnp.ndarray
    feedback_momentum_x: jnp.ndarray
    feedback_momentum_y: jnp.ndarray
    feedback_momentum_z: jnp.ndarray
    feedback_kinetic_energy: jnp.ndarray
    feedback_snf_energy: jnp.ndarray
    feedback_stellar_wind_energy: jnp.ndarray
    feedback_total_energy: jnp.ndarray
    feedback_clip_scale: jnp.ndarray
    # Optional CNN 2nd-head output (per-cell log feedback-energy modulation); None
    # for a 1-channel CNN. Defaulted so existing keyword constructions are unaffected.
    cnn_feedback_log_mod: "jnp.ndarray | None" = None


def evaluate_star_formation_diagnostics(
    U_gas: jnp.ndarray,
    params: dict,
    *,
    eq,
    code_to_kelvin_temp: float,
    density_threshold_mode: str,
    density_threshold: float,
    density_width: float,
    rho_unit_cgs: float,
    h_species: float,
    mH_cgs: float,
    temperature_threshold_k: float,
    temperature_width_k: float,
    sf_pi0: float | jnp.ndarray,
    sf_tau: float,
    sf_mass_scale: float,
    sf_max_fraction_per_step: float,
    dtau: float,
    mode: str = "deterministic",
    rng_key: jax.Array | None = None,
    sf_cnn_weights: dict | None = None,
    sf_cnn_lambda: float | jnp.ndarray = 1.0,
    cnn_extra_fields: tuple | None = None,
) -> tuple[StarFormationDiagnostics, jax.Array | None]:
    W = eq.get_primitives_from_conservatives(U_gas)
    rho = jnp.maximum(W[eq.mass_ids], getattr(eq, "eps", 1.0e-20))
    p = jnp.maximum(W[eq.energy_ids], getattr(eq, "eps", 1.0e-20))
    temp_code = eq.get_temperature(p, rho)
    temp_k = temp_code * jnp.asarray(float(code_to_kelvin_temp), dtype=jnp.float32)
    overdensity = rho / (jnp.mean(rho) + 1.0e-8)
    a = jnp.asarray(params.get("a", 1.0), dtype=jnp.float32)
    rho_phys = jnp.maximum(cosmo_conv.density_code_to_phys(rho, a), 1.0e-30)
    nH_cgs = (
        jnp.asarray(float(h_species), dtype=jnp.float32)
        * rho_phys
        * jnp.asarray(float(rho_unit_cgs), dtype=jnp.float32)
        / jnp.asarray(float(mH_cgs), dtype=jnp.float32)
    )

    mode_norm = str(density_threshold_mode).lower()
    if mode_norm == "physical_nh":
        density_field = nH_cgs
    elif mode_norm == "physical_nh_comoving":
        density_field = nH_cgs * (a**3)
    elif mode_norm == "overdensity":
        density_field = overdensity
    else:
        raise ValueError(f"Unsupported density_threshold_mode: {density_threshold_mode}")

    s_rho = smooth_sigmoid(density_field, density_threshold, density_width)
    s_temp = smooth_sigmoid(jnp.asarray(temperature_threshold_k, dtype=jnp.float32) - temp_k, 0.0, temperature_width_k)
    pi0 = jnp.clip(jnp.asarray(sf_pi0, dtype=jnp.float32) * s_rho * s_temp, 0.0, 1.0)

    cnn_feedback_log_mod = None
    if sf_cnn_weights is not None:
        # Hybrid SF model: pi0 = sigmoid(lambda * logit(pi0_phys) + CNN(fields)).
        # Zero-initialized CNN final layer + lambda=1 reproduces the physical model.
        from cosmo_feedback.sf_cnn import apply_sf_cnn, build_cnn_inputs

        metal_fraction, stellar_grid = cnn_extra_fields
        # 5 input channels = time-dependent variant: append the scale factor.
        cnn_in_ch = int(sf_cnn_weights["w0"].shape[1])
        channels = build_cnn_inputs(
            rho_phys, temp_k, metal_fraction, stellar_grid,
            scale_factor=a if cnn_in_ch == 5 else None,
        )
        # remat: the roll-based conv would otherwise save 27 shifted activation
        # copies per layer (~8 GB/step) as backward residuals
        cnn_out = jax.checkpoint(apply_sf_cnn)(sf_cnn_weights, channels)
        # 2-channel CNN: ch0 = pi0 logit residual, ch1 = log feedback-energy modulation.
        if cnn_out.ndim == 4:
            residual = cnn_out[0]
            cnn_feedback_log_mod = cnn_out[1]
        else:
            residual = cnn_out
            cnn_feedback_log_mod = None
        # 1e-8 keeps spurious Gumbel firings negligible (~0.02 cells/step at 128^3)
        # while the floored logit (~ -18.4) stays within reach of lambda + CNN output.
        eps = jnp.asarray(1.0e-8, dtype=jnp.float32)
        pi0_c = jnp.clip(pi0, eps, 1.0 - eps)
        logit_phys = jnp.log(pi0_c) - jnp.log1p(-pi0_c)
        lam = jnp.asarray(sf_cnn_lambda, dtype=jnp.float32)
        # finite_cotangent_sigmoid: a single non-finite downstream cotangent cell
        # (Gumbel ST derivative ~1/tau compounding over the late rollout) would
        # otherwise poison every CNN weight through the conv sum.
        pi0 = finite_cotangent_sigmoid(lam * logit_phys + residual)

    mode_norm = str(mode).lower()
    if mode_norm == "gumbel":
        if rng_key is None:
            rng_key = jr.PRNGKey(0)
        rng_key, z_hat = gumbel_relaxed_bernoulli(rng_key, pi0, sf_tau)
    else:
        z_hat = pi0

    dtau_arr = jnp.asarray(dtau, dtype=jnp.float32)
    formed_fraction = jnp.clip(
        jnp.asarray(float(sf_mass_scale), dtype=jnp.float32) * dtau_arr * z_hat,
        0.0,
        float(sf_max_fraction_per_step),
    )
    delta_star_grid = formed_fraction * rho
    return (
        StarFormationDiagnostics(
            overdensity=overdensity,
            nH_cgs=nH_cgs,
            temperature_k=temp_k,
            s_rho=s_rho,
            s_temp=s_temp,
            pi0=pi0,
            z_hat=z_hat,
            delta_star_grid=delta_star_grid,
            stellar_density_grid=jnp.zeros_like(delta_star_grid),
            feedback_source_grid=jnp.zeros_like(delta_star_grid),
            metal_deposition_grid=jnp.zeros_like(delta_star_grid),
            thermal_deposition_grid=jnp.zeros_like(delta_star_grid),
            feedback_momentum_x=jnp.zeros_like(delta_star_grid),
            feedback_momentum_y=jnp.zeros_like(delta_star_grid),
            feedback_momentum_z=jnp.zeros_like(delta_star_grid),
            feedback_kinetic_energy=jnp.zeros_like(delta_star_grid),
            feedback_snf_energy=jnp.zeros_like(delta_star_grid),
            feedback_stellar_wind_energy=jnp.zeros_like(delta_star_grid),
            feedback_total_energy=jnp.zeros_like(delta_star_grid),
            feedback_clip_scale=jnp.ones_like(delta_star_grid),
            cnn_feedback_log_mod=cnn_feedback_log_mod,
        ),
        rng_key,
    )


class StellarFeedbackTracerForce:
    def __init__(
        self,
        *,
        eq,
        code_to_kelvin_temp: float,
        mode: str = "deterministic",
        density_threshold_mode: str = "overdensity",
        density_threshold: float = 10.0,
        density_width: float = 1.0,
        rho_unit_cgs: float = 1.6e-24,
        h_species: float = 0.76,
        temperature_threshold_k: float = 2.0e4,
        temperature_width_k: float = 5.0e3,
        sf_pi0: float = 0.05,
        sf_tau: float = 0.3,
        sf_mass_scale: float = 0.05,
        sf_max_fraction_per_step: float = 0.25,
        star_step_start: int = 0,
        seed: int = 0,
        n_age_bins: int = 0,
        age_shift_steps: int = 0,
        enable_metal_source: bool = False,
        metal_yield: float = 0.02,
        stellar_paint_sigma_threshold: float | None = None,
        stellar_paint_min_mass: float | None = None,
        feedback_deposition_mode: str = "local",
        unified_feedback_kernel: bool = True,
        feedback_kernel_kind: str = "3x3x3",
        metal_kernel_width_cells: float = 1.0,
        thermal_kernel_width_cells: float = 1.0,
        momentum_kernel_width_cells: float = 1.0,
        enable_momentum_feedback: bool = False,
        feedback_momentum_scale: float = 0.0,
        snf_energy: float = 0.0,
        stellar_wind_energy: float = 0.0,
        feedback_energy_clip_fraction: float | None = None,
        store_diagnostics: bool = False,
    ):
        self.eq = eq
        self.code_to_kelvin_temp = float(code_to_kelvin_temp)
        self.mode = str(mode).lower()
        self.density_threshold_mode = str(density_threshold_mode).lower()
        self.density_threshold = float(density_threshold)
        self.density_width = float(density_width)
        self.rho_unit_cgs = float(rho_unit_cgs)
        self.h_species = float(h_species)
        self.temperature_threshold_k = float(temperature_threshold_k)
        self.temperature_width_k = float(temperature_width_k)
        self.sf_pi0 = float(sf_pi0)
        self.sf_tau = float(sf_tau)
        self.sf_mass_scale = float(sf_mass_scale)
        self.sf_max_fraction_per_step = float(sf_max_fraction_per_step)
        self.star_step_start = int(max(0, star_step_start))
        self.seed = int(seed)
        self.n_age_bins = int(max(0, n_age_bins))
        self.age_shift_steps = int(max(0, age_shift_steps))
        self.enable_metal_source = bool(enable_metal_source)
        self.metal_yield = float(metal_yield)
        self.stellar_paint_sigma_threshold = None if stellar_paint_sigma_threshold is None else float(stellar_paint_sigma_threshold)
        self.stellar_paint_min_mass = None if stellar_paint_min_mass is None else float(stellar_paint_min_mass)
        self.feedback_deposition_mode = str(feedback_deposition_mode).lower()
        self.unified_feedback_kernel = bool(unified_feedback_kernel)
        self.feedback_kernel_kind = str(feedback_kernel_kind)
        self.metal_kernel_width_cells = float(max(metal_kernel_width_cells, 1.0e-3))
        self.thermal_kernel_width_cells = float(max(thermal_kernel_width_cells, 1.0e-3))
        self.momentum_kernel_width_cells = float(max(momentum_kernel_width_cells, 1.0e-3))
        self.enable_momentum_feedback = bool(enable_momentum_feedback)
        self.feedback_momentum_scale = float(feedback_momentum_scale)
        self.snf_energy = float(max(snf_energy, 0.0))
        self.stellar_wind_energy = float(max(stellar_wind_energy, 0.0))
        self.feedback_energy_clip_fraction = None if feedback_energy_clip_fraction is None else float(feedback_energy_clip_fraction)
        self.store_diagnostics = bool(store_diagnostics)
        self.mH_cgs = 1.6735575e-24

    def timestep(self, U):
        del U
        return jnp.asarray(1.0e10, dtype=jnp.float32)

    def _shift_age_bins(self, age_bins: jnp.ndarray) -> jnp.ndarray:
        if self.n_age_bins <= 1:
            return age_bins
        youngest = jnp.zeros_like(age_bins[:, :1])
        middle = age_bins[:, :-2] if self.n_age_bins > 2 else jnp.zeros((age_bins.shape[0], 0), dtype=age_bins.dtype)
        oldest = age_bins[:, -1:] + age_bins[:, -2:-1]
        return jnp.concatenate([youngest, middle, oldest], axis=1)

    def force(self, i_step, U_gas, params, dtau):
        dm_params = params.get("dm", None)
        if dm_params is None or "x" not in dm_params:
            return U_gas, params

        hyper = params.get("feedback_hyper", {})
        density_threshold = hyper.get("sf_density_threshold", self.density_threshold)
        temperature_threshold_k = hyper.get("sf_temperature_threshold_k", self.temperature_threshold_k)
        snf_energy = jnp.maximum(
            jnp.asarray(hyper.get("snf_energy", self.snf_energy), dtype=jnp.float32),
            0.0,
        )
        stellar_wind_energy = jnp.maximum(
            jnp.asarray(hyper.get("stellar_wind_energy", self.stellar_wind_energy), dtype=jnp.float32),
            0.0,
        )
        feedback_momentum_scale = jnp.asarray(
            hyper.get("feedback_momentum_scale", self.feedback_momentum_scale),
            dtype=jnp.float32,
        )
        metal_yield = jnp.maximum(
            jnp.asarray(hyper.get("metal_yield", self.metal_yield), dtype=jnp.float32),
            0.0,
        )
        sf_pi0 = jnp.clip(
            jnp.asarray(hyper.get("sf_pi0", self.sf_pi0), dtype=jnp.float32),
            0.0,
            1.0,
        )

        i_step_arr = jnp.asarray(i_step, dtype=jnp.int32)
        rng_key = params.get("rng_sf", jr.PRNGKey(self.seed))
        dm_out = dict(dm_params)
        dm_positions = jax.lax.stop_gradient(jnp.asarray(dm_out["x"], dtype=jnp.float32))
        sf_active = i_step_arr >= jnp.asarray(self.star_step_start, dtype=jnp.int32)

        sf_cnn_weights = hyper.get("sf_cnn", None)
        sf_cnn_lambda = hyper.get("sf_cnn_lambda", 1.0)
        cnn_extra_fields = None
        if sf_cnn_weights is not None:
            from .metallicity import extract_metal_fields

            mesh_shape = tuple(int(v) for v in U_gas.shape[-3:])
            _, metal_fraction = extract_metal_fields(U_gas, self.eq)
            if metal_fraction is None:
                metal_fraction = jnp.zeros(mesh_shape, dtype=jnp.float32)
            star_mass_cur = jnp.asarray(
                dm_out.get("star_mass", jnp.zeros((dm_positions.shape[0],), dtype=jnp.float32)),
                dtype=jnp.float32,
            )
            stellar_grid = particle_to_grid_cic_paint(star_mass_cur, dm_positions, mesh_shape)
            cnn_extra_fields = (metal_fraction, stellar_grid)

        sf_diag, rng_next = evaluate_star_formation_diagnostics(
            U_gas,
            params,
            eq=self.eq,
            code_to_kelvin_temp=self.code_to_kelvin_temp,
            density_threshold_mode=self.density_threshold_mode,
            density_threshold=density_threshold,
            density_width=self.density_width,
            rho_unit_cgs=self.rho_unit_cgs,
            h_species=self.h_species,
            mH_cgs=self.mH_cgs,
            temperature_threshold_k=temperature_threshold_k,
            temperature_width_k=self.temperature_width_k,
            sf_pi0=sf_pi0,
            sf_tau=self.sf_tau,
            sf_mass_scale=self.sf_mass_scale,
            sf_max_fraction_per_step=self.sf_max_fraction_per_step,
            dtau=jnp.asarray(dtau, dtype=jnp.float32),
            mode=self.mode,
            rng_key=rng_key,
            sf_cnn_weights=sf_cnn_weights,
            sf_cnn_lambda=sf_cnn_lambda,
            cnn_extra_fields=cnn_extra_fields,
        )
        sf_diag = StarFormationDiagnostics(
            overdensity=sf_diag.overdensity,
            nH_cgs=sf_diag.nH_cgs,
            temperature_k=sf_diag.temperature_k,
            s_rho=sf_diag.s_rho,
            s_temp=sf_diag.s_temp,
            pi0=jnp.where(sf_active, sf_diag.pi0, jnp.zeros_like(sf_diag.pi0)),
            z_hat=jnp.where(sf_active, sf_diag.z_hat, jnp.zeros_like(sf_diag.z_hat)),
            delta_star_grid=jnp.where(sf_active, sf_diag.delta_star_grid, jnp.zeros_like(sf_diag.delta_star_grid)),
            stellar_density_grid=sf_diag.stellar_density_grid,
            feedback_source_grid=sf_diag.feedback_source_grid,
            metal_deposition_grid=sf_diag.metal_deposition_grid,
            thermal_deposition_grid=sf_diag.thermal_deposition_grid,
            feedback_momentum_x=sf_diag.feedback_momentum_x,
            feedback_momentum_y=sf_diag.feedback_momentum_y,
            feedback_momentum_z=sf_diag.feedback_momentum_z,
            feedback_kinetic_energy=sf_diag.feedback_kinetic_energy,
            feedback_snf_energy=sf_diag.feedback_snf_energy,
            feedback_stellar_wind_energy=sf_diag.feedback_stellar_wind_energy,
            feedback_total_energy=sf_diag.feedback_total_energy,
            feedback_clip_scale=sf_diag.feedback_clip_scale,
            cnn_feedback_log_mod=sf_diag.cnn_feedback_log_mod,
        )

        delta_star_particles = grid_to_particle_mass_increment(sf_diag.delta_star_grid, dm_positions)

        star_mass_old = jnp.asarray(
            dm_out.get("star_mass", jnp.zeros((dm_positions.shape[0],), dtype=jnp.float32)),
            dtype=jnp.float32,
        )
        star_mass_new = jnp.maximum(star_mass_old + delta_star_particles, 0.0)
        dm_out["star_mass"] = star_mass_new

        if self.n_age_bins > 0:
            star_age_bins = jnp.asarray(
                dm_out.get(
                    "star_age_bins",
                    jnp.zeros((dm_positions.shape[0], self.n_age_bins), dtype=jnp.float32),
                ),
                dtype=jnp.float32,
            )
            if self.age_shift_steps > 0:
                do_shift = sf_active & (((i_step_arr + 1) % self.age_shift_steps) == 0)
                star_age_bins = jax.lax.cond(
                    do_shift,
                    lambda bins: self._shift_age_bins(bins),
                    lambda bins: bins,
                    star_age_bins,
                )
            star_age_bins = star_age_bins.at[:, 0].add(delta_star_particles)
            dm_out["star_age_bins"] = jnp.maximum(star_age_bins, 0.0)
            dm_out["star_mass"] = jnp.sum(dm_out["star_age_bins"], axis=1)

        U_new = U_gas
        star_mass_painted = apply_particle_sigma_threshold(
            dm_out["star_mass"],
            self.stellar_paint_sigma_threshold,
            self.stellar_paint_min_mass,
        )
        stellar_density_grid = particle_to_grid_cic_paint(star_mass_painted, dm_positions, tuple(int(v) for v in U_gas.shape[1:]))
        use_kernel = self.feedback_deposition_mode == "kernel"
        metal_kernel_width = self.metal_kernel_width_cells if not self.unified_feedback_kernel else self.thermal_kernel_width_cells
        thermal_kernel_width = self.thermal_kernel_width_cells
        momentum_kernel_width = self.momentum_kernel_width_cells if not self.unified_feedback_kernel else self.thermal_kernel_width_cells
        source_kernel = make_feedback_kernel(width_cells=thermal_kernel_width, kind=self.feedback_kernel_kind) if use_kernel else None
        metal_kernel = make_feedback_kernel(width_cells=metal_kernel_width, kind=self.feedback_kernel_kind) if use_kernel else None
        momentum_kernel = make_feedback_kernel(width_cells=momentum_kernel_width, kind=self.feedback_kernel_kind) if use_kernel else None
        gated_delta_star_grid = sf_diag.delta_star_grid
        if use_kernel:
            feedback_source_grid = convolve_scalar_periodic(gated_delta_star_grid, source_kernel)
            metal_deposition_grid = convolve_scalar_periodic(gated_delta_star_grid, metal_kernel)
            wind_source_grid = convolve_scalar_periodic(stellar_density_grid, source_kernel)
            momentum_source_grid = convolve_scalar_periodic(gated_delta_star_grid, momentum_kernel)
        else:
            feedback_source_grid = gated_delta_star_grid
            metal_deposition_grid = gated_delta_star_grid
            wind_source_grid = stellar_density_grid
            momentum_source_grid = gated_delta_star_grid
        dtau_arr = jnp.asarray(dtau, dtype=jnp.float32)
        feedback_snf_energy = positive_source_product(snf_energy, feedback_source_grid)
        feedback_stellar_wind_energy = (
            positive_source_product(stellar_wind_energy, wind_source_grid) * dtau_arr
        )
        # (4) AGN-like thermal injection: lightweight net(stellar field) -> heating,
        # scaled by local gas energy (zero-init net -> no AGN). Routed through the
        # same clip below for stability.
        feedback_clip_scale = jnp.ones_like(feedback_snf_energy)
        if self.feedback_energy_clip_fraction is not None:
            i_energy = int(self.eq.energy_ids)
            energy_ref = jnp.maximum(jnp.asarray(U_new[i_energy], dtype=jnp.float32), getattr(self.eq, "eps", 1.0e-20))
            clip_limit = jnp.maximum(
                jnp.asarray(self.feedback_energy_clip_fraction, dtype=jnp.float32) * energy_ref,
                getattr(self.eq, "eps", 1.0e-20),
            )
            thermal_raw = feedback_snf_energy + feedback_stellar_wind_energy
            feedback_clip_scale = jnp.minimum(
                jnp.ones_like(thermal_raw),
                clip_limit / jnp.maximum(thermal_raw, getattr(self.eq, "eps", 1.0e-20)),
            )
            feedback_snf_energy = feedback_snf_energy * feedback_clip_scale
            feedback_stellar_wind_energy = feedback_stellar_wind_energy * feedback_clip_scale
        thermal_deposition_grid = feedback_snf_energy + feedback_stellar_wind_energy
        # (4) AGN-like thermal injection. CRITICAL: inject AFTER the SF energy clip,
        # not through it. Routing AGN through the min/max clip (clip_limit/thermal_raw)
        # makes the per-cell gradient singular -> NaN (the direct per-cell adjoint is
        # finite; the clip's division is the NaN source). Bound AGN smoothly with tanh
        # to <= clip_fraction * E_gas so it stays stable with a finite gradient.
        agn_w = hyper.get("agn_net", None)
        if agn_w is not None:
            from cosmo_feedback.sf_cnn import apply_agn_net
            _ie = int(self.eq.energy_ids)
            # Bound to a small fraction of cell energy per step (smooth tanh, finite
            # gradient -- not the min/max clip whose division is singular). 1% (was 2%):
            # heating-only AGN accumulates, and 2% destabilized the forward as the net
            # engaged -> step_scale ratcheted to a stall at 0.588 (job 7478).
            _bound = jax.lax.stop_gradient(
                jnp.asarray(0.01, dtype=jnp.float32)
                * jnp.maximum(jnp.asarray(U_new[_ie], dtype=jnp.float32), getattr(self.eq, "eps", 1.0e-20)))
            _agn_raw = jax.checkpoint(apply_agn_net)(agn_w, stellar_density_grid)  # >= 0
            agn_thermal = _bound * jnp.tanh(_agn_raw)   # smooth, in [0, bound)
            # Restrict AGN heating to the DENSEST gas (overdensity > 10 -- halo centers,
            # physical for SMBH accretion). Heating diffuse cells evacuates them ->
            # forward NaN; AGN bypasses the SF feedback clip's stability. stop_gradient
            # mask; gradient is already finite, so masked cells are a clean zero.
            _rho_now = jnp.asarray(U_new[int(self.eq.mass_ids)], dtype=jnp.float32)
            _dense = jax.lax.stop_gradient((_rho_now > 10.0 * jnp.mean(_rho_now)).astype(jnp.float32))
            agn_thermal = agn_thermal * _dense
            thermal_deposition_grid = thermal_deposition_grid + agn_thermal
        # (3) CNN feedback-energy head: per-cell learned thermal injection from the CNN
        # 2nd channel. Same clip-safe construction as AGN -- after the clip (the min/max
        # division is the NaN source), tanh-bounded, dense-gated. Bidirectional (tanh
        # in [-1,1]) so the CNN can boost OR suppress local feedback; zero at zero-init.
        _fb_mod = sf_diag.cnn_feedback_log_mod
        if _fb_mod is not None:
            _ie2 = int(self.eq.energy_ids)
            _fb_bound = jax.lax.stop_gradient(
                jnp.asarray(0.01, dtype=jnp.float32)
                * jnp.maximum(jnp.asarray(U_new[_ie2], dtype=jnp.float32), getattr(self.eq, "eps", 1.0e-20)))
            _rho2 = jnp.asarray(U_new[int(self.eq.mass_ids)], dtype=jnp.float32)
            _dense2 = jax.lax.stop_gradient((_rho2 > jnp.mean(_rho2)).astype(jnp.float32))
            thermal_deposition_grid = thermal_deposition_grid + _fb_bound * jnp.tanh(_fb_mod) * _dense2
        feedback_momentum_x = jnp.zeros_like(thermal_deposition_grid)
        feedback_momentum_y = jnp.zeros_like(thermal_deposition_grid)
        feedback_momentum_z = jnp.zeros_like(thermal_deposition_grid)
        feedback_kinetic_energy = jnp.zeros_like(thermal_deposition_grid)
        if self.enable_momentum_feedback:
            feedback_momentum_x, feedback_momentum_y, feedback_momentum_z = compute_momentum_feedback_fields(
                momentum_source_grid,
                momentum_scale=feedback_momentum_scale,
            )
            rho_cons = jnp.maximum(U_new[int(self.eq.mass_ids)], getattr(self.eq, "eps", 1.0e-20))
            mom_old = jnp.stack(
                [
                    jnp.asarray(U_new[1], dtype=jnp.float32),
                    jnp.asarray(U_new[2], dtype=jnp.float32),
                    jnp.asarray(U_new[3], dtype=jnp.float32),
                ],
                axis=0,
            )
            mom_new = mom_old + jnp.stack([feedback_momentum_x, feedback_momentum_y, feedback_momentum_z], axis=0)
            kin_old = jnp.sum(mom_old * mom_old, axis=0) / jnp.maximum(2.0 * rho_cons, 1.0e-30)
            kin_new = jnp.sum(mom_new * mom_new, axis=0) / jnp.maximum(2.0 * rho_cons, 1.0e-30)
            feedback_kinetic_energy = jnp.maximum(kin_new - kin_old, 0.0)
            U_new = U_new.at[1].add(feedback_momentum_x)
            U_new = U_new.at[2].add(feedback_momentum_y)
            U_new = U_new.at[3].add(feedback_momentum_z)
        feedback_total_energy = thermal_deposition_grid + feedback_kinetic_energy
        # Sanitize the per-cell injected-energy cotangent so the singular hydro
        # adjoint can't poison per-cell trainable weights (AGN net / CNN feedback head).
        feedback_total_energy = finite_cotangent_passthrough(feedback_total_energy)
        i_energy = int(self.eq.energy_ids)
        U_new = U_new.at[i_energy].add(feedback_total_energy)
        i_dual = getattr(self.eq, "dual_energy_ids", None)
        if i_dual is not None and int(i_dual) < int(U_new.shape[0]):
            U_new = U_new.at[int(i_dual)].add(thermal_deposition_grid)
        if self.enable_metal_source and bool(getattr(self.eq, "has_metallicity", lambda: False)()):
            metal_id = int(self.eq.metal_density_ids)
            U_new = U_new.at[metal_id].add(metal_yield * metal_deposition_grid)
            U_new = U_new.at[metal_id].set(jnp.maximum(U_new[metal_id], 0.0))

        params_out = dict(params)
        params_out["dm"] = dm_out
        params_out["rng_sf"] = rng_next if rng_next is not None else rng_key
        if self.store_diagnostics:
            params_out["sf_diagnostics"] = {
                "overdensity": sf_diag.overdensity,
                "nH_cgs": sf_diag.nH_cgs,
                "temperature_k": sf_diag.temperature_k,
                "s_rho": sf_diag.s_rho,
                "s_temp": sf_diag.s_temp,
                "pi0": sf_diag.pi0,
                "z_hat": sf_diag.z_hat,
                "delta_star_grid": sf_diag.delta_star_grid,
                "stellar_density_grid": stellar_density_grid,
                "feedback_source_grid": feedback_source_grid,
                "metal_deposition_grid": metal_deposition_grid,
                "thermal_deposition_grid": thermal_deposition_grid,
                "feedback_momentum_x": feedback_momentum_x,
                "feedback_momentum_y": feedback_momentum_y,
                "feedback_momentum_z": feedback_momentum_z,
                "feedback_kinetic_energy": feedback_kinetic_energy,
                "feedback_snf_energy": feedback_snf_energy,
                "feedback_stellar_wind_energy": feedback_stellar_wind_energy,
                "feedback_total_energy": feedback_total_energy,
                "feedback_clip_scale": feedback_clip_scale,
            }
        return U_new, params_out
