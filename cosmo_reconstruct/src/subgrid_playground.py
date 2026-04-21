from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .forward_model import make_lattice_positions, make_pk_sqrt
from .full_hydro_model import (
    FullHydroConfig,
    FullHydroSystem,
    _init_hydro_state_from_white_noise,
    build_full_hydro_system,
    build_lpt_cosmology,
    extract_subgrid_state,
)
from .metallicity import initialize_metal_density
from .star_formation import StellarFeedbackTracerForce, initialize_star_particle_fields


def _cfg_with_overrides(cfg: FullHydroConfig, **overrides) -> FullHydroConfig:
    clean = {k: v for k, v in overrides.items() if v is not None}
    if not clean:
        return cfg
    return FullHydroConfig(**{**cfg.__dict__, **clean})


def _clone_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _clone_tree(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        out = [_clone_tree(v) for v in tree]
        return type(tree)(out)
    try:
        return jnp.asarray(tree)
    except Exception:
        return tree


def _numpy_snapshot(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in snapshot.items()}


def _expand_conservative_channels(
    U: jnp.ndarray,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
) -> jnp.ndarray:
    U = jnp.asarray(U, dtype=jnp.float32)
    target_n_cons = int(system.eq.n_cons)
    if int(U.shape[0]) == target_n_cons:
        return U
    if int(U.shape[0]) > target_n_cons:
        return U[:target_n_cons]

    padded = jnp.zeros((target_n_cons, *U.shape[1:]), dtype=jnp.float32)
    padded = padded.at[: U.shape[0]].set(U)
    W = system.eq.get_primitives_from_conservatives(U)
    if system.eq.dual_energy_ids is not None and int(system.eq.dual_energy_ids) >= int(U.shape[0]):
        padded = padded.at[int(system.eq.dual_energy_ids)].set(jnp.asarray(W[int(system.eq.dual_energy_ids)], dtype=jnp.float32))
    if system.eq.metal_density_ids is not None and int(system.eq.metal_density_ids) >= int(U.shape[0]):
        rhoZ0 = initialize_metal_density(
            padded[system.eq.mass_ids],
            metal_init=float(cfg.metal_init),
            metal_floor=float(cfg.metal_floor),
            blob_amplitude=float(cfg.metal_blob_amplitude),
            blob_sigma_cells=float(cfg.metal_blob_sigma_cells),
        )
        padded = padded.at[int(system.eq.metal_density_ids)].set(rhoZ0)
    return padded


def _build_force_from_cfg(
    cfg: FullHydroConfig,
    system: FullHydroSystem,
    *,
    store_diagnostics: bool = True,
) -> StellarFeedbackTracerForce:
    return StellarFeedbackTracerForce(
        eq=system.eq,
        code_to_kelvin_temp=float(system.code_to_kelvin_temp),
        mode=str(cfg.star_formation_mode),
        density_threshold_mode=str(cfg.sf_density_threshold_mode),
        density_threshold=float(cfg.sf_density_threshold),
        density_width=float(cfg.sf_density_width),
        rho_unit_cgs=float(cfg.rho_unit_cgs),
        h_species=float(cfg.h_species),
        temperature_threshold_k=float(cfg.sf_temperature_threshold_k),
        temperature_width_k=float(cfg.sf_temperature_width_k),
        sf_pi0=float(cfg.sf_pi0 if cfg.enable_star_formation else 0.0),
        sf_tau=float(cfg.sf_tau),
        sf_mass_scale=float(cfg.sf_mass_scale),
        sf_max_fraction_per_step=float(cfg.sf_max_fraction_per_step),
        star_step_start=int(cfg.star_step_start),
        seed=int(cfg.sf_seed),
        n_age_bins=int(cfg.star_age_bins),
        age_shift_steps=int(cfg.star_age_shift_steps),
        enable_metal_source=bool(cfg.enable_metal_source),
        metal_yield=float(cfg.metal_yield),
        stellar_paint_sigma_threshold=cfg.stellar_paint_sigma_threshold,
        stellar_paint_min_mass=cfg.stellar_paint_min_mass,
        feedback_deposition_mode=str(cfg.feedback_deposition_mode),
        unified_feedback_kernel=bool(cfg.unified_feedback_kernel),
        feedback_kernel_kind=str(cfg.feedback_kernel_kind),
        metal_kernel_width_cells=float(cfg.metal_kernel_width_cells),
        thermal_kernel_width_cells=float(cfg.thermal_kernel_width_cells),
        momentum_kernel_width_cells=float(cfg.momentum_kernel_width_cells),
        enable_momentum_feedback=bool(cfg.enable_momentum_feedback),
        feedback_momentum_scale=float(cfg.feedback_momentum_scale),
        snf_energy=float(cfg.snf_energy),
        stellar_wind_energy=float(cfg.stellar_wind_energy),
        store_diagnostics=bool(store_diagnostics),
    )


def take_field_view(
    field: np.ndarray | jnp.ndarray,
    *,
    axis: int = 0,
    index: int | None = None,
    slab_half_width: int = 0,
    reducer: str = "sum",
) -> np.ndarray:
    arr = np.asarray(field)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D field, got shape {arr.shape}")
    axis = int(axis)
    if axis < 0 or axis > 2:
        raise ValueError("axis must be 0, 1, or 2")
    if index is None:
        index = arr.shape[axis] // 2
    index = int(np.clip(index, 0, arr.shape[axis] - 1))
    half = int(max(0, slab_half_width))
    slicer = [slice(None)] * 3
    slicer[axis] = slice(max(0, index - half), min(arr.shape[axis], index + half + 1))
    sub = arr[tuple(slicer)]
    reducer_norm = str(reducer).lower()
    if reducer_norm == "slice":
        if half != 0:
            raise ValueError("reducer='slice' requires slab_half_width=0")
        sub = np.take(arr, indices=index, axis=axis)
    elif reducer_norm == "sum":
        sub = np.sum(sub, axis=axis)
    elif reducer_norm == "mean":
        sub = np.mean(sub, axis=axis)
    elif reducer_norm == "max":
        sub = np.max(sub, axis=axis)
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")
    return np.asarray(sub)


def plot_field_grid(
    field_map: dict[str, np.ndarray | jnp.ndarray],
    *,
    names: list[str] | None = None,
    axis: int = 0,
    index: int | None = None,
    slab_half_width: int = 0,
    reducer: str | None = None,
    log10_fields: set[str] | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
):
    import math
    import matplotlib.pyplot as plt

    if names is None:
        names = list(field_map.keys())
    if not names:
        raise ValueError("No field names provided to plot.")
    reducer_eff = "slice" if reducer is None and int(slab_half_width) == 0 else (reducer or "sum")
    log10_fields = set() if log10_fields is None else set(log10_fields)

    n = len(names)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    if figsize is None:
        figsize = (4.5 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, name in zip(axes.ravel(), names):
        image = take_field_view(
            field_map[name],
            axis=axis,
            index=index,
            slab_half_width=slab_half_width,
            reducer=reducer_eff,
        )
        if name in log10_fields:
            image = np.log10(np.maximum(image, 1.0e-30))
        im = ax.imshow(image, origin="lower", cmap=cmap)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    return fig, axes


@dataclass
class FeedbackStepResult:
    cfg: FullHydroConfig
    dtau: float
    i_step: int
    before: dict[str, np.ndarray]
    after: dict[str, np.ndarray]
    delta: dict[str, np.ndarray]
    U_before: np.ndarray
    U_after: np.ndarray

    def plot(
        self,
        names: list[str],
        *,
        source: str = "after",
        axis: int = 0,
        index: int | None = None,
        slab_half_width: int = 0,
        reducer: str | None = None,
        log10_fields: set[str] | None = None,
        figsize: tuple[float, float] | None = None,
        cmap: str = "viridis",
    ):
        field_map = {"before": self.before, "after": self.after, "delta": self.delta}[str(source).lower()]
        return plot_field_grid(
            field_map,
            names=names,
            axis=axis,
            index=index,
            slab_half_width=slab_half_width,
            reducer=reducer,
            log10_fields=log10_fields,
            figsize=figsize,
            cmap=cmap,
        )


@dataclass
class FeedbackPlayground:
    cfg: FullHydroConfig
    system: FullHydroSystem
    U_gas: jnp.ndarray
    params: dict[str, Any]
    init_mesh: np.ndarray | None = None
    label: str | None = None

    @classmethod
    def from_white_noise(
        cls,
        *,
        cfg: FullHydroConfig | None = None,
        seed: int = 0,
        **cfg_overrides,
    ) -> "FeedbackPlayground":
        cfg_base = FullHydroConfig(
            mesh_n=32,
            hydro_steps=16,
            enable_star_formation=True,
            enable_metal_source=True,
            track_metallicity=True,
            store_subgrid_diagnostics=True,
        )
        cfg_use = _cfg_with_overrides(cfg or cfg_base, **cfg_overrides)
        cosmo = build_lpt_cosmology(cfg_use)
        system = build_full_hydro_system(cfg_use, cosmo)
        grid_pos = make_lattice_positions(int(cfg_use.mesh_n))
        pk_sqrt = make_pk_sqrt(cosmo, cfg_use)
        white_noise = jr.normal(jr.PRNGKey(int(seed)), (cfg_use.mesh_n, cfg_use.mesh_n, cfg_use.mesh_n), dtype=jnp.float32)
        U_gas, params, init_mesh = _init_hydro_state_from_white_noise(white_noise, pk_sqrt, grid_pos, system, cfg_use)
        return cls(cfg=cfg_use, system=system, U_gas=U_gas, params=params, init_mesh=np.asarray(init_mesh), label="white_noise")

    @classmethod
    def from_init_bundle(
        cls,
        bundle_path: str | Path,
        *,
        cfg: FullHydroConfig | None = None,
        state_u_key: str = "U_gas_init",
        state_dm_x_key: str = "dm_x_init",
        state_dm_p_key: str = "dm_p_or_v_init",
        state_dm_mass_key: str = "dm_mass_init",
        state_a_key: str = "a_init",
        state_init_mesh_key: str = "init_mesh",
        **cfg_overrides,
    ) -> "FeedbackPlayground":
        cfg_base = FullHydroConfig(
            mesh_n=128,
            enable_star_formation=True,
            enable_metal_source=True,
            track_metallicity=True,
            store_subgrid_diagnostics=True,
        )
        cfg_use = _cfg_with_overrides(cfg or cfg_base, **cfg_overrides)
        cosmo = build_lpt_cosmology(cfg_use)
        system = build_full_hydro_system(cfg_use, cosmo)
        d = np.load(Path(bundle_path).resolve())
        U_raw = np.asarray(d[state_u_key], dtype=np.float32)
        U_gas = _expand_conservative_channels(jnp.asarray(U_raw), system, cfg_use)

        dm_x = jnp.asarray(np.asarray(d[state_dm_x_key], dtype=np.float32))
        dm_p = jnp.asarray(np.asarray(d[state_dm_p_key], dtype=np.float32))
        if state_dm_mass_key in d:
            dm_mass = jnp.asarray(np.asarray(d[state_dm_mass_key], dtype=np.float32))
        else:
            dm_mass = jnp.ones((dm_x.shape[0],), dtype=jnp.float32) * jnp.asarray(1.0 - float(cfg_use.gas_mean_fraction), dtype=jnp.float32)
        if dm_mass.ndim == 0:
            dm_mass = jnp.ones((dm_x.shape[0],), dtype=jnp.float32) * dm_mass

        if state_a_key in d:
            a_init = float(np.asarray(d[state_a_key]).reshape(-1)[0])
        else:
            a_init = float(1.0 / (1.0 + float(cfg_use.z_init)))

        omega_m = float(system.cosmo_lpt.Omega_b + system.cosmo_lpt.Omega_c)
        dm_params = {
            "x": dm_x,
            "p_or_v": dm_p,
            "mass": dm_mass,
            "drift_factor": jnp.asarray(system.background.H0, dtype=jnp.float32),
            "kick_prefactor": jnp.asarray(1.5 * omega_m * system.background.H0 * float(cfg_use.dm_kick_scale), dtype=jnp.float32),
        }
        if cfg_use.gas_kick_factor is None:
            dm_params["gas_kick_prefactor"] = jnp.asarray(
                1.5 * omega_m * (system.background.H0**2) * float(cfg_use.gas_kick_scale),
                dtype=jnp.float32,
            )
        else:
            dm_params["gas_kick_factor"] = jnp.asarray(float(cfg_use.gas_kick_factor), dtype=jnp.float32)
        dm_params.update(
            initialize_star_particle_fields(
                dm_x,
                amplitude=float(cfg_use.synthetic_star_mass_amplitude),
                kind=str(cfg_use.synthetic_star_mass_kind),
                seed=int(cfg_use.sf_seed),
                n_age_bins=int(cfg_use.star_age_bins),
            )
        )
        params = {"a": jnp.asarray(a_init, dtype=jnp.float32), "dm": dm_params}
        if bool(cfg_use.enable_star_formation):
            params["rng_sf"] = jr.PRNGKey(int(cfg_use.sf_seed))
        init_mesh = np.asarray(d[state_init_mesh_key], dtype=np.float32) if state_init_mesh_key in d else None
        return cls(cfg=cfg_use, system=system, U_gas=U_gas, params=params, init_mesh=init_mesh, label=str(Path(bundle_path)))

    @classmethod
    def from_state(
        cls,
        U_gas: np.ndarray | jnp.ndarray,
        params: dict[str, Any],
        *,
        cfg: FullHydroConfig,
        system: FullHydroSystem | None = None,
        label: str | None = None,
    ) -> "FeedbackPlayground":
        if system is None:
            cosmo = build_lpt_cosmology(cfg)
            system = build_full_hydro_system(cfg, cosmo)
        return cls(cfg=cfg, system=system, U_gas=_expand_conservative_channels(jnp.asarray(U_gas, dtype=jnp.float32), system, cfg), params=_clone_tree(params), label=label)

    def snapshot(self) -> dict[str, np.ndarray]:
        return _numpy_snapshot(extract_subgrid_state(self.U_gas, self.params, self.system, self.cfg))

    def apply_feedback(
        self,
        *,
        dtau: float,
        i_step: int = 0,
        cfg_overrides: dict[str, Any] | None = None,
    ) -> FeedbackStepResult:
        cfg_step = _cfg_with_overrides(self.cfg, **(cfg_overrides or {}))
        force = _build_force_from_cfg(cfg_step, self.system, store_diagnostics=True)
        U_before = jnp.asarray(self.U_gas, dtype=jnp.float32)
        params_before = _clone_tree(self.params)
        before = _numpy_snapshot(extract_subgrid_state(U_before, params_before, self.system, cfg_step))
        U_after, params_after = force.force(
            jnp.asarray(int(i_step), dtype=jnp.int32),
            U_before,
            params_before,
            jnp.asarray(float(dtau), dtype=jnp.float32),
        )
        after = _numpy_snapshot(extract_subgrid_state(U_after, params_after, self.system, cfg_step))
        delta: dict[str, np.ndarray] = {}
        for key in sorted(set(before) | set(after)):
            if key in before and key in after and before[key].shape == after[key].shape and before[key].dtype.kind in {"f", "i", "u"}:
                delta[key] = np.asarray(after[key] - before[key])
        return FeedbackStepResult(
            cfg=cfg_step,
            dtau=float(dtau),
            i_step=int(i_step),
            before=before,
            after=after,
            delta=delta,
            U_before=np.asarray(U_before),
            U_after=np.asarray(U_after),
        )
