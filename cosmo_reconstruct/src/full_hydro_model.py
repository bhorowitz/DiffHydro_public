from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import diffhydro as dh
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from diffhydro.cosmology import (
    BackgroundExpansionForce,
    JaxPMCoupledGravityForce,
    LCDMBackground,
    SuperComovingEquationManager,
)
from diffhydro.cosmology import conversions as cosmo_conv
from diffhydro.cosmology.grackle_cooling_table import (
    GrackleCoolingRateInterpolator,
    GrackleCoolingTableData,
    build_grackle_cooling_table,
)
from diffhydro.cosmology.nyx_cooling_table import (
    NyxCoolingRateInterpolator,
    NyxCoolingTableData,
    build_nyx_cooling_table,
    parse_float_list,
    rho_crit0_cgs,
)
from jaxpm.painting import cic_paint
from jaxpm.growth import _growth_factor_ODE
from jaxpm.pm import lpt

from .forward_model import a_from_z, prime_growth_cache, white_noise_to_init_mesh
from .metallicity import extract_metal_fields, initialize_metal_density
from .star_formation import (
    StellarFeedbackTracerForce,
    apply_particle_sigma_threshold,
    initialize_star_particle_fields,
    particle_to_grid_cic_paint,
)


@dataclass(frozen=True)
class FullHydroConfig:
    mesh_n: int = 128
    box_size_mpc_h: float = 25.0
    z_init: float = 127.0
    z_target: float = 2.0
    lpt_order: int = 1

    omega_m: float = 0.3
    omega_b: float = 0.045
    h: float = 0.6711
    n_s: float = 0.9624
    sigma8: float = 0.8

    hydro_steps: int = 128
    dtau_min: float = 2.0e-7
    dtau_max: float = 8.0e-2
    # Optional solver/CFL cap on the a-schedule dtau (like cosmo_parallel's launcher):
    # dtau <= solver_dt_safety * sim.timestep(U). 0 disables (pure a-schedule, default).
    # Needed for dense cooling-collapse regions where the a-schedule dt violates CFL.
    solver_dt_safety: float = 0.0

    solver: str = "hll"
    # Hydro time integration: True = MOL/RK2 unsplit (default; all 3 axes reconstructed at once),
    # False = dimensionally-split sweep (one axis at a time -> ~3x less peak memory, needed to fit
    # 512^3 on 4x32GB). Split changes the operator-splitting but is a standard, valid scheme.
    use_mol: bool = True
    state_floor: float = 2.0e-8
    pressure_floor: float = 2.0e-8
    hydro_temp_floor_k: float = 0.0
    force_eps: float = 1.0e-8
    checkpoint: bool = True
    checkpoint_every: int = 1
    pmesh_shape: tuple[int, int, int] = (1, 1, 1)
    gravity_stop_gradient_source: bool = False
    # Multi-GPU distributed gravity (jaxdecomp env). "jaxpm" = single-device (default);
    # "jaxdecomp_fft" = sharded coupled PM gravity (needs field_sharding from make_meshes).
    gravity_backend: str = "jaxpm"
    jaxdecomp_halo_size: int = 32
    gravity_grad_mode: str = "forward_only"  # forward_only | differentiable (diff-DM later)
    # DM mass deposition for the distributed gravity: "relative" (jaxpm cic_paint_dx; needs
    # lattice-ordered DM, e.g. white-noise/jaxpm-decomp ICs) or "absolute_local" (single-device
    # CIC at absolute positions; order/clustering-agnostic, for external-IC bundles like CV0).
    jaxdecomp_dm_paint_mode: str = "relative"
    # Decoupled multi-resolution PM: run the distributed gravity FFT/painting at this resolution
    # (= DM particle resolution) instead of the hydro mesh_n. 0 = coupled (grav == hydro). Lets a
    # high-res hydro mesh (e.g. 512^3) use coarser DM gravity (e.g. 256^3) to fit memory.
    jaxdecomp_grav_mesh_n: int = 0
    projection_stop_gradient_correction: bool = False
    cooling_update_stop_gradient: bool = False
    hydro_flux_stop_gradient: bool = False
    dual_energy: bool = True
    track_metallicity: bool = False
    metal_init: float = 0.0
    metal_floor: float = 0.0
    metal_blob_amplitude: float = 0.0
    metal_blob_sigma_cells: float = 4.0

    temperature0: float = 1.0e4
    temperature_gamma: float = 2.0 / 3.0
    gas_mean_fraction: float = 1.58e-1

    dm_kick_scale: float = 1.0
    gas_kick_scale: float = 1.0
    gas_kick_factor: float | None = None

    enable_cooling: bool = False
    cooling_model: str = "nyx_table"
    #LEGACY COOLING
    cooling_stop_gradient: bool = True
    cooling_table: str = "data/m-00.cie"
    heating_rate_per_h: float = 1.0e-33
    nyx_heating_scale: float = 1.2
    cooling_rate_scale: float = 1.0
    cooling_temp_floor_k: float = 50.0
    cooling_subcycles: int = 8
    cooling_dtmax_s: float = 1.0e16
    #NYX COOLING (with UVB but no metallicity dependence)
    nyx_cooling_table_npz: str = "data/nyx_cooling_table.npz"
    nyx_cooling_treecool: str = "diffhydro/nyx_eos/TREECOOL_middle"
    nyx_cooling_z_nodes: str = "2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100"
    nyx_cooling_logdelta_min: float = -3.0
    nyx_cooling_logdelta_max: float = 3.0
    nyx_cooling_logdelta_n: int = 96
    nyx_cooling_logt_min: float = 2.5
    nyx_cooling_logt_max: float = 8.0
    nyx_cooling_logt_n: int = 120
    nyx_cooling_rebuild: bool = False
    nyx_cooling_eos_path: str = "diffhydro/nyx_eos"
    nyx_auto_rho_unit: bool = False
    # GRACKLE equilibrium cooling (tabulated offline through gracklepy)
    grackle_table_npz: str = "data/grackle_eq_cooling_table.npz"
    grackle_table_rebuild: bool = False
    grackle_data_file: str = "data/grackle_data_files/input/CloudyData_UVB=HM2012.h5"
    grackle_uvb_model: str = "HM2012"
    grackle_use_mu_table: bool = False
    grackle_lognH_min: float = -10.0
    grackle_lognH_max: float = 2.0
    grackle_lognH_n: int = 96
    grackle_logt_min: float = 1.0
    grackle_logt_max: float = 9.0
    grackle_logt_n: int = 120
    grackle_z_nodes: str = "0,1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100"
    grackle_z_sun: float = 0.01295
    grackle_heating_scale: float = 1.0
    grackle_metal_cooling_scale: float = 1.0
    tau_time_unit_s: float = 3.085677581e19
    rho_unit_cgs: float = 1.6e-24
    vel_unit_cms: float = 1.0e7
    mu_hydrogen: float = 1.0
    h_species: float = 0.76
    synthetic_star_mass_kind: str = "none"
    synthetic_star_mass_amplitude: float = 0.0
    enable_star_formation: bool = False
    star_formation_mode: str = "deterministic"
    sf_density_threshold_mode: str = "overdensity"
    sf_density_threshold: float = 10.0
    sf_density_width: float = 1.0
    sf_temperature_threshold_k: float = 2.0e4
    sf_temperature_width_k: float = 5.0e3
    sf_pi0: float = 0.05
    sf_tau: float = 0.3
    sf_mass_scale: float = 0.05
    sf_max_fraction_per_step: float = 0.25
    star_step_start: int = 0
    sf_seed: int = 0
    star_age_bins: int = 0
    star_age_shift_steps: int = 0
    enable_metal_source: bool = False
    metal_yield: float = 0.02
    stellar_paint_sigma_threshold: float | None = None
    stellar_paint_min_mass: float | None = None
    feedback_deposition_mode: str = "local"
    unified_feedback_kernel: bool = True
    feedback_kernel_kind: str = "3x3x3"
    metal_kernel_width_cells: float = 1.0
    thermal_kernel_width_cells: float = 1.0
    momentum_kernel_width_cells: float = 1.0
    enable_momentum_feedback: bool = False
    feedback_momentum_scale: float = 0.0
    snf_energy: float = 0.0
    stellar_wind_energy: float = 0.0
    feedback_energy_clip_fraction: float | None = None
    store_subgrid_diagnostics: bool = False
    enable_vacuum_momentum_cap: bool = False
    vacuum_momentum_rho_guard: float = 1.0e-7
    vacuum_momentum_kinetic_to_thermal_max: float = 1.0
    vacuum_momentum_internal_floor: float = 1.0e-8
    enable_hydro_state_repair: bool = False
    hydro_repair_rho_floor: float = 1.0e-12
    hydro_repair_pressure_floor: float = 1.0e-12
    hydro_repair_max_kinetic_to_thermal_ratio: float = 1.0e6
    fail_on_nonfinite: bool = True


@dataclass(frozen=True)
class FullHydroSystem:
    cosmo_lpt: jc.Cosmology
    background: LCDMBackground
    eq: SuperComovingEquationManager
    sim: dh.hydro
    hydro_vel_unit_cms: float
    kelvin_to_code_temp: float
    code_to_kelvin_temp: float


def build_lpt_cosmology(cfg: FullHydroConfig) -> jc.Cosmology:
    return jc.Planck15(
        Omega_c=cfg.omega_m - cfg.omega_b,
        Omega_b=cfg.omega_b,
        h=cfg.h,
        n_s=cfg.n_s,
        sigma8=cfg.sigma8,
    )


def _paint_dm_density(
    positions: jnp.ndarray,
    mesh_shape: tuple[int, int, int],
    weight: jnp.ndarray | None = None,
) -> jnp.ndarray:
    mesh = jnp.zeros(mesh_shape, dtype=jnp.float32)
    if weight is None:
        return cic_paint(mesh, positions)
    return cic_paint(mesh, positions, weight=weight)


def _paint_particle_scalar(
    positions: jnp.ndarray,
    scalar: jnp.ndarray,
    mesh_shape: tuple[int, int, int],
    sigma_threshold: float | None = None,
    min_mass: float | None = None,
) -> jnp.ndarray:
    scalar_filtered = apply_particle_sigma_threshold(scalar, sigma_threshold, min_mass)
    return particle_to_grid_cic_paint(scalar_filtered, positions, mesh_shape)


def _paint_dm_velocity_tilde(
    positions: jnp.ndarray,
    velocity_tilde: jnp.ndarray,
    mesh_shape: tuple[int, int, int],
    eps: float = 1.0e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rho_dm = _paint_dm_density(positions, mesh_shape)
    inv_rho = 1.0 / jnp.maximum(rho_dm, eps)

    vx = cic_paint(jnp.zeros(mesh_shape, dtype=jnp.float32), positions, weight=velocity_tilde[:, 0]) * inv_rho
    vy = cic_paint(jnp.zeros(mesh_shape, dtype=jnp.float32), positions, weight=velocity_tilde[:, 1]) * inv_rho
    vz = cic_paint(jnp.zeros(mesh_shape, dtype=jnp.float32), positions, weight=velocity_tilde[:, 2]) * inv_rho
    return vx, vy, vz


def _build_hydrosim(
    eq,
    forces,
    *,
    solver_name: str,
    use_mol: bool = True,
    dx_o: float = 1.0,
    enable_vacuum_momentum_cap: bool = False,
    vacuum_momentum_rho_guard: float = 1.0e-7,
    vacuum_momentum_kinetic_to_thermal_max: float = 1.0,
    vacuum_momentum_internal_floor: float = 1.0e-8,
    enable_hydro_state_repair: bool = False,
    hydro_repair_rho_floor: float = 1.0e-7,
    hydro_repair_pressure_floor: float = 1.0e-8,
    hydro_repair_max_kinetic_to_thermal_ratio: float = 1.0e6,
    hydro_flux_stop_gradient: bool = False,
    pmesh_shape: tuple[int, int, int] = (1, 1, 1),
):
    ss = dh.signal_speed_Einfeldt
    sname = solver_name.lower()
    if sname == "hllc":
        riemann_solver = dh.HLLC(equation_manager=eq, signal_speed=ss)
    elif sname in ("laxfriedrichs", "lf"):
        riemann_solver = dh.LaxFriedrichs(equation_manager=eq, signal_speed=ss)
    elif sname == "hll":
        riemann_solver = dh.HLL(equation_manager=eq, signal_speed=ss)
    elif sname in ("nyx_riemann", "nyx"):
        riemann_solver = dh.NyxRiemannEuler(
            equation_manager=eq,
            signal_speed=ss,
            eos_consistent_interface_energy=True,
        )
    else:
        raise ValueError(f"Unknown solver '{solver_name}'")

    conv_flux = dh.ConvectiveFlux(
        eq,
        riemann_solver,
        dh.PPM_CW(limiter="MINMOD", steepen=False),
        positivity=True,
    )
    conv_flux.dx_o = float(dx_o)

    sim = dh.hydro(
        n_super_step=1,
        max_dt=0.2,
        fluxes=[conv_flux],
        forces=list(forces),
        use_mol=bool(use_mol),
        pmesh_shape=tuple(int(v) for v in pmesh_shape),
    )
    sim.dx_o = float(dx_o)
    sim.enable_vacuum_momentum_cap = bool(enable_vacuum_momentum_cap)
    sim.vacuum_momentum_rho_guard = float(vacuum_momentum_rho_guard)
    sim.vacuum_momentum_kinetic_to_thermal_max = float(vacuum_momentum_kinetic_to_thermal_max)
    sim.vacuum_momentum_internal_floor = float(vacuum_momentum_internal_floor)
    sim.enable_hydro_state_repair = bool(enable_hydro_state_repair)
    sim.hydro_repair_rho_floor = float(hydro_repair_rho_floor)
    sim.hydro_repair_pressure_floor = float(hydro_repair_pressure_floor)
    sim.hydro_repair_max_kinetic_to_thermal_ratio = float(hydro_repair_max_kinetic_to_thermal_ratio)
    sim.stop_hydro_flux_grad = bool(hydro_flux_stop_gradient)
    for flux in sim.fluxes:
        if hasattr(flux, "dx_o"):
            flux.dx_o = float(dx_o)
    return sim


class CosmologicalHydrogenCoolingForce:
    """Cooling source term using Kelvin/cgs internally, mapped from supercomoving state."""

    def __init__(
        self,
        equation_manager,
        logT_table,
        logLambda_m20_table,
        *,
        rho_unit_cgs: float,
        vel_unit_cms: float,
        tau_time_unit_s: float,
        mu: float = 1.0,
        h_species: float = 0.76,
        heating_rate_per_h: float = 0.0,
        cooling_rate_scale: float = 1.0,
        temp_floor_k: float = 1.0e-2,
        subcycles: int = 8,
        dtmax_s: float = 1.0e16,
        stop_rate_grad: bool = False,
        stop_update_grad: bool = False,
        eps: float = 1.0e-30,
    ):
        self.eq = equation_manager
        self.logT_table = jnp.asarray(logT_table, dtype=jnp.float32)
        self.logLambda_m20_table = jnp.asarray(logLambda_m20_table, dtype=jnp.float32)
        self.rho_unit_cgs = float(rho_unit_cgs)
        self.vel_unit_cms = float(vel_unit_cms)
        self.p_unit_cgs = float(rho_unit_cgs) * float(vel_unit_cms) * float(vel_unit_cms)
        self.tau_time_unit_s = float(tau_time_unit_s)
        self.mu = float(mu)
        self.h_species = float(np.clip(h_species, 0.0, 1.0))
        self.heating_rate_per_h = float(heating_rate_per_h)
        self.cooling_rate_scale = float(max(cooling_rate_scale, 0.0))
        self.temp_floor_k = float(temp_floor_k)
        self.subcycles = int(max(1, subcycles))
        self.dtmax_s = float(dtmax_s)
        self.stop_rate_grad = bool(stop_rate_grad)
        self.stop_update_grad = bool(stop_update_grad)
        self.eps = float(eps)
        self.gamma = float(getattr(self.eq, "gamma", 5.0 / 3.0))
        self.i_rho = self.eq.mass_ids
        self.i_p = self.eq.energy_ids
        self.i_dual = getattr(self.eq, "dual_energy_ids", None)
        self.mH_cgs = 1.6735575e-24
        self.kB_cgs = 1.380649e-16

    def timestep(self, U):
        del U
        return jnp.asarray(1.0e10, dtype=jnp.float32)

    def _interp_log_lambda(self, logT):
        logT_min = self.logT_table[0]
        logT_max = self.logT_table[-1]
        logT_clip = jnp.clip(logT, logT_min, logT_max)
        logLm20_clip = jnp.interp(logT_clip, self.logT_table, self.logLambda_m20_table)
        dlogT = jnp.maximum(self.logT_table[1] - self.logT_table[0], 1.0e-6)
        slope_lo = (self.logLambda_m20_table[1] - self.logLambda_m20_table[0]) / dlogT
        logLm20_lo = self.logLambda_m20_table[0] + slope_lo * (logT - logT_min)
        logLm20 = jnp.where(logT < logT_min, logLm20_lo, logLm20_clip)
        return logLm20 - 20.0

    def _code_thermo_to_cgs(self, rho_code, p_code, a):
        rho_phys = jnp.maximum(cosmo_conv.density_code_to_phys(rho_code, a), self.eps)
        p_phys = jnp.maximum(cosmo_conv.pressure_code_to_phys(p_code, a), self.eps)
        rho_cgs = jnp.maximum(rho_phys * self.rho_unit_cgs, self.eps)
        p_cgs = jnp.maximum(p_phys * self.p_unit_cgs, self.eps)
        return rho_phys, p_phys, rho_cgs, p_cgs

    def temperature_kelvin_from_code(self, rho_code, p_code, a):
        _, _, rho_cgs, p_cgs = self._code_thermo_to_cgs(rho_code, p_code, a)
        t_kelvin = (self.mu * self.mH_cgs / self.kB_cgs) * p_cgs / jnp.maximum(rho_cgs, self.eps)
        return jnp.maximum(t_kelvin, self.temp_floor_k)

    def cooling_rate_cgs_from_code(self, rho_code, p_code, a):
        t_kelvin = self.temperature_kelvin_from_code(rho_code, p_code, a)
        logT = jnp.log10(jnp.maximum(t_kelvin, self.temp_floor_k))
        logLambda = self._interp_log_lambda(logT)
        Lambda = 10.0**logLambda
        nH = self.h_species * jnp.maximum(cosmo_conv.density_code_to_phys(rho_code, a), self.eps) * self.rho_unit_cgs / self.mH_cgs
        if self.stop_rate_grad:
            Lambda = jax.lax.stop_gradient(Lambda)
            nH = jax.lax.stop_gradient(nH)
        dotE = -self.cooling_rate_scale * nH * nH * Lambda + nH * self.heating_rate_per_h
        return jnp.nan_to_num(dotE, nan=0.0, posinf=0.0, neginf=0.0)

    def force(self, i_step, U, params, dtau):
        del i_step
        a = jnp.asarray(params.get("a", 1.0), dtype=jnp.float32)
        dtau = jnp.maximum(jnp.asarray(dtau, dtype=jnp.float32), 0.0)
        dt_s = jnp.minimum((a * a) * dtau * self.tau_time_unit_s, self.dtmax_s)
        dt_sub = dt_s / float(self.subcycles)

        W = self.eq.get_primitives_from_conservatives(U)
        rho_code = jnp.maximum(W[self.i_rho], self.eps)
        p_code = jnp.maximum(W[self.i_p], self.eps)

        _, _, rho_cgs, p_cgs = self._code_thermo_to_cgs(rho_code, p_code, a)
        Et = p_cgs / (self.gamma - 1.0)
        Et_floor = rho_cgs * self.kB_cgs * self.temp_floor_k / (
            self.mu * self.mH_cgs * (self.gamma - 1.0)
        )

        def body(_, Et_cur):
            p_cgs_cur = (self.gamma - 1.0) * jnp.maximum(Et_cur, self.eps)
            T_cur = (self.mu * self.mH_cgs / self.kB_cgs) * p_cgs_cur / jnp.maximum(rho_cgs, self.eps)
            logT = jnp.log10(jnp.maximum(T_cur, self.temp_floor_k))
            logLambda = self._interp_log_lambda(logT)
            Lambda = 10.0**logLambda
            nH = self.h_species * rho_cgs / self.mH_cgs
            if self.stop_rate_grad:
                Lambda = jax.lax.stop_gradient(Lambda)
                nH = jax.lax.stop_gradient(nH)
            dotE = -self.cooling_rate_scale * nH * nH * Lambda + nH * self.heating_rate_per_h
            dotE = jnp.nan_to_num(dotE, nan=0.0, posinf=0.0, neginf=0.0)
            Et_new = jnp.maximum(Et_cur + dotE * dt_sub, Et_floor)
            return Et_new

        Et_final = jax.lax.fori_loop(0, self.subcycles, body, Et)
        p_cgs_new = (self.gamma - 1.0) * jnp.maximum(Et_final, self.eps)
        p_phys_new = p_cgs_new / self.p_unit_cgs
        p_code_new = cosmo_conv.pressure_phys_to_code(p_phys_new, a)

        p_code_new = jnp.maximum(p_code_new, self.eq.eps)
        W_new = W.at[self.i_p].set(p_code_new)
        if self.i_dual is not None and int(self.i_dual) < int(W.shape[0]):
            rhoe_code_new = jnp.maximum(p_code_new / (self.gamma - 1.0), self.eq.eps)
            W_new = W_new.at[int(self.i_dual)].set(rhoe_code_new)
        U_new = self.eq.get_conservatives_from_primitives(W_new)
        if self.stop_update_grad:
            U_new = U + jax.lax.stop_gradient(U_new - U)
        return U_new, params


class NyxTabulatedCoolingForce:
    """Nyx-style tabulated cooling/heating source terms in cgs space."""

    def __init__(
        self,
        equation_manager,
        nyx_interp: NyxCoolingRateInterpolator,
        *,
        rho_unit_cgs: float,
        vel_unit_cms: float,
        tau_time_unit_s: float,
        mu: float = 1.0,
        cooling_rate_scale: float = 1.0,
        heating_rate_scale: float = 1.0,
        temp_floor_k: float = 1.0e-2,
        subcycles: int = 8,
        dtmax_s: float = 1.0e16,
        stop_rate_grad: bool = False,
        stop_update_grad: bool = False,
        eps: float = 1.0e-40,
    ):
        self.eq = equation_manager
        self.nyx = nyx_interp
        self.rho_unit_cgs = float(rho_unit_cgs)
        self.vel_unit_cms = float(vel_unit_cms)
        self.p_unit_cgs = float(rho_unit_cgs) * float(vel_unit_cms) * float(vel_unit_cms)
        self.tau_time_unit_s = float(tau_time_unit_s)
        self.mu = float(mu)
        self.cooling_rate_scale = float(max(cooling_rate_scale, 0.0))
        self.heating_rate_scale = float(max(heating_rate_scale, 0.0))
        self.temp_floor_k = float(temp_floor_k)
        self.subcycles = int(max(1, subcycles))
        self.dtmax_s = float(dtmax_s)
        self.stop_rate_grad = bool(stop_rate_grad)
        self.stop_update_grad = bool(stop_update_grad)
        self.eps = float(eps)
        self.gamma = float(getattr(self.eq, "gamma", 5.0 / 3.0))
        self.i_rho = self.eq.mass_ids
        self.i_p = self.eq.energy_ids
        self.i_dual = getattr(self.eq, "dual_energy_ids", None)
        self.mH_cgs = 1.6735575e-24
        self.kB_cgs = 1.380649e-16

    def timestep(self, U):
        del U
        return jnp.asarray(1.0e10, dtype=jnp.float32)

    def _code_thermo_to_cgs(self, rho_code, p_code, a):
        rho_phys = jnp.maximum(cosmo_conv.density_code_to_phys(rho_code, a), self.eps)
        p_phys = jnp.maximum(cosmo_conv.pressure_code_to_phys(p_code, a), self.eps)
        rho_cgs = jnp.maximum(rho_phys * self.rho_unit_cgs, self.eps)
        p_cgs = jnp.maximum(p_phys * self.p_unit_cgs, self.eps)
        return rho_phys, p_phys, rho_cgs, p_cgs

    def temperature_kelvin_from_code(self, rho_code, p_code, a):
        _, _, rho_cgs, p_cgs = self._code_thermo_to_cgs(rho_code, p_code, a)
        t_kelvin = (self.mu * self.mH_cgs / self.kB_cgs) * p_cgs / jnp.maximum(rho_cgs, self.eps)
        return jnp.maximum(t_kelvin, self.temp_floor_k)

    def cooling_rate_cgs_from_code(self, rho_code, p_code, a):
        _, _, rho_cgs, _ = self._code_thermo_to_cgs(rho_code, p_code, a)
        t_kelvin = self.temperature_kelvin_from_code(rho_code, p_code, a)
        z = (1.0 / jnp.maximum(jnp.asarray(a, dtype=jnp.float32), 1.0e-12)) - 1.0
        heat_cgs, cool_cgs, _ = self.nyx.evaluate(rho_cgs, t_kelvin, z)
        heat_cgs = jnp.nan_to_num(heat_cgs, nan=0.0, posinf=0.0, neginf=0.0)
        cool_cgs = jnp.nan_to_num(cool_cgs, nan=0.0, posinf=0.0, neginf=0.0)
        if self.stop_rate_grad:
            heat_cgs = jax.lax.stop_gradient(heat_cgs)
            cool_cgs = jax.lax.stop_gradient(cool_cgs)
        dotE = self.heating_rate_scale * heat_cgs - self.cooling_rate_scale * cool_cgs
        return jnp.nan_to_num(dotE, nan=0.0, posinf=0.0, neginf=0.0)

    def force(self, i_step, U, params, dtau):
        del i_step
        a = jnp.asarray(params.get("a", 1.0), dtype=jnp.float32)
        z = (1.0 / jnp.maximum(a, 1.0e-12)) - 1.0
        dtau = jnp.maximum(jnp.asarray(dtau, dtype=jnp.float32), 0.0)
        dt_s = jnp.minimum((a * a) * dtau * self.tau_time_unit_s, self.dtmax_s)
        dt_sub = dt_s / float(self.subcycles)

        W = self.eq.get_primitives_from_conservatives(U)
        rho_code = jnp.maximum(W[self.i_rho], self.eps)
        p_code = jnp.maximum(W[self.i_p], self.eps)

        _, _, rho_cgs, p_cgs = self._code_thermo_to_cgs(rho_code, p_code, a)
        Et = p_cgs / (self.gamma - 1.0)
        Et_floor = rho_cgs * self.kB_cgs * self.temp_floor_k / (
            self.mu * self.mH_cgs * (self.gamma - 1.0)
        )

        def body(_, Et_cur):
            p_cgs_cur = (self.gamma - 1.0) * jnp.maximum(Et_cur, self.eps)
            T_cur = (self.mu * self.mH_cgs / self.kB_cgs) * p_cgs_cur / jnp.maximum(rho_cgs, self.eps)
            heat_cgs, cool_cgs, _ = self.nyx.evaluate(rho_cgs, T_cur, z)
            heat_cgs = jnp.nan_to_num(heat_cgs, nan=0.0, posinf=0.0, neginf=0.0)
            cool_cgs = jnp.nan_to_num(cool_cgs, nan=0.0, posinf=0.0, neginf=0.0)
            if self.stop_rate_grad:
                heat_cgs = jax.lax.stop_gradient(heat_cgs)
                cool_cgs = jax.lax.stop_gradient(cool_cgs)
            dotE = self.heating_rate_scale * heat_cgs - self.cooling_rate_scale * cool_cgs
            dotE = jnp.nan_to_num(dotE, nan=0.0, posinf=0.0, neginf=0.0)
            Et_new = jnp.maximum(Et_cur + dotE * dt_sub, Et_floor)
            return Et_new

        Et_final = jax.lax.fori_loop(0, self.subcycles, body, Et)
        p_cgs_new = (self.gamma - 1.0) * jnp.maximum(Et_final, self.eps)
        p_phys_new = p_cgs_new / self.p_unit_cgs
        p_code_new = cosmo_conv.pressure_phys_to_code(p_phys_new, a)

        p_code_new = jnp.maximum(p_code_new, self.eq.eps)
        W_new = W.at[self.i_p].set(p_code_new)
        if self.i_dual is not None and int(self.i_dual) < int(W.shape[0]):
            rhoe_code_new = jnp.maximum(p_code_new / (self.gamma - 1.0), self.eq.eps)
            W_new = W_new.at[int(self.i_dual)].set(rhoe_code_new)
        U_new = self.eq.get_conservatives_from_primitives(W_new)
        valid_mask = _raw_hydro_cell_valid_mask(U, self.i_rho)
        U_new = jnp.where(valid_mask[jnp.newaxis, ...], U_new, U)
        if self.stop_update_grad:
            U_new = U + jax.lax.stop_gradient(U_new - U)
        return U_new, params


class GrackleTabulatedCoolingForce:
    """Grackle equilibrium cooling/heating source terms in cgs space."""

    def __init__(
        self,
        equation_manager,
        grackle_interp: GrackleCoolingRateInterpolator,
        *,
        rho_unit_cgs: float,
        vel_unit_cms: float,
        tau_time_unit_s: float,
        mu: float = 1.0,
        use_mu_table: bool = False,
        cooling_rate_scale: float = 1.0,
        heating_rate_scale: float = 1.0,
        metal_cooling_scale: float = 1.0,
        temp_floor_k: float = 1.0e-2,
        subcycles: int = 8,
        dtmax_s: float = 1.0e16,
        stop_rate_grad: bool = False,
        stop_update_grad: bool = False,
        eps: float = 1.0e-40,
    ):
        self.eq = equation_manager
        self.grackle = grackle_interp
        self.rho_unit_cgs = float(rho_unit_cgs)
        self.vel_unit_cms = float(vel_unit_cms)
        self.p_unit_cgs = float(rho_unit_cgs) * float(vel_unit_cms) * float(vel_unit_cms)
        self.tau_time_unit_s = float(tau_time_unit_s)
        self.mu = float(mu)
        self.use_mu_table = bool(use_mu_table)
        self.cooling_rate_scale = float(max(cooling_rate_scale, 0.0))
        self.heating_rate_scale = float(max(heating_rate_scale, 0.0))
        self.metal_cooling_scale = float(max(metal_cooling_scale, 0.0))
        self.temp_floor_k = float(temp_floor_k)
        self.subcycles = int(max(1, subcycles))
        self.dtmax_s = float(dtmax_s)
        self.stop_rate_grad = bool(stop_rate_grad)
        self.stop_update_grad = bool(stop_update_grad)
        self.eps = float(eps)
        self.gamma = float(getattr(self.eq, "gamma", 5.0 / 3.0))
        self.i_rho = self.eq.mass_ids
        self.i_p = self.eq.energy_ids
        self.i_dual = getattr(self.eq, "dual_energy_ids", None)
        self.mH_cgs = 1.6735575e-24
        self.kB_cgs = 1.380649e-16

    def timestep(self, U):
        del U
        return jnp.asarray(1.0e10, dtype=jnp.float32)

    def _code_thermo_to_cgs(self, rho_code, p_code, a):
        rho_phys = jnp.maximum(cosmo_conv.density_code_to_phys(rho_code, a), self.eps)
        p_phys = jnp.maximum(cosmo_conv.pressure_code_to_phys(p_code, a), self.eps)
        rho_cgs = jnp.maximum(rho_phys * self.rho_unit_cgs, self.eps)
        p_cgs = jnp.maximum(p_phys * self.p_unit_cgs, self.eps)
        return rho_phys, p_phys, rho_cgs, p_cgs

    def _temperature_from_mu(self, rho_cgs, p_cgs, mu_eff):
        t_kelvin = (jnp.maximum(mu_eff, 1.0e-8) * self.mH_cgs / self.kB_cgs) * p_cgs / jnp.maximum(rho_cgs, self.eps)
        return jnp.maximum(t_kelvin, self.temp_floor_k)

    def _temperature_and_mu(self, rho_code, p_code, a):
        _, _, rho_cgs, p_cgs = self._code_thermo_to_cgs(rho_code, p_code, a)
        temp_guess = self._temperature_from_mu(rho_cgs, p_cgs, jnp.asarray(self.mu, dtype=jnp.float32))
        if not self.use_mu_table:
            return temp_guess, jnp.ones_like(temp_guess, dtype=jnp.float32) * jnp.asarray(self.mu, dtype=jnp.float32)
        z = (1.0 / jnp.maximum(jnp.asarray(a, dtype=jnp.float32), 1.0e-12)) - 1.0
        nH_cgs = self.grackle.h_species * rho_cgs / self.mH_cgs
        _, _, _, mu_tab = self.grackle.evaluate(nH_cgs, temp_guess, z)
        mu_tab = jnp.maximum(mu_tab, 1.0e-6)
        temp_eff = self._temperature_from_mu(rho_cgs, p_cgs, mu_tab)
        return temp_eff, mu_tab

    def cooling_rate_cgs_from_code(self, rho_code, p_code, a, U=None):
        _, _, rho_cgs, _ = self._code_thermo_to_cgs(rho_code, p_code, a)
        temp_k, _ = self._temperature_and_mu(rho_code, p_code, a)
        z = (1.0 / jnp.maximum(jnp.asarray(a, dtype=jnp.float32), 1.0e-12)) - 1.0
        nH_cgs = self.grackle.h_species * rho_cgs / self.mH_cgs
        heat_prim, cool_prim, cool_metal_solar, _ = self.grackle.evaluate(nH_cgs, temp_k, z)
        if self.stop_rate_grad:
            heat_prim = jax.lax.stop_gradient(heat_prim)
            cool_prim = jax.lax.stop_gradient(cool_prim)
            cool_metal_solar = jax.lax.stop_gradient(cool_metal_solar)
        if U is not None:
            _, metal_fraction = extract_metal_fields(U, self.eq)
            if metal_fraction is None:
                z_ratio = jnp.zeros_like(heat_prim)
            else:
                z_ratio = jnp.maximum(metal_fraction / max(self.grackle.z_sun, 1.0e-30), 0.0)
        else:
            z_ratio = jnp.zeros_like(heat_prim)
        dotE = (
            self.heating_rate_scale * heat_prim
            - self.cooling_rate_scale * cool_prim
            - self.cooling_rate_scale * self.metal_cooling_scale * z_ratio * cool_metal_solar
        )
        return jnp.nan_to_num(dotE, nan=0.0, posinf=0.0, neginf=0.0)

    def force(self, i_step, U, params, dtau):
        del i_step
        a = jnp.asarray(params.get("a", 1.0), dtype=jnp.float32)
        z = (1.0 / jnp.maximum(a, 1.0e-12)) - 1.0
        dtau = jnp.maximum(jnp.asarray(dtau, dtype=jnp.float32), 0.0)
        dt_s = jnp.minimum((a * a) * dtau * self.tau_time_unit_s, self.dtmax_s)
        dt_sub = dt_s / float(self.subcycles)

        W = self.eq.get_primitives_from_conservatives(U)
        rho_code = jnp.maximum(W[self.i_rho], self.eps)
        p_code = jnp.maximum(W[self.i_p], self.eps)

        _, _, rho_cgs, p_cgs = self._code_thermo_to_cgs(rho_code, p_code, a)
        temp_k, mu_eff = self._temperature_and_mu(rho_code, p_code, a)
        nH_cgs = self.grackle.h_species * rho_cgs / self.mH_cgs
        _, metal_fraction = extract_metal_fields(U, self.eq)
        if metal_fraction is None:
            z_ratio = jnp.zeros_like(rho_code)
        else:
            z_ratio = jnp.maximum(metal_fraction / max(self.grackle.z_sun, 1.0e-30), 0.0)
        if self.stop_rate_grad:
            z_ratio = jax.lax.stop_gradient(z_ratio)

        Et = p_cgs / (self.gamma - 1.0)
        Et_floor = rho_cgs * self.kB_cgs * self.temp_floor_k / (
            jnp.maximum(mu_eff, 1.0e-8) * self.mH_cgs * (self.gamma - 1.0)
        )

        # Trainable cooling/heating: multipliers from feedback_hyper (default 1.0 =
        # identity, so a fresh run reproduces the fixed-rate model exactly).
        _hyper = params.get("feedback_hyper", {})
        cool_scale = self.cooling_rate_scale * jnp.asarray(
            _hyper.get("cooling_rate_mult", 1.0), dtype=jnp.float32)
        heat_scale = self.heating_rate_scale * jnp.asarray(
            _hyper.get("heating_rate_mult", 1.0), dtype=jnp.float32)

        def body(_, Et_cur):
            p_cgs_cur = (self.gamma - 1.0) * jnp.maximum(Et_cur, self.eps)
            temp_guess = self._temperature_from_mu(rho_cgs, p_cgs_cur, mu_eff)
            heat_prim, cool_prim, cool_metal_solar, mu_tab = self.grackle.evaluate(nH_cgs, temp_guess, z)
            if self.use_mu_table:
                mu_eval = jnp.maximum(mu_tab, 1.0e-6)
                temp_eval = self._temperature_from_mu(rho_cgs, p_cgs_cur, mu_eval)
                heat_prim, cool_prim, cool_metal_solar, _ = self.grackle.evaluate(nH_cgs, temp_eval, z)
            if self.stop_rate_grad:
                heat_prim = jax.lax.stop_gradient(heat_prim)
                cool_prim = jax.lax.stop_gradient(cool_prim)
                cool_metal_solar = jax.lax.stop_gradient(cool_metal_solar)
            dotE = (
                heat_scale * heat_prim
                - cool_scale * cool_prim
                - cool_scale * self.metal_cooling_scale * z_ratio * cool_metal_solar
            )
            dotE = jnp.nan_to_num(dotE, nan=0.0, posinf=0.0, neginf=0.0)
            Et_new = jnp.maximum(Et_cur + dotE * dt_sub, Et_floor)
            return Et_new

        Et_final = jax.lax.fori_loop(0, self.subcycles, body, Et)
        p_cgs_new = (self.gamma - 1.0) * jnp.maximum(Et_final, self.eps)
        p_phys_new = p_cgs_new / self.p_unit_cgs
        p_code_new = cosmo_conv.pressure_phys_to_code(p_phys_new, a)

        p_code_new = jnp.maximum(p_code_new, self.eq.eps)
        W_new = W.at[self.i_p].set(p_code_new)
        if self.i_dual is not None and int(self.i_dual) < int(W.shape[0]):
            rhoe_code_new = jnp.maximum(p_code_new / (self.gamma - 1.0), self.eq.eps)
            W_new = W_new.at[int(self.i_dual)].set(rhoe_code_new)
        U_new = self.eq.get_conservatives_from_primitives(W_new)
   #     valid_mask = _raw_hydro_cell_valid_mask(U, self.i_rho)
   #     U_new = jnp.where(valid_mask[jnp.newaxis, ...], U_new, U)
        if self.stop_update_grad:
            U_new = U + jax.lax.stop_gradient(U_new - U)
        return U_new, params


def _build_cooling_force(eq, bg: LCDMBackground, cfg: FullHydroConfig):
    if not bool(cfg.enable_cooling):
        return None

    cooling_model = str(cfg.cooling_model).lower()
    hydro_velocity_scale = float(bg.H0) * float(cfg.mesh_n) / max(float(cfg.box_size_mpc_h), 1.0e-30)
    hydro_vel_unit_cms = float(cfg.vel_unit_cms) / max(hydro_velocity_scale, 1.0e-30)

    rho_unit_cgs_used = float(cfg.rho_unit_cgs)
    if cooling_model == "nyx_table" and bool(cfg.nyx_auto_rho_unit):
        rho_b0_cgs = rho_crit0_cgs(float(cfg.h)) * float(cfg.omega_b)
        rho_unit_cgs_used = rho_b0_cgs / max(float(cfg.gas_mean_fraction), 1.0e-30)

    if cooling_model == "legacy":
        cooling_table_path = Path(cfg.cooling_table).resolve()
        if not cooling_table_path.exists():
            raise FileNotFoundError(f"Cooling table not found: {cooling_table_path}")
        cooling_table = np.genfromtxt(cooling_table_path)
        if cooling_table.ndim != 2 or cooling_table.shape[1] < 2:
            raise ValueError(
                f"Cooling table {cooling_table_path} must have at least 2 columns: logT and logLambda_m20."
            )
        return CosmologicalHydrogenCoolingForce(
            eq,
            cooling_table[:, 0],
            cooling_table[:, 1],
            rho_unit_cgs=float(rho_unit_cgs_used),
            vel_unit_cms=float(hydro_vel_unit_cms),
            tau_time_unit_s=float(cfg.tau_time_unit_s),
            mu=float(cfg.mu_hydrogen),
            h_species=float(cfg.h_species),
            heating_rate_per_h=float(cfg.heating_rate_per_h),
            cooling_rate_scale=float(cfg.cooling_rate_scale),
            temp_floor_k=float(cfg.cooling_temp_floor_k),
            subcycles=int(cfg.cooling_subcycles),
            dtmax_s=float(cfg.cooling_dtmax_s),
            stop_rate_grad=bool(cfg.cooling_stop_gradient),
            stop_update_grad=bool(cfg.cooling_update_stop_gradient),
        )

    if cooling_model == "nyx_table":
        z_nodes = parse_float_list(cfg.nyx_cooling_z_nodes)
        ld_nodes = np.linspace(
            float(cfg.nyx_cooling_logdelta_min),
            float(cfg.nyx_cooling_logdelta_max),
            int(max(2, cfg.nyx_cooling_logdelta_n)),
            dtype=np.float64,
        )
        lt_nodes = np.linspace(
            float(cfg.nyx_cooling_logt_min),
            float(cfg.nyx_cooling_logt_max),
            int(max(2, cfg.nyx_cooling_logt_n)),
            dtype=np.float64,
        )
        nyx_table_path = Path(cfg.nyx_cooling_table_npz).resolve()
        if bool(cfg.nyx_cooling_rebuild) or (not nyx_table_path.exists()):
            build_nyx_cooling_table(
                nyx_table_path,
                treecool_file=Path(cfg.nyx_cooling_treecool).resolve(),
                z_nodes=z_nodes,
                logdelta_nodes=ld_nodes,
                logT_nodes=lt_nodes,
                h=float(cfg.h),
                omega_b=float(cfg.omega_b),
                h_species=float(cfg.h_species),
                uvb_density_a=1.0,
                uvb_density_b=0.0,
                jh=1,
                jhe=1,
                eos_search_paths=[Path(cfg.nyx_cooling_eos_path).resolve()],
            )
        nyx_table = NyxCoolingTableData.load(nyx_table_path)
        nyx_interp = NyxCoolingRateInterpolator(nyx_table)
        return NyxTabulatedCoolingForce(
            eq,
            nyx_interp,
            rho_unit_cgs=float(rho_unit_cgs_used),
            vel_unit_cms=float(hydro_vel_unit_cms),
            tau_time_unit_s=float(cfg.tau_time_unit_s),
            mu=float(cfg.mu_hydrogen),
            heating_rate_scale=float(cfg.nyx_heating_scale),
            cooling_rate_scale=float(cfg.cooling_rate_scale),
            temp_floor_k=float(cfg.cooling_temp_floor_k),
            subcycles=int(cfg.cooling_subcycles),
            dtmax_s=float(cfg.cooling_dtmax_s),
            stop_rate_grad=bool(cfg.cooling_stop_gradient),
            stop_update_grad=bool(cfg.cooling_update_stop_gradient),
        )

    if cooling_model == "grackle_eq":
        z_nodes = parse_float_list(cfg.grackle_z_nodes)
        ln_nodes = np.linspace(
            float(cfg.grackle_lognH_min),
            float(cfg.grackle_lognH_max),
            int(max(2, cfg.grackle_lognH_n)),
            dtype=np.float64,
        )
        lt_nodes = np.linspace(
            float(cfg.grackle_logt_min),
            float(cfg.grackle_logt_max),
            int(max(2, cfg.grackle_logt_n)),
            dtype=np.float64,
        )
        grackle_table_path = Path(cfg.grackle_table_npz).resolve()
        if bool(cfg.grackle_table_rebuild) or (not grackle_table_path.exists()):
            print(
                "[info] Building new Grackle table: "
                f"path={grackle_table_path}, "
                f"z=[{min(z_nodes):.3g},{max(z_nodes):.3g}] ({len(z_nodes)} nodes), "
                f"lognH=[{float(cfg.grackle_lognH_min):.3g},{float(cfg.grackle_lognH_max):.3g}] ({int(max(2, cfg.grackle_lognH_n))} nodes), "
                f"logT=[{float(cfg.grackle_logt_min):.3g},{float(cfg.grackle_logt_max):.3g}] ({int(max(2, cfg.grackle_logt_n))} nodes), "
                f"uvb={cfg.grackle_uvb_model}"
            )
            build_grackle_cooling_table(
                grackle_table_path,
                grackle_data_file=Path(cfg.grackle_data_file).resolve(),
                z_nodes=z_nodes,
                lognH_nodes=ln_nodes,
                logT_nodes=lt_nodes,
                uvb_model=str(cfg.grackle_uvb_model),
                z_sun=float(cfg.grackle_z_sun),
                h_species=float(cfg.h_species),
            )
        grackle_table = GrackleCoolingTableData.load(grackle_table_path)
        gsum = grackle_table.summary_dict()
        print(
            "[info] Loaded Grackle table: "
            f"path={grackle_table_path}, "
            f"z=[{gsum['z_min']:.3g},{gsum['z_max']:.3g}] ({gsum['n_z']} nodes), "
            f"lognH=[{gsum['lognH_min']:.3g},{gsum['lognH_max']:.3g}] ({gsum['n_lognH']} nodes), "
            f"logT=[{gsum['logT_min']:.3g},{gsum['logT_max']:.3g}] ({gsum['n_logT']} nodes), "
            f"uvb={gsum['uvb_model']}"
        )
        grackle_interp = GrackleCoolingRateInterpolator(grackle_table)
        return GrackleTabulatedCoolingForce(
            eq,
            grackle_interp,
            rho_unit_cgs=float(rho_unit_cgs_used),
            vel_unit_cms=float(hydro_vel_unit_cms),
            tau_time_unit_s=float(cfg.tau_time_unit_s),
            mu=float(cfg.mu_hydrogen),
            use_mu_table=bool(cfg.grackle_use_mu_table),
            heating_rate_scale=float(cfg.grackle_heating_scale),
            cooling_rate_scale=float(cfg.cooling_rate_scale),
            metal_cooling_scale=float(cfg.grackle_metal_cooling_scale),
            temp_floor_k=float(cfg.cooling_temp_floor_k),
            subcycles=int(cfg.cooling_subcycles),
            dtmax_s=float(cfg.cooling_dtmax_s),
            stop_rate_grad=bool(cfg.cooling_stop_gradient),
            stop_update_grad=bool(cfg.cooling_update_stop_gradient),
        )

    raise ValueError(f"Unsupported cooling model: {cfg.cooling_model}")


def build_full_hydro_system(cfg: FullHydroConfig, cosmo_lpt: jc.Cosmology, field_sharding=None) -> FullHydroSystem:
    if bool(cfg.enable_metal_source) and not bool(cfg.track_metallicity):
        raise ValueError("enable_metal_source requires track_metallicity=True")

    omega_m = float(cosmo_lpt.Omega_b + cosmo_lpt.Omega_c)
    bg = LCDMBackground(
        h=float(cosmo_lpt.h),
        Omega_m=omega_m,
        Omega_b=float(cosmo_lpt.Omega_b),
        Omega_lambda=float(1.0 - omega_m - float(cosmo_lpt.Omega_k)),
        Omega_k=float(cosmo_lpt.Omega_k),
        n_s=float(cosmo_lpt.n_s),
        sigma8=float(cosmo_lpt.sigma8),
        w0=float(cosmo_lpt.w0),
        wa=float(cosmo_lpt.wa),
        use_jax_cosmo=True,
    )

    n_cons_requested = 5 + int(bool(cfg.dual_energy)) + int(bool(cfg.track_metallicity))
    base_eq = dh.equationmanager.EquationManager(
        n_cons=n_cons_requested,
        dual_energy=bool(cfg.dual_energy),
        has_metallicity=bool(cfg.track_metallicity),
    )
    base_eq.mesh_shape = [cfg.mesh_n, cfg.mesh_n, cfg.mesh_n]
    eq = SuperComovingEquationManager(base_eq, enforce_gamma_53=True)
    eq.eps = float(eq.eps)

    hydro_velocity_scale = float(bg.H0) * float(cfg.mesh_n) / max(float(cfg.box_size_mpc_h), 1.0e-30)
    hydro_vel_unit_cms = float(cfg.vel_unit_cms) / max(hydro_velocity_scale, 1.0e-30)
    mH_cgs = 1.6735575e-24
    kB_cgs = 1.380649e-16
    mu = float(cfg.mu_hydrogen)
    kelvin_to_code_temp = kB_cgs / (
        max(mu, 1.0e-12) * mH_cgs * float(hydro_vel_unit_cms) * float(hydro_vel_unit_cms)
    )
    code_to_kelvin_temp = 1.0 / max(kelvin_to_code_temp, 1.0e-30)

    bg_force = BackgroundExpansionForce(bg, a_init=a_from_z(cfg.z_init))
    if str(getattr(cfg, "gravity_backend", "jaxpm")) == "jaxdecomp_fft":
        # Distributed multi-GPU coupled gravity (jaxdecomp env). field_sharding is
        # built by the driver (cosmo_feedback.distributed.make_meshes) and passed in.
        from cosmo_parallel.coupled_gravity_forces import CoupledJaxDecompFFTGravityForce
        if field_sharding is None:
            raise ValueError("gravity_backend='jaxdecomp_fft' requires field_sharding (pass from make_meshes)")
        grav_force = CoupledJaxDecompFFTGravityForce(
            eq,
            mesh_shape=(cfg.mesh_n, cfg.mesh_n, cfg.mesh_n),
            subtract_mean=True,
            dm_drift_factor=bg.H0,
            dm_kick_factor=1.0,
            gas_kick_factor=(None if cfg.gas_kick_factor is None else float(cfg.gas_kick_factor)),
            eps=float(cfg.force_eps),
            sharding=field_sharding,
            halo_size=int(getattr(cfg, "jaxdecomp_halo_size", 32)),
            dm_pos_key="x",  # cosmo_feedback snapshot stores absolute DM positions under "x"
            gravity_grad_mode=str(getattr(cfg, "gravity_grad_mode", "forward_only")),
            dm_paint_mode=str(getattr(cfg, "jaxdecomp_dm_paint_mode", "relative")),
            grav_mesh_shape=((lambda g: (g, g, g) if g and g != cfg.mesh_n else None)(
                int(getattr(cfg, "jaxdecomp_grav_mesh_n", 0)))),
        )
    else:
        grav_force = JaxPMCoupledGravityForce(
            eq,
            mesh_shape=(cfg.mesh_n, cfg.mesh_n, cfg.mesh_n),
            subtract_mean=True,
            use_jaxpm=True,
            dm_drift_factor=bg.H0,
            dm_kick_factor=1.0,
            gas_kick_factor=(None if cfg.gas_kick_factor is None else float(cfg.gas_kick_factor)),
            eps=float(cfg.force_eps),
            stop_gradient_source=bool(cfg.gravity_stop_gradient_source),
        )

    force_stack = [bg_force, grav_force]
    cooling_force = _build_cooling_force(eq, bg, cfg)
    if cooling_force is not None:
        force_stack.append(cooling_force)
    if (
        bool(cfg.enable_star_formation)
        or bool(cfg.enable_metal_source)
        or int(cfg.star_age_shift_steps) > 0
        or bool(cfg.enable_momentum_feedback)
        or float(cfg.snf_energy) > 0.0
        or float(cfg.stellar_wind_energy) > 0.0
    ):
        force_stack.append(
            StellarFeedbackTracerForce(
                eq=eq,
                code_to_kelvin_temp=float(code_to_kelvin_temp),
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
                feedback_energy_clip_fraction=cfg.feedback_energy_clip_fraction,
                store_diagnostics=bool(cfg.store_subgrid_diagnostics),
            )
        )

    sim = _build_hydrosim(
        eq,
        force_stack,
        solver_name=cfg.solver,
        use_mol=bool(getattr(cfg, "use_mol", True)),
        dx_o=1.0,
        enable_vacuum_momentum_cap=cfg.enable_vacuum_momentum_cap,
        vacuum_momentum_rho_guard=cfg.vacuum_momentum_rho_guard,
        vacuum_momentum_kinetic_to_thermal_max=cfg.vacuum_momentum_kinetic_to_thermal_max,
        vacuum_momentum_internal_floor=cfg.vacuum_momentum_internal_floor,
        enable_hydro_state_repair=cfg.enable_hydro_state_repair,
        hydro_repair_rho_floor=cfg.hydro_repair_rho_floor,
        hydro_repair_pressure_floor=cfg.hydro_repair_pressure_floor,
        hydro_repair_max_kinetic_to_thermal_ratio=cfg.hydro_repair_max_kinetic_to_thermal_ratio,
        hydro_flux_stop_gradient=cfg.hydro_flux_stop_gradient,
        pmesh_shape=cfg.pmesh_shape,
    )
    return FullHydroSystem(
        cosmo_lpt=cosmo_lpt,
        background=bg,
        eq=eq,
        sim=sim,
        hydro_vel_unit_cms=float(hydro_vel_unit_cms),
        kelvin_to_code_temp=float(kelvin_to_code_temp),
        code_to_kelvin_temp=float(code_to_kelvin_temp),
    )


def _init_hydro_state_from_white_noise(
    white_noise: jnp.ndarray,
    pk_sqrt: jnp.ndarray,
    grid_pos: jnp.ndarray,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
) -> tuple[jnp.ndarray, dict, jnp.ndarray]:
    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    a_init = a_from_z(cfg.z_init)
    init_mesh = white_noise_to_init_mesh(white_noise, pk_sqrt)
    prime_growth_cache(system.cosmo_lpt, a_init)

    a_lpt = jnp.asarray([a_init], dtype=jnp.float32)
    dx, p_lpt, _ = lpt(system.cosmo_lpt, init_mesh, grid_pos, a=a_lpt, order=cfg.lpt_order)
    if dx.ndim > 2 and dx.shape[0] == 1:
        dx = dx[0]
    if p_lpt.ndim > 2 and p_lpt.shape[0] == 1:
        p_lpt = p_lpt[0]

    dm_x = jnp.mod(grid_pos + dx, jnp.asarray(mesh_shape, dtype=jnp.float32))
    dm_v_tilde = system.background.H0 * p_lpt

    rho_dm_mesh = _paint_dm_density(dm_x, mesh_shape)
    rho_dm_norm = rho_dm_mesh / (jnp.mean(rho_dm_mesh) + 1.0e-8)

    vx_t, vy_t, vz_t = _paint_dm_velocity_tilde(dm_x, dm_v_tilde, mesh_shape)
    vx_pec = cosmo_conv.velocity_code_to_phys(vx_t, a_init)
    vy_pec = cosmo_conv.velocity_code_to_phys(vy_t, a_init)
    vz_pec = cosmo_conv.velocity_code_to_phys(vz_t, a_init)

    gas_mean = jnp.asarray(float(cfg.gas_mean_fraction), dtype=jnp.float32)
    rho_gas_phys = gas_mean * rho_dm_norm

    t_floor_k = max(float(cfg.hydro_temp_floor_k), 0.0)
    if t_floor_k <= 0.0 and bool(cfg.enable_cooling):
        t_floor_k = max(float(cfg.cooling_temp_floor_k), 0.0)

    T0_k = jnp.asarray(float(cfg.temperature0), dtype=jnp.float32)
    T_k = T0_k * jnp.power(jnp.maximum(rho_gas_phys / jnp.maximum(gas_mean, 1.0e-8), 1.0e-4), float(cfg.temperature_gamma))
    T_k = jnp.maximum(T_k, jnp.asarray(float(t_floor_k), dtype=jnp.float32))
    T_phys = jnp.maximum(
        T_k * jnp.asarray(float(system.kelvin_to_code_temp), dtype=jnp.float32),
        jnp.asarray(1.0e-12, dtype=jnp.float32),
    )
    p_gas_phys = rho_gas_phys * system.eq.R * T_phys

    w_code = cosmo_conv.primitives_phys_to_code(
        rho_gas_phys,
        vx_pec,
        vy_pec,
        vz_pec,
        p_gas_phys,
        a_init,
    )
    U0 = system.eq.get_conservatives_from_primitives(w_code)
    if bool(cfg.track_metallicity) and getattr(system.eq, "metal_density_ids", None) is not None:
        rhoZ0 = initialize_metal_density(
            U0[system.eq.mass_ids],
            metal_init=float(cfg.metal_init),
            metal_floor=float(cfg.metal_floor),
            blob_amplitude=float(cfg.metal_blob_amplitude),
            blob_sigma_cells=float(cfg.metal_blob_sigma_cells),
        )
        U0 = U0.at[int(system.eq.metal_density_ids)].set(rhoZ0)

    dm_mass_fraction = float(max(1.0e-6, min(0.999999, 1.0 - float(cfg.gas_mean_fraction))))
    omega_m = float(system.cosmo_lpt.Omega_b + system.cosmo_lpt.Omega_c)
    dm_params = {
        "x": dm_x,
        "p_or_v": p_lpt,
        "mass": jnp.ones((dm_x.shape[0],), dtype=jnp.float32) * jnp.asarray(dm_mass_fraction, dtype=jnp.float32),
        "drift_factor": jnp.asarray(system.background.H0, dtype=jnp.float32),
        "kick_prefactor": jnp.asarray(1.5 * omega_m * system.background.H0 * float(cfg.dm_kick_scale), dtype=jnp.float32),
    }

    if cfg.gas_kick_factor is None:
        dm_params["gas_kick_prefactor"] = jnp.asarray(
            1.5 * omega_m * (system.background.H0**2) * float(cfg.gas_kick_scale),
            dtype=jnp.float32,
        )
    else:
        dm_params["gas_kick_factor"] = jnp.asarray(float(cfg.gas_kick_factor), dtype=jnp.float32)
    dm_params.update(
        initialize_star_particle_fields(
            dm_x,
            amplitude=float(cfg.synthetic_star_mass_amplitude),
            kind=str(cfg.synthetic_star_mass_kind),
            seed=int(cfg.sf_seed),
            n_age_bins=int(cfg.star_age_bins),
        )
    )

    params0 = {
        "a": jnp.asarray(a_init, dtype=jnp.float32),
        "dm": dm_params,
    }
    if bool(cfg.enable_star_formation):
        params0["rng_sf"] = jax.random.PRNGKey(int(cfg.sf_seed))
    return U0, params0, init_mesh


def _run_hydro_scan(
    U0: jnp.ndarray,
    params0: dict,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
) -> tuple[jnp.ndarray, dict, jnp.ndarray]:
    n_steps = int(cfg.hydro_steps)
    if n_steps < 1:
        raise ValueError("hydro_steps must be >= 1")

    def one_step(carry, i):
        U, params = carry
        return advance_full_hydro_step(U, params, system, cfg, i)

    checkpoint_every = max(1, int(getattr(cfg, "checkpoint_every", 1)))

    # Ensure the scan carry structure stays constant: advance_full_hydro_step adds
    # diagnostic keys to params dynamically, so pre-initialize them here (applies to
    # both the simple and the block-checkpoint scan branches below).
    for k in [
        "_raw_nonfinite_flag",
        "_raw_nonfinite_U_count",
        "_raw_nonpositive_rho_count",
        "_raw_nonfinite_dm_x_count",
        "_raw_nonfinite_dm_p_count",
        "_raw_nonfinite_star_mass_count",
        "_hydro_repair_count",
    ]:
        if k not in params0:
            if k == "_raw_nonfinite_flag":
                params0[k] = jnp.asarray(False, dtype=jnp.bool_)
            else:
                params0[k] = jnp.asarray(0, dtype=jnp.int32)

    if checkpoint_every <= 1:
        scan_step = jax.checkpoint(one_step) if cfg.checkpoint else one_step
        (Uf, paramsf), dt_hist = jax.lax.scan(
            scan_step,
            init=(U0, params0),
            xs=jnp.arange(n_steps, dtype=jnp.int32),
        )
        return Uf, paramsf, dt_hist

    block = int(checkpoint_every)
    n_blocks = n_steps // block
    remainder = n_steps % block

    def inner_step(carry_in, _):
        state, idx = carry_in
        state_out, dtau = one_step(state, idx)
        return (state_out, idx + jnp.asarray(1, dtype=jnp.int32)), dtau

    def block_step(carry_in, _):
        block_inner_step = jax.checkpoint(inner_step) if cfg.checkpoint else inner_step
        carry_out, dt_block = jax.lax.scan(block_inner_step, carry_in, xs=None, length=block)
        return carry_out, dt_block

    carry_with_idx = ((U0, params0), jnp.asarray(0, dtype=jnp.int32))
    dt_parts = []

    if n_blocks > 0:
        carry_with_idx, dt_blocks = jax.lax.scan(block_step, carry_with_idx, xs=None, length=n_blocks)
        dt_parts.append(jnp.reshape(dt_blocks, (-1,)))

    if remainder > 0:
        rem_step = jax.checkpoint(inner_step) if cfg.checkpoint else inner_step
        carry_with_idx, dt_rem = jax.lax.scan(rem_step, carry_with_idx, xs=None, length=remainder)
        dt_parts.append(jnp.reshape(dt_rem, (-1,)))

    (Uf, paramsf), _ = carry_with_idx
    if dt_parts:
        dt_hist = jnp.concatenate(dt_parts, axis=0)
    else:
        dt_hist = jnp.zeros((0,), dtype=jnp.float32)
    return Uf, paramsf, dt_hist


def _project_hydro_state_with_floors(
    U_new: jnp.ndarray,
    params_new: dict,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
) -> jnp.ndarray:
    t_floor_k = max(float(cfg.hydro_temp_floor_k), 0.0)
    if t_floor_k <= 0.0 and bool(cfg.enable_cooling):
        t_floor_k = max(float(cfg.cooling_temp_floor_k), 0.0)
    t_floor_code = jnp.asarray(float(system.kelvin_to_code_temp) * float(t_floor_k), dtype=jnp.float32)

    w_new = system.eq.get_primitives_from_conservatives(U_new)
    w_new = jnp.nan_to_num(w_new, nan=0.0, posinf=0.0, neginf=0.0)
    a_new = jnp.asarray(params_new.get("a", 1.0), dtype=jnp.float32)
    rho_floor = cosmo_conv.density_phys_to_code(jnp.asarray(float(cfg.state_floor), dtype=w_new.dtype), a_new)
    p_floor = cosmo_conv.pressure_phys_to_code(jnp.asarray(float(cfg.pressure_floor), dtype=w_new.dtype), a_new)
    if float(t_floor_k) > 0.0:
        rho_phys_new = cosmo_conv.density_code_to_phys(w_new[0], a_new)
        rho_phys_new = jnp.nan_to_num(
            rho_phys_new,
            nan=float(cfg.state_floor),
            posinf=float(cfg.state_floor),
            neginf=float(cfg.state_floor),
        )
        p_floor_temp_phys = rho_phys_new * jnp.asarray(float(system.eq.R), dtype=w_new.dtype) * t_floor_code
        p_floor_temp_code = cosmo_conv.pressure_phys_to_code(p_floor_temp_phys, a_new)
        p_floor = jnp.maximum(p_floor, p_floor_temp_code)

    w_new = w_new.at[0].set(jnp.maximum(w_new[0], rho_floor))
    w_new = w_new.at[4].set(jnp.maximum(w_new[4], p_floor))
    i_dual = getattr(system.eq, "dual_energy_ids", None)
    if i_dual is not None and int(i_dual) < int(w_new.shape[0]):
        w_new = w_new.at[int(i_dual)].set(jnp.maximum(w_new[int(i_dual)], system.eq.eps))
    for i_passive in tuple(getattr(system.eq, "true_passive_ids", ())):
        if int(i_passive) < int(w_new.shape[0]):
            w_new = w_new.at[int(i_passive)].set(jnp.maximum(w_new[int(i_passive)], 0.0))
    return system.eq.get_conservatives_from_primitives(w_new)


def _raw_hydro_cell_valid_mask(U_state: jnp.ndarray, i_rho: int) -> jnp.ndarray:
    U_state = jnp.asarray(U_state, dtype=jnp.float32)
    rho_raw = U_state[int(i_rho)]
    finite_mask = jnp.all(jnp.isfinite(U_state), axis=0)
    return finite_mask & (rho_raw > 0.0)


def _nonfinite_state_counts(
    U_state: jnp.ndarray,
    params_state: dict,
) -> dict[str, jnp.ndarray]:
    dm_state = params_state.get("dm", {})
    U_state = jnp.asarray(U_state, dtype=jnp.float32)
    rho_raw = U_state[0]
    counts = {
        "raw_nonfinite_U_count": jnp.sum(~jnp.isfinite(U_state), dtype=jnp.int32),
        "raw_nonpositive_rho_count": jnp.sum(rho_raw <= 0.0, dtype=jnp.int32),
        "raw_nonfinite_dm_x_count": jnp.asarray(0, dtype=jnp.int32),
        "raw_nonfinite_dm_p_count": jnp.asarray(0, dtype=jnp.int32),
        "raw_nonfinite_star_mass_count": jnp.asarray(0, dtype=jnp.int32),
    }
    if "x" in dm_state:
        counts["raw_nonfinite_dm_x_count"] = jnp.sum(~jnp.isfinite(jnp.asarray(dm_state["x"], dtype=jnp.float32)), dtype=jnp.int32)
    if "p_or_v" in dm_state:
        counts["raw_nonfinite_dm_p_count"] = jnp.sum(~jnp.isfinite(jnp.asarray(dm_state["p_or_v"], dtype=jnp.float32)), dtype=jnp.int32)
    if "star_mass" in dm_state:
        counts["raw_nonfinite_star_mass_count"] = jnp.sum(~jnp.isfinite(jnp.asarray(dm_state["star_mass"], dtype=jnp.float32)), dtype=jnp.int32)
    return counts


def _with_nonfinite_metadata(params_state: dict, counts: dict[str, jnp.ndarray]) -> dict:
    params_out = dict(params_state)
    has_invalid = (
        counts["raw_nonfinite_U_count"]
        + counts["raw_nonpositive_rho_count"]
        + counts["raw_nonfinite_dm_x_count"]
        + counts["raw_nonfinite_dm_p_count"]
        + counts["raw_nonfinite_star_mass_count"]
    ) > 0
    params_out["_raw_nonfinite_flag"] = jnp.asarray(has_invalid, dtype=jnp.bool_)
    for key, value in counts.items():
        params_out[f"_{key}"] = jnp.asarray(value, dtype=jnp.int32)
    return params_out


def advance_full_hydro_step(
    U: jnp.ndarray,
    params: dict,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
    i,
) -> tuple[tuple[jnp.ndarray, dict], jnp.ndarray]:
    n_steps = int(cfg.hydro_steps)
    i_f = jnp.asarray(i, dtype=jnp.float32)
    remaining = jnp.maximum(jnp.asarray(float(n_steps), dtype=jnp.float32) - i_f, 1.0)
    a_target = jnp.asarray(a_from_z(cfg.z_target), dtype=jnp.float32)

    a_now = jnp.asarray(params["a"], dtype=jnp.float32)
    da_dtau = system.background.da_dtau(a_now)
    if float(getattr(cfg, "solver_dt_safety", 0.0)) > 0.0:
        # Adaptive CFL-limited stepping (driver loops until a_target): step as large as
        # the solver's stable dt allows, but never overshoot a_target. Dense cooling-
        # collapse cells need local dt limiting that the pure a-schedule can't provide.
        dtau_to_target = (a_target - a_now) / jnp.maximum(da_dtau, 1.0e-12)
        dtau_cfl = jnp.asarray(float(cfg.solver_dt_safety), dtype=jnp.float32) * jnp.asarray(
            system.sim.timestep(U), dtype=jnp.float32)
        dtau = jnp.minimum(jnp.minimum(dtau_to_target, dtau_cfl), float(cfg.dtau_max))
        dtau = jnp.maximum(dtau, float(cfg.dtau_min))
    else:
        dtau_raw = (a_target - a_now) / jnp.maximum(da_dtau * remaining, 1.0e-12)
        dtau = jnp.clip(dtau_raw, float(cfg.dtau_min), float(cfg.dtau_max))

    U_new, params_new = system.sim._hydrostep(i, (U, params), dtau)
    nonfinite_counts = _nonfinite_state_counts(U_new, params_new)
    params_new = _with_nonfinite_metadata(params_new, nonfinite_counts)
    if bool(cfg.fail_on_nonfinite):
        U_new = jax.lax.cond(
            jnp.asarray(params_new["_raw_nonfinite_flag"], dtype=jnp.bool_),
            lambda u: u,
            lambda u: _project_hydro_state_with_floors(u, params_new, system, cfg),
            U_new,
        )
    else:
        U_projected = _project_hydro_state_with_floors(U_new, params_new, system, cfg)
        if bool(cfg.projection_stop_gradient_correction):
            U_new = U_new + jax.lax.stop_gradient(U_projected - U_new)
        else:
            U_new = U_projected
    return (U_new, params_new), dtau


def _extract_final_fields(
    Uf: jnp.ndarray,
    paramsf: dict,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    a_fin = jnp.asarray(paramsf["a"], dtype=jnp.float32)

    w_fin = system.eq.get_primitives_from_conservatives(Uf)
    rho_gas_phys = cosmo_conv.density_code_to_phys(w_fin[0], a_fin)
    p_gas_phys = cosmo_conv.pressure_code_to_phys(w_fin[4], a_fin)
    t_code = p_gas_phys / jnp.maximum(rho_gas_phys * system.eq.R, 1.0e-30)
    temp_map = t_code * jnp.asarray(float(system.code_to_kelvin_temp), dtype=jnp.float32)
    vx_phys = cosmo_conv.velocity_code_to_phys(w_fin[1], a_fin)
    vy_phys = cosmo_conv.velocity_code_to_phys(w_fin[2], a_fin)
    vz_phys = cosmo_conv.velocity_code_to_phys(w_fin[3], a_fin)
    hydro_vel_unit_cms = jnp.asarray(float(system.hydro_vel_unit_cms), dtype=jnp.float32)
    vx_cms = vx_phys * hydro_vel_unit_cms
    vy_cms = vy_phys * hydro_vel_unit_cms
    vz_cms = vz_phys * hydro_vel_unit_cms

    rho_gas_norm = rho_gas_phys / (jnp.mean(rho_gas_phys) + 1.0e-8)

    dm_positions = jnp.asarray(paramsf["dm"]["x"], dtype=jnp.float32)
    dm_weight = jnp.asarray(paramsf["dm"].get("mass", 1.0), dtype=jnp.float32)
    dm_mesh = _paint_dm_density(dm_positions, mesh_shape, weight=dm_weight)
    dm_norm = dm_mesh / (jnp.mean(dm_mesh) + 1.0e-8)

    return (
        dm_norm.astype(jnp.float32),
        rho_gas_norm.astype(jnp.float32),
        temp_map.astype(jnp.float32),
        vx_cms.astype(jnp.float32),
        vy_cms.astype(jnp.float32),
        vz_cms.astype(jnp.float32),
        a_fin,
    )


def extract_subgrid_state(
    U_state: jnp.ndarray,
    params_state: dict,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
) -> dict[str, jnp.ndarray]:
    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    a_now = jnp.asarray(params_state["a"], dtype=jnp.float32)
    w_state = system.eq.get_primitives_from_conservatives(U_state)

    rho_gas_phys = cosmo_conv.density_code_to_phys(w_state[0], a_now)
    p_gas_phys = cosmo_conv.pressure_code_to_phys(w_state[4], a_now)
    temp_code = p_gas_phys / jnp.maximum(rho_gas_phys * system.eq.R, 1.0e-30)
    temp_k = temp_code * jnp.asarray(float(system.code_to_kelvin_temp), dtype=jnp.float32)

    dm_positions = jnp.asarray(params_state["dm"]["x"], dtype=jnp.float32)
    dm_weight = jnp.asarray(params_state["dm"].get("mass", 1.0), dtype=jnp.float32)
    dm_density = _paint_dm_density(dm_positions, mesh_shape, weight=dm_weight)

    out = {
        "a": a_now,
        "z": (1.0 / jnp.maximum(a_now, 1.0e-12)) - 1.0,
        "gas_density": rho_gas_phys,
        "gas_temperature_k": temp_k,
        "dm_density": dm_density,
    }

    rhoZ, metal_fraction = extract_metal_fields(U_state, system.eq)
    if rhoZ is not None:
        out["metal_density"] = rhoZ
        out["metal_fraction"] = metal_fraction

    star_mass = params_state["dm"].get("star_mass", None)
    if star_mass is not None:
        star_mass = jnp.asarray(star_mass, dtype=jnp.float32)
        out["particle_star_mass"] = star_mass
        out["stellar_density"] = _paint_particle_scalar(
            dm_positions,
            star_mass,
            mesh_shape,
            sigma_threshold=cfg.stellar_paint_sigma_threshold,
            min_mass=cfg.stellar_paint_min_mass,
        )
    star_age_bins = params_state["dm"].get("star_age_bins", None)
    if star_age_bins is not None:
        out["star_age_bins"] = jnp.asarray(star_age_bins, dtype=jnp.float32)

    sf_diag = params_state.get("sf_diagnostics", None)
    if sf_diag is not None:
        for key, value in sf_diag.items():
            out[f"sf_{key}"] = jnp.asarray(value, dtype=jnp.float32)
    return out


def evolve_full_hydro_state(
    white_noise: jnp.ndarray,
    pk_sqrt: jnp.ndarray,
    grid_pos: jnp.ndarray,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
) -> tuple[jnp.ndarray, dict, jnp.ndarray, jnp.ndarray, dict, jnp.ndarray]:
    U0, params0, init_mesh = _init_hydro_state_from_white_noise(white_noise, pk_sqrt, grid_pos, system, cfg)
    Uf, paramsf, dt_hist = _run_hydro_scan(U0, params0, system, cfg)
    return U0, params0, init_mesh, Uf, paramsf, dt_hist


def forward_fields_full_hydro(
    white_noise: jnp.ndarray,
    pk_sqrt: jnp.ndarray,
    grid_pos: jnp.ndarray,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    _, _, init_mesh, Uf, paramsf, dt_hist = evolve_full_hydro_state(
        white_noise,
        pk_sqrt,
        grid_pos,
        system,
        cfg,
    )
    rho_dm, rho_gas, temp_map, vx_cms, vy_cms, vz_cms, a_fin = _extract_final_fields(Uf, paramsf, system, cfg)
    return (
        rho_dm,
        rho_gas,
        temp_map,
        paramsf["dm"]["x"],
        init_mesh,
        jnp.asarray([a_fin, jnp.mean(dt_hist)], dtype=jnp.float32),
        vx_cms,
        vy_cms,
        vz_cms,
    )


def make_hydro_density_nlogposterior(
    target_rho: jnp.ndarray,
    pk_sqrt: jnp.ndarray,
    grid_pos: jnp.ndarray,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
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
        _, rho_gas, _, _, _, _, _, _, _ = forward_fields_full_hydro(
            white_noise,
            pk_sqrt,
            grid_pos,
            system,
            cfg,
        )

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


def prime_system_growth_cache(system: FullHydroSystem, cfg: FullHydroConfig) -> None:
    prime_growth_cache(system.cosmo_lpt, a_from_z(cfg.z_init))
