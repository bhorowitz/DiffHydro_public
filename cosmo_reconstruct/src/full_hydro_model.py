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
from diffhydro.cosmology.nyx_cooling_table import (
    NyxCoolingRateInterpolator,
    NyxCoolingTableData,
    build_nyx_cooling_table,
    parse_float_list,
    rho_crit0_cgs,
)
from jaxpm.painting import cic_paint
from jaxpm.pm import lpt

from .forward_model import a_from_z, prime_growth_cache, white_noise_to_init_mesh


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

    solver: str = "hll"
    state_floor: float = 2.0e-8
    pressure_floor: float = 2.0e-8
    hydro_temp_floor_k: float = 0.0
    force_eps: float = 1.0e-8
    checkpoint: bool = True
    checkpoint_every: int = 1
    dual_energy: bool = True

    temperature0: float = 1.0e4
    temperature_gamma: float = 2.0 / 3.0
    gas_mean_fraction: float = 1.58e-1

    dm_kick_scale: float = 1.0
    gas_kick_scale: float = 1.0
    gas_kick_factor: float | None = None

    enable_cooling: bool = False
    cooling_model: str = "nyx_table"
    cooling_stop_gradient: bool = True
    cooling_table: str = "data/m-00.cie"
    heating_rate_per_h: float = 1.0e-33
    nyx_heating_scale: float = 1.2
    cooling_rate_scale: float = 1.0
    cooling_temp_floor_k: float = 1.0
    cooling_subcycles: int = 8
    cooling_dtmax_s: float = 1.0e16
    nyx_cooling_table_npz: str = "data/nyx_cooling_table.npz"
    nyx_cooling_treecool: str = "diffhydro/nyx_eos/TREECOOL_middle"
    nyx_cooling_z_nodes: str = "2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100"
    nyx_cooling_logdelta_min: float = -3.0
    nyx_cooling_logdelta_max: float = 3.0
    nyx_cooling_logdelta_n: int = 96
    nyx_cooling_logt_min: float = 0.0
    nyx_cooling_logt_max: float = 8.0
    nyx_cooling_logt_n: int = 120
    nyx_cooling_rebuild: bool = False
    nyx_cooling_eos_path: str = "diffhydro/nyx_eos"
    nyx_auto_rho_unit: bool = False
    tau_time_unit_s: float = 3.085677581e19
    rho_unit_cgs: float = 1.6e-24
    vel_unit_cms: float = 1.0e7
    mu_hydrogen: float = 1.0
    h_species: float = 0.76


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
    dx_o: float = 1.0,
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
        use_mol=True,
        pmesh_shape=(1, 1, 1),
    )
    sim.dx_o = float(dx_o)
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
        )

    raise ValueError(f"Unsupported cooling model: {cfg.cooling_model}")


def build_full_hydro_system(cfg: FullHydroConfig, cosmo_lpt: jc.Cosmology) -> FullHydroSystem:
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

    n_cons_requested = 6 if bool(cfg.dual_energy) else 5
    base_eq = dh.equationmanager.EquationManager(n_cons=n_cons_requested)
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
    grav_force = JaxPMCoupledGravityForce(
        eq,
        mesh_shape=(cfg.mesh_n, cfg.mesh_n, cfg.mesh_n),
        subtract_mean=True,
        use_jaxpm=True,
        dm_drift_factor=bg.H0,
        dm_kick_factor=1.0,
        gas_kick_factor=(None if cfg.gas_kick_factor is None else float(cfg.gas_kick_factor)),
        eps=float(cfg.force_eps),
    )

    force_stack = [bg_force, grav_force]
    cooling_force = _build_cooling_force(eq, bg, cfg)
    if cooling_force is not None:
        force_stack.append(cooling_force)

    sim = _build_hydrosim(eq, force_stack, solver_name=cfg.solver, dx_o=1.0)
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

    dx, p_lpt, _ = lpt(system.cosmo_lpt, init_mesh, grid_pos, a=jnp.asarray(a_init, dtype=jnp.float32), order=cfg.lpt_order)
    if dx.ndim == 3:
        dx = dx[0]
    if p_lpt.ndim == 3:
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

    params0 = {
        "a": jnp.asarray(a_init, dtype=jnp.float32),
        "dm": dm_params,
    }
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

    a_target = jnp.asarray(a_from_z(cfg.z_target), dtype=jnp.float32)
    t_floor_k = max(float(cfg.hydro_temp_floor_k), 0.0)
    if t_floor_k <= 0.0 and bool(cfg.enable_cooling):
        t_floor_k = max(float(cfg.cooling_temp_floor_k), 0.0)
    t_floor_code = jnp.asarray(float(system.kelvin_to_code_temp) * float(t_floor_k), dtype=jnp.float32)

    def one_step(carry, i):
        U, params = carry
        i_f = jnp.asarray(i, dtype=jnp.float32)
        remaining = jnp.maximum(jnp.asarray(float(n_steps), dtype=jnp.float32) - i_f, 1.0)

        a_now = jnp.asarray(params["a"], dtype=jnp.float32)
        da_dtau = system.background.da_dtau(a_now)
        dtau_raw = (a_target - a_now) / jnp.maximum(da_dtau * remaining, 1.0e-12)
        dtau = jnp.clip(dtau_raw, float(cfg.dtau_min), float(cfg.dtau_max))

        U_new, params_new = system.sim._hydrostep(i, (U, params), dtau)

        # Keep primitive state physically admissible with Nyx-style floor projection.
        w_new = system.eq.get_primitives_from_conservatives(U_new)
        w_new = jnp.nan_to_num(w_new, nan=0.0, posinf=0.0, neginf=0.0)
        a_new = jnp.asarray(params_new.get("a", a_now), dtype=jnp.float32)
        rho_floor = cosmo_conv.density_phys_to_code(jnp.asarray(float(cfg.state_floor), dtype=w_new.dtype), a_new)
        p_floor = cosmo_conv.pressure_phys_to_code(jnp.asarray(float(cfg.pressure_floor), dtype=w_new.dtype), a_new)
        if float(t_floor_k) > 0.0:
            rho_phys_new = cosmo_conv.density_code_to_phys(w_new[0], a_new)
            rho_phys_new = jnp.nan_to_num(rho_phys_new, nan=float(cfg.state_floor), posinf=float(cfg.state_floor), neginf=float(cfg.state_floor))
            p_floor_temp_phys = rho_phys_new * jnp.asarray(float(system.eq.R), dtype=w_new.dtype) * t_floor_code
            p_floor_temp_code = cosmo_conv.pressure_phys_to_code(p_floor_temp_phys, a_new)
            p_floor = jnp.maximum(p_floor, p_floor_temp_code)

        w_new = w_new.at[0].set(jnp.maximum(w_new[0], rho_floor))
        w_new = w_new.at[4].set(jnp.maximum(w_new[4], p_floor))
        for i_passive in tuple(getattr(system.eq, "passive_ids", ())):
            if int(i_passive) >= 5:
                w_new = w_new.at[int(i_passive)].set(jnp.maximum(w_new[int(i_passive)], system.eq.eps))
        U_new = system.eq.get_conservatives_from_primitives(w_new)

        return (U_new, params_new), dtau

    checkpoint_every = max(1, int(getattr(cfg, "checkpoint_every", 1)))

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

    if cfg.checkpoint:
        block_step = jax.checkpoint(block_step)

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
    U0, params0, init_mesh = _init_hydro_state_from_white_noise(white_noise, pk_sqrt, grid_pos, system, cfg)
    Uf, paramsf, dt_hist = _run_hydro_scan(U0, params0, system, cfg)
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
