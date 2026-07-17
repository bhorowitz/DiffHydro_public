from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import diffhydro as dh
import functools
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
    # Hard floor on the conformal step (0 disables). Caps the *max* step count when the CFL
    # cap is active; relies on the state ceilings below to keep the floored step stable.
    dtau_floor: float = 0.0
    # Per-cell state ceilings (0 disables each). A few numerically over-collapsed cells reach
    # absurd T / |v|, spiking the per-cell signal speed |v|+c_s -> they dominate the global CFL
    # min and collapse the timestep (and can diverge). Capping T (-> bounds c_s) and the peculiar
    # velocity each step bounds max(inv_dt), so the CFL-limited dtau stops collapsing (step count
    # returns to ~hydro_steps) and the runaway cells stay finite. Set far above the real WHIM
    # (~1e7 K) so only numerical runaways are touched.
    hydro_temp_ceiling_k: float = 0.0
    hydro_velocity_ceiling_kms: float = 0.0

    solver: str = "hll"
    # Reconstruction slope limiter (diffhydro LIMITER_DICT): "MINMOD" (default,
    # legacy), "VANLEER" (smoother -> less-kinky gradient), "MC", "SUPERBEE".
    hydro_limiter: str = "MINMOD"
    # Soft-HLL Riemann smoothing: 0 = standard HLL (default, exact). >0 replaces
    # the wave-speed clamps min(S_L,0)/max(S_R,0) and the piecewise flux selection
    # with softplus blends so the flux is C-infinity (no min/max/where kinks). The
    # value is the transition width as a fraction of the local sound speed; small
    # (e.g. 0.05-0.2) stays close to HLL while removing the gradient kinks.
    hydro_riemann_smooth: float = 0.0
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
    # Unified-mesh mode (Option C'): build the hydro shard_map on the SAME 2-axis
    # ('x','y') mesh as the jaxdecomp gravity (hydro pmesh (pd0,pd1,1), field spec
    # P(None,'x','y',None)). Lets hydro + gravity co-occur in ONE traced program,
    # enabling jit/scan/checkpoint BPTT through the distributed rollout. Requires
    # gravity_backend='jaxdecomp_fft' + field_sharding.
    unified_mesh: bool = False
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
    # UVB/reionization temperature floor (campaign Batch 2, default OFF = legacy).
    # The equilibrium cooling table cannot reheat diffuse gas within a Hubble time
    # (relaxation ~1e20 yr at void densities), so gas that never got the reionization
    # flash rides a cold adiabat to cooling_temp_floor_k while the real IGM sits near
    # T_eq(nH, z) ~ 1e4 K. This floors T at uvb_temp_floor_scale * T_eq(nH, z)
    # (primordial heat=cool from the loaded Grackle table), smoothly gated OFF above
    # uvb_temp_floor_delta_cut in overdensity so dense/metal-cooled gas is untouched.
    uvb_temp_floor: bool = False
    uvb_temp_floor_delta_cut: float = 5.0
    uvb_temp_floor_width_dex: float = 0.3
    uvb_temp_floor_scale: float = 1.0
    cooling_subcycles: int = 8
    cooling_dtmax_s: float = 1.0e16
    # Sub-step integrator for the cooling/heating source. "explicit" = forward
    # Euler (default, legacy). "exponential" = analytic relaxation of the locally
    # linearized ODE (Et_new = Et + dotE*dt*(exp(k*dt)-1)/(k*dt), k=d dotE/d Et):
    # unconditionally stable and gives a bounded, contractive reverse-mode
    # gradient (exp(k*dt) in (0,1) for cooling) instead of explicit Euler's
    # amplifying 1+k*dt -- smoother gradient for sampling.
    # "implicit_adjoint" = unconditionally-stable backward-Euler sub-steps with
    # the EXACT implicit-function-theorem adjoint (exact + smooth gradient, no
    # stop_gradient surrogate); the correct choice for HMC/NUTS sampling.
    cooling_integrator: str = "explicit"
    # Path to a trained NN cooling-rate surrogate npz (src/nn_cooling.py). When
    # set, replaces the tabulated NyxCoolingRateInterpolator with a C-infinity
    # tanh MLP whose curvature is training-bounded -> with
    # cooling_integrator="exponential_exact" the potential is conservative AND
    # leapfrog-integrable (the table's exact adjoint is chaos-amplified).
    cooling_nn_npz: str | None = None
    # >0 enables the C-inf sigmoid dual-energy selector with this relative
    # blend width around eta (e.g. 0.5). 0 keeps the hard where() (default).
    dual_energy_smooth: float = 0.0
    # C1 smoothstep interpolation of the Nyx cooling table (removes the bilinear
    # C0 gradient-kink comb). Recommended with cooling_integrator="implicit_adjoint".
    cooling_smooth_table: bool = False
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


class SoftHLL(dh.HLL):
    """HLL Riemann solver with C-infinity wave-speed clamps.

    Standard HLL has gradient kinks from min(S_L,0), max(S_R,0) and the nested
    `where` flux selection. Here the clamps are replaced by softplus blends:
        S_R =  s*softplus( S_R_raw/s) > 0     (~ max(S_R_raw, 0))
        S_L = -s*softplus(-S_L_raw/s) < 0     (~ min(S_L_raw, 0))
    with s = smooth_frac*(cs_L+cs_R). Because S_L<0<S_R STRICTLY, the HLL average
    flux is smooth everywhere and recovers F_L/F_R in the supersonic limits
    automatically, so the piecewise selection (and its kinks) is dropped. As
    smooth_frac -> 0 this converges to standard HLL.
    """

    def __init__(self, equation_manager, signal_speed, smooth_frac=0.1, **kwargs):
        super().__init__(equation_manager, signal_speed)
        self.smooth_frac = float(smooth_frac)

    def _solve_riemann_problem_xi_single_phase(
            self, primitives_L, primitives_R, conservatives_L, conservatives_R, axis, **kwargs):
        eq = self.equation_manager
        F_L = eq.get_fluxes_xi(primitives_L, conservatives_L, axis)
        F_R = eq.get_fluxes_xi(primitives_R, conservatives_R, axis)
        uL = primitives_L[self.velocity_ids[axis]]
        uR = primitives_R[self.velocity_ids[axis]]
        cs_L = eq.get_speed_of_sound(primitives_L[eq.energy_ids], primitives_L[eq.mass_ids])
        cs_R = eq.get_speed_of_sound(primitives_R[eq.energy_ids], primitives_R[eq.mass_ids])
        S_L_raw, S_R_raw = self.signal_speed(
            uL, uR, cs_L, cs_R,
            rho_L=primitives_L[self.mass_ids], rho_R=primitives_R[self.mass_ids],
            p_L=primitives_L[self.energy_ids], p_R=primitives_R[self.energy_ids],
            gamma=eq.gamma,
        )
        s = self.smooth_frac * (cs_L + cs_R + self.eps)
        S_R = s * jnp.logaddexp(0.0, S_R_raw / s)    # ~ max(S_R_raw, 0), > 0
        S_L = -s * jnp.logaddexp(0.0, -S_L_raw / s)  # ~ min(S_L_raw, 0), < 0
        denom = S_R - S_L  # strictly positive -> no guard / no where needed
        F_hll = (
            S_R * F_L - S_L * F_R + S_L * S_R * (conservatives_R - conservatives_L)
        ) / denom
        return F_hll, None, None


def _build_hydrosim(
    eq,
    forces,
    *,
    solver_name: str,
    limiter: str = "MINMOD",
    riemann_smooth: float = 0.0,
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
    mesh=None,
    field_spec=None,
):
    ss = dh.signal_speed_Einfeldt
    sname = solver_name.lower()
    if sname == "hllc":
        riemann_solver = dh.HLLC(equation_manager=eq, signal_speed=ss)
    elif sname in ("laxfriedrichs", "lf"):
        riemann_solver = dh.LaxFriedrichs(equation_manager=eq, signal_speed=ss)
    elif sname == "hll":
        if float(riemann_smooth) > 0.0:
            riemann_solver = SoftHLL(equation_manager=eq, signal_speed=ss, smooth_frac=float(riemann_smooth))
        else:
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
        dh.PPM_CW(limiter=str(limiter), steepen=False),
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
        mesh=mesh,
        field_spec=field_spec,
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


def _stable_temperature_kelvin(rho_cgs, p_cgs, *, mu, mH_cgs, kB_cgs, rho_unit_cgs, p_unit_cgs, eps):
    """T = (mu mH / kB) * p_cgs / rho_cgs, evaluated via O(1) code-unit ratios.

    rho_cgs ~ 1e-30 g/cm^3, so the direct cgs division has a forward value
    (~1e30) that fits float32 but a reverse-mode term ~1/rho_cgs^2 (~1e50+) that
    under/overflows float32 -> NaN gradient on every cell (rho_cgs**2 flushes to
    0). Since p_unit_cgs / rho_unit_cgs == vel_unit_cms^2 exactly, dividing the
    O(1) physical (code-unit) density/pressure is value-identical while keeping
    all gradient intermediates representable.
    """
    rho_phys = rho_cgs / rho_unit_cgs
    p_phys = p_cgs / p_unit_cgs
    floor_phys = eps / rho_unit_cgs
    return (mu * mH_cgs / kB_cgs) * (p_unit_cgs / rho_unit_cgs) * p_phys / jnp.maximum(rho_phys, floor_phys)


def _safe_denom(x, eps=1.0e-30):
    """1/x-safe denominator: replace |x|<eps by +/-eps keeping the sign."""
    return jnp.where(jnp.abs(x) < eps, jnp.where(x >= 0, eps, -eps), x)


def _implicit_energy_newton(rate_fn, E0, rho_cgs, z, dt, E_floor, n_newton=8):
    """Backward-Euler solve of  E1 = E0 + dt*rate_fn(E1, rho, z)  per cell.

    All Newton algebra runs in per-cell SCALED units: e = E/A with
    A = max(E0, E_floor), so the residual e - e0 - f, the scaled rate
    f = dt*R/A (the fractional energy change per sub-step), and the Jacobian
    1 - df/de are all O(1). cgs energies here are ~1e-18 erg/cm^3 and their
    jvp tangents reach ~1e22 when seeded with ones -- right where float32
    under/overflows and shreds gradient reproducibility. Seeding the rate
    derivative in e-space (tangent A instead of 1) keeps every intermediate
    representable, pure float32, no x64 needed.
    e is clamped to >= e_floor each iteration so T stays positive and the table
    never sees a non-physical state (a Newton overshoot -> T<0 -> NaN).
    """
    A = jnp.maximum(jnp.maximum(E0, E_floor), jnp.asarray(1.0e-30, E0.dtype))
    e0 = E0 / A                      # ~1 by construction
    ef = E_floor / A                 # <=1
    def f_scaled(e):
        # fractional energy change per sub-step; O(1) for a resolved sub-step
        return dt * rate_fn(A * e, rho_cgs, z) / A
    e = jnp.maximum(e0, ef)
    for _ in range(int(n_newton)):
        # value and derivative of the scaled rate in ONE jvp, seeded in e-space
        f, f_e = jax.jvp(f_scaled, (e,), (jnp.ones_like(e),))
        # saturated/extreme cells (T or rho clipped at a table edge) can produce
        # inf*0=NaN tangents; the true local sensitivity there is 0.
        f = jnp.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        f_e = jnp.nan_to_num(f_e, nan=0.0, posinf=0.0, neginf=0.0)
        F = e - e0 - f
        dF = 1.0 - f_e
        e = jnp.maximum(e - F / _safe_denom(dF), ef)
    return A * e


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _nyx_implicit_energy_step(rate_fn, E0, rho_cgs, z, dt, E_floor):
    """One implicit (backward-Euler) cooling sub-step with an EXACT, smooth VJP.

    Forward: solve E1 = E0 + dt*rate_fn(E1, rho, z) (Newton, per-cell O(1)
    scaled units -- see _implicit_energy_newton).
    Backward: the implicit-function-theorem adjoint of that fixed point --
      dE1/dE0   = 1 / (1 - dt*R_E)
      dE1/drho  = dt*R_rho / (1 - dt*R_E)
    where R_E = d(rate)/dE1, R_rho = d(rate)/drho are FIRST derivatives of the
    rate at the converged E1 (one forward-mode jvp each -- no 2nd-order AD through
    the table, hence no NaN, and no stop_gradient bias). The adjoint algebra is
    likewise formed on O(1) scaled quantities so float32 stays reproducible.
    This makes grad(U) the exact gradient of the value U for HMC, unlike the
    exponential-phi surrogate.
    Only `rate_fn` is nondiff (a callable); z/dt/E_floor are differentiable args
    carried for the solve but get zero cotangents (none is sampled).
    """
    return _implicit_energy_newton(rate_fn, E0, rho_cgs, z, dt, E_floor)


def _nyx_implicit_step_fwd(rate_fn, E0, rho_cgs, z, dt, E_floor):
    E1 = _implicit_energy_newton(rate_fn, E0, rho_cgs, z, dt, E_floor)
    return E1, (E1, rho_cgs, z, dt, E_floor)


def _nyx_implicit_step_bwd(rate_fn, residual, g):
    E1, rho_cgs, z, dt, E_floor = residual
    # Same per-cell O(1) scaling as the forward Newton: e = E/A. All adjoint
    # algebra (Jacobian, denominators, sensitivities) is formed on O(1) scaled
    # quantities; the only large/small factors are applied once at the end.
    A = jnp.maximum(jnp.maximum(E1, E_floor), jnp.asarray(1.0e-30, E1.dtype))
    e1 = E1 / A
    # d(scaled rate)/de at the converged solution, seeded in e-space (tangent 1
    # in e == tangent A in E -> O(1) intermediates through the table).
    def f_of_e(e):
        return dt * rate_fn(A * e, rho_cgs, z) / A
    _, f_e = jax.jvp(f_of_e, (e1,), (jnp.ones_like(e1),))
    # d(scaled rate)/dlog(rho), seeded with tangent rho (O(1) through log-density
    # table coordinates); converts to d/drho by a single 1/rho at the end.
    def f_of_rho(r):
        return dt * rate_fn(A * e1, r, z) / A
    _, f_lrho = jax.jvp(f_of_rho, (rho_cgs,), (rho_cgs,))
    # nan_to_num guards saturated cells (T/rho clipped at a table edge -> the
    # true sensitivity is 0 but float32 can form inf*0=NaN in the tangent).
    f_e = jnp.nan_to_num(f_e, nan=0.0, posinf=0.0, neginf=0.0)
    f_lrho = jnp.nan_to_num(f_lrho, nan=0.0, posinf=0.0, neginf=0.0)
    # IFT adjoint of e1 = e0 + f(e1, log rho):
    #   de1/de0     = 1/(1 - f_e)
    #   de1/dlogrho = f_lrho/(1 - f_e)
    # For a dissipative step f_e<=0 -> denom>=1 (contractive). Limit f_e<=0 so a
    # numerically-positive f_e (heating cells / table noise) cannot drive the
    # denominator toward 0 and blow the gradient up.
    f_e = jnp.minimum(f_e, 0.0)
    denom = 1.0 - f_e
    # dE1/dE0 = de1/de0 (A cancels); dE1/drho = A*(de1/dlogrho)/rho.
    gE0 = g / denom
    grho = gE0 * f_lrho * (A / jnp.maximum(rho_cgs, jnp.asarray(1.0e-35, rho_cgs.dtype)))
    # cells pinned to the energy floor have a (sub)gradient of 0 (E1 no longer
    # depends on the inputs there) -- zero their cotangents to stay consistent.
    active = (E1 > E_floor * (1.0 + 1.0e-4)).astype(g.dtype)
    gE0 = jnp.nan_to_num(gE0 * active, nan=0.0, posinf=0.0, neginf=0.0)
    grho = jnp.nan_to_num(grho * active, nan=0.0, posinf=0.0, neginf=0.0)
    # cotangents for (E0, rho_cgs, z, dt, E_floor); only E0, rho_cgs matter.
    return (gE0.astype(g.dtype), grho.astype(g.dtype),
            jnp.zeros_like(z), jnp.zeros_like(dt), jnp.zeros_like(E_floor))


_nyx_implicit_energy_step.defvjp(_nyx_implicit_step_fwd, _nyx_implicit_step_bwd)


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
        t_kelvin = _stable_temperature_kelvin(
            rho_cgs, p_cgs, mu=self.mu, mH_cgs=self.mH_cgs, kB_cgs=self.kB_cgs,
            rho_unit_cgs=self.rho_unit_cgs, p_unit_cgs=self.p_unit_cgs, eps=self.eps)
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
            T_cur = _stable_temperature_kelvin(
                rho_cgs, p_cgs_cur, mu=self.mu, mH_cgs=self.mH_cgs, kB_cgs=self.kB_cgs,
                rho_unit_cgs=self.rho_unit_cgs, p_unit_cgs=self.p_unit_cgs, eps=self.eps)
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
        integrator: str = "explicit",
        cooling_step_mode: str = None,
        smooth_table: bool = False,
        newton_iters: int = 8,
    ):
        self.eq = equation_manager
        self.nyx = nyx_interp
        # `cooling_step_mode` is an alias for `integrator` (explicit | exponential |
        # implicit_adjoint) kept for the standalone implicit-adjoint test suite.
        self.integrator = str(cooling_step_mode if cooling_step_mode is not None else integrator)
        self.newton_iters = int(newton_iters)
        # C1 smoothstep table interpolation -> smooth cooling-rate derivatives for
        # the implicit-adjoint gradient (default off preserves bilinear behavior).
        self.nyx.smooth_interp = bool(smooth_table)
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
        t_kelvin = _stable_temperature_kelvin(
            rho_cgs, p_cgs, mu=self.mu, mH_cgs=self.mH_cgs, kB_cgs=self.kB_cgs,
            rho_unit_cgs=self.rho_unit_cgs, p_unit_cgs=self.p_unit_cgs, eps=self.eps)
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

    def _rate_from_energy_cgs(self, E_cgs, rho_cgs, z):
        """Net heating-cooling rate dotE (cgs) as a function of internal energy
        density E_cgs and rho_cgs. T is recovered from E via
        E = rho*kB*T/(mu*mH*(gamma-1)); works in the caller's dtype (float64 in
        the implicit solve). Used by the implicit-adjoint integrator."""
        rho_safe = jnp.maximum(rho_cgs, self.eps)
        T = (self.gamma - 1.0) * E_cgs * (self.mu * self.mH_cgs) / (self.kB_cgs * rho_safe)
        T = jnp.maximum(T, self.temp_floor_k)
        heat_cgs, cool_cgs, _ = self.nyx.evaluate(rho_cgs, T, z)
        heat_cgs = jnp.nan_to_num(heat_cgs, nan=0.0, posinf=0.0, neginf=0.0)
        cool_cgs = jnp.nan_to_num(cool_cgs, nan=0.0, posinf=0.0, neginf=0.0)
        dotE = self.heating_rate_scale * heat_cgs - self.cooling_rate_scale * cool_cgs
        return jnp.nan_to_num(dotE, nan=0.0, posinf=0.0, neginf=0.0)

    def _implicit_energy_solve(self, E0, rho_cgs, z, dt, E_floor):
        """One implicit backward-Euler cooling sub-step with the exact IFT adjoint
        (see _nyx_implicit_energy_step), then apply the energy floor."""
        E1 = _nyx_implicit_energy_step(self._rate_from_energy_cgs, E0, rho_cgs, z, dt, E_floor)
        return jnp.maximum(E1, E_floor)

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

        exponential = self.integrator in ("exponential", "exponential_exact")
        # exponential_exact: SAME forward value as `exponential` (same surrogate
        # posterior), but the gradient flows through dotE, k and phi (no
        # stop_gradient), so the AD force IS the gradient of the evaluated
        # value -- a conservative force, as leapfrog-based MH samplers require
        # (the stop_gradient surrogate force does ~0.4 nats of phantom work per
        # unit path length, measured by --energy-probe force, which vetoes any
        # trajectory longer than ~2 sampling-coord units). Requires 2nd-order
        # AD through the cooling table: pair with cooling_smooth_table=True
        # (C1 interpolation) so d(dotE)/dT is continuous.
        exact_grad = (self.integrator == "exponential_exact")

        def _dotE_of_T(T):
            heat_cgs, cool_cgs, _ = self.nyx.evaluate(rho_cgs, T, z)
            heat_cgs = jnp.nan_to_num(heat_cgs, nan=0.0, posinf=0.0, neginf=0.0)
            cool_cgs = jnp.nan_to_num(cool_cgs, nan=0.0, posinf=0.0, neginf=0.0)
            d = self.heating_rate_scale * heat_cgs - self.cooling_rate_scale * cool_cgs
            return jnp.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        def body(_, Et_cur):
            p_cgs_cur = (self.gamma - 1.0) * jnp.maximum(Et_cur, self.eps)
            T_cur = _stable_temperature_kelvin(
                rho_cgs, p_cgs_cur, mu=self.mu, mH_cgs=self.mH_cgs, kB_cgs=self.kB_cgs,
                rho_unit_cgs=self.rho_unit_cgs, p_unit_cgs=self.p_unit_cgs, eps=self.eps)
            if exact_grad:
                # exponential_exact: same analytic-relaxation update as the
                # `exponential` branch below, but the gradient flows through
                # dotE, k and phi so the AD force IS the gradient of the
                # evaluated value (conservative -- required by leapfrog MH).
                # Computed in per-cell O(1) rescaled variables e = Et/A with
                # the huge dt/A factor FUSED into the differentiated function:
                # the naive formulation materializes backward partials like
                # (T/Et)^2 * dt ~ 1e42 which overflow float32 -> NaN. Here
                # dg/de = k*dt exactly, with all AD intermediates O(1..1e28).
                # A is a stop_gradient CONSTANT scale: the value is
                # A-independent, so the gradient stays exact.
                # Floor the scale at the PHYSICAL energy floor, not machine eps:
                # cells at Et ~ eps=1e-40 (reachable under sharp limiters) give
                # dtA = dt_sub/A ~ 1e13/1e-40 = f32 inf -> NaN gradient.
                A = jax.lax.stop_gradient(
                    jnp.maximum(Et_cur, jnp.maximum(Et_floor, jnp.asarray(1.0e-20, Et_cur.dtype)))
                )
                dtA = dt_sub / A
                e_cur = Et_cur / A

                def _g_of_e(e):
                    p_e = (self.gamma - 1.0) * jnp.maximum(A * e, self.eps)
                    T_e = _stable_temperature_kelvin(
                        rho_cgs, p_e, mu=self.mu, mH_cgs=self.mH_cgs,
                        kB_cgs=self.kB_cgs, rho_unit_cgs=self.rho_unit_cgs,
                        p_unit_cgs=self.p_unit_cgs, eps=self.eps)
                    return _dotE_of_T(T_e) * dtA

                g, kdt = jax.jvp(_g_of_e, (e_cur,), (jnp.ones_like(e_cur),))
                kdt = jnp.clip(kdt, -50.0, 50.0)
                kdt_safe = jnp.where(jnp.abs(kdt) > 1.0e-4, kdt, 1.0)
                phi = jnp.where(jnp.abs(kdt) > 1.0e-4, jnp.expm1(kdt) / kdt_safe, 1.0 + 0.5 * kdt)
                Et_new = A * jnp.maximum(e_cur + g * phi, Et_floor / A)
            elif exponential:
                # Exponential (analytic-relaxation) sub-step of the locally
                # linearized cooling ODE dEt/dt = dotE0 + k*(Et-Et0):
                #   Et_new = Et + dotE*dt * (exp(k*dt)-1)/(k*dt),  k = d dotE/d Et.
                # k = (d dotE/dT) * (dT/dEt); T is linear in Et so dT/dEt = T/Et.
                # The factor (exp(k*dt)-1)/(k*dt) -> 1 as k->0 (recovers Euler) and
                # is bounded in (0,1) for k<0 (cooling) -> unconditionally stable,
                # bounded contractive gradient.
                dotE, ddotE_dT = jax.jvp(_dotE_of_T, (T_cur,), (jnp.ones_like(T_cur),))
                if self.stop_rate_grad:
                    dotE = jax.lax.stop_gradient(dotE)
                # The relaxation coefficient is a numerically-derived RATE, not part
                # of the differentiated state map: stop its gradient. This (a) makes
                # the reverse pass first-order (no second-order AD through the table
                # -> no NaN), and (b) yields exactly the desired smooth, bounded
                # backward: d Et_new/d inputs = d Et_cur + dt*phi*d dotE, with the
                # contractive phi in (0,1) for cooling damping the gradient.
                k = jax.lax.stop_gradient(ddotE_dT * T_cur / jnp.maximum(Et_cur, self.eps))
                kdt = jnp.clip(k * dt_sub, -50.0, 50.0)
                kdt_safe = jnp.where(jnp.abs(kdt) > 1.0e-4, kdt, 1.0)
                phi = jax.lax.stop_gradient(
                    jnp.where(jnp.abs(kdt) > 1.0e-4, jnp.expm1(kdt) / kdt_safe, 1.0 + 0.5 * kdt)
                )
                Et_new = jnp.maximum(Et_cur + dotE * dt_sub * phi, Et_floor)
            else:
                dotE = _dotE_of_T(T_cur)
                if self.stop_rate_grad:
                    dotE = jax.lax.stop_gradient(dotE)
                Et_new = jnp.maximum(Et_cur + dotE * dt_sub, Et_floor)
            return Et_new

        if exact_grad:
            # Unrolled + per-substep remat: reverse-mode through the subcycle
            # fori_loop stores every substep's internals (with the NN rate
            # surrogate that is a ~15 GiB single tensor at 64^3 x 128 steps);
            # checkpointing the substep body stores only Et and recomputes.
            body_ckpt = jax.checkpoint(lambda e: body(0, e))
            Et_final = Et
            for _ in range(self.subcycles):
                Et_final = body_ckpt(Et_final)
        elif self.integrator == "implicit_adjoint":
            # Unconditionally-stable backward-Euler sub-steps with the exact
            # implicit-function-theorem adjoint (exact + smooth gradient, no
            # stop_gradient surrogate). Unrolled so the per-substep custom_vjp
            # chain composes explicitly under reverse-mode.
            Et_cur = Et
            for _ in range(self.subcycles):
                Et_cur = self._implicit_energy_solve(Et_cur, rho_cgs, z, dt_sub, Et_floor)
            Et_final = Et_cur
            # Boundary finite-guard: away from the MAP (e.g. during unadjusted
            # MCLMC warmup exploration) a single cell can drive the implicit
            # solve non-finite; unadjusted dynamics have no accept/reject to
            # reject it, so one NaN poisons the whole field and the run dies.
            # Fall back that cell to the (finite) pre-cooling energy. `where`
            # cuts the NaN branch so no non-finite value or cotangent leaks
            # forward or backward from those cells.
            Et_final = jnp.where(jnp.isfinite(Et_final), Et_final, Et)
        else:
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
        # Final cell-wise finite-guard: any cell whose updated conservative
        # state came out non-finite falls back to the (finite) input state, so
        # a single bad cooling cell cannot poison the whole loss/gradient. This
        # matters for unadjusted MCLMC, which cannot reject a NaN proposal.
        finite_mask = jnp.all(jnp.isfinite(U_new), axis=0)
        U_new = jnp.where(finite_mask[jnp.newaxis, ...], U_new, U)
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
        integrator: str = "explicit",
        uvb_floor: bool = False,
        uvb_floor_delta_cut: float = 5.0,
        uvb_floor_width_dex: float = 0.3,
        uvb_floor_scale: float = 1.0,
    ):
        self.eq = equation_manager
        self.grackle = grackle_interp
        # "explicit" = forward-Euler subcycles (fast but STIFF-UNSTABLE for a correct
        # cooling table; the old clipped table masked this). Anything else uses the
        # unconditionally-stable implicit backward-Euler step with the exact IFT
        # adjoint (_nyx_implicit_energy_step), same as the Nyx force.
        self.integrator = str(integrator)
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

        # UVB/reionization temperature floor: precompute logT_eq(z, lognH) where the
        # PRIMORDIAL heat=cool (first downward crossing) on the table nodes. Constant
        # w.r.t. the state -> no gradient through the floor value itself; the floor
        # enters via jnp.maximum (clean subgradient) with a smooth overdensity gate.
        self.uvb_floor = bool(uvb_floor)
        self.uvb_floor_delta_cut = float(uvb_floor_delta_cut)
        self.uvb_floor_width_dex = float(max(uvb_floor_width_dex, 1.0e-3))
        self.uvb_floor_scale = float(uvb_floor_scale)
        if self.uvb_floor:
            import numpy as _np
            zN = _np.asarray(self.grackle.z_nodes, _np.float64)
            nN = _np.asarray(self.grackle.lognH_nodes, _np.float64)
            tN = _np.asarray(self.grackle.logT_nodes, _np.float64)
            heat = _np.asarray(getattr(self.grackle, "heat_prim_cgs",
                                        getattr(self.grackle, "heat_prim")), _np.float64)
            cool = _np.asarray(getattr(self.grackle, "cool_prim_cgs",
                                       getattr(self.grackle, "cool_prim")), _np.float64)
            logTeq = _np.full((len(zN), len(nN)), tN[0])
            for i in range(len(zN)):
                for j in range(len(nN)):
                    net = heat[i, j, :] - cool[i, j, :]
                    s = _np.sign(net)
                    cr = _np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
                    if cr.size:
                        k = cr[0]
                        f = net[k] / (net[k] - net[k + 1] + 1e-300)
                        logTeq[i, j] = tN[k] + f * (tN[k + 1] - tN[k])
                    elif _np.all(s >= 0):
                        logTeq[i, j] = tN[-1]
            self._uvb_z_nodes = jnp.asarray(zN, dtype=jnp.float32)
            self._uvb_lognH_nodes = jnp.asarray(nN, dtype=jnp.float32)
            self._uvb_logTeq = jnp.asarray(logTeq, dtype=jnp.float32)

    def _uvb_floor_temp_k(self, nH_cgs, rho_code, z):
        """Per-cell UVB floor temperature: uvb_floor_scale * T_eq(nH, z), smoothly
        gated off above uvb_floor_delta_cut in overdensity (dense/metal-cooled gas
        untouched). Never below the global temp_floor_k."""
        zc = jnp.clip(jnp.asarray(z, dtype=jnp.float32),
                      self._uvb_z_nodes[0], self._uvb_z_nodes[-1])
        iz = jnp.clip(jnp.searchsorted(self._uvb_z_nodes, zc) - 1,
                      0, self._uvb_z_nodes.shape[0] - 2)
        w = (zc - self._uvb_z_nodes[iz]) / jnp.maximum(
            self._uvb_z_nodes[iz + 1] - self._uvb_z_nodes[iz], 1e-6)
        teq_row = (1.0 - w) * self._uvb_logTeq[iz] + w * self._uvb_logTeq[iz + 1]
        lognH = jnp.log10(jnp.maximum(nH_cgs, 1.0e-30))
        logTeq = jnp.interp(lognH, self._uvb_lognH_nodes, teq_row)
        delta = rho_code / (jnp.mean(rho_code) + 1.0e-30)
        gate = jax.nn.sigmoid(
            (jnp.log10(jnp.asarray(self.uvb_floor_delta_cut, dtype=jnp.float32))
             - jnp.log10(jnp.maximum(delta, 1.0e-12))) / self.uvb_floor_width_dex)
        t_eq = jnp.power(10.0, logTeq) * self.uvb_floor_scale
        return jnp.maximum(self.temp_floor_k, t_eq * gate)

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
        t_kelvin = _stable_temperature_kelvin(
            rho_cgs, p_cgs, mu=jnp.maximum(mu_eff, 1.0e-8), mH_cgs=self.mH_cgs,
            kB_cgs=self.kB_cgs, rho_unit_cgs=self.rho_unit_cgs,
            p_unit_cgs=self.p_unit_cgs, eps=self.eps)
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
        # Floor temperature: global scalar, or per-cell UVB floor T_eq(nH,z) when
        # enabled (uvb_floor). Same energy-floor formula either way.
        _t_floor = (self._uvb_floor_temp_k(nH_cgs, rho_code, z) if self.uvb_floor
                    else self.temp_floor_k)
        Et_floor = rho_cgs * self.kB_cgs * _t_floor / (
            jnp.maximum(mu_eff, 1.0e-8) * self.mH_cgs * (self.gamma - 1.0)
        )

        # Trainable cooling/heating: multipliers from feedback_hyper (default 1.0 =
        # identity, so a fresh run reproduces the fixed-rate model exactly).
        _hyper = params.get("feedback_hyper", {})
        cool_scale = self.cooling_rate_scale * jnp.asarray(
            _hyper.get("cooling_rate_mult", 1.0), dtype=jnp.float32)
        heat_scale = self.heating_rate_scale * jnp.asarray(
            _hyper.get("heating_rate_mult", 1.0), dtype=jnp.float32)

        # Exponential (analytic-relaxation) integrator, ported from the Nyx force:
        # net rate as a function of T at fixed rho, differentiated w.r.t. T so the
        # local relaxation coefficient k = (d dotE/dT)*(dT/dEt) is known. Uses the
        # fixed mu_eff for E<->T (use_mu_table refinement dropped here, as in the
        # implicit branch). "exponential_exact" shares the identical forward value.
        exponential = self.integrator in ("exponential", "exponential_exact")

        def _dotE_of_T(T):
            # NO stop_gradient here: the exponential branch needs the REAL d(dotE)/dT
            # from jax.jvp to form the relaxation coefficient k. (Applying stop_gradient
            # inside this closure zeros the jvp tangent -> k=0 -> phi=1 -> the update
            # silently degenerates to explicit Euler.) The outer-gradient detachment is
            # applied AFTER the jvp (to dotE and k below), mirroring the Nyx force.
            heat_prim, cool_prim, cool_metal_solar, _ = self.grackle.evaluate(nH_cgs, T, z)
            d = (heat_scale * heat_prim - cool_scale * cool_prim
                 - cool_scale * self.metal_cooling_scale * z_ratio * cool_metal_solar)
            return jnp.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        def body(_, Et_cur):
            p_cgs_cur = (self.gamma - 1.0) * jnp.maximum(Et_cur, self.eps)
            temp_cur = self._temperature_from_mu(rho_cgs, p_cgs_cur, mu_eff)
            if exponential:
                # dEt/dt = dotE0 + k*(Et-Et0);  Et_new = Et + dotE*dt*(e^{kdt}-1)/(kdt).
                # k = (d dotE/dT)*(T/Et) since T is linear in Et. For cooling (k<0) the
                # factor phi in (0,1) -> unconditionally stable, energy asymptotes to
                # equilibrium and can never overshoot negative (kills the explicit
                # runaway collapse). Recovers Euler as k->0. k and phi are RATES, not
                # part of the differentiated state map -> stop_gradient (first-order,
                # bounded contractive backward; no 2nd-order AD through the table).
                dotE, ddotE_dT = jax.jvp(_dotE_of_T, (temp_cur,), (jnp.ones_like(temp_cur),))
                if self.stop_rate_grad:
                    dotE = jax.lax.stop_gradient(dotE)
                k = jax.lax.stop_gradient(ddotE_dT * temp_cur / jnp.maximum(Et_cur, self.eps))
                kdt = jnp.clip(k * dt_sub, -50.0, 50.0)
                kdt_safe = jnp.where(jnp.abs(kdt) > 1.0e-4, kdt, 1.0)
                phi = jax.lax.stop_gradient(
                    jnp.where(jnp.abs(kdt) > 1.0e-4, jnp.expm1(kdt) / kdt_safe, 1.0 + 0.5 * kdt))
                Et_new = jnp.maximum(Et_cur + dotE * dt_sub * phi, Et_floor)
                return Et_new
            heat_prim, cool_prim, cool_metal_solar, mu_tab = self.grackle.evaluate(nH_cgs, temp_cur, z)
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

        if self.integrator in ("explicit", "exponential", "exponential_exact"):
            Et_final = jax.lax.fori_loop(0, self.subcycles, body, Et)
        else:
            # Unconditionally-stable implicit backward-Euler subcycles (exact IFT
            # adjoint via _nyx_implicit_energy_step). rate_fn(E,rho,z) recovers T
            # from E and reads the table; nH follows the perturbed rho so the
            # adjoint stays consistent. use_mu_table refinement dropped here (fixed
            # mu_eff for E<->T). Fixes the explicit-Euler stiff instability that a
            # correct (unclipped) cooling table exposes.
            def rate_fn(E, rcgs, zz):
                rsafe = jnp.maximum(rcgs, self.eps)
                nH = self.grackle.h_species * rsafe / self.mH_cgs
                T = (self.gamma - 1.0) * E * (mu_eff * self.mH_cgs) / (self.kB_cgs * rsafe)
                T = jnp.maximum(T, self.temp_floor_k)
                hp, cp, cm, _ = self.grackle.evaluate(nH, T, zz)
                if self.stop_rate_grad:
                    hp = jax.lax.stop_gradient(hp); cp = jax.lax.stop_gradient(cp)
                    cm = jax.lax.stop_gradient(cm)
                dE = (heat_scale * hp - cool_scale * cp
                      - cool_scale * self.metal_cooling_scale * z_ratio * cm)
                return jnp.nan_to_num(dE, nan=0.0, posinf=0.0, neginf=0.0)
            Et_final = Et
            for _ in range(self.subcycles):
                Et_final = jnp.maximum(
                    _nyx_implicit_energy_step(rate_fn, Et_final, rho_cgs, z, dt_sub, Et_floor),
                    Et_floor)
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
        if getattr(cfg, "cooling_nn_npz", None):
            from .nn_cooling import NNCoolingEvaluator

            nyx_interp = NNCoolingEvaluator(str(cfg.cooling_nn_npz))
            print(f"[cooling] NN rate surrogate active: {cfg.cooling_nn_npz}", flush=True)
        else:
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
            integrator=str(cfg.cooling_integrator),
            smooth_table=bool(cfg.cooling_smooth_table),
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
        # Domain-coverage guard: the interpolator clips out-of-range (nH,T,z) to the
        # table edge, which SILENTLY gives voids/halo gas the boundary rate. Warn
        # loudly when the loaded table under-covers the configured domain so the
        # clip-extrapolation failure mode is visible (see the IGM-temperature audit).
        _cov = []
        if gsum['lognH_min'] > float(cfg.grackle_lognH_min) + 1e-6:
            _cov.append(f"lognH_min {gsum['lognH_min']:.3g} > cfg {cfg.grackle_lognH_min:.3g}")
        if gsum['lognH_max'] < float(cfg.grackle_lognH_max) - 1e-6:
            _cov.append(f"lognH_max {gsum['lognH_max']:.3g} < cfg {cfg.grackle_lognH_max:.3g}")
        if gsum['logT_min'] > float(cfg.grackle_logt_min) + 1e-6:
            _cov.append(f"logT_min {gsum['logT_min']:.3g} > cfg {cfg.grackle_logt_min:.3g}")
        if gsum['logT_max'] < float(cfg.grackle_logt_max) - 1e-6:
            _cov.append(f"logT_max {gsum['logT_max']:.3g} < cfg {cfg.grackle_logt_max:.3g}")
        if _cov:
            print("[WARN] Grackle table UNDER-COVERS the configured domain; out-of-range "
                  "cells get clip-to-boundary rates: " + "; ".join(_cov), flush=True)
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
            integrator=str(cfg.cooling_integrator),
            uvb_floor=bool(getattr(cfg, "uvb_temp_floor", False)),
            uvb_floor_delta_cut=float(getattr(cfg, "uvb_temp_floor_delta_cut", 5.0)),
            uvb_floor_width_dex=float(getattr(cfg, "uvb_temp_floor_width_dex", 0.3)),
            uvb_floor_scale=float(getattr(cfg, "uvb_temp_floor_scale", 1.0)),
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
    _de_smooth = float(getattr(cfg, "dual_energy_smooth", 0.0))
    if _de_smooth > 0.0:
        # C-inf sigmoid blend of the dual-energy pressure selector (relative
        # window width around eta); removes the value-discontinuous where()
        # that breaks force-vs-value consistency for MH samplers.
        base_eq.dual_energy_smooth_width = _de_smooth
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

    # Unified-mesh mode (Option C'): hydro runs its shard_map on the SAME 2-axis
    # ('x','y') mesh as the jaxdecomp gravity, with field spec P(None,'x','y',None)
    # and pmesh (pd0, pd1, 1). One mesh across the whole step -> the rollout can be
    # jitted/scanned/checkpointed as a single traced program (distributed BPTT).
    hydro_mesh = None
    hydro_field_spec = None
    pmesh_shape = cfg.pmesh_shape
    if bool(getattr(cfg, "unified_mesh", False)):
        if str(getattr(cfg, "gravity_backend", "jaxpm")) != "jaxdecomp_fft" or field_sharding is None:
            raise ValueError("unified_mesh=True requires gravity_backend='jaxdecomp_fft' and field_sharding")
        from jax.sharding import PartitionSpec as _PSpec
        hydro_mesh = field_sharding.mesh
        hydro_field_spec = _PSpec(None, "x", "y", None)
        pd = tuple(int(s) for s in hydro_mesh.devices.shape)
        pmesh_shape = (pd[0], pd[1], 1)

    sim = _build_hydrosim(
        eq,
        force_stack,
        solver_name=cfg.solver,
        limiter=str(getattr(cfg, "hydro_limiter", "MINMOD")),
        riemann_smooth=float(getattr(cfg, "hydro_riemann_smooth", 0.0)),
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
        pmesh_shape=pmesh_shape,
        mesh=hydro_mesh,
        field_spec=hydro_field_spec,
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


def _apply_state_ceilings(
    U: jnp.ndarray,
    params: dict,
    system: FullHydroSystem,
    cfg: FullHydroConfig,
) -> jnp.ndarray:
    """Cap per-cell temperature and peculiar velocity so a handful of numerically
    over-collapsed cells can't dominate the global CFL timestep (or diverge). Bounds the
    per-cell signal speed |v|+c_s -> bounds max(inv_dt) -> the CFL-limited dtau stops
    collapsing. Physical and conservative: the caps sit far above the real WHIM, touching
    only runaway cells. No-op when both knobs are 0 (so default behavior is unchanged)."""
    t_ceil_k = float(getattr(cfg, "hydro_temp_ceiling_k", 0.0))
    v_ceil_kms = float(getattr(cfg, "hydro_velocity_ceiling_kms", 0.0))
    if t_ceil_k <= 0.0 and v_ceil_kms <= 0.0:
        return U
    w = system.eq.get_primitives_from_conservatives(U)
    a = jnp.asarray(params.get("a", 1.0), dtype=jnp.float32)
    rho_safe = jnp.maximum(w[0], jnp.asarray(system.eq.eps, dtype=w.dtype))
    if t_ceil_k > 0.0:
        # temperature ceiling -> pressure ceiling (mirror of the floor in _project_..._floors)
        t_ceil_code = jnp.asarray(float(system.kelvin_to_code_temp) * t_ceil_k, dtype=w.dtype)
        rho_phys = cosmo_conv.density_code_to_phys(rho_safe, a)
        p_ceil_phys = rho_phys * jnp.asarray(float(system.eq.R), dtype=w.dtype) * t_ceil_code
        p_ceil_code = cosmo_conv.pressure_phys_to_code(p_ceil_phys, a)
        w = w.at[4].set(jnp.minimum(w[4], p_ceil_code))
        # keep the dual-energy channel (rho*e_int passive) consistent with the capped pressure,
        # else the dual-energy recovery would re-expose the uncapped internal energy.
        i_dual = getattr(system.eq, "dual_energy_ids", None)
        if i_dual is not None and int(i_dual) < int(w.shape[0]):
            rhoe_ceil = rho_safe * system.eq.get_specific_energy(p_ceil_code, rho_safe)
            w = w.at[int(i_dual)].set(jnp.minimum(w[int(i_dual)], rhoe_ceil))
    if v_ceil_kms > 0.0:
        v_ceil_code = jnp.abs(cosmo_conv.velocity_phys_to_code(
            jnp.asarray(v_ceil_kms, dtype=w.dtype), a))
        for i_v in system.eq.vel_ids:
            w = w.at[int(i_v)].set(jnp.clip(w[int(i_v)], -v_ceil_code, v_ceil_code))
    return system.eq.get_conservatives_from_primitives(w)


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
        # Hard floor on the step (caps the max step count). The state ceilings keep this
        # floored step CFL-stable by bounding the per-cell signal speed. Never overshoot a_target.
        dtau_lo = max(float(cfg.dtau_min), float(getattr(cfg, "dtau_floor", 0.0)))
        dtau = jnp.maximum(jnp.minimum(dtau, dtau_to_target), dtau_lo)
        # Diagnostic: the dtau the pure a-schedule would have taken this step (so the driver
        # can report whether/how much the CFL cap is throttling the step, esp. at early time).
        _dtau_aschedule = jnp.clip(
            (a_target - a_now) / jnp.maximum(da_dtau * remaining, 1.0e-12),
            float(cfg.dtau_min), float(cfg.dtau_max))
        _dtau_diag = {"_dtau_cfl": dtau_cfl, "_dtau_aschedule": _dtau_aschedule}
    else:
        _dtau_diag = None
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
    # Cap runaway T / |v| so a few over-collapsed cells can't dominate the next step's CFL
    # timestep (or diverge). No-op unless hydro_temp_ceiling_k / velocity ceiling are set.
    U_new = _apply_state_ceilings(U_new, params_new, system, cfg)
    if _dtau_diag is not None:
        # eager (adaptive) path only — never taken under the scan carry, so no key mismatch
        params_new = {**params_new, **_dtau_diag}
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
