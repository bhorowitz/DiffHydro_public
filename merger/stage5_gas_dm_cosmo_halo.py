"""
stage5_gas_dm_cosmo_halo.py
===========================
Stage 5: Gas+DM cosmological evolution of a static isolated halo.

This builds the single-halo gas profile from the Stage 2 HSE model, converts it
to supercomoving variables, couples it to live DM particles, and writes
diagnostic plots aimed at checking cosmological hydro stability.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import jax

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp

import diffhydro as dh
from diffhydro.cosmology import (
    BackgroundExpansionForce,
    JaxPMCoupledGravityForce,
    LCDMBackground,
    SuperComovingEquationManager,
)
from diffhydro.cosmology import conversions as cosmo_conv
from diffhydro.equationmanager import EquationManager

from merger.halo_reference import HaloReference
from merger.halo_gas_ic import build_stage2_gas_grids, shell_profile_from_grid

try:
    from jaxpm.pm import cic_paint
except Exception:  # pragma: no cover
    cic_paint = None


OUT_DIR = os.path.join(_HERE, "outputs", "stage5")
os.makedirs(OUT_DIR, exist_ok=True)

GAMMA = 5.0 / 3.0
Z_INIT = 0.295
M200 = 5.0e14
CONC = 3.5
PRESSURE_SCALE = 0.01
H0_KM_S_MPC = 70.0
OMEGA_M = 0.3
OMEGA_B = 0.0486
OMEGA_L = 0.7
KM_S_TO_KPC_MYR = 1.0227121650537077e-3
G_PHYS = 4.3009172706e-6 * KM_S_TO_KPC_MYR**2


def parse_args():
    p = argparse.ArgumentParser(description="Stage 5 gas+DM cosmological single-halo validation")
    p.add_argument("--n-par", type=int, default=4096, help="Number of DM particles")
    p.add_argument("--n-grid", type=int, default=32, help="Mesh resolution")
    p.add_argument("--a-final", type=float, default=0.80, help="Final scale factor")
    p.add_argument("--max-steps", type=int, default=80, help="Maximum number of hydro+DM steps")
    p.add_argument("--snapshot-every", type=int, default=10, help="Snapshot cadence in steps")
    p.add_argument("--a-tol", type=float, default=1.0e-6, help="Stop once a is within this tolerance of a_final")
    p.add_argument("--max-dtau", type=float, default=1.0, help="Maximum supercomoving timestep")
    p.add_argument("--min-dtau", type=float, default=1.0e-3, help="Minimum supercomoving timestep")
    p.add_argument("--dtau-safety", type=float, default=0.8, help="Safety factor on the remaining-a timestep estimate")
    p.add_argument("--pressure-scale", type=float, default=PRESSURE_SCALE, help="Scale factor applied to the physical HSE pressure profile")
    p.add_argument("--velocity-scale", type=float, default=0.10, help="Rescaling of the initial DM peculiar velocities")
    p.add_argument("--cancel-hubble-flow", action="store_true", help="Initialize gas and DM peculiar velocities to cancel local Hubble flow for a static physical halo")
    p.add_argument("--dm-kick-scale", type=float, default=1.0, help="Rescaling of the DM kick prefactor")
    p.add_argument("--gas-kick-scale", type=float, default=1.0, help="Rescaling of the gas kick prefactor")
    p.add_argument("--gravity-method", type=str, default="direct", choices=["direct", "jaxpm"], help="Gravity solver for the live DM component")
    p.add_argument("--softening-cells", type=float, default=0.5, help="Plummer-like softening length for direct gravity, in comoving cell units")
    p.add_argument("--output-subdir", type=str, default=None, help="Optional run subdirectory inside merger/outputs/stage5")
    p.add_argument("--r-max", type=float, default=None, help="Particle truncation radius [kpc]")
    p.add_argument("--l-box", type=float, default=None, help="Initial physical box size [kpc]")
    return p.parse_args()


def hubble_to_myr_inv(h0_km_s_mpc: float) -> float:
    return float(h0_km_s_mpc) * KM_S_TO_KPC_MYR / 1000.0


class DirectNBodyCosmoForce:
    """Direct-summation DM force in the Stage 5 supercomoving convention."""

    def __init__(
        self,
        *,
        bg_h0,
        dx_com,
        n_grid,
        softening_cells=0.5,
        gas_grid_x_com=None,
        gas_grid_y_com=None,
        gas_grid_z_com=None,
        gas_center_com=None,
        gas_radius_table=None,
        gas_g_table=None,
        gas_eps=1.0e-20,
    ):
        self.bg_h0 = jnp.asarray(bg_h0, dtype=jnp.float32)
        self.dx_com = jnp.asarray(dx_com, dtype=jnp.float32)
        self.n_grid = float(n_grid)
        self.softening_cells = jnp.asarray(softening_cells, dtype=jnp.float32)
        self.gas_grid_x_com = None if gas_grid_x_com is None else jnp.asarray(gas_grid_x_com, dtype=jnp.float32)
        self.gas_grid_y_com = None if gas_grid_y_com is None else jnp.asarray(gas_grid_y_com, dtype=jnp.float32)
        self.gas_grid_z_com = None if gas_grid_z_com is None else jnp.asarray(gas_grid_z_com, dtype=jnp.float32)
        self.gas_center_com = None if gas_center_com is None else jnp.asarray(gas_center_com, dtype=jnp.float32)
        self.gas_radius_table = None if gas_radius_table is None else jnp.asarray(gas_radius_table, dtype=jnp.float32)
        self.gas_g_table = None if gas_g_table is None else jnp.asarray(gas_g_table, dtype=jnp.float32)
        self.gas_eps = float(gas_eps)

    def timestep(self, U):
        del U
        return jnp.asarray(1.0e10, dtype=jnp.float32)

    def force(self, i, U_gas, params, dtau):
        del i
        dtau = jnp.maximum(jnp.asarray(dtau), 0.0)
        dtau_half = 0.5 * dtau

        params_out = dict(params)
        a = jnp.asarray(params_out.get("a", 1.0), dtype=jnp.float32)
        dm_params = dict(params_out["dm"])
        U_out = U_gas

        if self.gas_grid_x_com is not None:
            U_out = self._apply_gas_gravity(U_out, a, dtau)

        x = jnp.asarray(dm_params["x"], dtype=jnp.float32)
        p = jnp.asarray(dm_params["p_or_v"], dtype=jnp.float32)
        m = jnp.asarray(dm_params.get("m_par", dm_params.get("mass", 1.0)), dtype=jnp.float32)
        if m.ndim == 0:
            m = jnp.ones((x.shape[0],), dtype=jnp.float32) * m

        accel_p_old = self._canonical_accel(x, m, a)
        p_half = p + dtau_half * accel_p_old
        x_drift = jnp.mod(x + dtau * self.bg_h0 * p_half, self.n_grid)

        accel_p_new = self._canonical_accel(x_drift, m, a)
        p_new = p_half + dtau_half * accel_p_new

        dm_params["x"] = x_drift
        dm_params["p_or_v"] = p_new
        params_out["dm"] = dm_params
        return U_out, params_out

    def _canonical_accel(self, x_cells, masses, a):
        delta = x_cells[:, None, :] - x_cells[None, :, :]
        delta = (delta + 0.5 * self.n_grid) % self.n_grid - 0.5 * self.n_grid
        r2_cells = jnp.sum(delta * delta, axis=-1)
        eps2 = self.softening_cells * self.softening_cells
        mask = 1.0 - jnp.eye(x_cells.shape[0], dtype=jnp.float32)
        inv_r3 = mask * jax.lax.rsqrt(jnp.maximum(r2_cells + eps2, 1.0e-12)) ** 3
        length2 = jnp.maximum((a * self.dx_com) ** 2, 1.0e-20)
        pair = -G_PHYS * masses[None, :, None] * delta * (inv_r3[..., None] / length2)
        g_phys = jnp.sum(pair, axis=1)
        return (a * a / (self.dx_com * self.bg_h0)) * g_phys

    def _apply_gas_gravity(self, U, a, dtau):
        rho = jnp.asarray(U[0], dtype=jnp.float32)

        dx_com = self.gas_grid_x_com - self.gas_center_com[0]
        dy_com = self.gas_grid_y_com - self.gas_center_com[1]
        dz_com = self.gas_grid_z_com - self.gas_center_com[2]
        r_com = jnp.sqrt(dx_com**2 + dy_com**2 + dz_com**2)
        r_phys = a * r_com
        g_r = jnp.interp(
            r_phys.reshape((-1,)),
            self.gas_radius_table,
            self.gas_g_table,
            left=self.gas_g_table[0],
            right=0.0,
        ).reshape(r_phys.shape)
        inv_r = 1.0 / jnp.maximum(r_com, 1.0e-30)
        gx_phys = jnp.where(r_com > 0.0, g_r * dx_com * inv_r, 0.0)
        gy_phys = jnp.where(r_com > 0.0, g_r * dy_com * inv_r, 0.0)
        gz_phys = jnp.where(r_com > 0.0, g_r * dz_com * inv_r, 0.0)

        ax_code = (a**3) * gx_phys
        ay_code = (a**3) * gy_phys
        az_code = (a**3) * gz_phys

        rho_safe = jnp.maximum(rho, self.gas_eps)
        rho_kin = jnp.maximum(rho_safe, 1.0e-10)
        mx_old, my_old, mz_old, E_old = U[1], U[2], U[3], U[4]

        dmx = rho * ax_code * dtau
        dmy = rho * ay_code * dtau
        dmz = rho * az_code * dtau
        mx_new = mx_old + dmx
        my_new = my_old + dmy
        mz_new = mz_old + dmz

        kin_old = 0.5 * (mx_old * mx_old + my_old * my_old + mz_old * mz_old) / rho_kin
        kin_new = 0.5 * (mx_new * mx_new + my_new * my_new + mz_new * mz_new) / rho_kin
        eint = jnp.maximum(E_old - kin_old, self.gas_eps)
        E_new = jnp.maximum(eint + kin_new, kin_new + self.gas_eps)

        return U.at[1].set(mx_new).at[2].set(my_new).at[3].set(mz_new).at[4].set(E_new)


def temperature_proxy(rho, p):
    return p / np.maximum(rho, 1.0e-30)


def entropy_proxy(rho, p):
    return p / np.maximum(rho, 1.0e-30) ** GAMMA


def primitives_from_U(eq, U):
    W = np.asarray(eq.get_primitives_from_conservatives(jnp.asarray(U)))
    return {"rho": W[0], "vx": W[1], "vy": W[2], "vz": W[3], "p": W[4]}


def minimal_image(delta_cells, n_grid):
    return (delta_cells + 0.5 * n_grid) % n_grid - 0.5 * n_grid


def build_initial_state(ref, args):
    a_init = 1.0 / (1.0 + Z_INIT)
    r_max = args.r_max if args.r_max else 2.5 * ref.r200
    l_box_phys = args.l_box if args.l_box else 4.0 * ref.r200
    l_box_com = l_box_phys / a_init
    dx_com = l_box_com / args.n_grid
    dx_phys = l_box_phys / args.n_grid

    bg = LCDMBackground(
        h=H0_KM_S_MPC / 100.0,
        Omega_m=OMEGA_M,
        Omega_b=OMEGA_B,
        Omega_lambda=OMEGA_L,
        Omega_k=0.0,
        use_jax_cosmo=True,
    )
    bg.H0 = hubble_to_myr_inv(H0_KM_S_MPC)

    base_eq = EquationManager()
    base_eq.gamma = GAMMA
    base_eq.mesh_shape = [args.n_grid, args.n_grid, args.n_grid]
    base_eq.box_size = (l_box_com, l_box_com, l_box_com)
    eq = SuperComovingEquationManager(base_eq, enforce_gamma_53=True)

    grids = build_stage2_gas_grids(ref, args.n_grid, l_box_phys)
    grids["p_g_1d"] = grids["profiles"]["pressure"] * float(args.pressure_scale)
    rho_phys = grids["rho_g"].astype(np.float32)
    p_phys = (grids["p_g"] * float(args.pressure_scale)).astype(np.float32)
    if args.cancel_hubble_flow:
        H_init = float(bg.H(a_init))
        vx_gas_phys = (-H_init * (grids["X"] - grids["center"][0])).astype(np.float32)
        vy_gas_phys = (-H_init * (grids["Y"] - grids["center"][1])).astype(np.float32)
        vz_gas_phys = (-H_init * (grids["Z"] - grids["center"][2])).astype(np.float32)
    else:
        vx_gas_phys = np.zeros_like(rho_phys)
        vy_gas_phys = np.zeros_like(rho_phys)
        vz_gas_phys = np.zeros_like(rho_phys)
    W_code = eq.physical_primitives_to_code(
        jnp.asarray(rho_phys),
        jnp.asarray(vx_gas_phys),
        jnp.asarray(vy_gas_phys),
        jnp.asarray(vz_gas_phys),
        jnp.asarray(p_phys),
        jnp.asarray(a_init, dtype=jnp.float32),
    )
    U0 = eq.get_conservatives_from_primitives(W_code)

    np.random.seed(42)
    particles = ref.hse.generate_dm_particles(args.n_par, r_max=r_max)
    pos_phys = np.array(particles[("dm", "particle_position")], dtype=np.float32)
    vel_phys = np.array(particles[("dm", "particle_velocity")], dtype=np.float32) * np.float32(args.velocity_scale)
    m_par = float(np.array(particles[("dm", "particle_mass")])[0])

    center_phys = np.full(3, 0.5 * l_box_phys, dtype=np.float32)
    center_cells = center_phys / dx_phys
    if args.cancel_hubble_flow:
        H_init = float(bg.H(a_init))
        vel_phys = vel_phys - H_init * pos_phys
    pos_cells = (pos_phys + center_phys[None, :]) / dx_phys
    vel_tilde_cells = (a_init * vel_phys) / dx_com
    p_or_v = vel_tilde_cells / bg.H0

    rho_dm_phys = np.clip(grids["rho_total"] - grids["rho_g"] - grids["rho_star"], 0.0, None)
    rho_dm_code = rho_dm_phys * a_init**3
    mean_target_dm = float(np.mean(rho_dm_code))
    mean_unit_deposit = float(args.n_par) / float(args.n_grid**3)
    dm_particle_weight = mean_target_dm / max(mean_unit_deposit, 1.0e-30)

    params0 = {
        "a": jnp.asarray(a_init, dtype=jnp.float32),
        "dm": {
            "x": jnp.asarray(pos_cells, dtype=jnp.float32),
            "p_or_v": jnp.asarray(p_or_v, dtype=jnp.float32),
            "mass": jnp.ones((args.n_par,), dtype=jnp.float32) * np.float32(dm_particle_weight),
            "m_par": jnp.asarray(m_par, dtype=jnp.float32),
            "drift_factor": jnp.asarray(bg.H0, dtype=jnp.float32),
            "kick_prefactor": jnp.asarray(1.5 * OMEGA_M * bg.H0 * float(args.dm_kick_scale), dtype=jnp.float32),
            "gas_kick_prefactor": jnp.asarray(1.5 * OMEGA_M * (bg.H0**2) * float(args.gas_kick_scale), dtype=jnp.float32),
        },
    }
    meta = {
        "a_init": float(a_init),
        "r_max": float(r_max),
        "l_box_phys": float(l_box_phys),
        "l_box_com": float(l_box_com),
        "dx_com": float(dx_com),
        "dx_phys": float(dx_phys),
        "center_cells": center_cells.astype(np.float32),
        "center_phys": center_phys.astype(np.float32),
        "m_par": float(m_par),
        "dm_particle_weight": float(dm_particle_weight),
        "grids": grids,
        "cancel_hubble_flow": bool(args.cancel_hubble_flow),
    }
    return eq, bg, U0, params0, meta


def make_sim(eq, bg, mesh_shape, *, gravity_method="direct", dx_com=None, softening_cells=0.5, gas_gravity_meta=None):
    solver = dh.HLLC(equation_manager=eq, signal_speed=dh.signal_speed_Rusanov)
    flux = dh.ConvectiveFlux(eq, solver, dh.MUSCL3(limiter="VANLEER"), positivity=True)
    if dx_com is None:
        raise ValueError("dx_com is required for Stage 5 hydro flux spacing")
    flux.dx_o = float(dx_com)
    bg_force = BackgroundExpansionForce(bg, a_init=1.0)
    if gravity_method == "direct":
        if dx_com is None:
            raise ValueError("dx_com is required for gravity_method='direct'")
        grav_force = DirectNBodyCosmoForce(
            bg_h0=bg.H0,
            dx_com=dx_com,
            n_grid=mesh_shape[0],
            softening_cells=softening_cells,
            gas_grid_x_com=None if gas_gravity_meta is None else gas_gravity_meta["x_com"],
            gas_grid_y_com=None if gas_gravity_meta is None else gas_gravity_meta["y_com"],
            gas_grid_z_com=None if gas_gravity_meta is None else gas_gravity_meta["z_com"],
            gas_center_com=None if gas_gravity_meta is None else gas_gravity_meta["center_com"],
            gas_radius_table=None if gas_gravity_meta is None else gas_gravity_meta["radius_table"],
            gas_g_table=None if gas_gravity_meta is None else gas_gravity_meta["g_table"],
        )
    else:
        grav_force = JaxPMCoupledGravityForce(
            eq,
            mesh_shape=mesh_shape,
            subtract_mean=True,
            use_jaxpm=True,
            dm_drift_factor=bg.H0,
            dm_kick_factor=1.0,
        )
    return dh.hydro(
        n_super_step=1,
        max_dt=0.2,
        fluxes=[flux],
        forces=[bg_force, grav_force],
        use_mol=True,
        pmesh_shape=(1, 1, 1),
        dx_o=float(dx_com),
    )


def snapshot_from_state(eq, U, params, meta, n_grid):
    a = float(params["a"])
    prim_code = primitives_from_U(eq, U)
    rho_phys = np.asarray(cosmo_conv.density_code_to_phys(jnp.asarray(prim_code["rho"]), a))
    p_phys = np.asarray(cosmo_conv.pressure_code_to_phys(jnp.asarray(prim_code["p"]), a))
    vx_phys = np.asarray(cosmo_conv.velocity_code_to_phys(jnp.asarray(prim_code["vx"]), a))
    vy_phys = np.asarray(cosmo_conv.velocity_code_to_phys(jnp.asarray(prim_code["vy"]), a))
    vz_phys = np.asarray(cosmo_conv.velocity_code_to_phys(jnp.asarray(prim_code["vz"]), a))

    pos_cells = np.asarray(params["dm"]["x"], dtype=np.float64)
    p_or_v = np.asarray(params["dm"]["p_or_v"], dtype=np.float64)
    delta_cells = minimal_image(pos_cells - meta["center_cells"][None, :], n_grid)
    pos_com = delta_cells * meta["dx_com"]
    pos_phys = pos_com * a
    vel_tilde_cells = p_or_v * meta["bg_h0"]
    vel_phys = vel_tilde_cells * meta["dx_com"] / a
    r_phys = np.sqrt(np.sum(pos_phys**2, axis=1))

    return {
        "a": a,
        "U": np.asarray(U),
        "rho_phys": rho_phys,
        "p_phys": p_phys,
        "T_phys": temperature_proxy(rho_phys, p_phys),
        "entropy_phys": entropy_proxy(rho_phys, p_phys),
        "vx_phys": vx_phys,
        "vy_phys": vy_phys,
        "vz_phys": vz_phys,
        "pos_phys": pos_phys,
        "pos_cells": pos_cells,
        "vel_phys_dm": vel_phys,
        "r_phys_dm": r_phys,
    }


def apply_state_floors(eq, U, a, rho_floor_phys=1.0e-10, p_floor_phys=1.0e-10, t_floor=1.0e-10):
    w = eq.get_primitives_from_conservatives(U)
    w = jnp.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    rho_floor = cosmo_conv.density_phys_to_code(jnp.asarray(rho_floor_phys, dtype=w.dtype), a)
    p_floor = cosmo_conv.pressure_phys_to_code(jnp.asarray(p_floor_phys, dtype=w.dtype), a)
    rho_phys = cosmo_conv.density_code_to_phys(w[0], a)
    p_floor_temp_phys = jnp.maximum(rho_phys, jnp.asarray(rho_floor_phys, dtype=w.dtype)) * jnp.asarray(eq.R * t_floor, dtype=w.dtype)
    p_floor_temp = cosmo_conv.pressure_phys_to_code(p_floor_temp_phys, a)
    p_floor = jnp.maximum(p_floor, p_floor_temp)
    w = w.at[0].set(jnp.maximum(w[0], rho_floor))
    w = w.at[4].set(jnp.maximum(w[4], p_floor))
    return eq.get_conservatives_from_primitives(w)


def paint_dm_density(positions_cells, weights, mesh_shape):
    if cic_paint is None:
        raise ImportError("jaxpm is required for the DM CIC diagnostic plots")
    mesh = jnp.zeros(mesh_shape, dtype=jnp.float32)
    return np.asarray(cic_paint(mesh, jnp.asarray(positions_cells, dtype=jnp.float32), weight=jnp.asarray(weights, dtype=jnp.float32)))


def current_physical_grid(meta, a):
    scale = float(a) / float(meta["a_init"])
    grids = meta["grids"]
    dx = float(meta["dx_phys"]) * scale
    x = (grids["X"] - grids["center"][0]) * scale
    y = (grids["Y"] - grids["center"][1]) * scale
    z = (grids["Z"] - grids["center"][2]) * scale
    r = grids["r3d"] * scale
    return dx, x, y, z, r


def choose_dtau(sim, U, bg, a_now, a_final, max_dtau, min_dtau, dtau_safety):
    da_dtau = float(bg.da_dtau(a_now))
    remaining = max(float(a_final) - float(a_now), 0.0)
    dtau_target = min(float(max_dtau), float(dtau_safety) * remaining / max(da_dtau, 1.0e-12))
    dtau_cfl = float(sim.timestep(U))
    dtau = min(dtau_cfl, dtau_target)
    if dtau_target > float(min_dtau):
        dtau = max(dtau, float(min_dtau))
        dtau = min(dtau, dtau_cfl, dtau_target)
    return max(dtau, min(dtau_cfl, dtau_target))


def validate_finite_state(U, params, where):
    U_np = np.asarray(U)
    a_val = float(params["a"])
    if not np.isfinite(U_np).all():
        raise FloatingPointError(f"Non-finite hydro state encountered at {where}.")
    if not np.isfinite(a_val):
        raise FloatingPointError(f"Non-finite scale factor encountered at {where}.")


def compute_snapshot_metrics(ref, snap, meta):
    grids = meta["grids"]
    prof = grids["profiles"]
    r_prof = prof["radius"]
    r_bins = np.logspace(np.log10(0.03 * ref.r200), np.log10(1.2 * ref.r200), 48)
    r_c = 0.5 * (r_bins[:-1] + r_bins[1:])
    dx_phys_now, x_phys, y_phys, z_phys, r_phys_grid = current_physical_grid(meta, snap["a"])

    rho_prof = shell_profile_from_grid(snap["rho_phys"], r_phys_grid, r_bins, statistic="mean")
    p_prof = shell_profile_from_grid(snap["p_phys"], r_phys_grid, r_bins, statistic="mean")
    ent_prof = shell_profile_from_grid(snap["entropy_phys"], r_phys_grid, r_bins, statistic="mean")
    temp_prof = shell_profile_from_grid(snap["T_phys"], r_phys_grid, r_bins, statistic="mean")

    dPdr = np.gradient(p_prof, r_c, edge_order=1)
    g_shell = np.interp(r_c, r_prof, prof["gravitational_field"])
    resid = (dPdr - rho_prof * g_shell) / (np.abs(dPdr) + np.abs(rho_prof * g_shell) + 1.0e-30)

    vx, vy, vz = snap["vx_phys"], snap["vy_phys"], snap["vz_phys"]
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    cs = np.sqrt(GAMMA * snap["p_phys"] / np.maximum(snap["rho_phys"], 1.0e-30))
    mach = vmag / np.maximum(cs, 1.0e-30)

    etherm = float(np.sum(snap["p_phys"] / (GAMMA - 1.0)) * dx_phys_now**3)
    ekin = float(0.5 * np.sum(snap["rho_phys"] * (vx**2 + vy**2 + vz**2)) * dx_phys_now**3)

    rr = np.maximum(r_phys_grid, 1.0e-20)
    vr_gas = (vx * x_phys + vy * y_phys + vz * z_phys) / rr
    shell_rho_vr = shell_profile_from_grid(snap["rho_phys"] * vr_gas, r_phys_grid, r_bins, statistic="mean")
    flux_r500 = 4.0 * np.pi * ref.r500**2 * np.interp(ref.r500, r_c, shell_rho_vr)
    flux_r200 = 4.0 * np.pi * ref.r200**2 * np.interp(ref.r200, r_c, shell_rho_vr)

    gas_shell_mass = shell_profile_from_grid(snap["rho_phys"] * dx_phys_now**3, r_phys_grid, r_bins, statistic="sum")
    gas_mass_cum = np.cumsum(gas_shell_mass)
    gas_r500 = float(np.interp(ref.r500, r_c, gas_mass_cum, left=gas_mass_cum[0], right=gas_mass_cum[-1]))
    dm_r500 = float(meta["m_par"] * np.sum(snap["r_phys_dm"] < ref.r500))
    star_r500 = float(np.interp(ref.r500, r_prof, prof["stellar_mass"]))
    gas_fraction_r500 = gas_r500 / max(gas_r500 + dm_r500 + star_r500, 1.0e-30)

    return {
        "a": float(snap["a"]),
        "hse_residual_rms": float(np.sqrt(np.nanmean(resid**2))),
        "hse_residual_p95": float(np.nanpercentile(np.abs(resid), 95.0)),
        "mach95": float(np.nanpercentile(mach, 95.0)),
        "kinetic_to_thermal": float(ekin / max(etherm, 1.0e-30)),
        "entropy_mean_r500": float(np.nanmean(ent_prof[r_c < ref.r500])),
        "gas_mass_flux_r500": float(flux_r500),
        "gas_mass_flux_r200": float(flux_r200),
        "gas_fraction_r500": float(gas_fraction_r500),
        "min_temperature": float(np.nanmin(snap["T_phys"])),
        "min_pressure": float(np.nanmin(snap["p_phys"])),
        "mean_temp_profile": float(np.nanmean(temp_prof)),
    }


def plot_profiles(ref, snapshots, meta, out_dir):
    grids = meta["grids"]
    prof = grids["profiles"]
    r_bins = np.logspace(np.log10(0.03 * ref.r200), np.log10(1.2 * prof["radius"][-1]), 60)
    r_c = 0.5 * (r_bins[:-1] + r_bins[1:])
    select_idx = np.unique(np.linspace(0, len(snapshots) - 1, min(4, len(snapshots))).astype(int))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()
    axes[0].loglog(prof["radius"], prof["gas_density"], "k--", lw=2, label="Target")
    axes[1].loglog(prof["radius"], grids["p_g_1d"], "k--", lw=2)
    axes[2].loglog(prof["radius"], temperature_proxy(prof["gas_density"], grids["p_g_1d"]), "k--", lw=2)
    axes[3].loglog(prof["radius"], entropy_proxy(prof["gas_density"], grids["p_g_1d"]), "k--", lw=2)

    for idx in select_idx:
        snap = snapshots[idx]
        dx_phys_now, x_phys, y_phys, z_phys, r_phys_grid = current_physical_grid(meta, snap["a"])
        rho = shell_profile_from_grid(snap["rho_phys"], r_phys_grid, r_bins, statistic="mean")
        p = shell_profile_from_grid(snap["p_phys"], r_phys_grid, r_bins, statistic="mean")
        t = shell_profile_from_grid(snap["T_phys"], r_phys_grid, r_bins, statistic="mean")
        s = shell_profile_from_grid(snap["entropy_phys"], r_phys_grid, r_bins, statistic="mean")
        vx, vy, vz = snap["vx_phys"], snap["vy_phys"], snap["vz_phys"]
        rr = np.maximum(r_phys_grid, 1.0e-20)
        vr = (vx * x_phys + vy * y_phys + vz * z_phys) / rr
        vr_prof = shell_profile_from_grid(vr, r_phys_grid, r_bins, statistic="mean")
        cs = np.sqrt(GAMMA * snap["p_phys"] / np.maximum(snap["rho_phys"], 1.0e-30))
        mach_prof = shell_profile_from_grid(np.sqrt(vx**2 + vy**2 + vz**2) / np.maximum(cs, 1.0e-30), r_phys_grid, r_bins, statistic="mean")
        gas_shell = shell_profile_from_grid(snap["rho_phys"] * dx_phys_now**3, r_phys_grid, r_bins, statistic="sum")
        gas_cum = np.cumsum(gas_shell)
        dm_cum = np.array([meta["m_par"] * np.sum(snap["r_phys_dm"] < rr_) for rr_ in r_c])
        label = f"a={snap['a']:.3f}"
        axes[0].loglog(r_c, rho, label=label)
        axes[1].loglog(r_c, p)
        axes[2].loglog(r_c, t)
        axes[3].loglog(r_c, s)
        axes[4].semilogx(r_c, vr_prof, label=label)
        axes[5].semilogx(r_c, mach_prof, label=label)
        axes[6].loglog(r_c, gas_cum, label=label)
        axes[7].loglog(r_c, dm_cum, label=label)

    axes[0].set_title("Gas density")
    axes[1].set_title("Pressure")
    axes[2].set_title("Temperature")
    axes[3].set_title("Entropy")
    axes[4].set_title("Radial velocity")
    axes[5].set_title("Mach profile")
    axes[6].set_title("Cumulative gas mass")
    axes[7].set_title("Cumulative DM mass")
    axes[7].loglog(prof["radius"], prof["dark_matter_mass"], "k--", lw=2, label="DM target")

    for ax in axes:
        ax.set_xlabel("r [kpc]")
    axes[4].axhline(0.0, color="k", lw=1)
    axes[5].axhline(1.0, color="k", lw=1, ls="--")
    axes[0].legend(fontsize=8)
    axes[4].legend(fontsize=8)
    axes[7].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "profiles_multi_a.png"), dpi=140)
    plt.close()


def plot_gas_slices(snapshots, out_dir):
    snap_i = snapshots[0]
    snap_f = snapshots[-1]
    N = snap_i["rho_phys"].shape[0]
    fields_i = [
        np.log10(snap_i["rho_phys"][:, :, N // 2] + 1.0e-30),
        np.log10(snap_i["p_phys"][:, :, N // 2] + 1.0e-30),
        np.log10(snap_i["T_phys"][:, :, N // 2] + 1.0e-30),
    ]
    fields_f = [
        np.log10(snap_f["rho_phys"][:, :, N // 2] + 1.0e-30),
        np.log10(snap_f["p_phys"][:, :, N // 2] + 1.0e-30),
        np.log10(snap_f["T_phys"][:, :, N // 2] + 1.0e-30),
    ]
    titles = [r"$\log_{10}\rho_g$", r"$\log_{10}p$", r"$\log_{10}T$"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for j in range(3):
        vmin = min(np.min(fields_i[j]), np.min(fields_f[j]))
        vmax = max(np.max(fields_i[j]), np.max(fields_f[j]))
        im = axes[0, j].imshow(fields_i[j], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"Initial {titles[j]}")
        plt.colorbar(im, ax=axes[0, j], fraction=0.046)
        im2 = axes[1, j].imshow(fields_f[j], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, j].set_title(f"Final {titles[j]}")
        plt.colorbar(im2, ax=axes[1, j], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gas_slices_initial_final.png"), dpi=140)
    plt.close()


def make_gas_slice_animation(snapshots, out_dir):
    if len(snapshots) < 2:
        return

    N = snapshots[0]["rho_phys"].shape[0]
    dens_slices = [np.log10(snap["rho_phys"][:, :, N // 2] + 1.0e-30) for snap in snapshots]
    temp_slices = [np.log10(snap["T_phys"][:, :, N // 2] + 1.0e-30) for snap in snapshots]
    dens_vmin = min(np.min(arr) for arr in dens_slices)
    dens_vmax = max(np.max(arr) for arr in dens_slices)
    temp_vmin = min(np.min(arr) for arr in temp_slices)
    temp_vmax = max(np.max(arr) for arr in temp_slices)

    frames = []
    for snap, dens, temp in zip(snapshots, dens_slices, temp_slices):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        im0 = axes[0].imshow(dens, origin="lower", cmap="viridis", vmin=dens_vmin, vmax=dens_vmax)
        axes[0].set_title(rf"Gas density  a={snap['a']:.3f}")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(temp, origin="lower", cmap="magma", vmin=temp_vmin, vmax=temp_vmax)
        axes[1].set_title(rf"Temperature  a={snap['a']:.3f}")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(os.path.join(out_dir, "gas_slices_evolution.gif"), frames, duration=0.8, loop=0)


def plot_dm_cic(snapshots, meta, out_dir):
    snap_i = snapshots[0]
    snap_f = snapshots[-1]
    mesh_shape = (snap_i["rho_phys"].shape[0],) * 3
    weights = np.full((snap_i["pos_cells"].shape[0],), meta["dm_particle_weight"], dtype=np.float32)
    rho_i = paint_dm_density(snap_i["pos_cells"], weights, mesh_shape)
    rho_f = paint_dm_density(snap_f["pos_cells"], weights, mesh_shape)
    proj_i = np.sum(rho_i, axis=2)
    proj_f = np.sum(rho_f, axis=2)
    sl_i = rho_i[:, :, mesh_shape[2] // 2]
    sl_f = rho_f[:, :, mesh_shape[2] // 2]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    panels = [
        (sl_i, "Initial DM CIC midplane"),
        (sl_f, "Final DM CIC midplane"),
        (proj_i, "Initial DM projected"),
        (proj_f, "Final DM projected"),
    ]
    for ax, (arr, title) in zip(axes.ravel(), panels):
        im = ax.imshow(np.log10(arr + 1.0e-12), origin="lower", cmap="magma")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dm_cic_slice_projection.png"), dpi=140)
    plt.close()


def make_dm_cic_animation(snapshots, meta, out_dir):
    if len(snapshots) < 2:
        return
    mesh_shape = (snapshots[0]["rho_phys"].shape[0],) * 3
    weights = np.full((snapshots[0]["pos_cells"].shape[0],), meta["dm_particle_weight"], dtype=np.float32)
    rho_list = [paint_dm_density(snap["pos_cells"], weights, mesh_shape)[:, :, mesh_shape[2] // 2] for snap in snapshots]
    log_list = [np.log10(rho + 1.0e-12) for rho in rho_list]
    vmin = min(float(np.min(arr)) for arr in log_list)
    vmax = max(float(np.max(arr)) for arr in log_list)
    frames = []
    for snap, rho, logrho in zip(snapshots, rho_list, log_list):
        frac = (rho - rho_list[0]) / np.maximum(rho_list[0], 1.0e-12)
        lim = max(0.1, float(np.nanpercentile(np.abs(frac), 99.0)))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        im0 = axes[0].imshow(logrho, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        axes[0].set_title(rf"DM CIC midplane  a={snap['a']:.3f}")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(frac, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
        axes[1].set_title("Fractional DM change")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(os.path.join(out_dir, "dm_density_evolution.gif"), frames, duration=0.8, loop=0)


def plot_time_series(history, out_dir):
    a = np.array([h["a"] for h in history])
    hse = np.array([h["hse_residual_rms"] for h in history])
    kt = np.array([h["kinetic_to_thermal"] for h in history])
    ent = np.array([h["entropy_mean_r500"] for h in history])
    gf = np.array([h["gas_fraction_r500"] for h in history])
    flux500 = np.array([h["gas_mass_flux_r500"] for h in history])
    mach95 = np.array([h["mach95"] for h in history])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(a, hse, "o-")
    axes[0, 0].set_title("HSE residual rms")
    axes[0, 1].plot(a, kt, "o-")
    axes[0, 1].set_title("Gas kinetic / thermal")
    axes[0, 2].plot(a, ent / max(ent[0], 1.0e-30), "o-")
    axes[0, 2].set_title("Entropy inside r500")
    axes[1, 0].plot(a, gf / max(gf[0], 1.0e-30), "o-")
    axes[1, 0].set_title("Gas fraction inside r500")
    axes[1, 1].plot(a, flux500, "o-")
    axes[1, 1].axhline(0.0, color="k", lw=1)
    axes[1, 1].set_title("Mass flux at r500")
    axes[1, 2].plot(a, mach95, "o-")
    axes[1, 2].set_title("Mach95")
    for ax in axes.ravel():
        ax.set_xlabel("a")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_series_metrics.png"), dpi=140)
    plt.close()


def write_history_bundle(history, out_dir):
    np.savez(
        os.path.join(out_dir, "snapshot_history.npz"),
        a=np.array([h["a"] for h in history], dtype=np.float32),
        hse_residual_rms=np.array([h["hse_residual_rms"] for h in history], dtype=np.float32),
        kinetic_to_thermal=np.array([h["kinetic_to_thermal"] for h in history], dtype=np.float32),
        entropy_mean_r500=np.array([h["entropy_mean_r500"] for h in history], dtype=np.float32),
        gas_mass_flux_r500=np.array([h["gas_mass_flux_r500"] for h in history], dtype=np.float32),
        gas_fraction_r500=np.array([h["gas_fraction_r500"] for h in history], dtype=np.float32),
        mach95=np.array([h["mach95"] for h in history], dtype=np.float32),
    )


def main():
    global OUT_DIR
    args = parse_args()
    if args.output_subdir:
        OUT_DIR = os.path.join(_HERE, "outputs", "stage5", args.output_subdir)
    os.makedirs(OUT_DIR, exist_ok=True)

    ref = HaloReference(z=Z_INIT, m200_msun=M200, conc=CONC)
    eq, bg, U, params, meta = build_initial_state(ref, args)
    meta["bg_h0"] = float(bg.H0)
    sim = make_sim(
        eq,
        bg,
        tuple(eq.mesh_shape),
        gravity_method=args.gravity_method,
        dx_com=meta["dx_com"],
        softening_cells=args.softening_cells,
        gas_gravity_meta={
            "x_com": meta["grids"]["X"] / meta["a_init"],
            "y_com": meta["grids"]["Y"] / meta["a_init"],
            "z_com": meta["grids"]["Z"] / meta["a_init"],
            "center_com": meta["grids"]["center"] / meta["a_init"],
            "radius_table": meta["grids"]["profiles"]["radius"],
            "g_table": meta["grids"]["profiles"]["gravitational_field"],
        },
    )

    snapshots = []
    history = []
    snap0 = snapshot_from_state(eq, U, params, meta, args.n_grid)
    snapshots.append(snap0)
    history.append(compute_snapshot_metrics(ref, snap0, meta))

    a_init = meta["a_init"]
    a_final = max(float(args.a_final), a_init + 1.0e-5)

    wall_t0 = time.time()
    step_i = 0
    while (a_final - float(params["a"])) > float(args.a_tol):
        if step_i >= args.max_steps:
            raise RuntimeError(
                f"Reached max_steps={args.max_steps} before a_final={a_final:.4f}. "
                "Increase --max-steps or --max-dtau."
            )
        a_now = float(params["a"])
        dtau = choose_dtau(sim, U, bg, a_now, a_final, args.max_dtau, args.min_dtau, args.dtau_safety)
        if step_i % 10 == 0:
            remaining_a = max(a_final - float(params["a"]), 0.0)
            print(
                f"Step {step_i}, a={float(params['a']):.6f}, "
                f"remaining_da={remaining_a:.3e}, "
                f"time elapsed: {time.time() - wall_t0:.2f} s, dtau={dtau:.4e}"
            )
        
        params_dm = dict(params["dm"])
        params_dm["drift_factor"] = jnp.asarray(bg.H0, dtype=jnp.float32)
        params = dict(params)
        params["dm"] = params_dm
        U, params = sim._hydrostep(step_i, (U, params), jnp.asarray(dtau, dtype=jnp.float32))
        U = apply_state_floors(eq, U, jnp.asarray(params["a"], dtype=jnp.float32))
        validate_finite_state(U, params, f"step {step_i}")
        step_i += 1

        if (step_i % args.snapshot_every) == 0 or float(params["a"]) >= a_final:
            snap = snapshot_from_state(eq, U, params, meta, args.n_grid)
            snapshots.append(snap)
            history.append(compute_snapshot_metrics(ref, snap, meta))

    wall = time.time() - wall_t0

    plot_profiles(ref, snapshots, meta, OUT_DIR)
    plot_gas_slices(snapshots, OUT_DIR)
    make_gas_slice_animation(snapshots, OUT_DIR)
    plot_dm_cic(snapshots, meta, OUT_DIR)
    make_dm_cic_animation(snapshots, meta, OUT_DIR)
    plot_time_series(history, OUT_DIR)
    write_history_bundle(history, OUT_DIR)

    scalars = {
        "a_init": float(a_init),
        "a_final": float(history[-1]["a"]),
        "pressure_scale": float(args.pressure_scale),
        "velocity_scale": float(args.velocity_scale),
        "gravity_method": str(args.gravity_method),
        "softening_cells": float(args.softening_cells),
        "steps_taken": int(step_i),
        "snapshot_count": int(len(snapshots)),
        "wall_time_s": float(wall),
        "hse_residual_rms_init": float(history[0]["hse_residual_rms"]),
        "hse_residual_rms_final": float(history[-1]["hse_residual_rms"]),
        "kinetic_to_thermal_init": float(history[0]["kinetic_to_thermal"]),
        "kinetic_to_thermal_final": float(history[-1]["kinetic_to_thermal"]),
        "gas_fraction_r500_ratio_final": float(history[-1]["gas_fraction_r500"] / max(history[0]["gas_fraction_r500"], 1.0e-30)),
        "mach95_final": float(history[-1]["mach95"]),
        "min_temperature_final": float(history[-1]["min_temperature"]),
        "min_pressure_final": float(history[-1]["min_pressure"]),
    }
    params_out = {
        "stage": 5,
        "description": "Single-halo gas+DM cosmological evolution in supercomoving variables",
        "halo": {
            "z_init": float(Z_INIT),
            "M200_Msun": float(M200),
            "conc": float(CONC),
            "r200_kpc": float(ref.r200),
            "r500_kpc": float(ref.r500),
            "r_max_kpc": float(meta["r_max"]),
        },
        "cosmology": {
            "H0_km_s_Mpc": float(H0_KM_S_MPC),
            "H0_Myr_inv": float(bg.H0),
            "Omega_m": float(OMEGA_M),
            "Omega_b": float(OMEGA_B),
            "Omega_lambda": float(OMEGA_L),
            "a_init": float(a_init),
            "a_final_target": float(a_final),
        },
        "run": {
            "n_grid": int(args.n_grid),
            "n_par": int(args.n_par),
            "l_box_phys_kpc": float(meta["l_box_phys"]),
            "l_box_com_kpc": float(meta["l_box_com"]),
            "dx_phys_kpc": float(meta["dx_phys"]),
            "dx_com_kpc": float(meta["dx_com"]),
            "snapshot_every": int(args.snapshot_every),
            "max_steps": int(args.max_steps),
            "max_dtau": float(args.max_dtau),
            "min_dtau": float(args.min_dtau),
            "dtau_safety": float(args.dtau_safety),
            "a_tol": float(args.a_tol),
            "pressure_scale": float(args.pressure_scale),
            "cancel_hubble_flow": bool(args.cancel_hubble_flow),
            "velocity_scale": float(args.velocity_scale),
            "gravity_method": str(args.gravity_method),
            "softening_cells": float(args.softening_cells),
            "dm_kick_scale": float(args.dm_kick_scale),
            "gas_kick_scale": float(args.gas_kick_scale),
        },
    }

    with open(os.path.join(OUT_DIR, "stage5_params.yaml"), "w") as f:
        yaml.dump(params_out, f, default_flow_style=False, sort_keys=False)
    with open(os.path.join(OUT_DIR, "scalars.txt"), "w") as f:
        f.write("Stage 5 Scalar Diagnostics\n")
        for k, v in scalars.items():
            f.write(f"{k}: {v}\n")

    print("[Stage 5] Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
