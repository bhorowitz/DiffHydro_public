#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmo_parallel.common import integrate_tau_target, parse_pmesh_shape, write_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run coupled hydro+DM cosmology from gadget-style Nyx ICs.")
    p.add_argument("--gpu", type=str, default="0,1,2,3")
    p.add_argument("--xla-preallocate", action="store_true", default=False)
    p.add_argument(
        "--nyx-ic-plotfile",
        type=Path,
        default=Path("/home/ben.horowitz/DiffHydro_public/Nyx/Exec/LyA/gadget_ic_runs/n128_z2/plt_init_00000"),
    )
    p.add_argument(
        "--nyx-dm-ic-h5",
        type=Path,
        default=Path("/home/ben.horowitz/DiffHydro_public/Nyx/Exec/LyA/gadget_ic_runs/n128_z2/plt_init_00000_particles.h5"),
    )
    p.add_argument(
        "--cached-ic-npz",
        type=Path,
        default=None,
        help="Optional pre-extracted gas IC cache (.npz) to avoid requiring yt in the runtime environment.",
    )
    p.add_argument("--z-target", type=float, default=10.0)
    p.add_argument("--solver", choices=["hll", "hllc", "lf", "laxfriedrichs", "nyx"], default="hllc")
    p.add_argument("--gravity-backend", choices=["fft", "multigrid", "jaxdecomp_fft"], default="fft")
    p.add_argument("--pmesh-shape", type=str, default="4x1x1")
    p.add_argument("--jaxdecomp-pdims", type=str, default="auto")
    p.add_argument("--jaxdecomp-halo-size", type=int, default=16)
    p.add_argument("--gas-mean-fraction", type=float, default=1.58e-1)
    p.add_argument("--hydro-temp-floor-k", type=float, default=0.0)
    p.add_argument("--nyx-velocity-divisor", type=float, default=100.0)
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--force-eps", type=float, default=1.0e-8)
    p.add_argument("--vel-unit-cms", type=float, default=1.0e7)
    p.add_argument("--mu-hydrogen", type=float, default=1.0)
    p.add_argument("--rho-unit-cgs", type=float, default=1.6e-24)
    p.add_argument("--h-species", type=float, default=0.76)
    p.add_argument("--tau-time-unit-s", type=float, default=3.085677581e19)
    p.add_argument("--enable-cooling", action="store_true", default=True)
    p.add_argument("--disable-cooling", dest="enable_cooling", action="store_false")
    p.add_argument("--cooling-rate-scale", type=float, default=1.0)
    p.add_argument("--nyx-heating-scale", type=float, default=1.0)
    p.add_argument("--cooling-temp-floor-k", type=float, default=1.0)
    p.add_argument("--cooling-subcycles", type=int, default=8)
    p.add_argument("--cooling-dtmax-s", type=float, default=1.0e16)
    p.add_argument("--nyx-cooling-table-npz", type=Path, default=Path("data/nyx_cooling_table.npz"))
    p.add_argument("--nyx-cooling-treecool", type=Path, default=Path("diffhydro/nyx_eos/TREECOOL_middle"))
    p.add_argument("--nyx-cooling-z-nodes", type=str, default="2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100")
    p.add_argument("--nyx-cooling-logdelta-min", type=float, default=-3.0)
    p.add_argument("--nyx-cooling-logdelta-max", type=float, default=3.0)
    p.add_argument("--nyx-cooling-logdelta-n", type=int, default=96)
    p.add_argument("--nyx-cooling-logt-min", type=float, default=0.0)
    p.add_argument("--nyx-cooling-logt-max", type=float, default=8.0)
    p.add_argument("--nyx-cooling-logt-n", type=int, default=120)
    p.add_argument("--nyx-cooling-rebuild", action="store_true", default=False)
    p.add_argument("--nyx-cooling-eos-path", type=Path, default=Path("diffhydro/nyx_eos"))
    p.add_argument(
        "--ic-import-density-a3",
        action="store_true",
        dest="ic_import_density_a3",
        help="Include an extra a^-3 factor when mapping Nyx density into physical gas density.",
    )
    p.add_argument(
        "--no-ic-import-density-a3",
        action="store_false",
        dest="ic_import_density_a3",
        help="Use the archived baseline import convention without the extra a^-3 factor.",
    )
    p.set_defaults(ic_import_density_a3=False)
    p.add_argument("--dtau-max", type=float, default=8.0e-2)
    p.add_argument("--dtau-min", type=float, default=2.0e-10)
    p.add_argument("--dtau-safety", type=float, default=0.3)
    p.add_argument("--relative-max-change-a", type=float, default=0.3)
    p.add_argument("--absolute-max-change-a", type=float, default=0.0)
    p.add_argument("--dtau-change-max", type=float, default=0.0)
    p.add_argument("--dtau-init-shrink", type=float, default=1.0)
    p.add_argument("--solver-dt-safety", type=float, default=0.9)
    p.add_argument("--disable-solver-dt", action="store_true", default=False)
    p.add_argument("--step-retries", type=int, default=10)
    p.add_argument("--dtau-retry-factor", type=float, default=0.5)
    p.add_argument("--retry-cap-mult", type=float, default=1.25)
    p.add_argument("--state-floor", type=float, default=2.0e-8)
    p.add_argument("--pressure-floor", type=float, default=2.0e-8)
    p.add_argument("--dm-kick-scale", type=float, default=1.0)
    p.add_argument("--gas-kick-scale", type=float, default=1.0)
    p.add_argument(
        "--gas-kick-mode",
        choices=["legacy_h0sq", "dm_match"],
        default="legacy_h0sq",
        help=(
            "How to build the gas kick prefactor. "
            "'legacy_h0sq' uses 1.5*Omega_m*H0^2*gas_kick_scale; "
            "'dm_match' uses the DM prefactor 1.5*Omega_m*H0*gas_kick_scale."
        ),
    )
    p.add_argument("--snapshot-every", type=int, default=0)
    p.add_argument(
        "--checkpoint-z-values",
        type=str,
        default="",
        help="Comma-separated redshifts to dump full field checkpoints when crossed, for example '10,2'.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ben.horowitz/DiffHydro_public/cosmo_parallel/results/2gadgetic_coupled_parallel"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if args.xla_preallocate else "false"
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import h5py
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import diffhydro as dh
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
    )
    from diffhydro.physics.multigrid_gs import poisson_multigrid

    try:
        from jaxpm.pm import cic_paint as jpm_cic_paint
    except Exception:
        jpm_cic_paint = None

    jpm_cic_paint_dx = None
    jpm_pm_forces = None
    NamedSharding = None
    P = None

    def _cic_paint_local(mesh, positions, weight=None):
        positions_e = jnp.expand_dims(positions, axis=1)
        floor = jnp.floor(positions_e)
        offsets = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=positions.dtype,
        )[None, :, :]
        neigh = floor + offsets
        kernel = 1.0 - jnp.abs(positions_e - neigh)
        kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]
        if weight is not None:
            kernel = kernel * jnp.expand_dims(jnp.asarray(weight), -1)
        neigh = jnp.mod(neigh.astype(jnp.int32), jnp.asarray(mesh.shape, dtype=jnp.int32))
        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1, 2),
            scatter_dims_to_operand_dims=(0, 1, 2),
        )
        return jax.lax.scatter_add(mesh, neigh, kernel, dnums)

    def _infer_levels(shape: tuple[int, ...]) -> int:
        dims = list(shape)
        levels = 0
        while all(d % 2 == 0 and d >= 4 for d in dims):
            levels += 1
            dims = [d // 2 for d in dims]
        return max(levels, 1)

    def _grad_centered_periodic(phi: jnp.ndarray, dx: float):
        dphidx = (jnp.roll(phi, -1, 0) - jnp.roll(phi, 1, 0)) / (2.0 * dx)
        dphidy = (jnp.roll(phi, -1, 1) - jnp.roll(phi, 1, 1)) / (2.0 * dx)
        dphidz = (jnp.roll(phi, -1, 2) - jnp.roll(phi, 1, 2)) / (2.0 * dx)
        return (-dphidx, -dphidy, -dphidz)

    def _compute_phys_floor_code(a, rho_floor_phys: float, p_floor_phys: float, dtype):
        a_arr = jnp.asarray(a, dtype=dtype)
        rho_floor_code = cosmo_conv.density_phys_to_code(jnp.asarray(rho_floor_phys, dtype=dtype), a_arr)
        p_floor_code = cosmo_conv.pressure_phys_to_code(jnp.asarray(p_floor_phys, dtype=dtype), a_arr)
        return rho_floor_code, p_floor_code

    def _code_temp_to_kelvin(t_code: jnp.ndarray, mu_hydrogen: float, vel_unit_cms: float):
        mH_cgs = 1.6735575e-24
        kB_cgs = 1.380649e-16
        return jnp.maximum(
            t_code * (mu_hydrogen * mH_cgs / kB_cgs) * (vel_unit_cms * vel_unit_cms),
            1.0e-30,
        )

    def _state_is_finite(U_state, params_state) -> bool:
        ok = bool(jnp.isfinite(jnp.sum(jnp.asarray(U_state, dtype=jnp.float32))))
        if not ok:
            return False
        a_val = float(params_state.get("a", 1.0))
        if not np.isfinite(a_val):
            return False
        dm_state = params_state.get("dm", None)
        if dm_state is not None:
            if "x" in dm_state and (not bool(jnp.isfinite(jnp.sum(jnp.asarray(dm_state["x"], dtype=jnp.float32))))):
                return False
            if "x_abs" in dm_state and (not bool(jnp.isfinite(jnp.sum(jnp.asarray(dm_state["x_abs"], dtype=jnp.float32))))):
                return False
            if "disp" in dm_state and (not bool(jnp.isfinite(jnp.sum(jnp.asarray(dm_state["disp"], dtype=jnp.float32))))):
                return False
            if "p_or_v" in dm_state and (not bool(jnp.isfinite(jnp.sum(jnp.asarray(dm_state["p_or_v"], dtype=jnp.float32))))):
                return False
        return True

    def _dm_density_mesh(dm_state):
        dm_mass = jnp.asarray(dm_state["mass"], dtype=jnp.float32)
        if "x_abs" in dm_state:
            return jpm_cic_paint(
                jnp.zeros(shape=mesh_shape, dtype=jnp.float32, device=jaxdecomp_field_sharding),
                dm_state["x_abs"],
                weight=dm_mass,
                halo_size=int(args.jaxdecomp_halo_size),
                sharding=jaxdecomp_field_sharding,
            )
        if "disp" in dm_state:
            return jpm_cic_paint_dx(
                dm_state["disp"],
                halo_size=int(args.jaxdecomp_halo_size),
                sharding=jaxdecomp_field_sharding,
                weight=dm_mass,
            )
        dm_x = jnp.asarray(dm_state["x"], dtype=jnp.float32)
        return (jpm_cic_paint if jpm_cic_paint is not None else _cic_paint_local)(
            jnp.zeros(mesh_shape, dtype=jnp.float32),
            dm_x,
            weight=dm_mass,
        )

    def _hierarchical_reconstruct_disp_and_momentum(abs_pos_grid: np.ndarray, p_grid: np.ndarray, mesh_n: int):
        idx_x = np.argsort(abs_pos_grid[:, 0], kind="stable")
        pos_x = abs_pos_grid[idx_x].reshape(mesh_n, mesh_n * mesh_n, 3)
        p_x = p_grid[idx_x].reshape(mesh_n, mesh_n * mesh_n, 3)
        pos_out = np.empty((mesh_n, mesh_n, mesh_n, 3), dtype=np.float32)
        p_out = np.empty((mesh_n, mesh_n, mesh_n, 3), dtype=np.float32)
        for i in range(mesh_n):
            order_y = np.argsort(pos_x[i, :, 1], kind="stable")
            slab_pos = pos_x[i, order_y].reshape(mesh_n, mesh_n, 3)
            slab_p = p_x[i, order_y].reshape(mesh_n, mesh_n, 3)
            for j in range(mesh_n):
                order_z = np.argsort(slab_pos[j, :, 2], kind="stable")
                pos_out[i, j] = slab_pos[j, order_z]
                p_out[i, j] = slab_p[j, order_z]

        grid_i, grid_j, grid_k = np.meshgrid(
            np.arange(mesh_n, dtype=np.float32),
            np.arange(mesh_n, dtype=np.float32),
            np.arange(mesh_n, dtype=np.float32),
            indexing="ij",
        )
        lattice = np.stack([grid_i, grid_j, grid_k], axis=-1)
        disp_out = ((pos_out - lattice + mesh_n / 2.0) % mesh_n) - mesh_n / 2.0
        return disp_out.astype(np.float32), p_out.astype(np.float32)

    def _dump_state(
        out_base: Path,
        U_state,
        params_state,
        *,
        dt_hist_local: list[float],
        label: str,
    ) -> dict[str, float]:
        out_base.mkdir(parents=True, exist_ok=True)
        Uf_np = np.asarray(jax.device_get(U_state), dtype=np.float32)
        a_out = float(np.asarray(jax.device_get(params_state["a"])))
        z_out = 1.0 / max(a_out, 1.0e-30) - 1.0
        w_out = np.asarray(jax.device_get(eq.get_primitives_from_conservatives(U_state)), dtype=np.float32)
        rho_out = np.asarray(jax.device_get(cosmo_conv.density_code_to_phys(w_out[0], a_out)), dtype=np.float32)
        p_out = np.asarray(jax.device_get(cosmo_conv.pressure_code_to_phys(w_out[4], a_out)), dtype=np.float32)
        temp_code_out = p_out / np.maximum(rho_out * float(eq.R), 1.0e-30)
        temp_out = np.asarray(
            jax.device_get(
                _code_temp_to_kelvin(
                    jnp.asarray(temp_code_out, dtype=jnp.float32),
                    mu_hydrogen=float(args.mu_hydrogen),
                    vel_unit_cms=float(hydro_vel_unit_cms),
                )
            ),
            dtype=np.float32,
        )
        dm_mesh_out = np.asarray(jax.device_get(_dm_density_mesh(params_state["dm"])), dtype=np.float32)
        dt_hist_np_local = np.asarray(dt_hist_local, dtype=np.float32)
        np.savez_compressed(
            out_base / f"{label}_fields.npz",
            conserved=Uf_np,
            rho_phys=rho_out,
            pressure_phys=p_out,
            temperature=temp_out,
            temperature_code=temp_code_out,
            dm_density=dm_mesh_out,
            dt_hist=dt_hist_np_local,
            a=a_out,
            z=z_out,
            box_size=float(box_size),
            n_grid=int(n_grid),
        )
        k = n_grid // 2
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        im0 = axes[0, 0].imshow(np.log10(np.maximum(density[:, :, k], 1.0e-30)), origin="lower", cmap="viridis")
        axes[0, 0].set_title("IC gas density")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        fig.colorbar(im0, ax=axes[0, 0], shrink=0.85)
        im1 = axes[0, 1].imshow(np.log10(np.maximum(rho_out[:, :, k], 1.0e-30)), origin="lower", cmap="viridis")
        axes[0, 1].set_title(f"Gas density z={z_out:.2f}")
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        fig.colorbar(im1, ax=axes[0, 1], shrink=0.85)
        im2 = axes[1, 0].imshow(np.log10(np.maximum(dm_mesh_out[:, :, k], 1.0e-30)), origin="lower", cmap="magma")
        axes[1, 0].set_title(f"DM density z={z_out:.2f}")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        fig.colorbar(im2, ax=axes[1, 0], shrink=0.85)
        im3 = axes[1, 1].imshow(np.log10(np.maximum(temp_out[:, :, k], 1.0e-30)), origin="lower", cmap="inferno")
        axes[1, 1].set_title(f"Temp[K] z={z_out:.2f}")
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        fig.colorbar(im3, ax=axes[1, 1], shrink=0.85)
        fig.savefig(out_base / f"{label}_slices_compare.png", dpi=160)
        plt.close(fig)
        return {
            "label": label,
            "a": float(a_out),
            "z": float(z_out),
            "rho_mean": float(np.mean(rho_out)),
            "rho_std": float(np.std(rho_out)),
            "temp_mean_k": float(np.mean(temp_out)),
            "temp_std_k": float(np.std(temp_out)),
            "dm_mean": float(np.mean(dm_mesh_out)),
            "dm_std": float(np.std(dm_mesh_out)),
        }

    class NyxTabulatedCoolingForce:
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
                dotE = self.heating_rate_scale * heat_cgs - self.cooling_rate_scale * cool_cgs
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

    def build_hydrosim(eq_local, forces_local, *, solver_name: str, pmesh_shape_local: tuple[int, int, int], dx_o: float = 1.0):
        ss = dh.signal_speed_Einfeldt
        sname = solver_name.lower()
        if sname == "hllc":
            riemann_solver = dh.HLLC(equation_manager=eq_local, signal_speed=ss)
        elif sname in ("laxfriedrichs", "lf"):
            riemann_solver = dh.LaxFriedrichs(equation_manager=eq_local, signal_speed=ss)
        elif sname == "hll":
            riemann_solver = dh.HLL(equation_manager=eq_local, signal_speed=ss)
        elif sname in ("nyx_riemann", "nyx"):
            riemann_solver = dh.NyxRiemannEuler(
                equation_manager=eq_local,
                signal_speed=ss,
                eos_consistent_interface_energy=True,
            )
        else:
            raise ValueError(f"Unknown solver '{solver_name}'")
        conv_flux_local = dh.ConvectiveFlux(
            eq_local,
            riemann_solver,
            dh.PPM_CW(limiter="MINMOD", steepen=False),
            positivity=True,
        )
        conv_flux_local.dx_o = float(dx_o)
        sim_local = dh.hydro(
            n_super_step=1,
            max_dt=0.2,
            fluxes=[conv_flux_local],
            forces=list(forces_local),
            use_mol=True,
            pmesh_shape=pmesh_shape_local,
            snapshot_every=(args.snapshot_every if args.snapshot_every > 0 else None),
            snapshot_dir=str(out_dir / "snapshots"),
        )
        sim_local.dx_o = float(dx_o)
        for flux in sim_local.fluxes:
            if hasattr(flux, "dx_o"):
                flux.dx_o = float(dx_o)
        return sim_local

    class CoupledMGGravityForce(JaxPMCoupledGravityForce):
        def __init__(self, *force_args, **force_kwargs):
            super().__init__(*force_args, use_jaxpm=False, **force_kwargs)
            self._mg_levels = _infer_levels(self.mesh_shape)

        def _mesh_acceleration_mg(self, delta):
            F = delta.astype(jnp.float32)
            phi0 = jnp.zeros_like(F, dtype=jnp.float32)
            phi = poisson_multigrid(
                F=F,
                U=phi0,
                l=self._mg_levels,
                v1=2,
                v2=2,
                mu=1,
                iter_cycle=2,
                eps=1.0e-6,
                h=1.0,
            )
            ax, ay, az = _grad_centered_periodic(phi, 1.0)
            return jnp.stack([ax, ay, az], axis=-1)

        def _sample_forces(self, positions, delta):
            acc = self._mesh_acceleration_mg(delta)
            idx = jnp.mod(jnp.floor(positions).astype(jnp.int32), jnp.asarray(self.mesh_shape, dtype=jnp.int32))
            return acc[idx[:, 0], idx[:, 1], idx[:, 2]]

        def _gas_mesh_acceleration(self, delta):
            return self._mesh_acceleration_mg(delta)

    class CoupledJaxDecompFFTGravityForce(JaxPMCoupledGravityForce):
        def __init__(self, *force_args, sharding, halo_size: int = 32, **force_kwargs):
            super().__init__(*force_args, use_jaxpm=False, **force_kwargs)
            self.sharding = sharding
            self.halo_size = int(halo_size)
            self.dm_sharding = NamedSharding(sharding.mesh, P("x", "y", None, None))

        def _make_delta_from_disp(self, rho_gas, dm_disp, dm_weight):
            rho_gas_sharded = jax.device_put(jnp.asarray(rho_gas, dtype=jnp.float32), self.sharding)
            dm_mesh = jpm_cic_paint_dx(
                jnp.asarray(dm_disp, dtype=jnp.float32),
                halo_size=self.halo_size,
                sharding=self.sharding,
                weight=jnp.asarray(dm_weight, dtype=jnp.float32),
            )
            total_rho = rho_gas_sharded + dm_mesh
            mean_total = jnp.mean(total_rho)
            source = total_rho - jnp.where(self.subtract_mean, mean_total, 0.0)
            return source / jnp.maximum(mean_total, self.eps)

        def _make_delta_from_absolute(self, rho_gas, dm_x, dm_weight):
            rho_gas_sharded = jax.device_put(jnp.asarray(rho_gas, dtype=jnp.float32), self.sharding)
            dm_mesh = jpm_cic_paint(
                jnp.zeros(shape=self.mesh_shape, dtype=jnp.float32, device=self.sharding),
                jnp.asarray(dm_x, dtype=jnp.float32),
                weight=jnp.asarray(dm_weight, dtype=jnp.float32),
                halo_size=self.halo_size,
                sharding=self.sharding,
            )
            total_rho = rho_gas_sharded + dm_mesh
            mean_total = jnp.mean(total_rho)
            source = total_rho - jnp.where(self.subtract_mean, mean_total, 0.0)
            return source / jnp.maximum(mean_total, self.eps)

        def _gas_mesh_acceleration(self, delta):
            zero_disp = jnp.zeros(self.mesh_shape + (3,), dtype=jnp.float32, device=self.dm_sharding)
            return jpm_pm_forces(
                zero_disp,
                mesh_shape=self.mesh_shape,
                delta=delta,
                r_split=0.0,
                paint_absolute_pos=False,
                halo_size=self.halo_size,
                sharding=self.sharding,
            )

        def force(self, i, U_gas, params, dtau):
            del i
            dtau = jnp.maximum(jnp.asarray(dtau), 0.0)
            dtau_half = 0.5 * dtau
            rho_gas = jnp.asarray(U_gas[self.i_rho], dtype=jnp.float32)

            params_out = dict(params)
            a_start = jnp.asarray(params_out.get("a", 1.0), dtype=jnp.float32)
            dm_params = params.get("dm", None)
            if dm_params is None:
                raise ValueError("jaxdecomp_fft backend requires params['dm']")

            use_absolute = "x_abs" in dm_params
            if use_absolute:
                dm_x = jnp.asarray(dm_params["x_abs"], dtype=jnp.float32)
                dm_mom = jnp.asarray(dm_params.get("p_or_v", jnp.zeros_like(dm_x)), dtype=jnp.float32)
                dm_weight = self._as_particle_weight(dm_x, dm_params.get("mass", self.dm_particle_mass))
            else:
                if "disp" not in dm_params:
                    raise ValueError("jaxdecomp_fft backend requires params['dm']['x_abs'] or params['dm']['disp']")
                dm_disp = jnp.asarray(dm_params["disp"], dtype=jnp.float32)
                dm_mom = jnp.asarray(dm_params.get("p_or_v", jnp.zeros_like(dm_disp)), dtype=jnp.float32)
                dm_weight = self._as_particle_weight(dm_disp, dm_params.get("mass", self.dm_particle_mass))
            drift_factor = jnp.asarray(dm_params.get("drift_factor", self.dm_drift_factor), dtype=jnp.float32)
            kick_prefactor = dm_params.get("kick_prefactor", None)
            if "kick_factor" in dm_params:
                kick_factor = jnp.asarray(dm_params["kick_factor"], dtype=jnp.float32)
            elif kick_prefactor is not None:
                kick_factor = jnp.asarray(kick_prefactor, dtype=jnp.float32) * a_start
            else:
                kick_factor = jnp.asarray(self.dm_kick_factor, dtype=jnp.float32)
            gas_kick_factor = dm_params.get("gas_kick_factor", self.gas_kick_factor)
            gas_kick_prefactor = dm_params.get("gas_kick_prefactor", None)
            if gas_kick_factor is None and gas_kick_prefactor is not None:
                gas_kick_factor = jnp.asarray(gas_kick_prefactor, dtype=jnp.float32) * a_start
            if gas_kick_factor is None:
                gas_kick_factor = drift_factor * kick_factor
            gas_kick_factor = jnp.asarray(gas_kick_factor, dtype=jnp.float32)

            delta_old = self._make_delta_from_absolute(rho_gas, dm_x, dm_weight) if use_absolute else self._make_delta_from_disp(rho_gas, dm_disp, dm_weight)
            gas_accel_old = self._gas_mesh_acceleration(delta_old)
            U_half = self._apply_gas_kick(U_gas, rho_gas, gas_accel_old, dtau_half, gas_kick_factor)

            dm_accel_old = jpm_pm_forces(
                dm_x if use_absolute else dm_disp,
                mesh_shape=self.mesh_shape,
                delta=delta_old,
                r_split=0.0,
                paint_absolute_pos=use_absolute,
                halo_size=self.halo_size,
                sharding=self.sharding,
            )
            dm_mom_half = dm_mom + dtau_half * kick_factor * dm_accel_old
            if use_absolute:
                mesh_shape = jnp.asarray(self.mesh_shape, dtype=dm_x.dtype)
                dm_x_drift = jnp.mod(dm_x + dtau * drift_factor * dm_mom_half, mesh_shape)
            else:
                dm_disp_drift = dm_disp + dtau * drift_factor * dm_mom_half

            delta_new = self._make_delta_from_absolute(rho_gas, dm_x_drift, dm_weight) if use_absolute else self._make_delta_from_disp(rho_gas, dm_disp_drift, dm_weight)
            gas_accel_new = self._gas_mesh_acceleration(delta_new)
            U_new = self._apply_gas_kick(U_half, rho_gas, gas_accel_new, dtau_half, gas_kick_factor)
            dm_accel_new = jpm_pm_forces(
                dm_x_drift if use_absolute else dm_disp_drift,
                mesh_shape=self.mesh_shape,
                delta=delta_new,
                r_split=0.0,
                paint_absolute_pos=use_absolute,
                halo_size=self.halo_size,
                sharding=self.sharding,
            )
            dm_mom_new = dm_mom_half + dtau_half * kick_factor * dm_accel_new

            dm_out = dict(dm_params)
            if use_absolute:
                dm_out["x_abs"] = dm_x_drift
            else:
                dm_out["disp"] = dm_disp_drift
            dm_out["p_or_v"] = dm_mom_new
            params_out["dm"] = dm_out
            return U_new, params_out

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pmesh_shape = parse_pmesh_shape(args.pmesh_shape)
    jaxdecomp_field_sharding = None
    if args.gravity_backend == "jaxdecomp_fft":
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from jaxpm.painting import cic_paint_dx as jpm_cic_paint_dx
        from jaxpm.pm import pm_forces as jpm_pm_forces

        if str(args.jaxdecomp_pdims).strip().lower() == "auto":
            ndev = int(jax.device_count())
            if ndev >= 4 and int(n_grid) % 2 == 0:
                pdims = (2, 2)
            elif ndev >= 4 and int(n_grid) % 4 == 0:
                pdims = (4, 1)
            elif ndev >= 2 and int(n_grid) % 2 == 0:
                pdims = (2, 1)
            else:
                pdims = (1, 1)
        else:
            raw = str(args.jaxdecomp_pdims).lower().replace(",", "x").strip()
            parts = [int(p) for p in raw.split("x") if p]
            if len(parts) != 2:
                raise ValueError(f"Expected --jaxdecomp-pdims like '2x2', got {args.jaxdecomp_pdims!r}")
            pdims = (parts[0], parts[1])
        if int(np.prod(pdims)) != int(jax.device_count()):
            raise ValueError(
                f"jaxdecomp_fft expects pdims product to match visible devices; got pdims={pdims}, devices={jax.device_count()}"
            )
        devices = np.asarray(jax.devices(), dtype=object).reshape(pdims)
        jaxdecomp_mesh = Mesh(devices, axis_names=("x", "y"))
        jaxdecomp_field_sharding = NamedSharding(jaxdecomp_mesh, P("x", "y", None))

    plotfile = args.nyx_ic_plotfile.resolve()
    dm_h5 = args.nyx_dm_ic_h5.resolve()
    if args.cached_ic_npz is None and not (plotfile / "Header").exists():
        raise FileNotFoundError(f"Missing Nyx plotfile Header in {plotfile}")
    if not dm_h5.exists():
        raise FileNotFoundError(f"Missing DM HDF5 file {dm_h5}")
    if args.cached_ic_npz is not None:
        cached = np.load(args.cached_ic_npz)
        density = np.asarray(cached["density"], dtype=np.float32)
        xmom = np.asarray(cached["xmom"], dtype=np.float32)
        ymom = np.asarray(cached["ymom"], dtype=np.float32)
        zmom = np.asarray(cached["zmom"], dtype=np.float32)
        temp_k = np.asarray(cached["temp_k"], dtype=np.float32)
        box_size = float(np.asarray(cached["box_size"]))
        n_grid = int(np.asarray(cached["n_grid"]))
        a_init = float(np.asarray(cached["a_init"]))
    else:
        import yt

        ds = yt.load(str(plotfile), hint="NyxDataset")
        cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
        density = np.asarray(cg[("boxlib", "density")], dtype=np.float32)
        xmom = np.asarray(cg[("boxlib", "xmom")], dtype=np.float32)
        ymom = np.asarray(cg[("boxlib", "ymom")], dtype=np.float32)
        zmom = np.asarray(cg[("boxlib", "zmom")], dtype=np.float32)
        temp_k = np.asarray(cg[("boxlib", "Temp")], dtype=np.float32)
        box_size = float(ds.domain_width[0].to_value("code_length"))
        n_grid = int(ds.domain_dimensions[0])
        a_init = float((plotfile / "comoving_a").read_text().strip())
    mesh_shape = (n_grid, n_grid, n_grid)
    a_target = 1.0 / (1.0 + float(args.z_target))
    if a_target <= a_init:
        raise ValueError(f"Target redshift must be lower than IC redshift; got a_target={a_target}, a_init={a_init}")

    with h5py.File(dm_h5, "r") as f:
        pos = np.stack(
            [np.asarray(f["x"], dtype=np.float32), np.asarray(f["y"], dtype=np.float32), np.asarray(f["z"], dtype=np.float32)],
            axis=1,
        )
        vel = np.stack(
            [np.asarray(f["vx"], dtype=np.float32), np.asarray(f["vy"], dtype=np.float32), np.asarray(f["vz"], dtype=np.float32)],
            axis=1,
        )
        mass = np.asarray(f["m"], dtype=np.float32) if "m" in f else np.ones((pos.shape[0],), dtype=np.float32)

    pos_grid = np.mod((pos / float(box_size)) * float(n_grid), float(n_grid))
    p_or_v = (vel / 100.0) * float(a_init) / float(box_size) * float(n_grid)

    omega_m = 0.3
    omega_b = 0.045
    bg = LCDMBackground(h=0.6711, Omega_m=omega_m, Omega_b=omega_b, Omega_lambda=0.7, use_jax_cosmo=False)

    base_eq = dh.equationmanager.EquationManager(n_cons=5)
    base_eq.mesh_shape = list(mesh_shape)
    eq = SuperComovingEquationManager(base_eq, enforce_gamma_53=True)

    gas_mass_fraction = float(args.gas_mean_fraction)
    dm_mass_fraction = max(1.0e-6, 1.0 - gas_mass_fraction)

    hydro_velocity_scale = float(bg.H0) * float(n_grid) / max(float(box_size), 1.0e-30)
    hydro_vel_unit_cms = float(args.vel_unit_cms) / max(hydro_velocity_scale, 1.0e-30)
    mH_cgs = 1.6735575e-24
    kB_cgs = 1.380649e-16
    kelvin_to_code_temp = kB_cgs / (
        max(float(args.mu_hydrogen), 1.0e-12) * mH_cgs * hydro_vel_unit_cms * hydro_vel_unit_cms
    )

    rho_norm_ic = jnp.asarray(density, dtype=jnp.float32) / jnp.maximum(jnp.mean(jnp.asarray(density, dtype=jnp.float32)), 1.0e-30)
    rho_gas_phys_ic = jnp.asarray(gas_mass_fraction, dtype=jnp.float32) * rho_norm_ic
    if bool(args.ic_import_density_a3):
        rho_gas_phys_ic = rho_gas_phys_ic / jnp.maximum(a_init**3, 1.0e-30)

    vx_pec = (jnp.asarray(xmom, dtype=jnp.float32) / jnp.maximum(jnp.asarray(density, dtype=jnp.float32), 1.0e-30)) / float(args.nyx_velocity_divisor)
    vy_pec = (jnp.asarray(ymom, dtype=jnp.float32) / jnp.maximum(jnp.asarray(density, dtype=jnp.float32), 1.0e-30)) / float(args.nyx_velocity_divisor)
    vz_pec = (jnp.asarray(zmom, dtype=jnp.float32) / jnp.maximum(jnp.asarray(density, dtype=jnp.float32), 1.0e-30)) / float(args.nyx_velocity_divisor)
    v_scale = jnp.asarray(hydro_velocity_scale, dtype=jnp.float32)
    vx_pec = vx_pec * v_scale
    vy_pec = vy_pec * v_scale
    vz_pec = vz_pec * v_scale

    temp_floor_k = max(float(args.hydro_temp_floor_k), 0.0)
    temp_code = jnp.maximum(jnp.asarray(temp_k, dtype=jnp.float32), temp_floor_k) * jnp.asarray(kelvin_to_code_temp, dtype=jnp.float32)
    p_gas_phys_ic = rho_gas_phys_ic * eq.R * temp_code

    w_code = cosmo_conv.primitives_phys_to_code(rho_gas_phys_ic, vx_pec, vy_pec, vz_pec, p_gas_phys_ic, a_init)
    U0 = eq.get_conservatives_from_primitives(w_code)

    bg_force = BackgroundExpansionForce(bg, a_init=a_init)
    if args.gravity_backend == "fft":
        grav_force = JaxPMCoupledGravityForce(
            eq,
            mesh_shape=mesh_shape,
            subtract_mean=True,
            dm_drift_factor=bg.H0,
            dm_kick_factor=1.0,
            gas_kick_factor=None,
            eps=float(args.force_eps),
        )
    elif args.gravity_backend == "multigrid":
        grav_force = CoupledMGGravityForce(
            eq,
            mesh_shape=mesh_shape,
            subtract_mean=True,
            dm_drift_factor=bg.H0,
            dm_kick_factor=1.0,
            gas_kick_factor=None,
            eps=float(args.force_eps),
        )
    else:
        grav_force = CoupledJaxDecompFFTGravityForce(
            eq,
            mesh_shape=mesh_shape,
            subtract_mean=True,
            dm_drift_factor=bg.H0,
            dm_kick_factor=1.0,
            gas_kick_factor=None,
            eps=float(args.force_eps),
            sharding=jaxdecomp_field_sharding,
            halo_size=int(args.jaxdecomp_halo_size),
        )

    force_stack = [bg_force, grav_force]
    if bool(args.enable_cooling):
        z_nodes = parse_float_list(args.nyx_cooling_z_nodes)
        ld_nodes = np.linspace(
            float(args.nyx_cooling_logdelta_min),
            float(args.nyx_cooling_logdelta_max),
            int(max(2, args.nyx_cooling_logdelta_n)),
            dtype=np.float64,
        )
        lt_nodes = np.linspace(
            float(args.nyx_cooling_logt_min),
            float(args.nyx_cooling_logt_max),
            int(max(2, args.nyx_cooling_logt_n)),
            dtype=np.float64,
        )
        nyx_table_path = args.nyx_cooling_table_npz.resolve()
        if bool(args.nyx_cooling_rebuild) or (not nyx_table_path.exists()):
            build_nyx_cooling_table(
                nyx_table_path,
                treecool_file=args.nyx_cooling_treecool.resolve(),
                z_nodes=z_nodes,
                logdelta_nodes=ld_nodes,
                logT_nodes=lt_nodes,
                h=float(bg.h),
                omega_b=float(bg.Omega_b),
                h_species=float(args.h_species),
                uvb_density_a=1.0,
                uvb_density_b=0.0,
                jh=1,
                jhe=1,
                eos_search_paths=[args.nyx_cooling_eos_path.resolve()],
            )
        nyx_table = NyxCoolingTableData.load(nyx_table_path)
        nyx_interp = NyxCoolingRateInterpolator(nyx_table)
        force_stack.append(
            NyxTabulatedCoolingForce(
                eq,
                nyx_interp,
                rho_unit_cgs=float(args.rho_unit_cgs),
                vel_unit_cms=float(hydro_vel_unit_cms),
                tau_time_unit_s=float(args.tau_time_unit_s),
                mu=float(args.mu_hydrogen),
                heating_rate_scale=float(args.nyx_heating_scale),
                cooling_rate_scale=float(args.cooling_rate_scale),
                temp_floor_k=float(args.cooling_temp_floor_k),
                subcycles=int(args.cooling_subcycles),
                dtmax_s=float(args.cooling_dtmax_s),
            )
        )

    active_solver = str(args.solver).lower()
    sim = build_hydrosim(eq, force_stack, solver_name=active_solver, pmesh_shape_local=pmesh_shape, dx_o=1.0)

    params0 = {
        "a": jnp.asarray(a_init, dtype=jnp.float32),
        "dm": {
            "x": jnp.asarray(pos_grid, dtype=jnp.float32),
            "p_or_v": jnp.asarray(p_or_v, dtype=jnp.float32),
            "mass": jnp.asarray(dm_mass_fraction, dtype=jnp.float32),
            "drift_factor": jnp.asarray(bg.H0, dtype=jnp.float32),
            "kick_prefactor": jnp.asarray(1.5 * omega_m * bg.H0 * float(args.dm_kick_scale), dtype=jnp.float32),
        },
        "_hydro_repair_count": jnp.int32(0)
    }
    if args.gravity_backend == "jaxdecomp_fft":
        if int(pos_grid.shape[0]) != int(np.prod(mesh_shape)):
            raise ValueError(
                "jaxdecomp_fft backend currently requires one DM particle per gas cell; "
                f"got n_dm={pos_grid.shape[0]} and gas cells={int(np.prod(mesh_shape))}."
            )
        dm_disp0_np, dm_p0_np = _hierarchical_reconstruct_disp_and_momentum(
            np.asarray(pos_grid, dtype=np.float32),
            np.asarray(p_or_v, dtype=np.float32),
            int(n_grid),
        )
        dm_sharding = NamedSharding(jaxdecomp_field_sharding.mesh, P("x", "y", None, None))
        params0["dm"] = {
            "disp": jax.device_put(jnp.asarray(dm_disp0_np, dtype=jnp.float32), dm_sharding),
            "p_or_v": jax.device_put(jnp.asarray(dm_p0_np, dtype=jnp.float32), dm_sharding),
            "mass": jnp.asarray(dm_mass_fraction, dtype=jnp.float32),
            "drift_factor": jnp.asarray(bg.H0, dtype=jnp.float32),
            "kick_prefactor": jnp.asarray(1.5 * omega_m * bg.H0 * float(args.dm_kick_scale), dtype=jnp.float32),
        }
    if args.gas_kick_mode == "legacy_h0sq":
        gas_kick_prefactor = 1.5 * omega_m * (bg.H0**2) * float(args.gas_kick_scale)
    else:
        gas_kick_prefactor = 1.5 * omega_m * bg.H0 * float(args.gas_kick_scale)
    params0["dm"]["gas_kick_prefactor"] = jnp.asarray(gas_kick_prefactor, dtype=jnp.float32)

    tau_target = integrate_tau_target(bg, a_init, a_target)
    checkpoint_rows: list[dict[str, float]] = []
    checkpoints_dir = out_dir / "checkpoints"
    z_checkpoint_list = []
    if str(args.checkpoint_z_values).strip():
        z_checkpoint_list = [float(x) for x in str(args.checkpoint_z_values).split(",") if x.strip()]
    z_checkpoint_list = sorted(set(z_checkpoint_list), reverse=True)
    checkpoint_targets = [(zz, 1.0 / (1.0 + zz)) for zz in z_checkpoint_list]
    checkpoint_next = 0

    step_i = 0
    tol = 1.0e-6
    dtau_retry_cap = float(args.dtau_max)
    prev_accepted_dtau = None
    retry_total = 0
    retry_steps_nonzero = 0
    max_retry_in_step = 0
    U = U0
    params = params0

    t0 = time.perf_counter()
    dt_hist = []
    while float(params["a"]) + tol < float(a_target):
        if step_i >= int(args.max_steps):
            break
        a_now = float(params["a"])
        da_dtau = float(bg.da_dtau(a_now))
        remaining = max(float(a_target) - a_now, 0.0)
        da_dtau_safe = max(da_dtau, 1.0e-12)
        dtau_by_target = float(args.dtau_safety) * remaining / da_dtau_safe
        dtau_by_rel_a = float("inf")
        if float(args.relative_max_change_a) > 0.0:
            dtau_by_rel_a = float(args.relative_max_change_a) * max(a_now, 1.0e-12) / da_dtau_safe
        dtau_by_abs_a = float("inf")
        if float(args.absolute_max_change_a) > 0.0:
            dtau_by_abs_a = float(args.absolute_max_change_a) / da_dtau_safe
        dtau_solver = float("inf")
        if not bool(args.disable_solver_dt):
            dtau_solver = float(args.solver_dt_safety) * float(sim.timestep(U))
        dtau = min(
            float(args.dtau_max),
            float(dtau_retry_cap),
            dtau_by_target,
            dtau_by_rel_a,
            dtau_by_abs_a,
            dtau_solver,
        )
        if prev_accepted_dtau is not None and float(args.dtau_change_max) > 0.0:
            dtau = min(dtau, float(args.dtau_change_max) * float(prev_accepted_dtau))
        if step_i == 0:
            dtau = dtau * float(args.dtau_init_shrink)
        dtau = max(dtau, float(args.dtau_min))

        accepted = False
        trial_dtau = float(dtau)
        retries_this_step = 0
        for retry_i in range(int(args.step_retries) + 1):
            U_try_raw, params_try = sim._hydrostep(step_i, (U, params), jnp.asarray(trial_dtau, dtype=jnp.float32))
            w_try = eq.get_primitives_from_conservatives(U_try_raw)
            a_try = params_try.get("a", params.get("a", 1.0))
            rho_floor, p_floor = _compute_phys_floor_code(
                a_try,
                float(args.state_floor),
                float(args.pressure_floor),
                w_try.dtype,
            )
            t_floor_k = max(float(args.hydro_temp_floor_k), 0.0)
            if t_floor_k <= 0.0 and bool(args.enable_cooling):
                t_floor_k = max(float(args.cooling_temp_floor_k), 0.0)
            if t_floor_k > 0.0:
                t_floor_code = jnp.asarray(kelvin_to_code_temp * t_floor_k, dtype=w_try.dtype)
                rho_phys_try = cosmo_conv.density_code_to_phys(w_try[0], a_try)
                p_floor_temp_phys = rho_phys_try * jnp.asarray(float(eq.R), dtype=w_try.dtype) * t_floor_code
                p_floor_temp_code = cosmo_conv.pressure_phys_to_code(p_floor_temp_phys, a_try)
                p_floor = jnp.maximum(p_floor, p_floor_temp_code)
            w_try = w_try.at[0].set(jnp.maximum(w_try[0], rho_floor))
            w_try = w_try.at[4].set(jnp.maximum(w_try[4], p_floor))
            U_try_proj = eq.get_conservatives_from_primitives(w_try)
            if _state_is_finite(U_try_proj, params_try):
                U, params = U_try_proj, params_try
                dt_hist.append(float(trial_dtau))
                if retry_i > 0:
                    dtau_retry_cap = min(float(dtau_retry_cap), float(trial_dtau) * float(args.retry_cap_mult))
                else:
                    dtau_retry_cap = min(float(args.dtau_max), float(dtau_retry_cap) * 1.1)
                dtau_retry_cap = max(float(args.dtau_min), float(dtau_retry_cap))
                accepted = True
                break
            retries_this_step = max(retries_this_step, retry_i + 1)
            retry_total += 1
            trial_dtau = trial_dtau * float(args.dtau_retry_factor)
            if trial_dtau < float(args.dtau_min):
                break
        if retries_this_step > 0:
            retry_steps_nonzero += 1
            max_retry_in_step = max(max_retry_in_step, retries_this_step)
        if not accepted:
            raise RuntimeError(f"Failed to take finite step at step={step_i}, a={a_now:.6f}")
        prev_accepted_dtau = float(trial_dtau)
        a_after = float(params["a"])
        while checkpoint_next < len(checkpoint_targets):
            z_req, a_req = checkpoint_targets[checkpoint_next]
            if (a_now - tol) < a_req <= (a_after + tol):
                label = f"snapshot_z{int(round(z_req))}" if abs(z_req - round(z_req)) < 1.0e-6 else f"snapshot_z{z_req:.3f}".replace(".", "p")
                checkpoint_rows.append(
                    _dump_state(
                        checkpoints_dir,
                        U,
                        params,
                        dt_hist_local=dt_hist,
                        label=label,
                    )
                )
                checkpoint_rows[-1]["z_requested"] = float(z_req)
                checkpoint_next += 1
            else:
                break
        step_i += 1

    jax.block_until_ready(U)
    elapsed = time.perf_counter() - t0

    Uf = U
    paramsf = params
    t_f = float(sum(dt_hist))
    n_steps = step_i

    Uf_np = np.asarray(jax.device_get(Uf), dtype=np.float32)
    a_fin = float(np.asarray(jax.device_get(paramsf["a"])))
    z_fin = 1.0 / max(a_fin, 1.0e-30) - 1.0
    w_fin = np.asarray(jax.device_get(eq.get_primitives_from_conservatives(Uf)), dtype=np.float32)
    rho_fin = np.asarray(jax.device_get(cosmo_conv.density_code_to_phys(w_fin[0], a_fin)), dtype=np.float32)
    p_fin = np.asarray(jax.device_get(cosmo_conv.pressure_code_to_phys(w_fin[4], a_fin)), dtype=np.float32)
    temp_code_fin = p_fin / np.maximum(rho_fin * float(eq.R), 1.0e-30)
    temp_fin = np.asarray(
        jax.device_get(
            _code_temp_to_kelvin(
                jnp.asarray(temp_code_fin, dtype=jnp.float32),
                mu_hydrogen=float(args.mu_hydrogen),
                vel_unit_cms=float(hydro_vel_unit_cms),
            )
        ),
        dtype=np.float32,
    )
    dt_hist_np = np.asarray(dt_hist, dtype=np.float32)

    dm_mesh = np.asarray(jax.device_get(_dm_density_mesh(paramsf["dm"])), dtype=np.float32)

    np.savez_compressed(
        out_dir / "final_fields.npz",
        conserved=Uf_np,
        rho_phys=rho_fin,
        pressure_phys=p_fin,
        temperature=temp_fin,
        temperature_code=temp_code_fin,
        dm_density=dm_mesh,
        dt_hist=dt_hist_np,
        a=a_fin,
        z=z_fin,
        box_size=float(box_size),
        n_grid=int(n_grid),
    )

    k = n_grid // 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    im0 = axes[0, 0].imshow(np.log10(np.maximum(density[:, :, k], 1.0e-30)), origin="lower", cmap="viridis")
    axes[0, 0].set_title("IC gas density")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.85)
    im1 = axes[0, 1].imshow(np.log10(np.maximum(rho_fin[:, :, k], 1.0e-30)), origin="lower", cmap="viridis")
    axes[0, 1].set_title(f"Final gas density z={z_fin:.2f}")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.85)
    im2 = axes[1, 0].imshow(np.log10(np.maximum(dm_mesh[:, :, k], 1.0e-30)), origin="lower", cmap="magma")
    axes[1, 0].set_title(f"Final DM density z={z_fin:.2f}")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.85)
    im3 = axes[1, 1].imshow(np.log10(np.maximum(temp_fin[:, :, k], 1.0e-30)), origin="lower", cmap="inferno")
    axes[1, 1].set_title(f"Final Temp[K] z={z_fin:.2f}")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.85)
    fig.savefig(out_dir / "slices_compare.png", dpi=160)
    plt.close(fig)

    summary = {
        "run": {
            "plotfile": (str(plotfile) if args.cached_ic_npz is None else None),
            "cached_ic_npz": (None if args.cached_ic_npz is None else str(args.cached_ic_npz.resolve())),
            "dm_h5": str(dm_h5),
            "n_grid": int(n_grid),
            "box_size": float(box_size),
            "a_init": float(a_init),
            "z_init": float(1.0 / a_init - 1.0),
            "a_target": float(a_target),
            "z_target": float(args.z_target),
            "a_final": float(a_fin),
            "z_final": float(z_fin),
            "gravity_backend": args.gravity_backend,
            "coupling_impl": (
                "core_jaxpm"
                if args.gravity_backend == "fft"
                else ("core_logic_mg" if args.gravity_backend == "multigrid" else "jaxdecomp_pm")
            ),
            "solver": args.solver,
            "dm_kick_scale": float(args.dm_kick_scale),
            "gas_kick_scale": float(args.gas_kick_scale),
            "gas_kick_mode": str(args.gas_kick_mode),
            "kick_prefactor_dm": float(1.5 * omega_m * bg.H0 * float(args.dm_kick_scale)),
            "kick_prefactor_gas": float(gas_kick_prefactor),
            "pmesh_shape": list(pmesh_shape),
            "jaxdecomp_pdims": (list(pdims) if args.gravity_backend == "jaxdecomp_fft" else None),
            "jaxdecomp_halo_size": (int(args.jaxdecomp_halo_size) if args.gravity_backend == "jaxdecomp_fft" else None),
            "ic_import_density_a3": bool(args.ic_import_density_a3),
            "gpu": args.gpu,
            "device_count_visible": int(jax.device_count()),
            "elapsed_s": float(elapsed),
            "n_steps_taken": int(n_steps),
            "tau_target": float(tau_target),
            "t_final": float(t_f),
            "dm_particles": int(pos_grid.shape[0]),
            "retry_total": int(retry_total),
            "retry_steps_nonzero": int(retry_steps_nonzero),
            "max_retry_in_step": int(max_retry_in_step),
        },
        "stats": {
            "rho_final_mean": float(np.mean(rho_fin)),
            "rho_final_std": float(np.std(rho_fin)),
            "temp_final_mean_k": float(np.mean(temp_fin)),
            "temp_final_std_k": float(np.std(temp_fin)),
            "dm_final_mean": float(np.mean(dm_mesh)),
            "dm_final_std": float(np.std(dm_mesh)),
            "dt_nonzero_count": int(np.count_nonzero(dt_hist_np > 0.0)),
        },
        "checkpoints": checkpoint_rows,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"[done] wrote gadget-IC coupled outputs to {out_dir}")


if __name__ == "__main__":
    main()
