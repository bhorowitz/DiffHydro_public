#!/usr/bin/env python3
"""Convert a run_gadgetic_coupled_parallel checkpoint into the fields npz
format expected by postprocess_lya_nhi_compare.py.

The coupled-parallel driver stores the supercomoving conserved state plus
derived physical fields in `<label>_fields.npz`. This script inverts the
driver's velocity convention (v_code -> peculiar velocity in cm/s) using the
same internal background (LCDMBackground h=0.6711 => H0=67.11) and velocity
unit (1e7 cm/s) that the driver hardcodes, and writes:

  gas_dh_final      baryon density normalized to its mean
  temp_dh_final     temperature in Kelvin
  v{x,y,z}_dh_final_cms  peculiar velocity in cm/s
  a_final_dh        scale factor of the checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp

import diffhydro as dh
from diffhydro.cosmology import SuperComovingEquationManager
from diffhydro.cosmology import conversions as cosmo_conv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-npz", type=Path, required=True,
                   help="snapshot_*_fields.npz or final_fields.npz from run_gadgetic_coupled_parallel.py")
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--bg-h", type=float, default=0.6711,
                   help="h of the LCDMBackground the coupled driver ran with (hardcoded 0.6711).")
    p.add_argument("--vel-unit-cms", type=float, default=1.0e7,
                   help="Velocity unit of the driver's Nyx-IC convention (v = (mom/rho)/100 in these units).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    d = np.load(args.checkpoint_npz)

    U = jnp.asarray(d["conserved"], dtype=jnp.float32)
    a = float(np.asarray(d["a"]))
    n_grid = int(np.asarray(d["n_grid"]))
    box_size = float(np.asarray(d["box_size"]))
    rho_phys = np.asarray(d["rho_phys"], dtype=np.float32)
    temp_k = np.asarray(d["temperature"], dtype=np.float32)

    base_eq = dh.equationmanager.EquationManager(n_cons=5)
    base_eq.mesh_shape = [n_grid, n_grid, n_grid]
    eq = SuperComovingEquationManager(base_eq, enforce_gamma_53=True)
    W = eq.get_primitives_from_conservatives(U)

    # Invert the driver's convention: v_code -> "phys" (internal grid units)
    # -> cm/s via hydro_velocity_scale = H0 * n_grid / box_size, H0 = 100*bg_h.
    hydro_velocity_scale = 100.0 * float(args.bg_h) * float(n_grid) / max(box_size, 1.0e-30)
    hydro_vel_unit_cms = float(args.vel_unit_cms) / max(hydro_velocity_scale, 1.0e-30)

    vx_cms = np.asarray(cosmo_conv.velocity_code_to_phys(W[1], a), dtype=np.float32) * hydro_vel_unit_cms
    vy_cms = np.asarray(cosmo_conv.velocity_code_to_phys(W[2], a), dtype=np.float32) * hydro_vel_unit_cms
    vz_cms = np.asarray(cosmo_conv.velocity_code_to_phys(W[3], a), dtype=np.float32) * hydro_vel_unit_cms

    gas_norm = rho_phys / max(float(np.mean(rho_phys)), 1.0e-30)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        gas_dh_final=gas_norm.astype(np.float32),
        temp_dh_final=temp_k,
        vx_dh_final_cms=vx_cms,
        vy_dh_final_cms=vy_cms,
        vz_dh_final_cms=vz_cms,
        a_final_dh=np.float32(a),
    )
    print(f"[done] wrote {args.output_npz}")
    print(f"  a={a:.6f} z={1.0 / a - 1.0:.4f} n_grid={n_grid} box={box_size:.4f}")
    print(f"  gas_norm mean={float(np.mean(gas_norm)):.4f} std={float(np.std(gas_norm)):.4f}")
    print(f"  temp_k p50={float(np.percentile(temp_k, 50)):.1f} K")
    print(f"  |v| p50={float(np.percentile(np.abs(vz_cms), 50)) / 1.0e5:.2f} km/s (z-comp)")


if __name__ == "__main__":
    main()
