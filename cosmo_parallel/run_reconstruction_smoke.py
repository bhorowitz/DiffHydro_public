#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmo_parallel.common import write_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small reconstruction / backprop smoke wrapper.")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--model", choices=["simple", "full_hydro"], default="simple")
    p.add_argument("--observable", choices=["lya_flux", "xray_proxy"], default="lya_flux")
    p.add_argument("--mesh-n", type=int, default=32)
    p.add_argument("--n-iters", type=int, default=1)
    p.add_argument("--hydro-steps", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=Path("cosmo_parallel/results/reconstruction_smoke"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    driver = (
        "cosmo_reconstruct/run_optimize_cv0_density_full_hydro.py"
        if args.model == "full_hydro"
        else "cosmo_reconstruct/run_optimize_cv0_density.py"
    )

    cmd = [
        sys.executable,
        driver,
        "--gpu",
        args.gpu,
        "--mesh-n",
        str(args.mesh_n),
        "--observable",
        args.observable,
        "--target-source",
        "self_consistent",
        "--self-target-seed",
        "7",
        "--self-target-scale",
        "1.0",
        "--self-target-noise-sigma",
        "0.0",
        "--optimizer",
        "adam",
        "--adam-lr",
        "1.0e-3",
        "--n-iters",
        str(args.n_iters),
        "--output-dir",
        str(args.output_dir),
    ]

    if args.model == "full_hydro":
        cmd.extend(
            [
                "--z-init",
                "20.0",
                "--z-target",
                "2.0",
                "--hydro-steps",
                str(args.hydro_steps),
                "--solver",
                "hll",
            ]
        )

    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)

    summary = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
        "output_dir": str(args.output_dir),
    }
    write_json(args.output_dir / "launcher_summary.json", summary)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print(f"[done] wrote reconstruction smoke outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
