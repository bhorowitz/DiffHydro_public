#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_stage_module():
    import importlib.util

    mod_path = Path(__file__).resolve().with_name("run_stellar_metallicity_stages.py")
    spec = importlib.util.spec_from_file_location("run_stellar_metallicity_stages", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read_config(path: Path) -> dict:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("YAML config requested but PyYAML is not installed.") from exc
        data = yaml.safe_load(text)
        return {} if data is None else dict(data)
    raise ValueError(f"Unsupported config file format: {path}")


def _default_stage_args(stage_mod) -> argparse.Namespace:
    saved_argv = sys.argv[:]
    try:
        sys.argv = [saved_argv[0]]
        return stage_mod._parse_args()
    finally:
        sys.argv = saved_argv


def _merge_config(base: argparse.Namespace, config: dict) -> argparse.Namespace:
    legacy_driver_keys = {
        "enable_vacuum_momentum_cap",
        "vacuum_momentum_rho_guard",
        "vacuum_momentum_kinetic_to_thermal_max",
        "vacuum_momentum_internal_floor",
        "enable_hydro_state_repair",
        "hydro_repair_rho_floor",
        "hydro_repair_pressure_floor",
        "hydro_repair_max_kinetic_to_thermal_ratio",
    }
    merged = vars(base).copy()
    for key, value in config.items():
        if key in legacy_driver_keys:
            continue
        if key not in merged:
            raise KeyError(f"Unknown config key: {key}")
        merged[key] = value
    return argparse.Namespace(**merged)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the CV0 feedback stage from a config file."
    )
    p.add_argument("--config", type=Path, required=True, help="JSON or YAML config file.")
    p.add_argument("--output-subdir", type=str, default=None, help="Override the output subdirectory name.")
    p.add_argument("--gpu", type=str, default=None, help="Override CUDA_VISIBLE_DEVICES.")
    p.add_argument("--xla-preallocate", choices=["true", "false"], default=None, help="Override XLA preallocation.")
    return p.parse_args()


def main() -> None:
    cli = _parse_args()
    stage_mod = _load_stage_module()
    base_args = _default_stage_args(stage_mod)

    config = _read_config(cli.config.resolve())
    args = _merge_config(base_args, config)
    args.stage = "cv0"

    if cli.gpu is not None:
        args.gpu = cli.gpu
    if cli.xla_preallocate is not None:
        args.xla_preallocate = (cli.xla_preallocate.lower() == "true")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if bool(args.xla_preallocate) else "false"
    os.environ.setdefault("MPLBACKEND", "Agg")

    t0 = time.perf_counter()
    stage_dirname = cli.output_subdir or config.get("output_subdir") or "stage_10_cv0"
    stage_dir = Path(args.output_root) / stage_dirname

    stage_mod._run_hydro_stage(args, "cv0", stage_dir)

    elapsed = time.perf_counter() - t0
    print(f"[ok] stage=cv0 wrote outputs to {stage_dir}")
    print(f"[ok] elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
