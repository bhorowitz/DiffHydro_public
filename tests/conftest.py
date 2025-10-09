# conftest.py
import os
import sys
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store",
        default=None,
        help=(
            "GPU(s) to use. Examples: "
            "'0' (single), '1', '0,2' (multi), or 'none' to force CPU."
        ),
    )

from typing import Optional

def _set_cuda_visible_devices_from_option(gpu_opt: Optional[str]):   
    """
    Apply the GPU selection early so libraries see the right devices when imported.
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    if gpu_opt is None:
        return  # don't touch env; use whatever the environment already has
    if gpu_opt.lower() == "none":
        # mask out all GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # e.g. "0" or "0,2"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_opt

def pytest_load_initial_conftests(args):
    """
    This hook runs very early, before most imports.
    It lets us set CUDA_VISIBLE_DEVICES before libraries like torch/cupy/jax are imported.
    """
    # Parse --gpu from raw args (no pytest config yet), fall back to env or default
    # Simple parse: find the --gpu argument manually
    gpu_opt = None
    for i, a in enumerate(args):
        if a.startswith("--gpu"):
            if a == "--gpu" and i + 1 < len(args):
                gpu_opt = args[i + 1]
            elif a.startswith("--gpu="):
                gpu_opt = a.split("=", 1)[1]
            break
    _set_cuda_visible_devices_from_option(gpu_opt)
