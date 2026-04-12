#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

N_GRID="${N_GRID:-64}"
N_PAR="${N_PAR:-4096}"    # Direct summation is O(N^2); 4096 is a safer single-GPU default
A_FINAL="${A_FINAL:-0.9500}"
MAX_STEPS="${MAX_STEPS:-800}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-10}"
MAX_DTAU="${MAX_DTAU:-50.0}"
MIN_DTAU="${MIN_DTAU:-1.0e-3}"
DTAU_SAFETY="${DTAU_SAFETY:-0.8}"
PRESSURE_SCALE="${PRESSURE_SCALE:-1.00}"
VELOCITY_SCALE="${VELOCITY_SCALE:-1.00}"
DM_KICK_SCALE="${DM_KICK_SCALE:-1.0}"
GAS_KICK_SCALE="${GAS_KICK_SCALE:-1.0}"
SOFTENING_CELLS="${SOFTENING_CELLS:-0.5}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-stage5_direct_example}"

python "$ROOT_DIR/merger/stage5_gas_dm_cosmo_halo.py" \
  --gravity-method direct \
  --cancel-hubble-flow \
  --n-grid "$N_GRID" \
  --n-par "$N_PAR" \
  --a-final "$A_FINAL" \
  --max-steps "$MAX_STEPS" \
  --snapshot-every "$SNAPSHOT_EVERY" \
  --max-dtau "$MAX_DTAU" \
  --min-dtau "$MIN_DTAU" \
  --dtau-safety "$DTAU_SAFETY" \
  --pressure-scale "$PRESSURE_SCALE" \
  --velocity-scale "$VELOCITY_SCALE" \
  --dm-kick-scale "$DM_KICK_SCALE" \
  --gas-kick-scale "$GAS_KICK_SCALE" \
  --softening-cells "$SOFTENING_CELLS" \
  --output-subdir "$OUTPUT_SUBDIR"
