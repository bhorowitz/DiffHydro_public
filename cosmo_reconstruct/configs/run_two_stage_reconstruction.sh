#!/usr/bin/env bash
set -euo pipefail

source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu

cd /home/ben.horowitz/DiffHydro_public

SIMPLE_OUT="cosmo_reconstruct/outputs/stage1d"
FULL_OUT="cosmo_reconstruct/outputs/stage2d"
RECON_IC_POWER_SUPPRESSION=${RECON_IC_POWER_SUPPRESSION:-1.0}

# Stage A: fast approximate optimizer (painted DM->gas model)
CUDA_VISIBLE_DEVICES=3 python cosmo_reconstruct/run_optimize_cv0_density.py \
  --mesh-n 64 \
  --kdk-steps 64 \
  --checkpoint-every 4 \
  --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
  --optimizer adam \
  --adam-lr 0.003 \
  --lbfgs-linesearch zoom \
  --lbfgs-lr 1.0 \
  --lbfgs-memory 10 \
  --n-iters 200 \
    --save-every 20 \
  --output-dir "${SIMPLE_OUT}"

# Stage B: full hydro refinement, warm-started from Stage A
CUDA_VISIBLE_DEVICES=3 python cosmo_reconstruct/run_optimize_cv0_density_full_hydro.py \
  --mesh-n 64 \
  --z-init 127.0 \
  --z-target 2.0 \
  --hydro-steps 128 \
  --solver hll \
  --dual-energy \
  --rho-unit-cgs 1.6e-24 \
  --enable-cooling \
  --cooling-model nyx_table \
  --cooling-stop-gradient \
  --sanitize-nonfinite-grads \
  --nyx-heating-scale 1.0 \
  --nyx-cooling-table-npz diffhydro_gadgetic_n128_z127to2_hll/nyx_cooling_table.npz \
  --nyx-cooling-z-nodes 2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100 \
  --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
  --optimizer lbfgs \
  --lbfgs-linesearch zoom \
  --lbfgs-lr 0.5 \
  --lbfgs-memory 4 \
  --n-iters 100 \
  --init-white-noise-npy "${SIMPLE_OUT}/optimized_white_noise.npy" \
  --output-dir "${FULL_OUT}-dry-run"\
  --save-every 10 \
  --dry-run

# Stage B: full hydro refinement, warm-started from Stage A
CUDA_VISIBLE_DEVICES=3 python cosmo_reconstruct/run_optimize_cv0_density_full_hydro.py \
  --mesh-n 64 \
  --z-init 127.0 \
  --z-target 2.0 \
  --hydro-steps 128 \
  --solver hll \
  --dual-energy \
  --rho-unit-cgs 1.6e-24 \
  --enable-cooling \
  --cooling-model nyx_table \
  --cooling-stop-gradient \
  --sanitize-nonfinite-grads \
  --nyx-heating-scale 1.0 \
  --nyx-cooling-table-npz diffhydro_gadgetic_n128_z127to2_hll/nyx_cooling_table.npz \
  --nyx-cooling-z-nodes 2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100 \
  --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
  --optimizer lbfgs \
  --lbfgs-linesearch zoom \
  --lbfgs-lr 0.5 \
  --lbfgs-memory 10 \
  --n-iters 1000 \
  --init-white-noise-npy "${SIMPLE_OUT}/optimized_white_noise.npy" \
  --save-every 10 \
  --output-dir "${FULL_OUT}"
