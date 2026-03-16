#!/usr/bin/env bash
set -euo pipefail

source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu

cd /home/ben.horowitz/DiffHydro_public
RECON_IC_POWER_SUPPRESSION=${RECON_IC_POWER_SUPPRESSION:-1.0}
CUDA_VISIBLE_DEVICES=3 python cosmo_reconstruct/run_optimize_cv0_density_full_hydro.py \
  --mesh-n 128 \
  --z-init 127.0 \
  --z-target 2.0 \
  --hydro-steps 128 \
  --solver hllc \
  --dual-energy \
  --rho-unit-cgs 1.6e-24 \
  --enable-cooling \
  --cooling-model nyx_table \
  --cooling-stop-gradient \
  --sanitize-nonfinite-grads \
  --nyx-heating-scale 1.2 \
  --nyx-cooling-table-npz diffhydro_gadgetic_n128_z127to2_hll/nyx_cooling_table.npz \
  --nyx-cooling-z-nodes 2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100 \
  --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
  --temperature0 1.0e4 \
  --temperature-gamma 0.6666666667 \
  --gas-mean-fraction 0.158 \
  --optimizer lbfgs \
  --lbfgs-linesearch zoom \
  --lbfgs-lr 0.5 \
  --lbfgs-memory 10 \
  --n-iters 10 \
  --compare-space log \
  --noise-sigma 0.05 \
  --prior-weight 1.0 \
  --init-white-noise-npy cosmo_reconstruct/outputs/step2_3_optimize/optimized_white_noise.npy \
  --output-dir cosmo_reconstruct/outputs/step2_3_optimize_full_hydro
