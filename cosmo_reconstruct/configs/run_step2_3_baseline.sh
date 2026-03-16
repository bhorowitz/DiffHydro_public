#!/usr/bin/env bash
set -euo pipefail

source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu

cd /home/ben.horowitz/DiffHydro_public
RECON_IC_POWER_SUPPRESSION=${RECON_IC_POWER_SUPPRESSION:-1.0}
CUDA_VISIBLE_DEVICES=3 python cosmo_reconstruct/run_optimize_cv0_density.py \
  --mesh-n 128 \
  --box-size-mpc-h 25.0 \
  --z-init 127.0 \
  --z-target 2.0 \
  --kdk-steps 16 \
  --checkpoint-every 4 \
  --h 0.6711 \
  --omega-m 0.3 \
  --omega-b 0.045 \
  --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
  --compare-space log \
  --noise-sigma 0.35 \
  --prior-weight 1.0 \
  --n-iters 5 \
  --adam-lr 2e-4 \
  --init-random-scale 0.7 \
  --output-dir cosmo_reconstruct/outputs/step2_3_optimize128_k16_iter5
