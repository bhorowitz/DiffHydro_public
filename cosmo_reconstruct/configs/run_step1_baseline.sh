#!/usr/bin/env bash
set -euo pipefail

source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu

cd /home/ben.horowitz/DiffHydro_public
RECON_IC_POWER_SUPPRESSION=${RECON_IC_POWER_SUPPRESSION:-1.0}
CUDA_VISIBLE_DEVICES=3 python cosmo_reconstruct/run_forward_model_cv0.py \
  --mesh-n 128 \
  --box-size-mpc-h 25.0 \
  --z-init 127.0 \
  --z-target 2.0 \
  --kdk-steps 64 \
  --checkpoint-every 4 \
  --h 0.6711 \
  --omega-m 0.3 \
  --omega-b 0.045 \
  --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
  --init-random-scale 1.0 \
  --output-dir cosmo_reconstruct/outputs/step1_forward
