#!/bin/bash
#SBATCH --job-name=dh-n400lf
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=main

set -euo pipefail

source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jaxdecomp

cd /home/ben.horowitz/DiffHydro_public

python cosmo_parallel/run_gadgetic_coupled_parallel.py \
  --gpu 0,1,2,3 \
  --cached-ic-npz cosmo_parallel/results/cache/music25_n400_true_init_fields.npz \
  --nyx-dm-ic-h5 Nyx/Exec/LyA/music25_runs/n400_true_init/plt_init_00000_particles.h5 \
  --pmesh-shape 2x2x1 \
  --jaxdecomp-pdims 2x2 \
  --jaxdecomp-halo-size 60 \
  --solver lf \
  --gravity-backend jaxdecomp_fft \
  --no-ic-import-density-a3 \
  --gas-kick-mode legacy_h0sq \
  --checkpoint-z-values 10,5,2 \
  --z-target 0 \
  --max-steps 20000 \
  --output-dir cosmo_parallel/results/2music25_n400_lf_jaxdecomp_fft_4gpu_z2_rankfix_halo32_pdims22
