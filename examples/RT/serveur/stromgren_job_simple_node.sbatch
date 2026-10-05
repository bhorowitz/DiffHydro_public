#!/usr/bin/env bash
#SBATCH --job-name=stromgren_validation
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpu-bind=none
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=pbs/%x_%j.out
#SBATCH --error=pbs/%x_%j.err
#SBATCH --mem-per-cpu=30GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=adrien.bernard@etu.u-bordeaux.fr
#SBATCH --mail-type=BEGIN,END,FAIL


export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1


source ~/.bashrc
cd /home/adrien/DiffHydro_public

eval "$(conda shell.bash hook)"
conda activate astro-env

export NCCL_LIB_DIR="/home/adrien/miniconda3/envs/astro-env/lib/python3.11/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$LD_LIBRARY_PATH"

srun env N=150 python examples/RT/stromgren_validation_simple_node.py
srun env python examples/RT/RT_run_chat_v5.py
