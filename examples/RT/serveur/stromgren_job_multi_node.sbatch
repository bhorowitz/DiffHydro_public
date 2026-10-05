#!/usr/bin/env bash
#SBATCH --job-name=stromgren_validation

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=none
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --output=pbs/%x_%j.out
#SBATCH --error=pbs/%x_%j.err
#SBATCH --mem-per-cpu=18GB
#SBATCH --cpus-per-task=9
#SBATCH --mail-user=adrien.bernard@etu.u-bordeaux.fr
#SBATCH --mail-type=BEGIN,END,FAIL


export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno3


source ~/.bashrc
cd /home/adrien/DiffHydro_public

eval "$(conda shell.bash hook)"
conda activate astro-env

export NCCL_LIB_DIR="/home/adrien/miniconda3/envs/astro-env/lib/python3.11/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$LD_LIBRARY_PATH"

srun env N=150 python examples/RT/stromgren_validation_multi_node.py