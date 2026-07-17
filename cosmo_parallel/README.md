## Cosmology Parallel Workspace

This folder collects the first-pass setup for multi-GPU cosmology work in DiffHydro.

### What is here

- `common.py`
  Shared helpers for parsing mesh shapes, integrating the background, and saving JSON.
- `run_hydro_only_cosmo_parallel.py`
  Hydro-only cosmological run with optional gas self-gravity backend selection.
- `run_hydro_only_smoke_matrix.py`
  Small local test matrix:
  - 4-GPU hydro-only background expansion
  - 1-GPU hydro + FFT gravity
  - 1-GPU hydro + multigrid gravity
- `run_dm_jaxdecomp_smoke.py`
  Multi-GPU dark-matter-only smoke test using the `jaxdecomp` environment.
- `run_reconstruction_smoke.py`
  Thin wrapper around the existing reconstruction workflow for small LyA/X-ray smoke tests.
- `batch/`
  SLURM launchers for 4-GPU node runs.
- `results/`
  Output artifacts and summaries from the smoke tests.

### Current scope

This setup intentionally separates the work into three tracks:

1. Hydro-only multi-GPU cosmology
   This uses the existing DiffHydro sharded hydro machinery.
2. JAX Decomp dark-matter evolution
   This is a separate prototype path in the `jaxdecomp` env, based on the existing sandbox examples.
3. Reconstruction / backprop smoke tests
   This reuses `cosmo_reconstruct/` through a launcher with small, self-consistent targets.

### Important limitation

The current gas gravity backends used here (`fft` and `multigrid`) are validated on replicated meshes.
They are useful for backend comparison, but they are not yet wired into a fully sharded cosmological gravity solve.

So the intended usage right now is:

- Multi-GPU hydro-only smoke: `--gravity-backend none`
- Single-GPU backend comparison: `--gravity-backend fft` and `--gravity-backend multigrid`
- Multi-GPU DM FFT path: `run_dm_jaxdecomp_smoke.py`

### Environments

Hydro / reconstruction:

```bash
source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu
```

JAX Decomp:

```bash
source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jaxdecomp
```

### Local smoke tests

Hydro-only matrix:

```bash
source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu
python cosmo_parallel/run_hydro_only_smoke_matrix.py
```

JAX Decomp DM smoke:

```bash
source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jaxdecomp
python cosmo_parallel/run_dm_jaxdecomp_smoke.py \
  --mesh-n 128 \
  --pdims 4x1 \
  --output-dir cosmo_parallel/results/dm_jaxdecomp_local
```

Reconstruction smoke:

```bash
source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu
python cosmo_parallel/run_reconstruction_smoke.py \
  --observable lya_flux \
  --output-dir cosmo_parallel/results/recon_lya_smoke
```
