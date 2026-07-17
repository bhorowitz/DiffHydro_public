# Cosmological Gas Reconstruction (CV0)

This folder contains a differentiable reconstruction pipeline that maps
`white noise -> LPT ICs -> PM evolution -> DM density -> gas density (+temperature)`.
It also includes a full-hydrodynamics refinement stage:
`white noise -> LPT ICs -> coupled DM+gas hydro evolution -> gas density`.

## Layout

- `run_forward_model_cv0.py`: Step 1 forward-model build + diagnostics + plots.
- `run_optimize_cv0_density.py`: Step 2/3 likelihood + backprop optimization against CV0 gas density.
- `run_optimize_cv0_density_full_hydro.py`: full-hydro refinement optimizer (warm-start from simple model).
- `src/forward_model.py`: differentiable model, checkpointed KDK integration, likelihood.
- `src/full_hydro_model.py`: differentiable coupled JaxPM+DiffHydro forward model.
- `src/diagnostics.py`: data loading, reference calibration, stats, plots.
- `configs/run_step1_baseline.sh`: baseline Step 1 launcher (GPU 3, `jax-gpu`).
- `configs/run_step2_3_baseline.sh`: baseline Step 2/3 launcher (GPU 3, `jax-gpu`).
- `configs/run_step2_3_full_hydro_baseline.sh`: baseline full-hydro refinement launcher.
- `configs/run_two_stage_reconstruction.sh`: full two-stage workflow (simple warm start -> full hydro).

## Environment

Use GPU 3 and conda env `jax-gpu`:

```bash
source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu
```

## Step 1: Forward Model + Diagnostics

```bash
CUDA_VISIBLE_DEVICES=3 python cosmo_reconstruct/run_forward_model_cv0.py \
  --mesh-n 128 \
  --kdk-steps 64 \
  --checkpoint-every 4 \
  --output-dir cosmo_reconstruct/outputs/step1_forward
```

This writes:

- `forward_fields.npz`
- `forward_stats.json`
- plot set (slices/projections/hists/scatter/power spectra)

## Step 2/3: Likelihood + Backprop Through Forward Model

```bash
CUDA_VISIBLE_DEVICES=3 python cosmo_reconstruct/run_optimize_cv0_density.py \
  --mesh-n 128 \
  --kdk-steps 64 \
  --checkpoint-every 4 \
  --optimizer adam \
  --n-iters 20 \
  --save-every 5 \
  --adam-lr 1.5e-2 \
  --noise-sigma 0.35 \
  --prior-weight 1.0 \
  --compare-space log \
  --output-dir cosmo_reconstruct/outputs/step2_3_optimize
```

L-BFGS option (with related knobs):

```bash
CUDA_VISIBLE_DEVICES=3 python cosmo_reconstruct/run_optimize_cv0_density.py \
  --mesh-n 128 \
  --kdk-steps 64 \
  --checkpoint-every 4 \
  --optimizer lbfgs \
  --lbfgs-lr 1.0 \
  --lbfgs-memory 10 \
  --lbfgs-linesearch zoom \
  --n-iters 20 \
  --output-dir cosmo_reconstruct/outputs/step2_3_optimize_lbfgs
```

This writes:

- `optimize_outputs.npz`
- `optimize_stats.json`
- `optimized_white_noise.npy`
- `optimization_history.png`
- forward-model comparison plot set

With `--save-every N`, the same full artifact set is also written periodically to:

- `OUTPUT_DIR/checkpoints/iter_XXXXXX/`

## Step 4: Full-Hydro Refinement

Use the optimized white noise from Step 2/3 as a warm start:

```bash
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
  --optimizer lbfgs \
  --lbfgs-linesearch zoom \
  --lbfgs-lr 0.5 \
  --lbfgs-memory 10 \
  --n-iters 10 \
  --init-white-noise-npy cosmo_reconstruct/outputs/step2_3_optimize/optimized_white_noise.npy \
  --output-dir cosmo_reconstruct/outputs/step2_3_optimize_full_hydro
```

This writes the same artifact set (`optimize_outputs.npz`, `optimize_stats.json`,
plots, and `optimized_white_noise.npy`) but using full hydro evolution in the forward pass.

Two-stage workflow helper:

```bash
bash cosmo_reconstruct/configs/run_two_stage_reconstruction.sh
```

## Notes

- CV0 density likelihood uses field-0 from
  `/gpfs02/work/diffusion/IllustrisTNG/Grids_Mgas_IllustrisTNG_CV_128_z=2.0.npy`,
  normalized to mean density 1.
- Gas and temperature mapping defaults are calibrated from
  `diffhydro_gadgetic_n128_z127to2_hll/snapshots/fields_ic_final.npz`.
- KDK integration supports checkpointing via `--checkpoint-every` for lower VJP memory.
- Growth caches are primed before optimization, so `--optimizer lbfgs --lbfgs-linesearch zoom`
  is supported without tracer-leak failures.
- For full-hydro optimization, `--lbfgs-linesearch zoom` can be significantly slower than
  `--lbfgs-linesearch none` because each line-search probe runs another full hydro forward/grad pass.
