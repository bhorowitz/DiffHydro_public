# Lyman-alpha Forest Tools in DiffHydro

This document summarizes the Lyman-alpha (LyA) forest machinery in this repo: the
core postprocessing library, the simulation drivers (single- and multi-GPU), the
Nyx-comparison analysis scripts, and the validation results at z=2.

## Core library

- `diffhydro/postprocess/lya.py` — the differentiable LyA forward model. Key entry points:
  - `build_nyx_eos_interpolator` — Nyx EOS (`eos_t` backend, TREECOOL UV background)
    interpolator on a (log delta_b, log T) grid at fixed z.
  - `compute_nhi_number_density` — neutral-hydrogen number density from
    (baryon overdensity, temperature) via the EOS interpolator.
  - `tau_cube_from_nhi` — real- or redshift-space optical-depth cube from
    (n_HI, T, v_los), Gimlet-style Doppler-profile integration along the LOS.
  - `flux_from_tau`, `column_density_map_nhi`, `mean_map_along_los` — flux and map utilities.

  The tau kernel matches Gimlet's `calc_tau.f90` (validated below). Note: at 256^3 it
  materializes ~2.6 GiB intermediates per cube and can OOM a 32 GB GPU — run large
  postprocessing with `JAX_PLATFORMS=cpu` (a full comparison takes ~2 min on CPU).

## Initial conditions (music25 suite)

25 Mpc/h box, h=0.71, Omega_m=0.27, Omega_b=0.045, sigma_8=0.8, n_s=0.96, z_start=100.
MUSIC configs and binaries live in `Nyx/Exec/LyA/music25_ics/` (nested Nyx checkout,
not tracked here); the config for the matched realization is copied to
`cosmo_parallel/ic_configs/music25_n512_true.conf`.

Note, you will need to install and configure MUSIC seperately.

**Realization map (important when comparing fields):**

(These aren't included since they are very big, but I include the information here in case you run NYX yourself and want to compare)
| IC file | Base seed | Realization |
|---|---|---|
| `music25_n64/128/256.nyx` | seed[8]=34567 (levelmin 8) | independent "A" |
| `music25_n512_true_generic.hdf5` == `music25_matched_n512` | seed[9]=45678 (levelmin 9) | matched "B" |
| `music25_matched_n256.nyx` (downsampled from matched n512) | — | matched "B" |
| `music25_n400_true.nyx` (resampled from n512_true) | — | matched "B" |

Nyx baseline runs `music25_runs/n256_gpu` (z=2 plotfile `plt00464`) and `n512_gpu`
(`plt00583`) use realization B. DiffHydro runs seeded from `n256_init`/`n512_init`
use realization A — comparing across realizations gives ~zero field-level correlation
by construction; only one-point statistics are meaningful there.

IC preparation tools:
- `cosmo_parallel/resample_music_generic.py` — Fourier resample of a MUSIC generic HDF5.
- `music_generic_to_nyx_binary.py` — MUSIC generic → Nyx binary particle file.
- `cosmo_parallel/prepare_music25_n400_true.sh` — n512_true → n400_true chain.
- `cosmo_parallel/cache_nyx_ic_fields.py` — Nyx IC plotfile → compact npz for yt-free runtimes.

## Simulation drivers

### Single GPU (Nyx benchmark driver)

- `diffhydro_jaxpm_nyx.py` — runs DiffHydro from a Nyx IC plotfile to a target Nyx
  plotfile redshift, with coupled DM, Nyx-table cooling/heating, and produces
  `snapshots/fields_ic_final.npz` + `metrics.npz` field-level comparison metrics.
- `run_music25_n128_post.sh` — end-to-end example: waits for the Nyx n128 run,
  converts plotfiles, runs the driver to z=10 and z=2 (`--solver hllc`).
  The `diffhydro_music25/n{64,128}_z2*` directories are solver variants of this
  command (`hll`, `hllc`, `lf`, `nyx`); e.g. `n128_z2hll` used `--solver hll`.

### Multi GPU (4-GPU jaxdecomp, production path)

- `cosmo_parallel/run_gadgetic_coupled_parallel.py` — coupled hydro+DM cosmology from
  Nyx-style IC plotfiles, sharded with jaxdecomp FFT gravity (`pdims 2x2`, halo 32-48).
  Dumps full-field checkpoints at requested redshifts (`--checkpoint-z-values 10,2`).
- `cosmo_parallel/batch/music25_n400_true_jaxdecomp_fft_4gpu_z2_rankfix_halo32_pdims22.sh`
  — the SLURM submission that produced the validated n400_true run
  (`cosmo_parallel/results/music25_n400_true_jaxdecomp_fft_4gpu_z2_rankfix_halo32_pdims22/`).
- See `cosmo_parallel/README.md` for the broader multi-GPU workspace.

**Unit conventions in the coupled driver** (needed to interpret checkpoints):
Nyx plotfile velocity `(mom/rho)/100` is in units of 1e7 cm/s; internal supercomoving
code velocity uses `hydro_velocity_scale = bg.H0 * n_grid / box_size` with the driver's
*hardcoded* internal background `LCDMBackground(h=0.6711, Omega_m=0.3)` — not the
music25 cosmology. Invert with h=0.6711 (as `export_coupled_fields_for_lya.py` does);
use h=0.71 for the LyA physics.

## Nyx comparison / analysis scripts

- `postprocess_lya_nhi_compare.py` — **the main comparison.** Loads a Nyx plotfile
  (via yt) and a DiffHydro fields npz, computes n_HI/tau/flux for both through the
  identical pipeline, and writes flux PDFs, 1D flux power spectra (real + redshift
  space), N_HI column maps, mean/min-flux maps, slice comparisons, and
  `summary_metrics.txt` (mean flux, tau percentiles, map-level Pearson correlations).
- `cosmo_parallel/export_coupled_fields_for_lya.py` — converts a coupled-parallel
  checkpoint npz into the fields format the comparison expects (density norm, T[K],
  peculiar velocity in cm/s).
- `cosmo_parallel/resample_lya_fields_npz.py` — Fourier regridding to a common grid
  for cross-resolution comparisons. Resamples positive fields in **log space with a
  Gaussian anti-aliasing taper**; plain sharp truncation rings around dense structures,
  floors ~4-6% of cells (1 K cells = zero thermal broadening) and produces saturated
  flux speckle. Do not use sharp truncation on rho/T.
- `cosmo_parallel/plot_lya_field_slices_compare.py` — density / temperature / LOS-velocity
  slice comparisons + 3D Pearson correlations.
- `compare_gimlet_kernel_vs_diffhydro_lya.py` — kernel-level unit check: DiffHydro's
  `tau_cube_from_nhi` vs a NumPy port of Gimlet's `calc_tau.f90` on identical n_HI.
- `run_zgimlet_compare.py` — runs the external `zgimlet lya_fields` executable on the
  same grids and compares against the DiffHydro mapping.
- `run_illustristng_lya_realspace.py` — real-space LyA postprocessing validated against
  IllustrisTNG CV grids (secondary baseline).

## Example workflow: multi-GPU run vs Nyx at z=2

```bash
# 1. Simulate (4 GPUs, SLURM)
sbatch cosmo_parallel/batch/music25_n400_true_jaxdecomp_fft_4gpu_z2_rankfix_halo32_pdims22.sh

R=cosmo_parallel/results/music25_n400_true_jaxdecomp_fft_4gpu_z2_rankfix_halo32_pdims22

# 2. Convert the z=2 checkpoint to the comparison format
python cosmo_parallel/export_coupled_fields_for_lya.py \
  --checkpoint-npz $R/checkpoints/snapshot_z2_fields.npz \
  --output-npz $R/lya_vs_nyx/fields_for_lya_z2_n400.npz

# 3. Regrid to the Nyx baseline resolution
python cosmo_parallel/resample_lya_fields_npz.py \
  --input-npz $R/lya_vs_nyx/fields_for_lya_z2_n400.npz \
  --output-npz $R/lya_vs_nyx/fields_for_lya_z2_n256.npz --n-out 256

# 4. Full LyA comparison (CPU: the tau kernel OOMs 32 GB GPUs at 256^3)
JAX_PLATFORMS=cpu python postprocess_lya_nhi_compare.py \
  --nyx-plotfile Nyx/Exec/LyA/music25_runs/n256_gpu/plt00464 \
  --diffhydro-fields $R/lya_vs_nyx/fields_for_lya_z2_n256.npz \
  --output-dir $R/lya_vs_nyx \
  --treecool-file Nyx/Exec/LyA/music25_runs/n256_gpu/TREECOOL_middle \
  --h 0.71 --omega-m 0.27 --omega-b 0.045

# 5. Field slice comparisons (density / T / v_los)
JAX_PLATFORMS=cpu python cosmo_parallel/plot_lya_field_slices_compare.py \
  --nyx-plotfile Nyx/Exec/LyA/music25_runs/n256_gpu/plt00464 \
  --diffhydro-fields $R/lya_vs_nyx/fields_for_lya_z2_n256.npz \
  --output-dir $R/lya_vs_nyx
```

## Validation results (z=2, matched ICs)

Multi-GPU DiffHydro n400_true (regridded to 256^3) vs native Nyx 256^3, with a
same-code control (Nyx 512^3 through the identical regrid pipeline) quantifying the
cross-resolution systematic:

| Metric | DiffHydro 4-GPU vs Nyx-256 | Control: Nyx-512 vs Nyx-256 |
|---|---|---|
| Pearson log10 N_HI column | 0.667 | 0.784 |
| mean flux (real space) | 0.821 vs 0.857 | 0.801 vs 0.857 |
| tau p99 (real space) | 1.48 vs 2.68 | 3.00 vs 2.68 |
| 3D Pearson log10 density / log10 T / v_los | 0.90 / 0.85 / 0.98 | — |

DiffHydro reaches ~85% of the same-code cross-resolution ceiling, and its mean-flux
offset is smaller than the regrid systematic itself. Same-grid single-GPU baseline
(`diffhydro_music25/n128_z2hll`, HLL vs Nyx n128): Pearson log10 N_HI column = 0.815.

Known differences / open items:
- The n400_true DiffHydro run has void temperatures down to ~0.4 K (Nyx floor ~1700 K
  from UV photoheating) — check heating configuration for production runs.
- LOS velocity agrees at 0.98 Pearson with a coherent ~few-percent amplitude excess,
  plausibly the hardcoded internal h=0.6711 vs music25 h=0.71 (see unit conventions).
- Cleanest future benchmark: same-grid 4-GPU n512 run from `music25_matched_n512.nyx`
  with heating enabled, vs Nyx `n512_gpu/plt00583`, no regridding.

## Standalone workflow (no Nyx required)

DiffHydro can generate LyA forest statistics fully self-contained — no Nyx checkout,
no MUSIC — via the differentiable forward model in `cosmo_reconstruct/`. This is the
pipeline used for IC optimization and posterior sampling against LyA flux, so it is
the most heavily exercised LyA path in the repo.

**How it works:**

1. **ICs from white noise** — `cosmo_reconstruct/src/forward_model.py`: a white-noise
   cube is colored with a `jax_cosmo` linear power spectrum (`make_pk_sqrt` →
   `white_noise_to_init_mesh`). Because the ICs are a differentiable function of the
   white noise, gradients flow end-to-end through the whole chain.
2. **Evolution** — `cosmo_reconstruct/src/full_hydro_model.py`: coupled JaxPM dark
   matter + DiffHydro supercomoving gas from the init mesh
   (`_init_hydro_state_from_white_noise`), with optional Nyx-table cooling/heating.
3. **LyA observable** — `cosmo_reconstruct/src/observable_utils.py`: the *same*
   kernel as the Nyx comparison (`build_nyx_eos_interpolator` →
   `compute_nhi_number_density` → `tau_cube_from_nhi` → flux). The required UV
   background table defaults to `diffhydro/nyx_eos/TREECOOL_middle`, which is bundled
   with the package (a data file, not Nyx code; Fortran EOS source + build notes in
   `diffhydro/nyx_eos/`).

**Entry points:**

```bash
# Forward model + IC optimization against a LyA flux target.
# --target-source self runs a fully self-consistent synthetic experiment
# (DiffHydro generates the target too); alternatively point it at TNG CV0 fields.
python cosmo_reconstruct/run_optimize_cv0_density_full_hydro.py \
  --observable lya_flux --target-source self ...

# NUTS / MCLMC posterior sampling of ICs given LyA flux
python cosmo_reconstruct/run_sample_cv0_density_full_hydro.py ...

# Quick smoke test of the whole chain
python cosmo_parallel/run_reconstruction_smoke.py \
  --observable lya_flux --output-dir cosmo_parallel/results/recon_lya_smoke

# Phase-matched white noise across resolutions (convergence tests)
python -m cosmo_feedback.gen_matched_white_noise --hi 256 --lo 128 --seed 0
```

See `cosmo_reconstruct/README.md` and `cosmo_reconstruct/ExampleScripts.md` for the
full option surface.

**Validation status:** the tau/flux kernel is validated three independent, Nyx-free
ways (Gimlet `calc_tau.f90` port, external `zgimlet` executable, IllustrisTNG CV
grids with provided HI fields); the hydro core is validated against Nyx (previous
section); the RSD velocity normalization was verified against Zel'dovich
expectations. The one untested link is the internal white-noise IC generator itself,
which has not been cross-checked against a MUSIC realization.

**MUSIC directly:** currently MUSIC ICs reach DiffHydro only through a Nyx detour
(generic HDF5 → `music_generic_to_nyx_binary.py` → Nyx init run → plotfile →
`cache_nyx_ic_fields.py`), because Nyx generates the gas fields from the particles.
A direct MUSIC-generic → DiffHydro IC converter (painting the displacements/velocities
already read by `resample_music_generic.py` onto the grid) is the missing piece if
you want a MUSIC-based workflow without any Nyx build.
