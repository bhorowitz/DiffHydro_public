# RT Beam Injection Debug

This folder contains a script version of the beam-injection part of
`examples/athena/blast-athena.ipynb`. It intentionally skips the Athena /
Sedov-Taylor blast setup and starts from a clean four-field radiation state:

```text
U = [E_gamma, F_gamma_x, F_gamma_y, F_gamma_z]
```

The current convention is that `F_gamma_*` is the physical radiation flux, not
`F/E`.

## Recommended Run

The least-diffusive diagnostic setup so far uses:

- 16^3 grid
- source moved away from the periodic-looking x boundary
- physical beam momentum scaling
- `fbeam = 0.999`
- experimental local-HLL RT solver
- non-periodic flux divergence

```bash
GPU=2 conda run -n dhrt python examples/rt_debug/beam_injection_debug.py \
  --mesh-size 16 \
  --t-target 55.8 \
  --n-super-step 25 \
  --max-steps 25 \
  --beam-reduced-flux 0.999 \
  --star-x 4 \
  --rt-solver hll-local \
  --run-name hll_local_16_25steps_center_x4_f0999
```

The most useful image for judging beam structure is usually:

```text
images/slices/0_E_gamma_z_center_log10_relative_floor1e-8.png
```

The absolute `log10(abs(E_gamma))` image can look very diffusive because it
shows tiny numerical tails down near `1e-20` or lower.

## Outputs

Outputs are written to:

```text
examples/rt_debug/runs/<run-name>/
```

Folder structure:

- `images/slices/`: center-z field slices, log slices, zooms, and relative-floor views.
- `images/profiles/`: beam-axis profiles through `y=z=center`.
- `images/metrics/`: M1 cone diagnostic images.
- `data/final_state.npz`: final state, active dt history, and M1 ratio array.
- `data/beam_axis_profiles.npz`: beam-axis field profiles.
- `metrics.json`: run parameters and scalar diagnostics.
- `run.log`: present for redirected runs.

## Main Findings

### 1. RT State Convention Was Mixed

The old code mixed two interpretations of `sol[1:4]`:

- physical flux `F`
- reduced or velocity-like flux `F/E`

The code now uses physical moments consistently:

```text
conservatives = primitives = [E, Fx, Fy, Fz]
```

The reduced flux invariant is:

```text
|F| / (c E) <= 1
```

not:

```text
|F| = c^2 E^2
```

### 2. Beam Momentum Scaling Was Diffusion-Like

The old beam momentum injection used:

```text
dFx = source^2 * c^2 * weights
```

while energy used:

```text
dE = source * weights
```

For small `source`, this makes `|F|/(cE)` tiny at injection, so the M1 closure
starts in the diffusion limit.

The default is now:

```text
dFx = fbeam * c * dE
```

The comparison script confirmed:

```text
physical scaling:
  max |F|/(cE) after injection ~= 0.95

legacy source^2*c^2 scaling:
  max |F|/(cE) after injection ~= 0.033
```

### 3. Removed Equality Projection

Old code projected fluxes toward:

```text
|F| = c^2 E^2
```

in multiple places. That over-constrains the state and is not the M1 invariant.

The replacement is a safety cone limiter:

```text
|F| <= fmax * c * E
```

This is applied after source injection. It limits only violations; it does not
force equality.

### 4. Removed Flux Sign Clipping

The RT Lax-Friedrichs solver previously clipped fluxes with:

```python
jnp.maximum(fluxes_L, 1e-12)
jnp.maximum(fluxes_R, 1e-12)
```

That destroys negative fluxes and can create artificial positive leakage. This
has been removed.

### 5. Removed `jnp.abs()` After Beam Energy Injection

The old beam energy path applied `jnp.abs()` to the full state after injection.
That can silently flip flux signs and hide bugs. It has been removed.

### 6. Periodic Flux Divergence Was Creating Wraparound

The conservative update uses a flux difference involving a roll. On one GPU,
that roll is `jnp.roll`, which is periodic. With the source at `x=0`, radiation
could immediately wrap to `x=N-1`, creating an opposite-edge lobe.

The debug script now defaults to non-periodic flux divergence. Pass
`--periodic-flux-divergence` only when you intentionally want the old periodic
flux difference.

### 7. Local HLL Reduces Beam Diffusion

An experimental `HLL_Radiative_transfer_Local` solver was added. It estimates
local M1 wave speeds using the normal pressure component, so transverse waves
are much slower for a strongly x-beamed field.

For the centered 16^3, 25-step, `fbeam=0.999` case:

```text
Rusanov:
  center-line energy fraction ~= 0.923
  off-beam z-slice energy fraction ~= 8.9e-6

HLL-local:
  center-line energy fraction ~= 0.997
  off-beam z-slice energy fraction ~= 1.1e-6
```

This is the most beam-aligned setup tried so far.

## Script API

Main script:

```bash
GPU=2 conda run -n dhrt python examples/rt_debug/beam_injection_debug.py [options]
```

Options:

- `--mesh-size INT`
  Grid size in each dimension. Default: `100`.

- `--t-target FLOAT`
  Target simulation time. Default: `18.6`.

- `--max-steps INT`
  Optional hard cap on steps. Useful for controlled debug runs.

- `--n-super-step INT`
  Internal maximum step buffer / loop cap. Default: `1000`.

- `--light-speed FLOAT`
  Reduced light speed used by the RT equation manager. Default: `2.0`.

- `--stromgren-rate FLOAT`
  Photon source rate for `injection_mode="stromgren"`. Default: `10.0`.

- `--beam-length-cells INT`
  Beam deposition length in cells for `beam_x`. Default: `1`.

- `--beam-reduced-flux FLOAT`
  Desired injected reduced flux `|F|/(cE)`. Default: `0.95`.
  Use `0.999` for a more beam-like M1 state.

- `--beam-momentum-scaling {physical,legacy_c2_source2}`
  Momentum injection convention. Default: `physical`.
  `legacy_c2_source2` reproduces the old `source^2*c^2` scaling.

- `--limiter STR`
  Reconstruction limiter passed to `dh.PLM`. Default: `VANLEER`.

- `--rt-solver {rusanov,hll-local}`
  RT Riemann solver. Default: `rusanov`.
  `hll-local` is the experimental less-diffusive M1 local-HLL solver.

- `--periodic-flux-divergence`
  Use the old periodic roll-based flux divergence. By default, the debug script
  uses non-periodic flux divergence to avoid wraparound artifacts.

- `--geometry {beam_x,2D,3D}`
  Injection geometry. Default: `beam_x`.

- `--no-momentum`
  Disable momentum injection.

- `--debug-force`
  Enable verbose force-level debug printing.

- `--debug-fixed-dt FLOAT`
  Force a maximum fixed dt for debug stepping.

- `--run-name STR`
  Output run folder name.

- `--star-x INT`, `--star-y INT`, `--star-z INT`
  Star/source position. Defaults to `x=0`, `y=z=mesh_size//2`.
  For beam debugging, use `--star-x 4` on a 16^3 grid to avoid immediate
  boundary wraparound.

- `--output-root PATH`
  Root directory for run outputs. Default: `examples/rt_debug/runs`.

## Comparison Script

Use this to compare physical vs legacy beam momentum scaling after injection
and after one hydro step:

```bash
GPU=2 conda run -n dhrt python examples/rt_debug/compare_beam_scaling.py \
  --mesh-size 16
```

It writes:

```text
examples/rt_debug/runs/compare_beam_scaling.json
```

## Quick Smoke Test

```bash
GPU=2 conda run -n dhrt python examples/rt_debug/beam_injection_debug.py \
  --mesh-size 16 \
  --t-target 1e-6 \
  --max-steps 1 \
  --run-name smoke
```

## Verbose Debug Mode

The underlying RT code still contains many direct `print` / `jax.debug.print`
statements. For notebook-style verbose diagnostics:

```bash
GPU=2 DIFFHYDRO_DEBUG_CHECKS=True conda run -n dhrt python \
  examples/rt_debug/beam_injection_debug.py --debug-force
```
