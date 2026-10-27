# diffhydro tutorials: from finite volumes to inverse-solving turbulence

Six notebooks that build on each other for tutorials for diffhydro. Part I writes everything by hand in `jax.numpy`, so the numerical methods
and the autodiff machinery are transparent. Part II switches to `diffhydro` and ends with turbulence that writes a letter.

| # | notebook | CFD | autodiff / inverse problems | CPU time* |
|---|---|---|---|---|
| 01 | [Finite volumes in 1D](01_finite_volume_1d.ipynb) | conservation form, upwind / Lax–Friedrichs, CFL, numerical diffusion, Burgers shocks | first `jax.jit`, `jax.grad`, `jax.jacfwd` | < 1 min |
| 02 | [Autodiff through time](02_autodiff_through_time.ipynb) | viscous Burgers as a test bed | forward vs reverse mode, checkpointing, gradient checks, NaN traps, Adam, coarse-to-fine losses, priors and reparametrisation | ~1.5 min |
| 03 | [Euler equations in 1D](03_euler_1d.ipynb) | Riemann problems, Rusanov / HLL / HLLC, MUSCL + limiters, SSP-RK, Sod convergence | inferring $p_L$ and $\gamma$ through shocks | ~2 min |
| 04 | [diffhydro](04_diffhydro_intro.ipynb) | the same solver as components, solver zoo, 2D Kelvin–Helmholtz | gradients through `diffhydro` | ~3 min |
| 05 | [Decaying turbulence in 2D](05_decaying_turbulence_2d.ipynb) | Gaussian random fields, Helmholtz decomposition, decaying turbulence, spectra, lognormal PDF | sensitivity maps (reverse mode), perturbation growth (forward mode) | ~2.5 min |
| 06 | [Making turbulence write a letter](06_letter_inverse_solve.ipynb) | — | full inverse problem: white noise → turbulent IC → hydro → image | ~8.5 min at 48² (~5 min on a V100 at 128²) |

\* Measured on a 36-core Linux server. Expect a laptop to be 2–3× slower.

## Running

The notebooks need only diffhydro's own dependencies plus matplotlib:

```bash
pip install -e ".[jax]" scipy matplotlib jupyter     # from the repository root; pick the jax wheel for your CPU/GPU
cd examples/tutorials
jupyter lab
```

Run them from this directory: they import the shared helpers in `tutorial_utils.py`, and they fall back to `../..`
if `diffhydro` is not installed. The committed outputs of notebooks 01–05 were produced on CPU and notebook 06 on a
single V100 GPU (at $128^2$). Notebook 06 picks its resolution from the JAX backend (`N = 128` on GPU, `48` on CPU).

Note, if you run it locally on mac you should look into mac metal for GPU acceleration.

**Why is notebook 06 so much slower on CPU?** Reverse-mode gradients through the hydro solver cost ~3× a forward pass
on a GPU but 15–40× on CPU. XLA's CPU backend runs the many small elementwise kernels of the adjoint one at a time;
plain JAX code shows the same ratio. Notebooks 01–05 keep their gradient experiments small enough not to notice.

## Files

* `0*_*.ipynb`: the notebooks, with outputs.
* `tutorial_utils.py`: helpers derived in one notebook and reused in later ones: exact Riemann solver, Adam, the
  notebook-03 Euler solver, Fourier tools, a 2D `diffhydro` builder and the turbulent-IC generator.
* `letter_D.gif`: the movie written by notebook 06.
