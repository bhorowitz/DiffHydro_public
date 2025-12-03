## diffhydro

### Differentiable Hydrodynamic Code in JAX

A fully differentiable, GPU-accelerated hydrodynamics framework for astrophysical simulations, built in JAX. This is a significant expansion and reorganization of the diffhydro code (see https://arxiv.org/abs/2502.02294), with a focus on non-cosmological astrophysics. I/we hope to include a number of the key features of codes like Athena(K), with added differentiability and ease of development.

### What is diffhydro?

diffhydro is a modern computational fluid dynamics code designed for astrophysical applications—like simulating exploding stars, turbulent gas clouds, and galaxy formation. What makes it special is that it’s fully differentiable: you can automatically compute how changing any input (like initial conditions or physical parameters) affects the final outcome.

Think of it like this: traditional simulation codes are like one-way streets—you set up initial conditions, run the simulation forward, and see what happens. diffhydro is a two-way street—you can also work backward from a desired outcome to figure out what initial conditions/parameters would produce it.

### Why does this matter?

* **Inverse problems:** Instead of guessing initial conditions and hoping they match observations, you can optimize them directly to reproduce what telescopes actually see.
* **Machine learning integration:** Train neural networks inside your simulations to learn and correct physics at scales you can’t resolve.
* **Parameter inference:** Efficiently explore how uncertain physical parameters affect predictions.
* **Data-driven discovery:** Connect simulations directly to observational data.

### Key Features

* **GPU-native:** Runs efficiently on modern GPUs and TPUs, with multi-device scaling tested up to 1024³ resolution.
* **Fully differentiable:** End-to-end automatic differentiation through the entire simulation.
* **Physics modules:** Self-gravity, radiative cooling/heating, turbulence driving, and more.
* **Modular design:** Easily swap reconstruction methods, Riemann solvers, and integrators.

### Example Applications

The paper demonstrates several novel use cases:

* **Initial condition reconstruction:** Recover complex initial conditions from final states, even through highly nonlinear dynamics.
* **Solver-in-the-loop ML:** Train neural networks to correct numerical errors while maintaining physical accuracy.
* **(Less novel) Forward modeling:** Standard hydrodynamics simulations of supernova remnants, turbulence, and self-gravitating systems.

### Getting Started

Built on the JAX ecosystem for scientific computing, diffhydro provides a Python-first, interactive workflow that integrates naturally with modern data-science tools and machine-learning frameworks.

- [Intro to JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)

Whether you’re doing traditional forward modeling or cutting-edge simulation-based inference, diffhydro offers a flexible platform that bridges classical computational astrophysics with differentiable programming and machine learning.

We provide a few examples in this repository, but if you don’t want to wade through the notebooks, there are some basic quick-start examples in this Colab notebook:

- [Turbulence and Blast Wave](https://colab.research.google.com/drive/14GuxwW_s4_OfuUXNsYOShAuySOoUay_b?usp=sharing)
- [Basic Gradient Use for Optimization](https://colab.research.google.com/drive/1E8ATlxOiwS6RkOljdfp2SSHcX0pk83KD?usp=sharing)

Basic install:
```bash
git clone https://github.com/bhorowitz/DiffHydro_public.git
cd ./DiffHydro_public
pip install -e .
```

You can run tests with:
```bash
pytest -q
```

Note: depending on your system this might try to use all GPUs and break the test default API, which assumes one GPU. You can use a command like the one below to specify a single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 pytest -q
```

### Note on Evolve Methods

The code provides a number of (probably confusingly named) evolve methods. Depending on what you are trying to optimize for (readability, pure speed, long-run memory conservation, embedding within larger inference loops), different APIs might be better or worse. We’ll make a cleaner automatic selector-type function soon. In general, we recommend the following:

**`hydrosim.evolve()`** or **`evolve_memory_efficient()`** for most integrated applications (i.e., optimization, solver-in-the-loop, etc.)

**`hydrosim.evolve_with_callbacks()`** for debugging and providing snapshots (read to CPU and to `.npy`) beyond final-time output.

**`hydrosim.evolve_till_time()`** if you **must** have a dynamic number of timesteps. This likely has the worst performance in terms of compile time and backprop. `jax.while` loops aren’t great yet (at least in the specific JAX versions I was using).

**`hydrosim.evolve_with_dt_schedule()`** for niche applications where you want to provide an array of `dt` per timestep. I used it for solver-in-the-loop applications. In this setup there is **no CFL checking**, so you will run into NaNs quickly if you aren’t conservative/careful.

### Some general strategy notes for backpropagation, optimization, and numerical stability

So far, the best optimization package/implementation I found for both IC reconstruction and solver-in-the-loop training is [optax](https://optax.readthedocs.io/en/latest/). Some general strategy notes:

- Include a failsafe “jiggler”-type function in your optimization loops; if you hit NaNs in derivative calculations you can sometimes just multiply/add a tiny factor and it will keep going without issue. If it happens more than a couple times, there is probably something fundamentally numerically unstable that you should investigate.

- If you find your optimization quickly runs into NaNs, try a different flux limiter. Certain limiters may behave better or worse during backprop depending on the problem. I usually default to `vanleer`.

- For backprop through gravity, I frequently ended up using SSPRK3 for the integrator; RK2 was sometimes a bit more sensitive.

- HLLC is a decent amount more memory-intensive than LaxFriedrichs. If you are running out of memory often during backprop, switching to LF could save you a lot. Of course, LF is much more diffusive, so certain features (i.e., instabilities) will not be resolved properly.

- As described in the paper, I found multigrid Poisson generally behaved more nicely than FFT Poisson. It is possible that [jaxdecomp](https://jaxdecomp.readthedocs.io/) might be even better, but I was unable to get it working properly on my system. If you do play around with it, be careful with direction conventions.

- It is always good to be conservative with CFL, particularly for backprop. Certain examples in the paper let me get away with `cfl = 0.8`, but many ended up closer to `0.2`. There are separate timestep safety scalings in other forcing functions, so be careful with those choices. `HeatingCooling` is particularly sensitive.

- If you run into memory issues while backpropagating through certain forcing functions with dynamic timestepping, it might be beneficial to use `stop_gradients` (i.e., `U = lax.stop_gradient(U)`) in the timestep so you don’t backprop through the timestep choice itself. Even more drastically, you can use `evolve_with_dt` and a fixed-timestep scheme.

### Getting Involved

diffhydro is under active development, and we’re looking to build a broader team. Let me know if you want to be involved or add features! Email me at ben.horowitz@ipmu.jp or open a pull request. :D

### Rough Roadmap

Main new features complete or under development:

- [x] Dimensionally unsplit solver for total flux calculations  
- [x] Various Riemann solvers (LaxFriedrichs, HLLC, HLL_MHD, HLLD_MHD) with an easy interface for positivity and limiters (inspired by JAXFLUIDS)  
- [x] Demonstration of solver/corrector-in-the-loop approach for speeding up simulations and capturing (possibly unknown) physics  
- [x] MHD with various possible numerical schemes for divergence control (Constrained Transport, Corner Transport Upwind, Positivity-Preserving CT)  
- [ ] More testing of MHD through various Athena benchmarks  
- [x] Self-gravity via multigrid methods  
- [ ] Adaptive mesh refinement schemes in 3D (the biggest challenge!)

Let me know if you want to be involved or add any features! ben.horowitz@ipmu.jp or open a pull request. :D

### Animations

2D shocks (1024²)

<img src="https://github.com/bhorowitz/DiffHydro_public/blob/main/animations/diffhydro_rho_web.gif?raw=true" width="500" />

3D gravity with Gaussian Random Field prior (256³)

<img src="https://github.com/bhorowitz/DiffHydro_public/blob/main/animations/gauss_grav_recon.gif?raw=true" width="500" />
