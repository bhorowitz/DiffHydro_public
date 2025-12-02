## diffhydro

### Differentialable Hydrodynamic Code in JAX

A fully differentiable, GPU-accelerated hydrodynamics framework for astrophysical simulations, built in JAX. This is a significant expansion and reorganization of the diffhydro code (see https://arxiv.org/abs/2502.02294) with a focus on non-cosmological astrophysics. I/We hope to include a number of the key features of codes like Athena(K) with added differentiability and ease of development.

### What is diffhydro?

diffhydro is a modern computational fluid dynamics code designed for astrophysical applications—like simulating exploding stars, turbulent gas clouds, and galaxy formation. What makes it special is that it's fully differentiable: you can automatically compute how changing any input (like initial conditions or physical parameters) affects the final outcome.

Think of it like this: traditional simulation codes are like one-way streets—you set up initial conditions, run the simulation forward, and see what happens. diffhydro is a two-way street—you can also work backward from a desired outcome to figure out what initial conditions/parameters would produce it.

### Why does this matter?

* Inverse problems: Instead of guessing initial conditions and hoping they match observations, you can optimize them directly to reproduce what telescopes actually see
* Machine learning integration: Train neural networks inside your simulations to learn and correct physics at scales you can't resolve
* Parameter inference: Efficiently explore how uncertain physical parameters affect predictions
* Data-driven discovery: Connect simulations directly to observational data

### Key Features

* GPU-native: Runs efficiently on modern GPUs and TPUs, with multi-device scaling tested up to 1024³ resolution
* Fully differentiable: End-to-end automatic differentiation through the entire simulation
* Physics modules: Self-gravity, radiative cooling/heating, turbulence driving, and more
* Modular design: Easily swap reconstruction methods, Riemann solvers, and integrators

### Example Applications
The paper demonstrates several novel use cases:

* Initial condition reconstruction: Recover complex initial conditions from final states, even through highly nonlinear dynamics
* Solver-in-the-loop ML: Train neural networks to correct numerical errors while maintaining physical accuracy
* Forward modeling: Standard hydrodynamics simulations of supernova remnants, turbulence, and self-gravitating systems

### Getting Started

Built on the JAX ecosystem for scientific computing, diffhydro provides a Python-first, interactive workflow that integrates naturally with modern data science tools and machine learning frameworks.

Whether you're doing traditional forward modeling or cutting-edge simulation-based inference, diffhydro offers a flexible platform that bridges classical computational astrophysics with differentiable programming and machine learning.

We provide a few examples in this repository, but if you don't want to wade through the notebooks there are some basic quick-start examples in this colab notebook:

- [Turbulence and Blast Wave](https://colab.research.google.com/drive/14GuxwW_s4_OfuUXNsYOShAuySOoUay_b?usp=sharing)

Basic install is 
```
git clone https://github.com/bhorowitz/DiffHydro_public.git
cd ./DiffHydro_public
pip install -e
```

You can run tests with 
```
pytest -q
```
Note, depending on your system this might take all GPUs and break the tests default API which assumes one GPU. You can use a command like below to specifiy one GPU:
```
CUDA_VISIBLE_DEVICES=0 pytest -q
```

### Getting Involved

diffhydro is under active development and we are looking to build a broader team! Let me know if you want to be involved and add any features! ben.horowitz@ipmu.jp or open a pull request! :D 

### Rough Roadmap

Main new features complete or under development
 - [x] Dimensionally unsplit solver for total flux calculations
 - [x] Various Riemann solvers (LaxFriedrichs, HLLC, HLL_MHD, HLLD_MHD) with easy interface for positivity and limiters (inspired by JAXFLUIDS)
 - [x] Demonstration of solver/corrector-in-loop approach for speeding up simulations and capturing (possibly unknown) physics
 - [x] MHD with various possible numerical schemes for divergence (Constrained Transport, Corner Transport Upwind, Positivity Preserving CT)
 - [ ] More testing of MHD through various Athena benchmarks
 - [X] Self Gravity via multigrid methods
 - [ ] Adaptive Mesh Refinement schemes in 3d (the biggest challenge!)

Let me know if you want to be involved and add any features! ben.horowitz@ipmu.jp or open a pull request! :D 

### Animations

2d shocks (1024^2)

<img src="https://github.com/bhorowitz/DiffHydro_public/blob/main/animations/diffhydro_rho_web.gif?raw=true" width="500" />

3d gravity with Gaussian Random Field Prior (256^3)

<img src="https://github.com/bhorowitz/DiffHydro_public/blob/main/animations/gauss_grav_recon.gif?raw=true" width="500" />
