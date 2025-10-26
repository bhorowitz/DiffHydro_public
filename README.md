## diffhydro

### Differentialable Hydrodynamic Code in JAX

Welcome to the landing page! This is a significant expansion and reorganization of the diffhydro code (see https://arxiv.org/abs/2502.02294) with a focus on non-cosmological astrophysics. I/We hope to include a number of the key features of codes like Athena(K) with added differentiability and ease of development.

Main new features complete or under development
 - [x] Dimensionally unsplit solver for total flux calculations
 - [x] Various Riemann solvers (LaxFriedrichs, HLLC, HLL_MHD, HLLD_MHD) with easy interface for positivity and limiters (inspired by JAXFLUIDS)
 - [x] Demonstration of solver/corrector-in-loop approach for speeding up simulations and capturing (possibly unknown) physics
 - [x] MHD with various possible numerical schemes for divergence (Constrained Transport, Corner Transport Upwind, Positivity Preserving CT)
 - [ ] More testing of MHD through various Athena benchmarks
 - [ ] Self Gravity via jaxdecomp and multigrid methods, both as forcing and as new flux term
 - [ ] Adaptive Mesh Refinement schemes in 3d (the biggest challenge!)

Let me know if you want to be involved and add any features! ben.horowitz@ipmu.jp or open a pull request! :D 