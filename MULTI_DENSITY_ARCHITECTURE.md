# Multi-Density Euler Architecture

## Overview

The DiffHydro framework supports **multi-density Euler equations** where multiple density fields (`rho`, `rho_test`, `rho_candidate`, etc.) share:
- **One velocity field** (vx, vy, vz)
- **One pressure field** derived from total density
- **Full hydrodynamic coupling** (momentum, energy)

Each density evolves via the continuity equation:
```
d(rho_i)/dt + div(rho_i * u) = 0
```

Implementation: `diffhydro/equationmanager_radiative_transf.py`

---

## Data Structure

### Conservative Variables (Storage Order)

For `n_dens` densities:

```
Index Range           Variable
[0...n_dens-1]       rho_1, rho_2, ... rho_n
[n_dens]             rho_total * vx
[n_dens+1]           rho_total * vy
[n_dens+2]           rho_total * vz
[n_dens+3]           E_total
```

**Total conservative variables:** `n_cons = n_dens + 4`

### Example: 2 Densities (rho, rho_test)

```
Index  Conservative Variable
  0    rho
  1    rho_test
  2    (rho + rho_test) * vx
  3    (rho + rho_test) * vy
  4    (rho + rho_test) * vz
  5    E_total
```

### Primitive Variables Layout

```
Index Range           Variable
[0...n_dens-1]       rho_1, rho_2, ... rho_n
[n_dens]             vx
[n_dens+1]           vy
[n_dens+2]           vz
[n_dens+3]           p
```

---

## Usage

### Single Density (Standard Euler)

```python
from diffhydro.equationmanager_radiative_transf import EquationManager

eq = EquationManager(density_names=("rho",))
# n_dens = 1, n_cons = 5 — identical to classic Euler
```

### Two Densities

```python
eq = EquationManager(density_names=("rho", "rho_test"))
# n_dens = 2, n_cons = 6
# Each density evolved independently, shared velocity and pressure
```

### N Densities

```python
eq = EquationManager(density_names=("rho", "rho_test", "rho_candidate"))
# n_dens = 3, n_cons = 7
```

### Initialization with Turbulence

```python
from diffhydro.physics.turbulence_radiative_transf import init_turbulent_velocity

eq = EquationManager(density_names=("rho", "rho_test"))
eq.mesh_shape = [100, 100, 100]

U = init_turbulent_velocity(eq, Lbox=100, rho0=1.0, p0=0.1)
# U.shape = (6, 100, 100, 100)
```

---

## Governing Equations

### Continuity (Each Density)
```
d(rho_i)/dt + div(rho_i * u) = 0
```

### Momentum (Shared, Total Density)
```
d(rho_total * u)/dt + div(rho_total * u (x) u + p * I) = 0
where rho_total = sum(rho_i)
```

### Energy (Total)
```
dE/dt + div((E + p) * u) = 0
```

### Equation of State
```
p = (gamma - 1) * rho_total * e
```

---

## Key Properties

| Property | Single Density | Multi-Density |
|----------|---------------|---------------|
| Density variables | 1 | N (scalable) |
| Velocity field | shared | shared |
| Pressure field | p(rho, e) | p(rho_total, e) |
| Momentum | rho * u | rho_total * u |
| Continuity fluxes | rho * u | each rho_i * u |
| Conservation | integral(rho) = const | integral(rho_i) = const each |

---

## Index Accessors

```python
eq.n_dens              # Number of densities
eq.n_cons              # Total conservative variables (n_dens + 4)
eq.density_names       # Tuple of density names
eq.density_map         # Dict: name -> index
eq.vel_idx_offset      # Where velocity starts (= n_dens)
eq.energy_idx          # Where energy is (= n_dens + 3)
eq.vel_ids             # Tuple: (vx_idx, vy_idx, vz_idx)
eq.density_slice       # slice(0, n_dens)
eq.momentum_slice      # slice(n_dens, n_dens+3)
eq.active_slice        # slice(0, 5) — for Riemann solver
eq.passive_slice       # slice(5, n_cons) — extra densities
```

---

## Riemann Solver Interface

The Riemann solver operates on 5 active variables: `[rho, vx, vy, vz, p]`.
Extra densities beyond the first are treated as passive scalars.

`get_fluxes_xi()` handles both modes:
- **Active-only** (shape[0]=5): standard Euler flux from Riemann solver
- **Full** (shape[0]=n_cons): multi-density flux with each rho advected independently

---

## API Reference

```python
EquationManager(
    gamma=1.4,
    eps=1e-12,
    isothermal=False,
    isothermal_sound_speed=1.0,
    density_names=("rho",)  # key parameter
)

# Conversions
primitives = eq.get_primitives_from_conservatives(conservatives)
conservatives = eq.get_conservatives_from_primitives(primitives)

# Fluxes
fluxes = eq.get_fluxes_xi(primitives, conservatives, axis=0)
sL, sR = eq.get_wavespeeds_xi(primitives, axis=0)

# Helpers
rho_total = eq.get_rho_total(conservatives)
```

---

## Files

| File | Role |
|------|------|
| `diffhydro/equationmanager_radiative_transf.py` | Multi-density equation manager |
| `diffhydro/physics/turbulence_radiative_transf.py` | Multi-density turbulence init |
| `diffhydro/physics/radiative_transfer.py` | Radiative transfer stubs |
| `diffhydro/equationmanager.py` | Original single-density equation manager |
