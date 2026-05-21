# Solution : Ajouter `psi` avec HLLC

## Le problème
Tu voulais ajouter une variable `psi` et l'utiliser avec HLLC, en faisant `self.mass_ids = self.active_map["psi"]`.  
**Cette approche casse tout le code.**

## Pourquoi c'est mauvais
1. `mass_ids` est utilisé **partout** dans le code:
   - `get_primitives_from_conservatives()` 
   - `get_pressure()`
   - `get_sound_speed()`
   - HLLC lui-même (`solver.mass_ids`)
   
2. Si tu le changes pour pointer vers `psi`, le code s'attend plus à ce que la **vraie masse (rho)** soit à l'index 0.

3. Tu finirais par avoir des conversions primitives/conservatives cassées, des calculs de pression incorrects, et HLLC calculerait les flux avec `psi` au lieu de `rho`.

## La bonne approche : `psi` comme variable PASSIVE

Au lieu de traiter `psi` comme une **variable active** (avec son propre système d'équations), traite-le comme une **variable passive transportée par rho** :

### Structure
```
Variables actives (5) : rho, vx, vy, vz, p
  → Ces 5 variables sont résolues par HLLC
  → Conservateurs: [rho, rho*vx, rho*vy, rho*vz, E_total]

Variables passives (1+) : psi, s_k, ...
  → Ces variables sont transportées par le flux de masse rho
  → Conservateurs: [rho*psi, rho*s_k, ...]
  → Équation pour psi: psi_t + div(rho*psi*u) = 0
```

### Implémentation
```python
# EquationManager
active_names = ("rho", "vx", "vy", "vz", "p")  # 5 seules
n_cons = 6  # 5 actives + 1 psi passif

# Variables
self.mass_ids = 0  # rho (toujours!)
self.psi_passive_idx = 0  # Position de psi dans passive_slice
```

### Avantages
1. ✓ HLLC continue à utiliser `mass_ids = 0` (rho) → **aucun changement**
2. ✓ Les conversions primitives/conservatives marchent correctement
3. ✓ Le flux passif est automatiquement géré par `ConvectiveFlux.flux()` :
   ```python
   # Dans fluxes.py
   mass_flux = F_act[eq.mass_ids]  # flux de rho
   passive_face = where(mass_flux >= 0, psi_L, psi_R)
   F_passive = mass_flux * passive_face  # Flux de psi = rho*u * psi
   ```

## Exemple d'utilisation
```python
eq = EquationManager(n_cons=6)  # 5 actives + 1 psi

# Les conversions marchent
prim = eq.get_primitives_from_conservatives(sol)  # psi est extrait automatiquement
cons = eq.get_conservatives_from_primitives(prim)

# HLLC utilise toujours rho comme masse
solver = dh.HLLC(equation_manager=eq, signal_speed=ss)
cf = dh.ConvectiveFlux(eq, solver, dh.MUSCL3())
hydrosim = dh.hydro(fluxes=[cf])

# Ça marche!
q = hydrosim.evolve_till_time(sol, {}, t_final)
```

## Si tu veux vraiment deux systèmes couplés (rho + psi avec leurs propres énergies)
C'est **beaucoup plus compliqué** et tu aurais besoin de :
1. Créer deux `EquationManager` distincts
2. Créer deux solveurs HLLC distincts
3. Coupler les deux systèmes dans le RHS
4. Gérer les flux d'énergie croisés manuellement

**Ce n'est pas recommandé** sauf si c'est vraiment physiquement motivé.

## Résumé
**Garder `psi` comme variable passive** est la solution la plus simple, la plus robuste, et la plus compatible avec HLLC.
