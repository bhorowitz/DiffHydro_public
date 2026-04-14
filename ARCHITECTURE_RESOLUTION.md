# Architecture de Résolution des Équations - DiffHydro

## 1. FONCTIONS PRINCIPALES QUI RÉSOLVENT LES ÉQUATIONS

### A. Intégrateurs Temporels (Cœur de la résolution)
**Fichier:** [`diffhydro/solver/integrator.py`](diffhydro/solver/integrator.py)

- `rk2_step()` - Runge-Kutta 2ème ordre
- `ssprk3_step()` - Strong Stability Preserving RK3
- `rk4_step()` - Runge-Kutta 4ème ordre

Ces fonctions appliquent: `u_new = u + dt * L(u)` où L est l'opérateur RHS

### B. Pas de Temps (Sweeps Directionnels)
**Fichier:** [`diffhydro/hydro_core.py`](diffhydro/hydro_core.py#L327)

- `split_solve_step()` - Effectue un step RK2 selon une direction (x, y, ou z)
  - Calcule le flux à face `fu1 = self.flux(sol, ax, params)`
  - Applique: `sol = sol - (fu_j+1/2 - fu_j-1/2) * dt / dx`

- `_hydrostep()` - Boucle sur les 3 directions (x, y, z)
- `evolve_till_time()` - Résout jusqu'à un temps cible avec ajustement de dt

### C. Calcul des Flux Physiques
**Fichier:** [`diffhydro/fluxes.py`](diffhydro/fluxes.py#L68)

- `ConvectiveFlux.flux()` - Orchestrate principal:
  1. Reconstruit les variables primitives aux faces (PLT/PPM)
  2. Applique un **solveur de Riemann** aux interfaces
  3. Retourne les flux numériques

**Fichier:** [`diffhydro/equationmanager.py`](diffhydro/equationmanager.py#L195)

- `EquationManager.get_fluxes_xi()` - Calcule les **flux physiques** Euler compressibles:
  ```
  F_rho = rho * u_i
  F_mom = rho * u_i * u_j + p * δij   (pression selon direction)
  F_E   = u_i * (E_tot + p)           (flux d'énergie)
  F_s_k = rho * s_k * u_i             (scalaires passifs - AUTOMATIQUE)
  ```

### D. Solveurs de Riemann (Interfacial Fluxes)
**Fichier:** [`diffhydro/solver/riemann_solver.py`](diffhydro/solver/riemann_solver.py)

Classe `RiemannSolver` abstraite avec implémentations:

| Solveur | Fichier | Description |
|---------|---------|-------------|
| `LaxFriedrichs` | L.173 | Simple, robuste |
| `LaxFriedrichs_safe` | L.190 | Avec protections NaN |
| `HLLC` | L.325 | 3 ondes (Toro) |
| `HLL` | L.362 | 2 ondes, robuste |
| `HLL_MHD` | L.446 | Version MHD |
| `HLLD_MHD_old` | L.691 | 4 ondes MHD (expérimental) |

Chaque solveur implémente:
```python
def _solve_riemann_problem_xi_single_phase(
    primitives_L, primitives_R,
    conservatives_L, conservatives_R,
    axis: int
) -> Tuple[fluxes_xi, None, None]
```

---

## 2. PARTIES À MODIFIER POUR AJOUTER UNE NOUVELLE DENSITÉ (SCALAIRE PASSIF)

### Étape 1: Augmenter le nombre de variables conservatives
**Fichier:** [`diffhydro/equationmanager.py`](diffhydro/equationmanager.py#L23)

```python
@dataclass
class EquationManager:
    gamma: float = 1.4
    n_cons: int = 5  # ← MODIFIER: passer de 5 à 6 pour 1 scalaire passif
    # n_cons = 5 + n_passive où n_passive = nombre de traceurs supplémentaires
```

**Location précise:** [`equationmanager.py` ligne 23](diffhydro/equationmanager.py#L23)

### Étape 2: Vérifier la structure des variables (DÉJÀ IMPLÉMENTÉE)

L'architecture supporte déjà les variables passives:

- **Variables actives** (0-4): [rho, vx, vy, vz, p]
- **Variables passives** (5+): [s_1, s_2, ...] (densités supplémentaires)

Utilise les slices:
```python
@property
def active_slice(self):
    return slice(0, self.n_active)  # [0:5]

@property
def passive_slice(self):
    return slice(self.n_active, self.n_cons)  # [5:n_cons]
```

### Étape 3: Les flux sont calculés AUTOMATIQUEMENT
**Fichier:** [`equationmanager.py` ligne 221](diffhydro/equationmanager.py#L221)

```python
def get_fluxes_xi(self, primitives, conservatives, axis: int):
    # ... calcul des flux actifs (rho, momenta, énergie) ...
    flux_a = jnp.stack([rho_ui, fx_rhou, fx_rhov, fx_rhow, fx_E], axis=0)
    
    # ← AUTOMATIQUE pour les scalaires passifs:
    if conservatives.shape[0] > self.n_active:
        cons_p = conservatives[self.passive_slice]  # récupère ρs_k
        flux_p = cons_p * ui                         # flux = ρs_k * u_i
        return jnp.concatenate([flux_a, flux_p], axis=0)
    
    return flux_a
```

### Étape 4: Conditions aux limites (si nécessaire)
**Fichier:** [`diffhydro/boundary/boundary.py`](diffhydro/boundary/boundary.py)

Vérifier que les halos incluent les variables passives (devrait être automatique via le slicing)

### Exemple d'implémentation complète:

```python
# Avant (1 densité, 3 vélocités, 1 pression)
eq = EquationManager(n_cons=5, gamma=1.4)

# Après (+ 1 scalaire passif traceur, ex. fraction massique de polluant)
eq = EquationManager(n_cons=6, gamma=1.4)  
# Shape: [rho, vx, vy, vz, p, rho*tracer]
```

### Limitations/Points à vérifier:

✅ **Automatiques:** flux, reconstructions (PPM/PLT), solveurs de Riemann  
⚠️ **À vérifier manuellement:**
- Conditions aux limites pour le nouvel scalaire
- Terme source d'advection-réaction si applicable
- Initialisation des conditions initiales

---

## 3. DIAGRAMME DE FLUX

```
evolve_till_time()
    ↓
_hydrostep() 
    ↓ (boucle x, y, z)
split_solve_step()  [RK2]
    ├─ flux(sol, ax) 
    │   ├─→ Reconstruction PPM/PLT
    │   ├─→ Riemann Solver 
    │   │   └─→ get_fluxes_xi() [VARIABLES ACTIVES + PASSIVES]
    │   └─→ Retourne F_i+1/2
    └─ Mise à jour: sol = sol - (F_i+1/2 - F_i-1/2) * dt/dx
```

---

## 4. FICHIERS CLÉ

| Fichier | Rôle |
|---------|------|
| [`equationmanager.py`](diffhydro/equationmanager.py) | Gère variables actives/passives, calcule flux |
| [`equationmanager_mhd.py`](diffhydro/equationmanager_mhd.py) | Variante MHD |
| [`hydro_core.py`](diffhydro/hydro_core.py) | Boucles temporelles, steps directionnels |
| [`fluxes.py`](diffhydro/fluxes.py) | Orchestre reconstruction + Riemann |
| [`solver/riemann_solver.py`](diffhydro/solver/riemann_solver.py) | Implémentations des solveurs |
| [`solver/integrator.py`](diffhydro/solver/integrator.py) | RK2, SSPRK3, RK4 |
| [`boundary/boundary.py`](diffhydro/boundary/boundary.py) | Conditions aux limites |

