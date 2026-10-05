# GUIDE PRATIQUE: Ajouter une Nouvelle Densité (Scalaire Passif)

## Résumé Rapide

Pour ajouter **1 nouvelle densité** (ex: traceur pollutant, fraction de mélange):

1. **Modifier `n_cons`** dans `EquationManager` (5 → 6)
2. **Initialiser** le champ dans les conditions initiales
3. **Appliquer les conditions aux limites** appropriées
4.  **Les flux se calculent automatiquement**

---

## Exemple Pas-à-Pas: Ajouter un Traceur Pollutant

### ÉTAPE 1: Augmenter n_cons
**Fichier:** [`diffhydro/equationmanager.py` L.23-25](diffhydro/equationmanager.py#L23)

**Avant:**
```python
@dataclass
class EquationManager:
    gamma: float = 1.4
    n_cons: int = 5              # rho, rho*vx, rho*vy, rho*vz, E_tot
    eps: float = 1e-12
```

**Après:**
```python
@dataclass
class EquationManager:
    gamma: float = 1.4
    n_cons: int = 6              # +1 pour rho*pollutant
    eps: float = 1e-12
```

**Résultat:** 
```
Variable 0: ρ
Variable 1: ρ v_x
Variable 2: ρ v_y  
Variable 3: ρ v_z
Variable 4: E_totale
Variable 5: ρ * c_pollutant  (NOUVEAU)
```

### ÉTAPE 2: Initialiser le Champ Passif

**Exemple dans un script d'initialisation** (ex: `examples/my_tracer_test.py`):

```python
import jax.numpy as jnp
from diffhydro import hydro, EquationManager

# Configuration
eq = EquationManager(n_cons=6, gamma=1.4)
hydro_solver = hydro(eq, ...)

# Domaine [0, 1]³
nx, ny, nz = 64, 64, 64
x = jnp.linspace(0, 1, nx)
y = jnp.linspace(0, 1, ny)
z = jnp.linspace(0, 1, nz)

# Initialiser les 5 variables actives
rho = jnp.ones((nx, ny, nz)) * 1.0          # densité
vx = jnp.ones((nx, ny, nz)) * 0.1
vy = jnp.ones((nx, ny, nz)) * 0.0
vz = jnp.ones((nx, ny, nz)) * 0.0
p = jnp.ones((nx, ny, nz)) * 0.1            # pression

# Convertir en conservatifs actifs
E_tot = p / (eq.gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)
fields_active = jnp.stack([rho, rho*vx, rho*vy, rho*vz, E_tot], axis=0)

# NOUVEAU: Initialiser le scalaire passif pollutant
# Exemple: concentration nulle sauf dans une sphère
cx, cy, cz = 0.5, 0.5, 0.5  # centre
radius = 0.2
r2 = (x[:, None, None] - cx)**2 + (y[None, :, None] - cy)**2 + (z[None, None, :] - cz)**2
pollutant_conc = jnp.where(r2 < radius**2, 0.5, 0.0)  # fraction massique [0,1]
rho_pollutant = rho * pollutant_conc  # conservative: ρ * c

# Combiner actifs + passif
fields = jnp.concatenate([fields_active, rho_pollutant[None, ...]], axis=0)

print(f"Shape fields: {fields.shape}")
# Output: Shape fields: (6, 64, 64, 64)

# Résoudre
t_final = 0.1
output_fields = hydro_solver.evolve_till_time(fields, ...)
```

### ÉTAPE 3: Conditions aux Limites (si nécessaire)

**Fichier:** [`diffhydro/boundary/boundary.py`](diffhydro/boundary/boundary.py)

Si vous utilisez des conditions aux limites **autres que périodiques**, vérifier/adapter:

```python
# Exemple: si vous avez une classe Boundary custom
class MyBoundary:
    def apply(self, fields, axis, side):
        """
        fields: shape (n_cons, nx, ny, nz)
        """
        # Les conditions se copient pour TOUTES les variables
        # (incluant le passif automatiquement)
        active = fields[:5]      # variables physiques
        passive = fields[5:]     # pollutant
        
        # Appliquer les mêmes BCs au passif
        # (ou personnaliser selon besoin)
        return fields  # avec BC appliquées
```

### ÉTAPE 4: Récupération du Traceur après Simulation

```python
# output_fields shape: (6, nx, ny, nz)
rho_final = output_fields[0]
vx_final = output_fields[1] / rho_final
vy_final = output_fields[2] / rho_final
vz_final = output_fields[3] / rho_final
p_final = (1.4 - 1.0) * (output_fields[4] - 0.5*rho_final*(vx_final**2 + vy_final**2 + vz_final**2))

rho_pollutant_final = output_fields[5]            # ρ*c
pollutant_conc_final = rho_pollutant_final / rho_final  # c

print("Pollutant concentration min-max:", pollutant_conc_final.min(), pollutant_conc_final.max())
```

---

## Cas 2: Ajouter PLUSIEURS Traceurs (ex: 3)

```python
# Configuration
eq = EquationManager(n_cons=8, gamma=1.4)  # 5 actifs + 3 passifs

# Initialisation
fields_active = jnp.stack([rho, rho*vx, rho*vy, rho*vz, E_tot], axis=0)  # (5, ...)

# 3 traceurs
tracer1 = rho * 0.1  # 10% traceur 1
tracer2 = rho * 0.2  # 20% traceur 2
tracer3 = rho * 0.05 # 5% traceur 3

fields = jnp.concatenate([fields_active, tracer1[None,...], tracer2[None,...], tracer3[None,...]], axis=0)
# Shape: (8, nx, ny, nz)
```

---

## Vérification: Ce qui est AUTOMATIQUE

 **Advection** du traceur (flux calculé par `get_fluxes_xi()`)  
 **Solveurs de Riemann** appliqués (HLL, HLLC, etc.)  
 **Conservation de masse** du traceur  
 **Reconstructions PPM/PLT** inclus  
 **Parallélisation JAX** (jit, pmap, pjit)

---

## Vérification: Ce qui NÉCESSITE du Travail

 **Terme source** (si le traceur a une dynamique propre)  
 **Réactions chimiques** ou transformations  
 **Diffusivité** (si voulue)  
 **Conditions aux limites non-périodiques**  

**Exemple:** Si le traceur a une **source**, modifier `hydro_core._hydrostep()`:

```python
def _hydrostep(self, i, state, dt):
    # ... sweep x, y, z ...
    state = self.sweep_stack(state, dt, 0)  # sweep X
    state = self.sweep_stack(state, dt, 1)  # sweep Y
    state = self.sweep_stack(state, dt, 2)  # sweep Z
    
    # ← AJOUTER terme source pour le traceur (index 5)
    state = self.add_source_term(state, dt)
    
    return state

def add_source_term(self, state, dt):
    """Ajoute une source au traceur passif (index 5)"""
    rho = state[0]
    # Exemple: production dans une région
    source_region = (x > 0.3) & (x < 0.7)
    state_with_source = state.at[5].add(rho * source_coeff * dt * source_region)
    return state_with_source
```

---

## Validation

Pour vérifier que le traceur se propage correctement:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Coupes transversales à diferentes étapes
for i, (t, field) in enumerate([(0, fields_init), (t_mid, fields_mid), (t_final, fields_final)]):
    rho = field[0]
    rho_poll = field[5]
    c = rho_poll / (rho + 1e-12)  # concentration (évite NaN)
    
    axes[i].imshow(c[:, :, nz//2], origin='lower', cmap='viridis')
    axes[i].set_title(f't = {t:.3f}')
    axes[i].colorbar(label='c_pollutant')

plt.show()
```

---

## Troubleshooting

### Erreur: "shapes don't match" ou NaN dans output

**Cause probable:** `n_cons` non synchronisé dans `EquationManager` créé à différents endroits

**Solution:**
```bash
grep -r "n_cons" diffhydro/ examples/
# Vérifier que n_cons=6 PARTOUT où EquationManager est instancié
```

### Traceur "disparaît" (converge vers 0)

**Possibilités:**
1. Problème d'initialisation (vérifier `fields[5]` n'est pas 0)
2. Divisant par `rho_final` sans protections (si rho → 0)
3. Conditions aux limites appliquent zéro au traceur (vérifier Boundary)

### Performance dégradée

- Ajouter des variables n'augmente que légèrement le coût (même architecture JAX)
- Si lenteur apparaît, vérifier `jit` compilation (D'habitude rapide après 1ère run)

---

## Cas 3: Ajouter `rho_test` (Copie Identique de ρ para Tests)

Ce cas montre comment ajouter une **copie exacte de la densité** appelée `rho_test`. 
Cela permet de tester des modifications sur une variable sans affecter ρ physicalement.

### ÉTAPE 1: Augmenter `n_cons` dans EquationManager

**Fichier:** [`diffhydro/equationmanager.py` L.23-25](diffhydro/equationmanager.py#L23)

**Avant:**
```python
@dataclass
class EquationManager:
    gamma: float = 1.4
    n_cons: int = 5              # rho, rho*vx, rho*vy, rho*vz, E_tot
    eps: float = 1e-12
```

**Après:**
```python
@dataclass
class EquationManager:
    gamma: float = 1.4
    n_cons: int = 6              # +1 pour rho_test (copie de rho)
    eps: float = 1e-12
```

**Ce que ça fait:** Alloue de la place pour stocker une 6ème variable conservative. DiffHydro appliquera **automatiquement** les mêmes équations de conservation (flux, Riemann) à `rho_test`.

---

### ÉTAPE 2: Initialiser `rho_test` = `rho` aux Conditions Initiales

**Fichier:** Votre script/notebook de simulation (ex: `examples/my_test.py`)

**Code complet:**
```python
import jax.numpy as jnp
from diffhydro import hydro, EquationManager, HLLC, ConvectiveFlux, MUSCL3

# ───────────────────────────────────────────────
# 1. CRÉATION du solveur avec n_cons=6
# ───────────────────────────────────────────────
eq = EquationManager(n_cons=6, gamma=1.4)  # ← CLEF: n_cons augmenté

# Initialiser le flux
ss = diffhydro.solver.signal_speeds.DefaultSignalSpeeds(eq)
solver = HLLC(equation_manager=eq, signal_speed=ss)
cf = ConvectiveFlux(eq, solver, MUSCL3(limiter="VANLEER"), positivity=False)
hydro_solver = hydro(n_super_step=1000, fluxes=[cf], use_mol=True, integrator="SSPRK3")

# ───────────────────────────────────────────────
# 2. INITIALISATION DES CHAMPS
# ───────────────────────────────────────────────
nx, ny, nz = 128, 128, 128
x = jnp.linspace(0, 1, nx)
y = jnp.linspace(0, 1, ny)
z = jnp.linspace(0, 1, nz)

# Variables physiques (5 premières)
rho = jnp.ones((nx, ny, nz)) * 1.0
vx = jnp.ones((nx, ny, nz)) * 0.1
vy = jnp.zeros((nx, ny, nz))
vz = jnp.zeros((nx, ny, nz))
p = jnp.ones((nx, ny, nz)) * 0.1

# Convertir en variables conservatives
E_tot = p / (eq.gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)
fields_active = jnp.stack([rho, rho*vx, rho*vy, rho*vz, E_tot], axis=0)  # Shape: (5, nx, ny, nz)

# ───────────────────────────────────────────────
# 3. AJOUTER rho_test (COPIE IDENTIQUE DE rho)
# ───────────────────────────────────────────────
# rho_test commence EXACTEMENT igual à rho conservative
rho_test = rho.copy()  # Début identique

# Empiler avec les 5 variables actives
fields = jnp.concatenate([fields_active, rho_test[None, ...]], axis=0)

print(f"Shape fields: {fields.shape}")  # (6, 128, 128, 128) ✓
print(f"fields[0] (rho) et fields[5] (rho_test) identiques: {jnp.allclose(fields[0], fields[5])}")  # True ✓
```

**Explications ligne par ligne:**

| Ligne | Importance | Raison |
|-------|-----------|--------|
| `n_cons: int = 6` | CRITIQUE | Alloue la mémoire et déclare 6 variables conservatives |
| `rho_test = rho.copy()` | CRITIQUE | Initialise rho_test = rho (copie indépendante) |
| `jnp.concatenate([..., rho_test[None, ...]], axis=0)` | CRITIQUE | Ajoute rho_test comme 6ème composante des fields |
| `fields[5]` est maintenant indexable | Important | Permet d'accéder à rho_test après simulation |

---

### ÉTAPE 3: Appeler `evolve` et Récupérer `rho_test`

```python
# ───────────────────────────────────────────────
# 4. RÉSOUDRE (evolve calcule AUTOMATIQUEMENT rho_test)
# ───────────────────────────────────────────────
params = {}
output_fields, output_params = hydro_solver.evolve(fields, params)
# output_fields shape: (6, 128, 128, 128)
# output_fields[0] = ρ finale
# output_fields[5] = ρ_test finale (soumise aux MÊMES calculs que ρ)

# ───────────────────────────────────────────────
# 5. EXTRAIRE ET UTILISER rho_test
# ───────────────────────────────────────────────
rho_final = output_fields[0]
rho_test_final = output_fields[5]

vx_final = output_fields[1] / rho_final
vy_final = output_fields[2] / rho_final
vz_final = output_fields[3] / rho_final
E_thermal = output_fields[4] - 0.5 * rho_final * (vx_final**2 + vy_final**2 + vz_final**2)

print(f"ρ final: min={jnp.min(rho_final):.6f}, max={jnp.max(rho_final):.6f}")
print(f"ρ_test final: min={jnp.min(rho_test_final):.6f}, max={jnp.max(rho_test_final):.6f}")
print(f"Différence (rho_test - rho): max={jnp.max(jnp.abs(rho_test_final - rho_final)):.2e}")
# Typiquement max diff ~1e-14 (erreur de précision numérique)
```

---

### ÉTAPE 4: Modifier les Calculs de `rho_test` (Plus tard)

Une fois que ça fonctionne, vous pouvez modifier **seulement** rho_test:

**Option A: Via un ForçageMoyen (Forcing):**
```python
class TestForce:
    def __call__(self, sol, ax, params):
        """Modifie uniquement rho_test (index 5)"""
        source = jnp.zeros_like(sol[5])  # structure de rho_test
        # Ajouter une logique personnalisée ICI
        return source  # retourner terme source

forces = [TestForce()]
hydro_solver = hydro(..., forces=forces)
```

**Option B: Post-processing après simulation:**
```python
# Modifier rho_test APRÈS evolve basé sur ρ finale
rho_test_modified = rho_test_final * (1.0 + decay_factor)
```

---

### Résumé des Fichiers à Modifier pour `rho_test`

| Fichier | Ligne | Avant | Après | Raison |
|---------|-------|-------|-------|--------|
| `diffhydro/equationmanager.py` | 25 | `n_cons: int = 5` | `n_cons: int = 6` | Déclare 6ème variable |
| Votre script d'init | (~40ème) | `fields = jnp.stack([rho, ...], axis=0)` | `fields = jnp.concatenate([..., rho_test[None, ...]], axis=0)` | Inclut rho_test dans fields |
| Extraction de sortie | (~10ème) | `rho = output[0]` | `rho_test = output[5]` | Accès à rho_test final |

---

## Ressources

- Main equation manager: [`equationmanager.py`](diffhydro/equationmanager.py)
- Flux implementation: [`equationmanager.py` L.195-230](diffhydro/equationmanager.py#L195)
- Example simulations: [`examples/`](examples/)
