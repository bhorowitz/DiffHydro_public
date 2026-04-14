# GUIDE PRATIQUE: Ajouter une Nouvelle Densité (Scalaire Passif)

## Résumé Rapide

Pour ajouter **1 nouvelle densité** (ex: traceur pollutant, fraction de mélange):

1. **Modifier `n_cons`** dans `EquationManager` (5 → 6)
2. **Initialiser** le champ dans les conditions initiales
3. **Appliquer les conditions aux limites** appropriées
4. ✅ **Les flux se calculent automatiquement**

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
    n_cons: int = 6              # ↑ +1 pour rho*pollutant
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

✅ **Advection** du traceur (flux calculé par `get_fluxes_xi()`)  
✅ **Solveurs de Riemann** appliqués (HLL, HLLC, etc.)  
✅ **Conservation de masse** du traceur  
✅ **Reconstructions PPM/PLT** inclus  
✅ **Parallélisation JAX** (jit, pmap, pjit)

---

## Vérification: Ce qui NÉCESSITE du Travail

❌ **Terme source** (si le traceur a une dynamique propre)  
❌ **Réactions chimiques** ou transformations  
❌ **Diffusivité** (si voulue)  
❌ **Conditions aux limites non-périodiques**  

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

## Ressources

- Main equation manager: [`equationmanager.py`](diffhydro/equationmanager.py)
- Flux implementation: [`equationmanager.py` L.195-230](diffhydro/equationmanager.py#L195)
- Example simulations: [`examples/`](examples/)
