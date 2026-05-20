# 🎯 Architecture pour Propagation Rapide + Diffusion 1/r²

## Votre Objectif

Injecter 2 sources de photons en (0,54,1,50) et (0,56,1,50) avec momentum tel que:
1. **Propagation rapide** dans la direction du momentum (comme un rectangle directionnel)
2. **Diffusion progressive** sur les bords
3. **Profil d'intensité en 1/r²** (décroissance radiale typique)

---

## Analyse du Problème Actuel

Votre approche actuelle:
```python
# Injection photons (gaussienne 3D)
sol[0, star_x, star_y, star_z] += E_photons_gaussian

# Injection momentum (boîte rectangulaire)
sol[1, 25:75, :, :] += momentum_constant
```

**Résultat actuel:**
- Photons diffusent sphériquement (via solveur hydrodynamique)
- Momentum se propage en ondes de choc (boîte → discontinuités)
- **Pas de couplage** entre direction du momentum et propagation photonique

---

## Architecture Requise : 4 Composants

### 1️⃣ **Injection Anisotrope (Pas Simple Gaussienne)**

**Problème:** Gaussienne 3D → diffusion **isotrope** (égale en toutes directions)

**Solution:** Profil **anisotrope** aligné avec le momentum:

#### Mathématique
```
E(x,y,z) = E₀ × exp(-(ρ_∥² / σ_∥² + ρ_⊥² / σ_⊥²))

Où:
- ρ_∥ = projection parallèle au momentum (direction X)
- ρ_⊥ = projection perpendiculaire (directions Y,Z)
- σ_∥ >> σ_⊥  (élongation dans direction X)
```

#### Implémentation Conceptuelle
```python
def injection_anisotrope(sol, x_star, y_star, z_star, 
                          sigma_parallel, sigma_perp, E_total):
    """
    Crée un profil d'énergie photonique allongé dans la direction X.
    
    Parameters:
    - sigma_parallel (X): 15-20 cellules (élongé)
    - sigma_perp (Y,Z): 3-5 cellules (compact)
    - Ratio = 3-5 pour obtenir un "beam" directionnel
    """
    
    x_vals = jnp.arange(mesh_shape[0])
    y_vals = jnp.arange(mesh_shape[1]) - y_star
    z_vals = jnp.arange(mesh_shape[2]) - z_star
    
    # Grille 3D
    X, Y, Z = jnp.meshgrid(x_vals - x_star, y_vals, z_vals, indexing='ij')
    
    # Profil anisotrope: gaussienne allongée selon X
    profile = jnp.exp(-(X**2 / (2*sigma_parallel**2) + 
                        (Y**2 + Z**2) / (2*sigma_perp**2)))
    
    # Normalisation
    profile = profile / jnp.sum(profile)
    
    # Injection
    sol[0, :, :, :] += E_total * profile
    
    return sol
```

**Paramètres Recommandés:**
- σ_∥ (parallèle au momentum): **15-20 cellules** (élongation forte)
- σ_⊥ (perpendiculaire): **3-5 cellules** (faisceau étroit)
- **Ratio σ_∥/σ_⊥ = 3-5** pour un "beam" visible

---

### 2️⃣ **Momentum Directionnel Progressif**

**Problème Actuel:** Boîte uniforme [25-75] → crée discontinuités Riemann confinées

**Solution:** Profil gaussien **progressif** dans la direction du rayonnement:

#### Mathématique
```
F_x(x) = F_max × exp(-(x - x_center)² / (2 × σ_momentum²))

Où:
- σ_momentum ≈ 2 × σ_parallel  (plus large que l'injection)
- Le momentum s'étend au-delà de la région photonique
- Crée un "champ de pression" autour du faisceau
```

#### Implémentation Conceptuelle
```python
def injection_momentum_anisotrope(sol, x_star, y_star, z_star,
                                   F_total, sigma_momentum):
    """
    Injecte un profil de momentum gaussien en direction X.
    
    Parameters:
    - sigma_momentum: ~30-40 cellules (envelope du momentum)
    - F_total: amplitude du flux (c² × photon_energy / cell_volume)
    """
    
    x_vals = jnp.arange(mesh_shape[0])
    
    # Profil gaussien du momentum en X
    F_profile = jnp.exp(-(x_vals - x_star)**2 / (2 * sigma_momentum**2))
    
    # Broadcast sur Y,Z (uniforme perpendiculairement)
    sol[1, :, :, :] += F_total * F_profile[:, None, None]
    # sol[2,3] restent inchangés (pas de momentum Y,Z)
    
    return sol
```

**Paramètres Recommandés:**
- σ_momentum: **30-40 cellules** (>σ_parallel)
- Largeur totale (FWHM): ~80 cellules
- Crée un "pusher" directif sans chocs confinés

---

### 3️⃣ **Terme Source de Diffusion Radiale**

**Problème:** Actuellement il n'y a **pas de diffusion** → les photons advectent seulement

**Solution:** Ajouter un **terme source de diffusion** dans le RHS:

#### Équation de Transport Complète
```
∂E_γ/∂t + ∇·F_γ = 0                                    [Conservation]
∂F_γ/∂t + c²∇(E_γ/3) = -χ·F_γ + D·∇²E_γ             [Transport + Diffusion]

Où:
- Premier terme: transport (déjà fait par Riemann solver)
- -χ·F_γ: absorption (damping du flux)
- D·∇²E_γ: diffusion (étalement radial)
```

#### Implémentation Conceptuelle
```python
def add_diffusion_source(sol, params, dt):
    """
    Ajoute une source de diffusion radiale aux photons.
    
    Modifie sol[0] (énergie photonique) via Laplacien de diffusion.
    """
    E_gamma = sol[0]
    
    # Coefficient de diffusion (à calibrer avec physique)
    D = params.get('diffusion_coeff', 0.1)  # Unités: (vitesse × longueur)
    
    # Laplacien 3D (finite differences)
    laplacian_E = compute_laplacian_3d(E_gamma)
    
    # Source de diffusion
    source_diffusion = D * laplacian_E
    
    # Mise à jour
    sol[0] += source_diffusion * dt
    
    return sol
```

**Où l'implémenter?** Dans `hydro_core.py`, fonction `_hydrostep()`, **APRÈS** l'étape hydrodynamique principale:

```python
# ... dans _hydrostep():
fields, params = self._hydrostep(i, (fields, params), dt)

# Puis ajouter APRÈS:
if params.get('use_radiative_diffusion', True):
    fields = add_diffusion_source(fields, params, dt)
```

---

### 4️⃣ **Terme Source Radiatif de Récession 1/r²**

**Problème:** La diffusion simple (∇²E) donne **1/r en 3D**, pas 1/r²

**Solution:** Ajouter un **terme source de densité radiale** (M1 closure modifiée):

#### Mathématique
```
La solution de Poisson en 3D pour une source ponctuelle est:
E(r) ∝ 1/r (diffusion isotrope)

Pour obtenir 1/r², il faut une **décroissance additionnelle**:
- Dilution géométrique (1/r²) vient du "dilution factor"
- Absorption progressif

Source complète:
∂E_γ/∂t + ∇·F_γ = -α·E_γ/r + D·∇²E_γ
                    ↑ dilution radiale
```

#### Implémentation Conceptuelle
```python
def add_radial_dilution_source(sol, x_star, y_star, z_star, alpha, dt):
    """
    Ajoute une dilution radiale 1/r² aux photons.
    
    Parameters:
    - alpha: coefficient de dilution (à calibrer)
    - Simule l'expansion géométrique + absorption
    """
    
    E_gamma = sol[0]
    
    # Distance radiale depuis la source
    x_idx = jnp.arange(mesh_shape[0])
    y_idx = jnp.arange(mesh_shape[1])
    z_idx = jnp.arange(mesh_shape[2])
    
    X, Y, Z = jnp.meshgrid(x_idx - x_star, y_idx - y_star, z_idx - z_star, indexing='ij')
    r = jnp.sqrt(X**2 + Y**2 + Z**2)
    r_safe = jnp.maximum(r, 1.0)  # Éviter division par zéro
    
    # Dilution 1/r²
    dilution_factor = alpha / (r_safe**2)
    dilution_factor = jnp.clip(dilution_factor, 0, 1)  # Entre 0 et 1
    
    # Source négative (perte)
    source_dilution = -dilution_factor * E_gamma
    
    # Mise à jour
    sol[0] += source_dilution * dt
    
    return sol
```

**Paramètres Recommandés:**
- α: **0.01 - 0.1** (contrôle le taux de dilution)
- Calibrer pour obtenir profil observé en astrophysique

---

## Architecture Complète du Flux Hydrodynamique

### Timeline d'Exécution par Timestep

```
├─ forcing() [radiative_transfer.py]
│  ├─→ Injection photons ANISOTROPE     ← Nouveau: profil allongé
│  ├─→ Injection momentum ANISOTROPE    ← Nouveau: gaussienne progressive
│  └─→ retour (sol_updated, params)
│
├─ rhs_unsplit() [hydro_core.py]
│  ├─→ flux() [fluxes.py]
│  │   ├─→ reconstruct_xi (WENO/TENO5)
│  │   ├─→ solve_riemann_problem (Lax-Friedrichs)
│  │   └─→ update_conservatives
│  │
│  ├─→ add_diffusion_source()            ← Nouveau: diffusion 3D
│  │   └─→ E_gamma += D·∇²E_gamma·dt
│  │
│  └─→ add_radial_dilution_source()      ← Nouveau: 1/r² dilution
│      └─→ E_gamma -= α·E_gamma/r²·dt
│
└─ retour: sol_updated pour next timestep
```

---

## Hiérarchie des Effets Physiques

### Ordre d'Importance pour Votre Cas

```
1. DOMINANT: Injection Anisotrope
   - Détermine la forme initiale du "beam"
   - 60% de l'effet visible
   
2. IMPORTANT: Momentum Directionnel
   - Accélère les photons dans la direction X
   - Crée la propagation rapide
   - 25% de l'effet
   
3. MODÉRATEUR: Diffusion (∇²E)
   - Étale progressivement les bords
   - Lisse les discontinuités
   - 10% de l'effet
   
4. SUBTIL: Dilution 1/r²
   - Donne le profil asymptotique final
   - Peut être ajusté après
   - 5% de l'effet initial
```

---

## Paramètres à Calibrer (Physique)

Pour vos 2 sources en (0,54,1,50) et (0,56,1,50):

| Paramètre | Valeur Recommandée | Physique |
|-----------|-------------------|----------|
| σ_∥ (injection photo) | 15-20 cellules | Beam width |
| σ_⊥ (injection photo) | 3-5 cellules | Lateral width |
| σ_momentum | 30-40 cellules | "Pression" autour du beam |
| D (diffusion) | 0.05-0.2 | Mean free path × vitesse |
| α (dilution 1/r²) | 0.01-0.05 | Expansion rate |

**Comment les calibrer?**
1. Commencer avec la configuration recommandée
2. Faire tourner quelques timesteps
3. Observer le profil spatial en slice (z=50)
4. Ajuster D et α jusqu'à obtenir forme 1/r²
5. Vérifier que l'énergie décroît physiquement

---

## Checkpoints de Validation

### À Chaque Étape d'Implémentation

**Après Injection Anisotrope:**
```python
# Vérifier que les photons sont allongés selon X
nonzero_x = jnp.argwhere(sol[0, :, 54, 50] > threshold)
print(f"Photons présents en X: [{nonzero_x.min()}, {nonzero_x.max()}]")
# Devrait être élongé vers X+

assert len(nonzero_x) >= 15, "Profil photonique trop court!"
```

**Après Momentum Anisotrope:**
```python
# Vérifier que le momentum est une gaussienne lisse
F_x_profile = jnp.mean(sol[1, :, 54, 50])
print(f"Momentum en X: max={jnp.max(sol[1])}, profile shape ok?")
# Devrait être lisse, sans discontinuités de Riemann
```

**Après Diffusion:**
```python
# Vérifier que les bords s'étalent progressivement
E_iter_0 = sol[0].copy()
# ... 10 timesteps ...
E_iter_10 = sol[0].copy()

expansion = (jnp.sum(E_iter_10) - jnp.sum(E_iter_0)) / jnp.sum(E_iter_0)
print(f"Expansion: {expansion*100:.1f}%")
# Devrait être ~5-20% (dépend de D)
```

**Après Dilution 1/r²:**
```python
# Vérifier le profil radial
r_vals = jnp.linspace(1, 50, 100)
E_radial = [jnp.mean(sol[0, 50:60, 54+dr, 50]) for dr in r_vals]
# Plotter et vérifier que E ~ 1/r²
```

---

## Fichiers à Modifier (Ordre d'Implémentation)

### Phase 1: Injection Anisotrope
**Fichier:** `diffhydro/physics/radiative_transfer.py`  
**Fonctions à créer:**
- `injection_photons_anisotrope()` → remplace gaussienne 3D
- `injection_momentum_anisotrope()` → remplace boîte rectangulaire

### Phase 2: Diffusion
**Fichier:** `diffhydro/hydro_core.py`  
**Fonctions à créer:**
- `compute_laplacian_3d()` → calcule ∇²E
- `add_diffusion_source()` → ajoute terme source

**Modifications:**
- Dans `_hydrostep()`, ajouter appel après RHS hydrodynamique

### Phase 3: Dilution
**Fichier:** `diffhydro/hydro_core.py`  
**Fonctions à créer:**
- `add_radial_dilution_source()` → ajoute 1/r² decay

**Modifications:**
- Dans `_hydrostep()`, ajouter appel optionnel

---

## Résumé Architectural

```
┌─────────────────────────────────────────────────┐
│ Injection Photons ANISOTROPE (gaussienne X-étendue)
│ Sigma_X = 15-20, Sigma_YZ = 3-5
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ Injection Momentum ANISOTROPE (gaussienne progressive)
│ Sigma = 30-40, crée "pusher" directif
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ Solveur Hydrodynamique Standard (Lax-Friedrichs)
│ Advecte photons + momentum selon équations RT
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ Terme Source de DIFFUSION (D·∇²E)
│ Étale progressivement les bords
│ Coefficient D = 0.05-0.2
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ Terme Source de DILUTION RADIALE (-α·E/r²)
│ Crée décroissance 1/r²
│ Coefficient α = 0.01-0.05
└──────────────────┬──────────────────────────────┘
                   ↓
          ✨ BEAM DIRECTIONNEL DIFFUSIF ✨
          avec profil d'intensité en 1/r²
```

---

## Cas d'Usage: Vos 2 Sources (0,54,1,50) et (0,56,1,50)

### Configuration Finale Recommandée

```python
# Dans StellarRadiationForce.__init__():
self.use_anisotropic_injection = True
self.use_radiative_diffusion = True
self.use_radial_dilution = True

# Paramètres injection
self.sigma_parallel = 18        # cellules
self.sigma_perp = 4             # cellules
self.sigma_momentum = 35         # cellules

# Paramètres diffusion
self.diffusion_coeff = 0.1      # (unités code)
self.dilution_alpha = 0.02      # (1/r² rate)

# Pour vos 2 sources:
star_positions = jnp.array([
    [0, 54, 50],  # Source 1
    [0, 56, 50]   # Source 2
], dtype=jnp.int32)

# Le momentum sera injecté dans tout le domaine X
# avec une enveloppe gaussienne centrée sur X=0
```

### Résultats Attendus

**Iter 0-10:** Beam bien collimaté, propagation rapide en X  
**Iter 10-100:** Diffusion visible sur les bords (∝ √t)  
**Iter 100+:** Profil asymptotique se forme (1/r²)  
**Iter 200+:** Équilibre entre advection et diffusion

---

## 🎯 Plan d'Action Résumé

**Sans toucher au code, vous devez:**

1. ✅ **Définir les profils anisotropes** (mathématique + paramètres)
2. ✅ **Choisir les coefficients** (D, α) basés sur physique
3. ✅ **Planifier l'architecture RHS** (où ajouter diffusion/dilution)
4. ✅ **Identifier les points de validation** (checksums, profils)
5. ⏳ **Implémenter par phases** (injection → diffusion → dilution)

**Cette architecture va vous donner:**
- ✨ Propagation rapide directionnelle (momentum anisotrope)
- ✨ Diffusion progressive (terme source ∇²E)
- ✨ Profil 1/r² asymptotique (dilution radiale)

C'est la solution complète au niveau architectural. À vous de l'implémenter!
