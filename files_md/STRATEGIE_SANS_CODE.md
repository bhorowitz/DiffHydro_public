# 📋 Stratégie d'Implémentation : Propagation Rapide + Diffusion 1/r²

## Votre Demande Résumée

**Injection photons:** 2 sources ponctuelles (0,54,50) et (0,56,50)  
**Injection momentum:** Direction X (comme actuellement)

**Comportement désiré:**
1. Propagation **rapide** des photons dans direction X (comme un faisceau)
2. Diffusion **progressive** sur les bords (lissage temporel)
3. Profil spatial final en **1/r²** (décroissance radiale)

---

## Le Problème avec l'Approche Actuelle

### Observations
- Photons injectés à (0,54,50) → diffusent **isotropiquement** en boule sphérique
- Momentum injecté comme boîte → crée **ondes de choc confinées** aux frontières
- **Pas de couplage** : le momentum n'accélère pas vraiment les photons dans direction X

### Racine Physique
L'équation complète de transport radiatif est:

```
∂E_γ/∂t + ∇·F_γ = 0                                [Énergie photonique]
∂F_γ/∂t + c²∇P_γ = -σ_a·F_γ + S_F                 [Flux photonique]
```

Actuellement votre code:
- ✅ Résout le premier terme (∂E_γ/∂t + ∇·F_γ) via solveur Riemann
- ✅ Injecte les deux termes (E et F)
- ❌ N'a **pas** de terme source de diffusion (pas de D·∇²E)
- ❌ N'a **pas** de dilution radiale (pas de 1/r²)

---

## Les 4 Piliers de la Solution

### Pilier 1️⃣ : Injection Photonique ANISOTROPE

**Concept:** Au lieu d'injecter une gaussienne 3D isotrope, injecter un **profil allongé** selon X.

```
Profil Isotrope (Actuel):        Profil Anisotrope (Proposé):
    ◯                                 ▬▬▬▬▬
   ○○○                              ▬▬▬▬▬▬▬▬
    ◯                                 ▬▬▬▬▬

Rayon égal en toutes directions   Allongé selon X (beam)
Diffusion sphérique              Diffusion directionnelle
```

**Mathématiquement:**

```
E_isotrope(r) = E₀ × exp(-r²/(2σ²))

E_anisotrope(x,y,z) = E₀ × exp(-(x²/(2σ_x²) + (y²+z²)/(2σ_yz²)))
                               ↑ coeff différent selon axe
```

**Paramètres clés:**
- σ_x (parallèle au momentum) = **15-20 cellules** (étendu)
- σ_yz (perpendiculaire) = **3-5 cellules** (étroit)
- Ratio = 3-5 crée un "beam" visible

**Effet:** Les photons sont déjà allongés dans la bonne direction, prêts pour advection rapide en X.

---

### Pilier 2️⃣ : Injection Momentum PROGRESSIF (Pas Boîte)

**Concept:** Au lieu de boîte [25-75] qui crée des chocs, injecter une **gaussienne lisse** qui s'étend au-delà des photons.

```
Momentum Boîte (Actuel):         Momentum Gaussien (Proposé):
  0 ┌─────────────┐ 0             0 ╱─────────────╲ 0
    │      1      │                 ╱             ╲
  0 └─────────────┘ 0             0                  0

Discontinuités aux x=25 et 75    Profil lisse, pas de choc
Ondes confinées                  Propagation uniforme
```

**Mathématiquement:**

```
F_boîte(x) = {  1 si 25 ≤ x ≤ 75
            {  0 sinon

F_gaussien(x) = F_max × exp(-(x - x_center)²/(2σ_F²))
```

**Paramètres clés:**
- σ_F = **30-40 cellules** (larger than photon injection)
- Pas de discontinuité → pas de shock de Riemann
- Le gradient doux crée une "pression" progressive

**Effet:** Momentum devient un "pusher" directionnel sans artefacts numériques.

---

### Pilier 3️⃣ : Terme Source de DIFFUSION (∇²E)

**Concept:** Les photons doivent **s'étaler progressivement** sur les bords. Cela vient d'une **diffusion physique** (mean free path).

```
Timestep t:         Timestep t+1:       Timestep t+2:
  ███               ████▓               █████▓▓
  ███               ███████             ██████████
  ███               ████▓               █████▓▓

Pas d'étalement     Étalement lent      Diffusion progressive
```

**Équation source:**

```
∂E_γ/∂t |_diffusion = D · ∇²E_γ

Où ∇²E = ∂²E/∂x² + ∂²E/∂y² + ∂²E/∂z²
```

**Implémentation:**
- Calculer le **Laplacien 3D** de E_gamma aux points de grille
- Ajouter comme **terme source** dans le RHS, après hydrodynamique
- Coefficient D = 0.05-0.2 (à calibrer)

**Où ajouter:**
```
_hydrostep():
    ├─→ forcing()              (injection photons + momentum)
    ├─→ rhs_unsplit()          (hydrodynamique standard)
    ├─→ add_diffusion()        ← NOUVEAU: E += D·∇²E·dt
    └─→ add_dilution()         (voir pilier 4)
```

**Effet:** Les bords du faisceau s'étalent graduellement, créant un "diffusion cloud" autour du beam.

---

### Pilier 4️⃣ : Dilution RADIALE en 1/r²

**Concept:** Au-delà de la simple diffusion (∇²E), les photons doivent **décroître** en s'éloignant radialement. C'est l'expansion géométrique + absorption.

```
Densité photonique vs distance:

E(r) ∝ 1/r²     ← Profil en 1/r² (astrophysique standard)
    ╲
     ╲
      ╲
       ╲___
```

**Équation source:**

```
∂E_γ/∂t |_dilution = -α · E_γ / r²

Où r = √((x-x₀)² + (y-y₀)² + (z-z₀)²)
```

**Implémentation:**
- Calculer distance **radiale** depuis chaque source
- Ajouter un **terme puits** proportionnel à E/r²
- Coefficient α = 0.01-0.05 (à calibrer)

**Où ajouter:** Même ligne que diffusion, dans `_hydrostep()`.

**Effet:** Profil spatial asymptotique converge vers 1/r², signature d'une source ponctuelle en astrophysique.

---

## Hiérarchie de Priorités

### En Termes d'Importance Visuelle

```
1. CRUCIAL (80% de l'effet): 
   → Injection photons anisotrope + momentum progressif
      C'est là que se joue la "forme" du faisceau

2. IMPORTANT (15% de l'effet):
   → Diffusion (∇²E)
      Crée l'étalement visible des bords

3. SUBTIL (5% de l'effet):
   → Dilution 1/r²
      Donne le profil asymptotique fine-tuned
```

---

## Calibrage Physique des Paramètres

### Pour Vos 2 Sources en (0,54,50) et (0,56,50)

**Injection Photonique:**
```
σ_x   = 18 cellules    (demi-largeur à 1/e²)
σ_yz  = 4 cellules
Ratio = 18/4 = 4.5     (beam bien collimaté)

Largeur totale (FWHM) ≈ 2.355·σ
→ En X: ~43 cellules
→ En YZ: ~9 cellules
```

**Injection Momentum:**
```
σ_F = 35 cellules      (enveloppe gaussian du flux)
x_center = 50          (centre du beam)

Cela crée une "zone d'influence" autour du beam
sans discontinuité de Riemann
```

**Diffusion:**
```
D = 0.1                (coefficient de diffusion)
Unités: [cellules]² / [timestep]

Ratio (D·dt)/dx² ≈ 0.1 × 0.067 / 1² ≈ 0.007
→ Diffusion lente et stable (critère CFL respecté)
```

**Dilution:**
```
α = 0.02               (taux de dilution 1/r²)

Perte fractionnelle par timestep ≈ α·dt/r²
À r=10: perte ≈ 0.02×0.067/100 ≈ 0.00013 (très faible)
À r=2: perte ≈ 0.02×0.067/4 ≈ 0.003 (modéré)
```

### Comment Calibrer?

1. **Commencer avec valeurs recommandées**
2. **Exécuter 100 timesteps, observer les slices XY/XZ à z=50**
3. **Mesurer le profil radial** : E(r) vs r
4. **Comparer avec 1/r²** : si trop plat → ↑α, si trop aigü → ↓α
5. **Ajuster D** pour étalement progressif (pas trop rapide ni trop lent)

---

## Checklist d'Implémentation (Phase par Phase)

### Phase 1️⃣ : Injection Anisotrope (Criticial)

**À faire:**
- [ ] Créer fonction `injection_photons_anisotrope(x,y,z,σ_x,σ_yz)`
- [ ] Remplacer l'injection gaussienne 3D isotrope
- [ ] Ajouter paramètres `sigma_parallel`, `sigma_perp` à config
- [ ] Tester: vérifier que photons sont bien allongés en X

**Validation:**
```
Afficher profil X en coupe (y=54, z=50):
```
▲ E_gamma
│      ╱╲
│     ╱  ╲
│    ╱    ╲
└───╱──────╲─── X
  25  50  75
```
Devrait être gaussienne centrée, étendue en X.
```

### Phase 2️⃣ : Momentum Progressif (Critical)

**À faire:**
- [ ] Remplacer boîte [25-75] par gaussienne
- [ ] Créer fonction `injection_momentum_anisotrope(σ_F)`
- [ ] Vérifier pas de discontinuités de Riemann
- [ ] Tester: profil momentum doit être lisse

**Validation:**
```
Pas d'ondes de choc aux x=25, 75 après 1 timestep
Les photons doivent avancer rapidement en X
```

### Phase 3️⃣ : Diffusion (Important)

**À faire:**
- [ ] Créer fonction `compute_laplacian_3d(E)`
- [ ] Créer fonction `add_diffusion_source(E, D, dt)`
- [ ] Intégrer dans `_hydrostep()` après RHS hydrodynamique
- [ ] Calibrer D (0.05-0.2)

**Validation:**
```
Mesurer largeur du faisceau vs temps:
width(t=0) ≈ 9 cellules
width(t=100) ≈ 15-20 cellules (fonction de D)

√(initial_width² + 2·D·t) ≈ width(t)
```

### Phase 4️⃣ : Dilution 1/r² (Subtle)

**À faire:**
- [ ] Créer fonction `add_radial_dilution_source(E, r_centers, α, dt)`
- [ ] Ajouter dans `_hydrostep()` après diffusion
- [ ] Calibrer α (0.01-0.05)

**Validation:**
```
Extraire profil radial E(r) à time=200
Fit E(r) = A/r^n
Vérifier n ≈ 2 (1/r²)
```

---

## Équations Complètes à Implémenter

### Laplacien Discret 3D (Finite Differences)

```
∇²E[i,j,k] = (E[i+1,j,k] - 2·E[i,j,k] + E[i-1,j,k])/dx²
           + (E[i,j+1,k] - 2·E[i,j,k] + E[i,j-1,k])/dy²
           + (E[i,j,k+1] - 2·E[i,j,k] + E[i,j,k-1])/dz²

Où dx = dy = dz = 1 (unités grille)
```

### Source de Diffusion

```
E_new[i,j,k] = E_old[i,j,k] + D·∇²E[i,j,k]·dt

Condition CFL: D·dt / dx² < 0.25 (pour stabilité)
```

### Source de Dilution Radiale

```
r[i,j,k] = √((i-i₀)² + (j-j₀)² + (k-k₀)²)

E_new[i,j,k] = E_old[i,j,k] - α·E_old[i,j,k]/(r[i,j,k]² + ε)·dt

Où ε = 1 (avoid division par zéro)
```

---

## Résumé : Ce que Vous Devez Faire (Sans Code)

### Architecture Globale

```
Actuellement:
  Photons injectés (gaussienne isotrope) + Momentum injecté (boîte)
  → Résultat: diffusion sphérique + chocs de Riemann confinés

Proposé:
  Photons injectés (gaussienne anisotrope) + Momentum (gaussienne progressive)
  + Diffusion (terme source ∇²E) + Dilution (terme source 1/r²)
  → Résultat: beam directionnel avec diffusion progressive et profil 1/r²
```

### Plan d'Implémentation

1. ✅ **Injection anisotrope** → change la forme initiale du faisceau
2. ✅ **Momentum progressif** → supprime les artefacts numériques
3. ✅ **Diffusion** → étale progressivement les bords
4. ✅ **Dilution 1/r²** → crée le profil asymptotique

### Paramètres Critiques à Calibrer

- σ_x (beam width) = 15-20
- σ_yz (beam height) = 3-5
- σ_F (momentum width) = 30-40
- D (diffusion) = 0.05-0.2
- α (dilution) = 0.01-0.05

### Validation à Chaque Étape

- Phase 1: Vérifier beam allongé en X
- Phase 2: Pas de chocs de Riemann
- Phase 3: Bords qui s'étalent avec √t
- Phase 4: Profil radial → 1/r²

---

## Fichiers Impliqués

```
diffhydro/physics/radiative_transfer.py
├─→ Modifier: injection photons (anisotrope)
└─→ Modifier: injection momentum (gaussienne)

diffhydro/hydro_core.py
├─→ Créer: compute_laplacian_3d()
├─→ Créer: add_diffusion_source()
├─→ Créer: add_radial_dilution_source()
└─→ Modifier: _hydrostep() (ajouter 2 termes source)

diffhydro/equationmanager_radiative_transf.py
└─→ Ajouter paramètres: D, α, sigma_x, sigma_yz, sigma_F
```

---

## 🎯 Bottom Line

**Pour obtenir exactement ce que vous demandez:**

1. **Injection photons anisotrope** (beam collimated)
2. **Momentum gaussien progressif** (pusher uniforme sans choc)
3. **Terme source diffusion** (étalement progressif)
4. **Terme source dilution 1/r²** (profil asymptotique)

C'est une **architecture physiquement correcte** et **numériquement stable**, utilisée standard en astrophysique numérique pour simuler sources de rayonnement.

Vous êtes prêt à implémenter! 🚀
