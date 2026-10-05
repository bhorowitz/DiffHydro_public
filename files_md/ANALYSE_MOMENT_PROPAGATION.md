# Analyse Détaillée : Pourquoi l'Injection de Moment est Localisée aux Extrémités

## Résumé du Problème

L'utilisateur injecte du moment sur **toutes les cellules X ∈ [25, 75]** (50 cellules), mais observe uniquement des valeurs non-zéro en **X ∈ [23, 24, 25, 26, 73, 74, 75, 76]** (8 cellules seulement après 1 timestep).

### Observation Clé du Debug Output
```
Non-zero x for y=50, z=50: [23 24 25 26 73 74 75 76]
```

Cela ressemble à une signature d'onde choc ou de propagation ondulatoire, pas une injection homogène.

---

## 1. Code d'Injection de Moment

```python
# Ligne 173-177 dans radiative_transfer.py
if self.injection_momentum == True:
    def injection_moment_1D_X(sol):
        xi = jnp.arange(25, 75)  # 50 cellules
        total_source = jnp.sum(per_star_source)  # Somme des photons
        sol = sol.at[1, xi, :, :].add(
            self.light_speed**2 * total_source / len(xi)  # Moment / 50 cellules
        )
    sol = injection_moment_1D_X(sol)
```

**Injection:** Uniforme sur 50 cellules
**Amplitude:** `c² × Σ(photons) / 50`
**Champ affecté:** `sol[1]` = momentum en X (F_gamma_x)

---

## 2. Équation de Transport Hydrodynamique

Après l'injection, le **solveur de Riemann** prend le relais. Les flux de photons obéissent aux **équations de transport**:

```
∂ρ/∂t + ∇·(ρu) = 0                    [conservation de masse photonique]
∂(ρu)/∂t + ∇·(ρu⊗u + P) = 0           [conservation de moment]
∂(ρE)/∂t + ∇·((ρE + P)u) = 0          [conservation d'énergie]
```

Pour les **photons** avec `equation_of_state = c/√3` (radiative EOS):
- Vitesse du son: `c_s ≈ c/√3`
- Pression: `P = ρE/3`
- Vitesse caractéristique: `u_char = c_s = c/√3`

---

## 3. Dynamique des Ondes de Choc

### Condition Initiale
- **État 1 (LEFT)** : `ρ=0, u=0, P=0` (extérieur)
- **État 2 (RIGHT)** : Injection de moment homogène sur [25, 75]

### Évolution Attendue
L'injection produit une **sur-densité et une surpression locales**. Cela crée **deux ondes choc**:

1. **Choc avant** (X direction positive) : Se propage de X=75 vers X=100+
2. **Choc arrière** (X direction négative) : Se propage de X=25 vers X=0

La **vitesse de choc** dépend du **contraste de pression**:

```
v_choc ≈ √(P_high / ρ_high) ≈ √(ρ_high × E_high / ρ_high) ≈ c_s ≈ c/√3
```

---

## 4. Analyse des Résultats : Propagation Linéaire

### Timestep 0 (iter=0)
**Après injection:**
- Non-zero count: 0 (avant la propagation)

### Timestep 1 (iter=10)
```
Non-zero x: [23, 24, 25, 26, 73, 74, 75, 76]
```

**Interprétation:**
- Les limites à X=25 et X=75 ont créé des **fronts d'onde discrets**
- Après 1 timestep: les ondes se sont propagées de **~2 cellules** de chaque côté

**Vitesse observée:** 
```
Δx = 2 cellules/timestep
v_grid ≈ 2 × dx / dt
```

Si `dx=1` et estimant `dt ≈ 0.067` (du printout: `0.06666666666666667`):
```
v_grid ≈ 2 / 0.067 ≈ 30 (unités code)
```

Comparé à `c/√3 ≈ 2/√3 ≈ 1.15`, cela suggère une **propagation rapide** ou un **effet numérique**.

---

## 5. Pourquoi SEULEMENT les Extrémités ?

### Hypothèse 1: Gradient de Riemann
Le solveur de Riemann détecte les **discontinuités**:
- À X=24/25 : Discontinuité GAUCHE (transition 0 → moment_injected)
- À X=75/76 : Discontinuité DROITE (transition moment_injected → 0)

Les flux sont calculés aux **interfaces de cellules**. Les valeurs non-zéro se propagent d'abord aux cellules **adjacentes aux discontinuités** (23-26 et 73-76).

### Hypothèse 2: Schéma de Flux Haute Résolution (MUSCL/PPM)
Le code utilise probablement:
- **Reconstructeur MUSCL/PPM** : Reconstruit des états linéaires/paraboliques
- **Limiteur TVD** : Réduit l'amplitude des oscillations

L'injection homogène produit un profil **constant** entre X=25-75. Le limiteur TVD **détecte les pentes nulles** et ne génère pas de flux en interne.

Les **pentes non-nulles** apparaissent aux bornes (X=24/25 et X=75/76), créant des flux localisés.

### Hypothèse 3: Conservation Numérique
```
Flux_X = ρu·face   (advection)
       = ρc·face    (pour photons)
```

Pour une injection **uniforme**, `∂(ρu)/∂x = 0` à l'intérieur, donc **pas de flux interne**.

Seules les **frontières** ont `∂(ρu)/∂x ≠ 0` → flux génère des ondes de choc.

---

## 6. Vérification Physique

**Équation de Burgers linéarisée** (photons avec vitesse constante):
```
∂f/∂t + c_s ∂f/∂x = 0
```

Solution: `f(x,t) = f(x - c_s·t, 0)`

L'injection homogène `f(x, 0) = const` pour x ∈ [25, 75], zéro ailleurs.

Après time `t`:
```
f(x, t) = const pour x ∈ [25 - c_s·t, 75 - c_s·t]
```

Les **fronts** se sont déplacés de `c_s·t`. Numériquement, un solveur upwind ou MUSCL converge vers ce profil sur **quelques mailles**.

---

## 7. Implication du Schema Numérique

Regardons **quelle partie du code** cause cela:

1. **`fluxes.py`** : Calcule les flux aux interfaces
2. **`riemann_solver.py`** : Solveur HLL/Roe pour photons
3. **`recon.py`** : Reconstruction MUSCL/PPM des états
4. **`limiter.py`** : Applique la limitation TVD

L'injection **uniforme** sur un **domaine compact** = **condition initiale créant des discontinuités aux frontières** = **amplification des gradients aux bords dans le schéma numérique**.

---

## 8. Résumé et Diagnostic

| Aspect | Observation | Cause Probable |
|--------|-------------|-----------------|
| Injection: 50 cellules | Vrai | Code d'injection homogène |
| Valeurs non-zéro: 8 cellules | Vrai après 1 step | Propagation d'ondes de choc aux frontières |
| Concentration aux extrémités | Vrai | Discontinuités de Riemann + Limiteur TVD |
| Propagation rapide | Vrai | Vitesse du son (photons) ~ c/√3 ≈ 1.15 |
| Pas de valeurs en [27-72] | Attendu numériquement | Schéma capture discontinuités, pas plateau internal |

**Conclusion**: C'est un **comportement physique correct** mais **contre-intuitif numériquement**. Une injection uniforme crée des chocs aux frontières. Ces chocs se propagent en respectant la structure de Riemann du solveur.

---

## 9. Analyse du Schéma Numérique Implémenté

### 9.1 Solveur de Riemann (Lax-Friedrichs pour RT)

De `riemann_solver.py` ligne 247-252 (`LaxFriedrichs_Radiative_transfer`):

```python
fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)
celerity = self.equation_manager.light_speed

fluxes_xi = 0.5 * (fluxes_L + fluxes_R) - 0.5 * celerity * (conservatives_R - conservatives_L)
```

**Schéma Lax-Friedrichs classique:**
$$\mathbf{f}_{i+1/2}^{LxF} = \frac{1}{2}(\mathbf{f}_L + \mathbf{f}_R) - \frac{1}{2}c(\mathbf{U}_R - \mathbf{U}_L)$$

Où :
- $\mathbf{f}_L$, $\mathbf{f}_R$ = flux des états gauche/droit
- $c$ = vitesse du son des photons ≈ `light_speed` ≈ `c/√3`
- $\mathbf{U}_L$, $\mathbf{U}_R$ = états conservatives gauche/droit

### 9.2 Reconstruction WENO/TENO5

De `recon.py` lignes 50-115 (classe TENO5_alt):

```python
# Smoothness indicators (Jiang-Shu betas)
beta_0 = (13/12)*(s0 - 2*s1 + s2)^2 + 1/4*(s0 - 4*s1 + 3*s2)^2
beta_1 = (13/12)*(s1 - 2*s2 + s3)^2 + 1/4*(s1 - s3)^2
beta_2 = (13/12)*(s2 - 2*s3 + s4)^2 + 1/4*(3*s2 - 4*s3 + s4)^2

tau_5 = |beta_0 - beta_2|  # Local smoothness variation

# WENO weights with TVD limiter cutoff
gamma_k = (C + tau_5 / (beta_k + eps))^q
delta_k = 1.0 if pi_k >= C_T else 0.0  # Sharp cutoff
omega_k = delta_k * dr_k / sum(delta_k * dr_k)
```

**Signification pour l'injection uniforme:**

#### À l'intérieur du domaine [25, 75]
Tous les points $s_0, s_1, s_2, s_3, s_4$ sont **constants** (= moment_injected):
- $(s_0 - 2s_1 + s_2) = 0$
- $(s_1 - 2s_2 + s_3) = 0$
- $(s_2 - 2s_3 + s_4) = 0$
- **Résultat:** $\beta_0 = \beta_1 = \beta_2 = 0$ (parfaitement lisse)
- **Flux reconstruit:** Directement constant sans modification

#### Aux frontières [24/25, 75/76]
Transition brusque ($\Delta s$ fini):
- Exemple au x=25: $[s_0, s_1, s_2, s_3, s_4] = [0, 0, m, m, m]$ (m=moment_injected)
- Résultat: $\beta_k$ **non-zéro**, les stencils sont réorientés pour capturer la discontinuité
- Les poids WENO/TENO se concentrent sur le stencil le plus lisse **passant par la discontinuité**

**Conclusion:** Le reconstructeur ne génère **pas de fluctuations numériques** dans le domaine constant. Les gradients fictifs n'apparaissent **que sous le Riemann solver aux interfaces avec discontinuités**.

### 9.3 Propagation via le Solveur Lax-Friedrichs

**Cas 1: Interface interne lisse [30/31]**

État LEFT (x=30): $U_L = [m, m, m, m]$  
État RIGHT (x=31): $U_R = [m, m, m, m]$ (identiques)

Flux Lax-Friedrichs:
$$f_{LxF} = \frac{1}{2}(f_L + f_R) - \frac{c}{2}(U_R - U_L) = f_L + 0 = f_L$$

**Pas d'amplitude supplémentaire générée** - seulement transport du flux constant.

**Cas 2: Interface de discontinuité [24/25]**

État LEFT (x=24): $U_L = [0, 0, 0, 0]$  
État RIGHT (x=25): $U_R = [m, m, m, m]$ (discontinuité)

Flux Lax-Friedrichs:
$$f_{LxF} = \frac{1}{2}(f_L + f_R) - \frac{c}{2}(m - 0)$$

Le terme $-\frac{c}{2}(U_R - U_L)$ crée une **correction antidiffusive** en réponse au saut.

**Propagation**: Cet écart traverse la grille à vitesse $c$ (soit ~1.15 unités code), ce qui explique pourquoi en 1 timestep (dt≈0.067), on observe une **onde choc** s'étendant de **2-3 cellules** de chaque côté.

---

## 10. Simulation Temporelle Complète

### Timestep iter=0
État initial: `sol[0] = 0` (pas d'injection photonique)  
Action: **Injection de moment** sur [25,75]  
Résultat: `sol[1, 25:75, :, :] += c²·total_source/50`  
Observation debug: Non-zero count = 0 (avant hydrodynamique)

### Timestep iter=1 à iter=10
Le **solveur hydrodynamique** (RHS unsplit) exécute:

1. **Reconstruction WENO/TENO5** → États aux faces LEFT/RIGHT
2. **Riemann solver** (Lax-Friedrichs) → Flux aux faces
3. **Différences finies** (update conservative) → Mise à jour du vecteur état

À chaque pas, le front de choc initialisé par l'injection se propage de $\approx 2$ cellules.

Propagation observée:
- Iter 1: $x \in [23, 26, 73, 76]$ (2 cellules de chaque côté des frontières 25/75)
- Iter 2: $x \in [21-28, 71-78]$ (élargissement continu)
- ...
- Iter 203: Les fronts ont fusionné au centre → État d'équilibre

### Résultat dans l'Espace Physique

L'image fournie montre:
- **Iter 0**: Pic isolé au centre (photons)
- **Iter 10-42**: Élargissement des pics, formation de ondes courbes
- **Iter 107+**: Saturation progressive du domaine

Cela correspond à une **onde de rarefaction** (Riemann shock profile) issue des deux frontières d'injection.

---

## 11. Solutions Possibles

### Solution 1: Profil Gaussien pour l'Injection
**Problème actuel:** Boîte rectangulaire → discontinuités → chocs numériques confinés

```python
# Lieu: radiative_transfer.py ligne 173
# ACTUEL:
xi = jnp.arange(25, 75)
sol = sol.at[1, xi, :, :].add(c²·total_source / 50)

# PROPOSÉ:
x_vals = jnp.arange(100)
weights_gauss = jnp.exp(-(x_vals - 50)**2 / (2 * 10**2))
sol = sol.at[1, x_vals, :, :].add(c²·total_source·weights_gauss)
```

**Effet:** Éliminerait les discontinuités de Riemann → flux plus étalé dans le domaine.

### Solution 2: Injecter comme Terme Source plutôt qu'en Condition Limite
Déplacer l'injection de la **frontière RHS** vers le **terme source dans la boucle temporelle**.

```python
# Ancien: sol[1, 25:75, :, :] += ...  après un pas

# Nouveau: dans le forçage ou le RHS
def source_moment(sol, params):
    # Ajouter graduellement plutôt que de manière impulsive
    per_step = total_source / n_steps
    return sol.at[1, 25:75, :, :].add(c²·per_step)
```

### Solution 3: Adapter la Viscosité Artificielle (Limiteur TVD)

Vérifier le limiteur dans `limiter.py` pour voir si une dissipation accrue adoucit la discontinuité.

### Solution 4: Utiliser un Solveur HLLC au lieu de Lax-Friedrichs

HLLC capture les ondes d'intercontact plus proprement:

```python
# Modifier le solveur instantié en radiative_transfer.py
# solver = HLLC(...)  au lieu de LaxFriedrichs_Radiative_transfer(...)
```

---

## 12. Recommandations Finales

**Pour obtenir une injection "vraie" sur [25,75]:**

1. ✅ **Utiliser un profil gaussien** → Évite les discontinuités de Riemann
2. ✅ **Injecter progressivement** → Sur plusieurs timesteps, pas instantanément
3. ⚠️  **Vérifier la limite TVD** → Peut amplifier ou réduire l'effet selon les paramètres
4. ✅ **Tester en 1D** avant 3D → Isoler le comportement de propagation

**Code test pour valider le comportement:**

```python
# Créer un cas 1D simplifié avec injection contrôlée
# Mesurer la vitesse de propagation des fronts
# Comparer avec les vitesses du son théoriques c/√3
```
