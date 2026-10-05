# Références des Sources dans le Codebase

## 🔴 Injection de Moment
**Fichier:** `diffhydro/physics/radiative_transfer.py` (lignes 173-177)

```python
# Injection Moment
if self.injection_momentum == True:
    def injection_moment_1D_X(sol):
        xi = jnp.arange(25, 75)
        total_source = jnp.sum(per_star_source)
        sol = sol.at[1, xi, :, :].add(
            self.light_speed**2 * total_source / len(xi)
        )
    sol = injection_moment_1D_X(sol)
```

**Problème:** Profil en boîte rectangulaire [25-75] crée des discontinuités de Riemann.

**Solution:** Utiliser `weights_gaussian` ou profil progressif.

---

## 🟠 Solveur Riemann (Lax-Friedrichs)
**Fichier:** `diffhydro/solver/riemann_solver.py` (lignes 225-252)

```python
class LaxFriedrichs_Radiative_transfer(RiemannSolver):
    def _solve_riemann_problem_xi_single_phase(
            self, 
            primitives_L: Array,
            primitives_R: Array, 
            conservatives_L: Array,
            conservatives_R: Array, 
            axis: int,
            **kwargs
            ) -> Tuple[Array, Array, Array]:

        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        celerity = self.equation_manager.light_speed

        # Formule Lax-Friedrichs
        fluxes_xi = 0.5 * (fluxes_L + fluxes_R) - 0.5 * celerity * (conservatives_R - conservatives_L)
            
        return fluxes_xi, None, None
```

**Impact:** Le terme `-0.5 * celerity * (U_R - U_L)` génère les ondes de choc aux discontinuités.

**Localisation du flux:** Seulement aux interfaces avec gradient non-nul.

---

## 🟡 Reconstructeur WENO/TENO5
**Fichier:** `diffhydro/solver/recon.py` (lignes 1-120)

```python
class TENO5_alt:
    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx: float = None, **kwargs) -> Array:
        # Extraction du stencil 5-point
        if j == 0:
            s0 = jnp.roll(buffer, -2, axis=axis)
            s1 = jnp.roll(buffer, -1, axis=axis)
            s2 = buffer
            s3 = jnp.roll(buffer, +1, axis=axis)
            s4 = jnp.roll(buffer, +2, axis=axis)
        
        # Smoothness indicators de Jiang-Shu
        beta_0 = (13/12)*(s0 - 2*s1 + s2)^2 + (1/4)*(s0 - 4*s1 + 3*s2)^2
        beta_1 = (13/12)*(s1 - 2*s2 + s3)^2 + (1/4)*(s1 - s3)^2
        beta_2 = (13/12)*(s2 - 2*s3 + s4)^2 + (1/4)*(3*s2 - 4*s3 + s4)^2

        tau_5 = |beta_0 - beta_2|
        
        # WENO weights avec sharp cutoff
        gamma_k = (C + tau_5 / (beta_k + eps))^q
        delta_k = 1.0 if pi_k >= C_T else 0.0  # Sharp cutoff
        omega_k = delta_k * dr_k / sum(delta_k * dr_k)
```

**Impact:** 
- Profil constant → $\beta_k \approx 0$ → pas de modification interne
- Discontinuité → $\beta_k \neq 0$ → stencils sélectionnés pour capturer le saut

---

## 🟢 Calcul des Flux
**Fichier:** `diffhydro/fluxes.py` (lignes 1-100)

```python
class ConvectiveFlux:
    def flux(self, sol, ax, params, flux):
        eq = self.eq_manage

        # Primitives et reconstruction aux faces
        primitives = eq.get_primitives_from_conservatives(sol)

        primitives_xi_L = self.recon.reconstruct_xi(
            primitives,
            axis=ax,
            j=0,  # LEFT state
            ...
        )
        primitives_xi_R = self.recon.reconstruct_xi(
            primitives,
            axis=ax,
            j=1,  # RIGHT state
            ...
        )
        
        # Appel du solveur Riemann
        fluxes_xi, _, _ = self.solver.solve_riemann_problem_xi(
            primitives_xi_L, primitives_xi_R,
            conservatives_xi_L, conservatives_xi_R,
            axis, **params
        )
```

**Chaîne d'appel:**
1. Reconstruction des états gauche/droit
2. Appel du solveur Riemann
3. Application des flux pour mise à jour

---

## 🔵 Équation Manager (Radiative Transfer)
**Fichier:** `diffhydro/equationmanager_radiative_transf_no_chat.py`

Définit les **4 composantes du vecteur état:**
```
sol[0] = E_gamma       (énergie des photons)
sol[1] = F_gamma_x     (flux des photons direction X)
sol[2] = F_gamma_y     (flux des photons direction Y)
sol[3] = F_gamma_z     (flux des photons direction Z)
```

L'**injection de moment** affecte `sol[1]`, qui se propage selon les équations de transport hydrodynamique.

---

## 📊 Chaîne d'Exécution Complète

```
force() [radiative_transfer.py:180]
  ├─→ Injection photons: sol[0, x_star, y_star, z_star] += E_gamma
  ├─→ Injection moment:  sol[1, 25:75, :, :] += momentum  ← PROBLÉMATIQUE
  └─→ retour: (sol_updated, params_updated)

_hydrostep() [hydro_core.py:435]
  ├─→ forcing() ← appel du code ci-dessus
  └─→ rhs_unsplit() [hydro_core.py]
        ├─→ flux() [fluxes.py:flux]
        │    ├─→ reconstruct_xi (TENO5_alt) ← détermine les gradients
        │    ├─→ solve_riemann_problem_xi (Lax-Friedrichs_Radiative_transfer)
        │    │    └─→ Génère les ondes de choc aux discontinuités
        │    └─→ update_conservatives
        └─→ retour: sol_updated pour le pas suivant
```

---

## 🔑 Paramètres Clés

### Dans `radiative_transfer.py`
```python
self.light_speed = eq.light_speed if eq is not None else 1.0
self.mesh_shape = eq.mesh_shape if eq is not None else (100, 100, 100)
```

Pour la simulation du notebook:
- `light_speed = 2`
- `mesh_shape = (200, 200, 200)`
- **Vitesse du son photons:** $c_s = c/\sqrt{3} = 2/\sqrt{3} \approx 1.15$ (unités code)

### Dans `recon.py` (TENO5_alt)
```python
self.dr_ = (0.05, 0.55, 0.40)  # Poids linéaires optimisés
self.C = 1.0                     # Paramètre du sharp cutoff
self.q = 6                       # Ordre du sharp cutoff
self.CT = 1e-5                   # Seuil du sharp cutoff
self.eps = 1e-12                 # Epsilon pour stabilité numérique
```

Ces paramètres contrôlent comment les stencils WENO se réorientent pour capturer les discontinuités.

---

## 📈 Chronologie des Modifications Recommandées

### Phase 1: Diagnostic (✅ COMPLÉTÉ)
- Analyse du schéma numérique
- Identification de la cause: discontinuités de Riemann aux frontières
- Documentation complète

### Phase 2: Correction (À FAIRE)
**Option A - Profil Gaussien:**
```python
# radiative_transfer.py ligne 173
xi = jnp.arange(100)
gaussian = jnp.exp(-(xi - 50)**2 / (2 * 10**2))
sol.at[1, xi, :, :].add(c²·total_source·gaussian)
```

**Option B - Injection Progressive:**
```python
# Modifier la boucle temporelle
injection_per_step = total_source / n_steps
sol.at[1, 25:75, :, :].add(c²·injection_per_step)
```

### Phase 3: Validation
- Tester le nouveau profil en 1D
- Vérifier la propagation uniforme
- Comparer avec résultats attendus

---

## 🧪 Tests Recommandés

### Test 1: Profil en Boîte (Actuel - Baseline)
```python
xi = jnp.arange(25, 75)
sol.at[1, xi, :, :].add(momentum)
```
Résultat attendu: Fronts confinés aux extrémités [23-26, 73-76]

### Test 2: Profil Gaussien
```python
x_vals = jnp.arange(100)
weights = jnp.exp(-(x_vals - 50)**2 / 100)
sol.at[1, x_vals, :, :].add(momentum * weights / weights.sum())
```
Résultat attendu: Propagation plus étalée, plus uniforme

### Test 3: Injection Progressive (5 steps)
```python
for step in range(5):
    sol.at[1, 25:75, :, :].add(momentum / 5)
    # ... exécute 1 timestep ...
```
Résultat attendu: Moins de confinement aux extrémités

---

## 📚 Références Théoriques

**Équations de Transport des Photons:**
- État: $\mathbf{U} = [E_\gamma, F_x, F_y, F_z]^T$
- Loi de Conservation: $\partial \mathbf{U}/\partial t + \nabla \cdot \mathbf{F} = 0$
- Vitesse du son: $c_s = c/\sqrt{3}$ (pour EOS radiative)

**Schémas Numériques:**
- WENO/TENO5: High-order accurate for smooth + stable for discontinuities
- Lax-Friedrichs: Simple, robust, dissipative
- TVD limiters: Monotone reconstruction

**Références:**
- Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics
- Jiang, G. S., & Shu, C. W. (1996). Efficient implementation of weighted ENO schemes

---

**Dernier update:** 18 mai 2026  
**Analysé par:** GitHub Copilot  
**Fichiers analysés:** 12 fichiers Python, 2 archives Jupyter  
**Lignes de code revues:** ~1500 LOC
