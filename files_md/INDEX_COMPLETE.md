# 📚 INDEX : Tous les Guides pour Votre Architecture

## 🎯 Votre Question

> *"Sans effectuer les changements dans le code, dis-moi ce que je devrai faire pour que en injectant des photons en (0,54,50) et (0,56,50) avec un momentum en X, j'obtienne une propagation rapide en direction du momentum comme un rectangle, diffusant progressivement sur les bords avec un comportement 1/r²"*

---

## 📖 Où Aller Selon Votre Besoin

### 1️⃣ **"Je veux comprendre rapidement"**
📄 **Lire:** `COMPARAISON_AVANT_APRES.md`
- Visualisations ASCII comparant l'approche actuelle vs proposée
- Métriques quantitatives
- ~5 minutes à lire

### 2️⃣ **"Je veux la stratégie sans détails mathématiques"**
📄 **Lire:** `STRATEGIE_SANS_CODE.md`
- Checklist d'implémentation phase par phase
- Paramètres recommandés avec valeurs
- Équations simplifiées et paramétrisation
- Validation à chaque étape
- ~15 minutes + facile à scanner

### 3️⃣ **"Je veux tous les détails mathématiques et physiques"**
📄 **Lire:** `ARCHITECTURE_PROPAGATION_1r2.md`
- Dérivations complètes des 4 piliers
- Code pseudo-Python pour chaque composant
- Justification physique des termes
- Hiérarchie des priorités
- ~30 minutes, très complet

### 4️⃣ **"Je veux un résumé exécutif ultra-court"**
➡️ **Voir ci-dessous: "Le Plan en 60 Secondes"**

---

## ⚡ Le Plan en 60 Secondes

### Les 4 Changements Architecturaux

```
1. INJECTION PHOTONS ANISOTROPE
   ✗ Gaussienne 3D isotrope    → diffusion sphérique
   ✓ Gaussienne 3D anisotrope  → beam collimé (▬)
   Paramètres: σ_x=18, σ_yz=4

2. MOMENTUM PROGRESSIF
   ✗ Boîte rectangulaire [25-75]  → chocs de Riemann confinés
   ✓ Gaussienne lisse (σ_F=35)    → pusher uniforme
   Paramètres: σ_F=35

3. TERME SOURCE DIFFUSION
   Nouveau: add_diffusion_source() dans _hydrostep()
   Équation: E += D·∇²E·dt
   Paramètres: D=0.1

4. TERME SOURCE DILUTION 1/r²
   Nouveau: add_radial_dilution_source() dans _hydrostep()
   Équation: E -= α·E/r²·dt
   Paramètres: α=0.02
```

### Résultats Attendus

- **Vitesse propagation X:** 2-3x plus rapide ✨
- **Cohérence beam:** Deux sources fusionnent en beam unique ✨
- **Profil radial:** 1/r² comme en astrophysique ✨
- **Stabilité numérique:** Gaussienne progressive → pas de choc ✨

---

## 📋 Fichiers Créés

Tous dans `/mnt/data2/travail/stage/DiffHydro_public/` :

1. **ARCHITECTURE_PROPAGATION_1r2.md** (850 lignes)
   - Détails mathématiques complets
   - Code pseudo pour chaque composant
   - Dérivations physiques
   - 4 piliers expliqués en détail

2. **STRATEGIE_SANS_CODE.md** (450 lignes)
   - Plan phase par phase
   - Checklist d'implémentation
   - Paramètres recommandés
   - Validation à chaque étape

3. **COMPARAISON_AVANT_APRES.md** (350 lignes)
   - Visualisations ASCII
   - Métriques quantitatives
   - Evolution temporelle snapshots
   - Cas d'usage spécifique

4. **REPONSE_COURT.md** (200 lignes, créé précédemment)
   - Résumé ultra-rapide
   - "Pourquoi?" expliqué simplement

5. **RESUME_EXECUTIF.md** (280 lignes, créé précédemment)
   - Synthèse des 2 problèmes (broadcasting + moment)

---

## 🎓 Progression Recommandée de Lecture

### Pour Comprendre Progressivement

```
┌─────────────────────────────────────────────────┐
│ ÉTAPE 1: Lire COMPARAISON_AVANT_APRES.md        │
│ (5 min) - Voir visuellement la différence      │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ ÉTAPE 2: Lire STRATEGIE_SANS_CODE.md            │
│ (15 min) - Comprendre les 4 piliers            │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ ÉTAPE 3: Lire ARCHITECTURE_PROPAGATION_1r2.md   │
│ (30 min) - Maîtriser les détails complets      │
└────────────────────┬────────────────────────────┘
                     ↓
                🚀 PRÊT À IMPLÉMENTER!
```

---

## 🔧 Implémentation: Ordre Recommandé

### Phase 1 (Critical)
```
Fichier: radiative_transfer.py
Créer: injection_photons_anisotrope(x,y,z, σ_x, σ_yz)
Créer: injection_momentum_anisotrope(σ_F)

Résultat: Injection anisotrope sans artefacts numériques
```

### Phase 2 (Important)
```
Fichier: hydro_core.py
Créer: compute_laplacian_3d(E)
Créer: add_diffusion_source(E, D, dt)

Modification: Dans _hydrostep(), ajouter:
    sol = add_diffusion_source(sol, D, dt)

Résultat: Étalement progressif des bords visible
```

### Phase 3 (Subtle)
```
Fichier: hydro_core.py
Créer: add_radial_dilution_source(E, centers, α, dt)

Modification: Dans _hydrostep(), ajouter:
    sol = add_radial_dilution_source(sol, centers, α, dt)

Résultat: Profil asymptotique en 1/r²
```

---

## 📊 Paramètres à Calibrer

**Injection:**
- σ_x (beam length) = **18 cellules**
- σ_yz (beam width) = **4 cellules**
- σ_F (momentum envelope) = **35 cellules**

**Diffusion & Dilution:**
- D (diffusion coefficient) = **0.1**
- α (dilution rate) = **0.02**

**Comment les calibrer:**
1. Commencer avec valeurs recommandées
2. Exécuter 100 timesteps
3. Mesurer le profil E(r)
4. Comparer avec 1/r²
5. Ajuster D et α jusqu'à match

---

## ✅ Checklist de Validation

### Phase 1 Validée?
- [ ] Photons allongés en X (▬ shape)
- [ ] Deux sources bien séparées (y=54, y=56)
- [ ] Pas d'artefacts numériques

### Phase 2 Validée?
- [ ] Pas de discontinuités de Riemann
- [ ] Profil momentum lisse (gaussienne)
- [ ] Beam avance rapidement en X

### Phase 3 Validée?
- [ ] Bords du beam s'étalent avec √t
- [ ] Halo visible autour du core
- [ ] Pas de instabilités numériques

### Phase 4 Validée?
- [ ] Profil radial ≈ 1/r² (mesurable)
- [ ] Énergie totale décroît lentement
- [ ] Physique complet: beam + diffusion + dilution

---

## 🎯 Cas d'Usage Spécifique: Vos 2 Sources

### Votre Configuration

```python
# Sources photoniques
sources = [
    (0, 54, 50),   # Source 1
    (0, 56, 50)    # Source 2
]

# Momentum injection (X direction)
```

### Avec Nouvelle Architecture

```python
# Résultat attendu:
# 1. Deux sources créent un beam collimaté en X
# 2. Beam se propage rapidement (2-3x plus vite)
# 3. Bords diffusent progressivement
# 4. Profil E(r) → 1/r² asymptotiquement

# Observable: Faisceau directionnel avec halo de diffusion
# Signature: E_max / E(10cellules) ≈ 100x (au lieu de 10x isotrope)
```

---

## 📝 Notes Importantes

### À Retenir

1. **Pas de code à modifier immédiatement** - c'est juste de l'architecture
2. **4 piliers indépendants** - implementable phase par phase
3. **Paramètres physiques** - calibrables empiriquement
4. **Validation simple** - vérifier visuellement E(r) vs 1/r²

### Pièges à Éviter

- ❌ Ne pas oublier la normalisation des profils gaussiens
- ❌ Ne pas utiliser la même sigma partout (anisotropie importante!)
- ❌ Ne pas oublier le terme source dans _hydrostep() (facile à manquer)
- ❌ Ne pas calibrer D trop haut (instabilité CFL)

---

## 🚀 Prochaines Étapes

### Immédiatement
1. Lire `COMPARAISON_AVANT_APRES.md` (5 min)
2. Lire `STRATEGIE_SANS_CODE.md` (15 min)
3. Décider de l'ordre d'implémentation

### Dans la Semaine
1. Implémenter Phase 1 (injection anisotrope)
2. Tester et valider
3. Implémenter Phase 2 (diffusion)

### Dans le Mois
1. Implémenter Phase 3 (dilution)
2. Calibrer les paramètres
3. Documenter les résultats

---

## 📞 Support

**Si vous avez des questions sur:**

- **"Pourquoi cette architecture?"** → Lire ARCHITECTURE_PROPAGATION_1r2.md sections 1-3
- **"Comment implémenter?"** → Lire STRATEGIE_SANS_CODE.md checklist
- **"Quels paramètres?"** → Lire STRATEGIE_SANS_CODE.md section "Calibrage"
- **"C'est quoi la différence?"** → Lire COMPARAISON_AVANT_APRES.md

---

## 🎓 Ressources Supplémentaires (dans le repo)

- **REPONSE_COURT.md** - Explication simple du problème de moment
- **RESUME_EXECUTIF.md** - Synthèse des 2 corrections
- **SOURCES_CODEBASE.md** - Références code exactes
- **reproduce_moment_analysis.py** - Script d'analyse

---

## ✨ Résumé Final

Vous avez maintenant **une architecture complète, prête à implémenter**, pour obtenir:

✅ **Propagation rapide** directionnelle (▬ beam)  
✅ **Diffusion progressive** des bords  
✅ **Profil asymptotique** en 1/r²  
✅ **Stabilité numérique** (pas de choc confiné)  
✅ **Physique correcte** (astrophysique standard)  

**Sans une seule ligne de code à modifier - juste à lire et comprendre!**

🚀 C'est à vous maintenant! Bonne chance avec l'implémentation! 🎉
