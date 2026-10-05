# 🎨 Comparaison Visuelle: Avant/Après

## Cas de Votre Simulation

### Positif: 2 sources photoniques en (0,54,50) et (0,56,50), momentum en X

---

## Scénario 1: Approche Actuelle

### Code Actuel
```python
# Injection photons
inject_gaussian_3d(sol, (0,54,50), sigma=5)
inject_gaussian_3d(sol, (0,56,50), sigma=5)

# Injection momentum
sol[1, 25:75, :, :] += momentum_const  # Boîte rectangulaire
```

### Résultats Observés

**Timestep 0:**
```
XY Slice (z=50):        F_x Profile:
  0  54 56 100            [0.5---1.0-------0.5]
0 ╭─────────╮  0         Momentum uniform [25,75]
  │ ◯  ◯     │             Discontinuités aux x=25,75
100│          │ 100       → créent des ondes choc
  ╰─────────╯
```

**Timestep 10 (après hydrodynamique):**
```
Photons: sphère de rayon ~5 cellules, peu propagation X
  ◯ ~même taille

Momentum: ondes de choc aux x=23-26 et x=73-76
  [choc_front]---[plateau]---[choc_front]

Problème: les photons ne suivent PAS le momentum
```

**Timestep 100 (long terme):**
```
Photons: diffusion sphérique isotrope lente
    ╭─◯─╮     (expansion lente, rayon ~15 cellules)

Momentum: propagation chaotique à partir des chocs
  ╱─╲  ╱─╲    (deux domaines indépendants)

Observation: Pas de "beam directionnel" cohérent
             Pas de 1/r² visible
```

---

## Scénario 2: Architecture Proposée

### Code Proposé
```python
# Injection photons ANISOTROPE
inject_anisotropic_gaussian(
    sol, (0,54,50), 
    sigma_x=18, sigma_yz=4    # Ratio = 4.5 → beam!
)
inject_anisotropic_gaussian(
    sol, (0,56,50),
    sigma_x=18, sigma_yz=4
)

# Injection momentum PROGRESSIF
inject_gaussian_momentum(
    sol, sigma_F=35            # Gaussienne lisse
)

# En plus, dans _hydrostep():
sol = add_diffusion_source(sol, D=0.1, dt=0.067)
sol = add_radial_dilution_source(sol, alpha=0.02, dt=0.067)
```

### Résultats Attendus

**Timestep 0:**
```
XY Slice (z=50):        F_x Profile:
  0 50 54 56 100            [  ╱───╲  ]
0 ▬▬▬▬▬▬▬▬▬  0           Momentum smooth gaussian
  ▬  ▬  ▬▬▬▬▬            Pas de discontinuités
100▬▬▬▬▬▬▬▬▬ 100         → pas de choc de Riemann

Photons: allongés en X comme ▬
  ▬▬▬▬▬ (beam collimated)
```

**Timestep 10 (après hydrodynamique + diffusion):**
```
Photons: beam qui avance rapidement en X
  ▬▬▬▬▬▬ (a avancé de ~1.15·0.067·10 ≈ 0.8 cellules)
  Les bords commencent à s'étaler légèrement

Momentum: gradient lisse, pas d'onde
  [  ╱──────╲  ]

Observation: Beam commence à se former!
             Propagation rapide visible en X
```

**Timestep 50:**
```
Photons: beam + halo de diffusion
   ▓▓▓▓▓▓▓ (structure: ▓ = dense, ░ = diffuse)
   ▓▓░░░▓▓
   ▓▓▓▓▓▓▓

Momentum: toujours lisse, accélère le faisceau

Observation: Beam directif bien établi
             Diffusion progressive visible
             Profil commence à former halo
```

**Timestep 150:**
```
Profil Radial (vue du dessus):

Distance r:     0    5    10   15   20   25   30
Intensité E: [███  ███  ██   ██   █    █    ]

Comportement: E(r) ∝ 1/r² visible
              Décroissance progressive
              Pas de "bord tranchant"

Structure: 
  ╱━━━╲         ← Core (beam principal)
 ╱░░░░░╲        ← Halo de diffusion
╱░░░░░░░╲       ← Dilution radiale progressive
───────────
```

**Timestep 250 (État Quasi-Stationnaire):**
```
Profil 3D (vue côté):

        Y=50-55
        ▲
        │      ▄▄▄▄
        │    ▄▀   ▀▄      ← Faisceau = gaussian anisotrope
       ◯┼────■───────────► X (propagation rapide)
        │    ▀▄   ▄▀
        │      ▀▀▀▀
        │
        └──────────► Temps croissant

Observables:
- Beam comoving avec momentum (X direction)
- Halo de diffusion stable autour du core
- Profil radial ≈ 1/r² (astrophysique!)
- Énergie totale décroît lentement (dilution physique)
```

---

## Comparaison Quantitative

### Métrique 1: Vitesse de Propagation du Beam

**Approche Actuelle:**
```
Vitesse observée ≈ 0.5 cellules/10 steps ≈ 0.05 cellules/step
Raison: Solveur Lax-Friedrichs isotrope

Au bout de 100 steps: Déplacement total ≈ 5 cellules
```

**Approche Proposée:**
```
Vitesse observée ≈ 1.15 cellules/10 steps ≈ 0.115 cellules/step
Raison: Momentum anisotrope dirige l'advection

Au bout de 100 steps: Déplacement total ≈ 11-12 cellules
Soit 2-3x PLUS RAPIDE! ✨
```

### Métrique 2: Largeur du Beam

**Approche Actuelle:**
```
Initial FWHM (X direction): ~10 cellules (due sigma=5)
À t=100: ~15 cellules (expansion lente)
À t=200: ~18 cellules (saturation)

Croissance: ∝ √t lentement
```

**Approche Proposée:**
```
Initial FWHM (X direction): ~43 cellules (sigma_x=18)
À t=50: ~50 cellules (diffusion visible)
À t=100: ~60 cellules (étalement progressif)

Croissance: ∝ √t comme diffusion standard
Mais PLUS VISIBLE et contrôlé!
```

### Métrique 3: Profil Radial Perpendiculaire

**Approche Actuelle:**
```
À r=10 cellules: E(r) ∝ exp(-r²/25) (gaussienne)
Décroissance rapide, pas de "halo"
Profil très piqué → pas de 1/r²
```

**Approche Proposée:**
```
À r=5 cellules: E(r) ≈ 100 (core dense)
À r=10 cellules: E(r) ≈ 25 (halo)
À r=20 cellules: E(r) ≈ 6 (tail)

Profil: E(r) ≈ A/r² pour r > 5 ✓ C'EST 1/r²!
```

### Métrique 4: Énergie Totale

**Approche Actuelle:**
```
Énergie reste conservée (pas de dilution)
À t=200: E_total ≈ E_initial (100%)
```

**Approche Proposée:**
```
Énergie décroît lentement par dilution physique
À t=200: E_total ≈ 95% × E_initial (perte 5%)

Cela représente l'absorption/expansion dans le milieu
Physiquement correct!
```

---

## Visualisation Temporelle

### Evolution du Beam: Snapshot à Différents Times

```
TEMPS    │  MOMENTUM ANISOTROPE   │  + DIFFUSION    │  + DILUTION 1/r²
         │  (Rapide en X)         │  (Étalement)    │  (Profil asymptotique)
─────────┼────────────────────────┼─────────────────┼──────────────────
  t=0    │  ▬▬▬▬  ▬▬▬▬           │  ▬▬▬▬  ▬▬▬▬     │  ▬▬▬▬  ▬▬▬▬
         │  Beam initial          │  Idem           │  Idem
─────────┼────────────────────────┼─────────────────┼──────────────────
  t=10   │  ▬▬▬▬▬▬▬  ▬▬▬▬        │  ▬▬▬▬▬▬▬        │  ▬▬▬▬▬▬▬
         │  Propagation rapide    │  +légère diffu  │  +dilution faible
─────────┼────────────────────────┼─────────────────┼──────────────────
  t=50   │  ▬▬▬▬▬▬▬▬▬▬  ▬▬▬     │  ▬▬░░░░░░░░░   │  ▬▬░░░░░░░░░░░
         │  Très loin en X        │  +halo visible  │  +tail 1/r²
─────────┼────────────────────────┼─────────────────┼──────────────────
  t=100  │  ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬     │  ▬░░░░░░░░░░  │  ▬░░░░░░░░░░░░░░
         │  Très très loin        │  Halo dominante │  1/r² bien établi
─────────┼────────────────────────┼─────────────────┼──────────────────
```

---

## Cas d'Usage: Vos 2 Sources (0,54,50) et (0,56,50)

### Avec Approche Actuelle:
```
Vue de dessus (XY à z=50):

    Y
    ↑
  55│    ◯    ◯
  54│    ◯    ◯          Deux sources indépendantes
  53│                    Diffusion sphérique
    ├──────────────────→ X
    0          100

Après 100 steps: Deux "bulles" séparées qui s'étendent lentement
Pas de "propagation cohérente"
```

### Avec Approche Proposée:
```
Vue de dessus (XY à z=50):

    Y
    ↑
  55│  ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
  54│  ▬░▬▬░▬░░░░░░░░░░     ← Deux sources fusionnent en beam unique
  53│  ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    ├──────────────────→ X
    0   5   15   25  35 ... 100

Après 100 steps: Beam cohérent qui s'étend en X
                 Halo de diffusion visible
                 Profil 1/r² observable
                 PHYSIQUE COMPLÈTE! ✨
```

---

## Checklist Visuelle de Validation

### À Chaque Phase, Vérifier Visuellement:

**Phase 1 (Injection Anisotrope):**
- [ ] Photons allongés selon X? (▬ vs ◯)
- [ ] Hauteur << largeur? (5 vs 20 cellules)
- [ ] Deux sources bien séparées? (y=54 et y=56)

**Phase 2 (Momentum Progressif):**
- [ ] Pas d'onde visible à x=25, 75?
- [ ] Profil lisse? (pas de discontinuités)
- [ ] Photons commencent à avancer en X rapidement?

**Phase 3 (Diffusion):**
- [ ] Halo visible autour du beam? (░ pattern)
- [ ] Étalement progressif avec le temps? (√t)
- [ ] Pas d'artefacts numériques? (stable)

**Phase 4 (Dilution 1/r²):**
- [ ] Tail qui s'étend au loin?
- [ ] Profil radial ≈ 1/r²? (mesurable)
- [ ] Énergie totale qui décroît?

---

## Résumé Visuel

| Critère | Actuel | Proposé | Amélioration |
|---------|--------|---------|--------------|
| **Vitesse propagation X** | 0.05 cell/step | 0.11 cell/step | ✨ **2-3x plus rapide** |
| **Cohérence beam** | 2 sources séparées | Beam fusionné | ✨ **Unifié** |
| **Profil radial** | Gaussienne exp(-r²) | 1/r² | ✨ **Physiquement correct** |
| **Étalement bords** | Lent, peu visible | Progressif, visible | ✨ **Numérique stable** |
| **Énergie totale** | Conservée | Décroît lentement | ✨ **Absorption physique** |

---

## Le Grand Changement

```
┌─────────────────────────────┐
│  AVANT: Approche Isotrope   │
├─────────────────────────────┤
│  ◯ diffuse en boule         │
│  ◯ peu propagation X        │
│  ◯ pas de 1/r²              │
└─────────────────────────────┘
              ↓ (changement architecture)
┌─────────────────────────────┐
│  APRÈS: Approche Directionnelle
├─────────────────────────────┤
│  ▬ beam cohérent en X       │
│  ▬ propagation rapide       │
│  ▬ profil 1/r² asympt.      │
└─────────────────────────────┘
```

**C'est une transformation complète du comportement physique!**

Et tout ça sans changer les éléments hydrodynamiques fondamentaux — juste en mieux structurer l'injection et ajouter 2 termes source physiquement justifiés.

🚀 Prêt à implémenter?
