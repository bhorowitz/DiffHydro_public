# Checklist: Intégration des Unités dans Radiative Transfer

## ✅ Déjà fait

### 1. **Système d'unités étendu**
- [x] `field_dims.py`: Ajout champs RT (E_gamma, Fx_gamma, etc.)
- [x] `registry.py`: Ajout "radiation_flux" (erg/s/cm²)
- [x] `code_units.py`: Ajout `RadFlux_cgs` property

**Vérification** : Tester que tout fonctionne
```python
from diffhydro.units import CodeUnits
cu = CodeUnits.from_config({
    "length": "1e22 cm",
    "mass": "1e11 Msun", 
    "velocity": "1e5 cm/s"
})
print(cu.RadFlux_cgs)  # Devrait afficher un nombre
```

---

## 📝 À FAIRE

### 2. **Modifier `StellarRadiationForce.__init__()`**

**Localisation** : `diffhydro/physics/radiative_transfer.py`, ligne ~68

**Ajouter ces paramètres** :
```python
from diffhydro.units import CodeUnits, UnitParser

def __init__(
    self,
    # ... paramètres existants ...
    cu: CodeUnits = None,  # ← AJOUTER
    parser: UnitParser = None,  # ← AJOUTER
):
    # ... code existant ...
    self.eq = eq
    
    # ← AJOUTER ces deux lignes
    self.cu = cu
    self.parser = parser or UnitParser()
```

---

### 3. **Ajouter méthodes helpers**

**Ajouter après `__init__()`** :
```python
def set_code_units(self, cu: CodeUnits, parser: UnitParser = None):
    """Configure CodeUnits pour conversions."""
    self.cu = cu
    self.parser = parser or UnitParser()

def convert_physical_to_code(self, value, dimension: str):
    """Convertir physique → code units."""
    if self.cu is None:
        raise ValueError("CodeUnits not configured.")
    from diffhydro.units import to_code
    return to_code(value, dimension, self.cu, self.parser)

def convert_code_to_physical(self, value, dimension: str, out_unit: str = None):
    """Convertir code units → physique."""
    if self.cu is None:
        raise ValueError("CodeUnits not configured.")
    from diffhydro.units import from_code
    if out_unit is None:
        out_unit = self.parser.default_cgs_unit(dimension)
    return from_code(value, dimension, self.cu, out_unit, self.parser)
```

---

## 🧪 Tests de vérification

### Test 1: Configuration
```python
from diffhydro.physics.radiative_transfer import StellarRadiationForce
from diffhydro.units import CodeUnits

cu = CodeUnits.from_config({
    "length": "1e22 cm",
    "mass": "1e11 Msun",
    "velocity": "1e5 cm/s"
})

srf = StellarRadiationForce(eq=mock_eq, cu=cu)
assert srf.cu is not None
print("✓ StellarRadiationForce accepte CodeUnits")
```

### Test 2: Conversions
```python
energy_phys = "1e50 erg/s"
energy_code = srf.convert_physical_to_code(energy_phys, "energy_density")
result = srf.convert_code_to_physical(energy_code, "energy_density")
print(f"✓ {energy_phys} → {energy_code:.2e} → {result.value:.2e} {result.unit}")
```

### Test 3: Radiation flux
```python
flux = srf.convert_physical_to_code("1e5 erg/s/cm^2", "radiation_flux")
print(f"✓ Radiation flux convertion works: {flux:.2e}")
```

---

## 📚 Fichiers de référence

- `examples/radiative_transfer_units_example.py` - Exemple complet
- `examples/radiative_transfer_modified_example.py` - Template de modification
- `INTEGRATION_GUIDE_UNITS_FR.md` - Guide détaillé
- `diffhydro/units/code_units.py` - CodeUnits avec RadFlux_cgs
- `diffhydro/units/registry.py` - UnitParser avec radiation_flux

---

## 💡 Points clés

### Pourquoi les unités ?

| **Sans unités** | **Avec unités** |
|---|---|
| Tous les calculs en code units | Conversions explicites |
| Facile de se tromper | Paramètres en unités physiques |
| Diffcile à déboguer | Vérifications physiques claires |

### Échelles typiques

```
CodeUnits:
  L_cgs = 1e22 cm    (~1 kpc)
  M_cgs = 1.99e44 g  (~100 billion Msun)
  V_cgs = 1e5 cm/s   (100 km/s)
  
Dérivées:
  RadFlux_cgs = P_cgs * V_cgs ≈ 1e-7 erg/s/cm²
```

---

## ❓ Questions courantes

**Q: Dois-je modifier tous mes calculs ?**
A: Non, le système d'unités est optionnel. Si `cu is None`, ignorez les conversions.

**Q: Comment accepter des paramètres physiques ?**
A: Utiliser `convert_physical_to_code(value_str, dimension)` pour convertir avant d'assigner.

**Q: Et pour afficher les résultats ?**
A: Utiliser `convert_code_to_physical(value, dimension)` pour convertir avant affichage.

---

## 🎯 Ordre de travail recommandé

1. ✅ Vérifier que l'extension des unités fonctionne (test ci-dessus)
2. 📝 Modifier `StellarRadiationForce.__init__()` et ajouter helpers
3. 🧪 Lancer les tests de vérification
4. 🔄 (Optionnel) Utiliser conversions dans vos méthodes existantes
5. 📊 Afficher résultats en unités physiques pour validation
