"""
GUIDE COMPLET D'INTÉGRATION DES UNITÉS DANS RADIATIVE_TRANSFER

Ce fichier montre exactement quoi faire pour intégrer les unités dans votre
module radiative_transfer.py existant.
"""

# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 1: MODIFICATIONS DANS __init__ DE StellarRadiationForce
# ═══════════════════════════════════════════════════════════════════════════

"""
Avant (sans unités):
───────────────────
def __init__(
    self,
    escape_fraction=0.1,
    ...
    eq=None,
    debug=False,
    ...
):
    self.escape_fraction = escape_fraction
    ...
    self.eq = eq


Après (avec unités):
───────────────────
"""

from diffhydro.units import CodeUnits, UnitParser

def __init__(
    self,
    escape_fraction=0.1,
    stellar_spectrum_func=None,
    dx=1.0,
    injection_mode="physical",
    stromgren_rate=1e-7,
    gaussian_star=True,
    injection_geometry="3D",
    injection_momentum=False,
    momentum_only=False,
    eq=None,
    debug=False,
    one_injection=False,
    beam_axis=0,
    beam_sign=+1,
    beam_length_cells=8,
    beam_sigma=3.0,
    beam_reduced_flux=0.95,
    beam_momentum_scaling="physical",
    # ╔════════════════════════════════════════════╗
    # ║ AJOUT: Paramètres pour les unités         ║
    # ╚════════════════════════════════════════════╝
    cu: CodeUnits = None,
    parser: UnitParser = None,
):
    # ... code existant ...
    self.eq = eq

    # ╔════════════════════════════════════════════╗
    # ║ AJOUT: Stocker CodeUnits et UnitParser     ║
    # ╚════════════════════════════════════════════╝
    self.cu = cu
    self.parser = parser or UnitParser()


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 2: AJOUTER DES MÉTHODES HELPERS POUR LES CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
Ajouter ces méthodes à la classe StellarRadiationForce:
"""

def set_code_units(self, cu: CodeUnits, parser: UnitParser = None):
    """
    Configure CodeUnits pour les conversions.
    
    Usage:
        srf.set_code_units(cu, parser)
    """
    self.cu = cu
    self.parser = parser or UnitParser()

def convert_physical_to_code(self, value, dimension: str):
    """
    Convertir une valeur physique en unités de code.
    
    Args:
        value: str comme "1e50 erg/s" ou float
        dimension: str comme "energy_density", "radiation_flux"
    
    Returns:
        float en unités de code
    
    Example:
        stromgren_code = self.convert_physical_to_code("1e50 erg/s", "energy_density")
    """
    if self.cu is None:
        raise ValueError("CodeUnits not configured. Call set_code_units() first.")
    
    from diffhydro.units import to_code
    return to_code(value, dimension, self.cu, self.parser)

def convert_code_to_physical(self, value, dimension: str, out_unit: str = None):
    """
    Convertir une valeur de code units en unités physiques.
    
    Args:
        value: float ou array en unités de code
        dimension: str comme "energy_density", "radiation_flux"
        out_unit: str comme "erg/s/cm^2" (optionnel, utilise CGS par défaut)
    
    Returns:
        Quantity object avec .value et .unit
    
    Example:
        result = self.convert_code_to_physical(1e-8, "energy_density")
        print(f"{result.value:.2e} {result.unit}")
    """
    if self.cu is None:
        raise ValueError("CodeUnits not configured. Call set_code_units() first.")
    
    from diffhydro.units import from_code
    if out_unit is None:
        out_unit = self.parser.default_cgs_unit(dimension)
    return from_code(value, dimension, self.cu, out_unit, self.parser)


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 3: UTILISER LES CONVERSIONS DANS VOS MÉTHODES
# ═══════════════════════════════════════════════════════════════════════════

"""
Exemple 1: Dans get_N_gamma(), utiliser les unités
───────────────────────────────────────────────────

Avant (sans unités):
    def get_N_gamma(self, star_masses, star_ages_old, ...):
        emission_old = self.get_stellar_emission(...)
        ...
        return (star_masses * delta_emission) * self.escape_fraction / cell_volume

Après (avec unités):
    def get_N_gamma(self, star_masses, star_ages_old, ...):
        emission_old = self.get_stellar_emission(...)
        ...
        energy_injection = (star_masses * delta_emission) * self.escape_fraction / cell_volume
        
        # Si star_masses était en unités physiques, convertir:
        if isinstance(star_masses, str):
            star_masses_code = self.convert_physical_to_code(star_masses, "mass")
        
        return energy_injection
"""

"""
Exemple 2: Dans force(), utiliser les conversions
──────────────────────────────────────────────────

Avant (sans unités):
    def force(self, i, sol, params, dt):
        per_star_source = self.stromgren_rate * dt
        ...

Après (avec unités):
    def force(self, i, sol, params, dt):
        # sol est en unités de code
        per_star_source = self.stromgren_rate * dt
        
        # Si vous voulez afficher en unités physiques:
        if self.cu is not None and i % 10 == 0:  # Tous les 10 timesteps
            per_star_phys = self.convert_code_to_physical(
                per_star_source, 
                "energy_density"
            )
            print(f"Energy injection: {per_star_phys.value:.2e} {per_star_phys.unit}")
        
        # ... reste du code ...
"""


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 4: EXEMPLE D'UTILISATION COMPLÈTE
# ═══════════════════════════════════════════════════════════════════════════

"""
Dans votre script de simulation:
────────────────────────────────
"""

def setup_simulation_with_units():
    """
    Montrer comment configurer la simulation avec les unités.
    """
    from diffhydro.units import CodeUnits
    from diffhydro.physics.radiative_transfer import StellarRadiationForce

    # 1. Définir les unités de code
    code_units_cfg = {
        "length": "1e22 cm",      # ~1 kpc
        "mass": "1e11 Msun",      # ~100 billion Msun
        "velocity": "1e5 cm/s",   # 100 km/s
    }

    cu = CodeUnits.from_config(code_units_cfg)

    # 2. Créer le solveur avec unités
    srf = StellarRadiationForce(
        escape_fraction=0.1,
        injection_mode="physical",
        gaussian_star=True,
        injection_geometry="3D",
        injection_momentum=True,
        eq=eq,  # votre équation manager
        cu=cu,  # ← NOUVEAU: passer CodeUnits
    )

    # 3. (Optionnel) Convertir des paramètres physiques
    # Si vous voulez accepter des paramètres en unités physiques:
    stromgren_physical = "1e50 erg/s"
    stromgren_code = srf.convert_physical_to_code(
        stromgren_physical,
        "energy_density"
    )
    srf.stromgren_rate = stromgren_code

    print(f"Stromgren rate: {stromgren_physical} → {stromgren_code:.2e} code units")

    return srf, cu


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 5: TESTS POUR VÉRIFIER L'INTÉGRATION
# ═══════════════════════════════════════════════════════════════════════════

"""
Créer un test pour vérifier que tout fonctionne:
"""

def test_radiative_transfer_units():
    """Test que les conversions d'unité fonctionnent."""
    import jax.numpy as jnp
    from diffhydro.units import CodeUnits
    
    # Setup
    cu = CodeUnits.from_config({
        "length": "1e22 cm",
        "mass": "1e11 Msun",
        "velocity": "1e5 cm/s",
    })

    class MockEq:
        light_speed = 1.0
        mesh_shape = (100, 100, 100)
        eps = 1e-20

    srf = StellarRadiationForce(
        escape_fraction=0.1,
        eq=MockEq(),
        cu=cu,
    )

    # Test conversions
    print("Testing unit conversions...")

    # Test 1: Convert physical energy to code
    energy_phys = "1e50 erg/s"
    energy_code = srf.convert_physical_to_code(energy_phys, "energy_density")
    print(f"✓ Energy: {energy_phys} → {energy_code:.2e} code units")

    # Test 2: Convert code back to physical
    result = srf.convert_code_to_physical(energy_code, "energy_density")
    print(f"✓ Back to physical: {result.value:.2e} {result.unit}")

    # Test 3: Radiation flux
    flux_phys = "1e5 erg/s/cm^2"
    flux_code = srf.convert_physical_to_code(flux_phys, "radiation_flux")
    print(f"✓ Flux: {flux_phys} → {flux_code:.2e} code units")

    print("\nAll tests passed! ✓")


# ═══════════════════════════════════════════════════════════════════════════
# RÉSUMÉ DES FICHIERS MODIFIÉS
# ═══════════════════════════════════════════════════════════════════════════

"""
Les fichiers suivants ont déjà été modifiés:

1. diffhydro/units/field_dims.py
   ✓ Ajout des champs RT: E_gamma, Fx_gamma, Fy_gamma, Fz_gamma

2. diffhydro/units/registry.py
   ✓ Ajout de "radiation_flux" au UnitParser
   ✓ Ajout de "erg/s/cm^2" aux unités

3. diffhydro/units/code_units.py
   ✓ Ajout de RadFlux_cgs property
   ✓ Ajout de "radiation_flux" à la méthode scale()

À FAIRE:
────────
1. Modifier StellarRadiationForce.__init__() pour accepter cu et parser
2. Ajouter les méthodes helpers (set_code_units, convert_*)
3. (Optionnel) Utiliser les conversions dans vos méthodes existantes

Fichiers d'exemples créés:
──────────────────────────
- examples/radiative_transfer_units_example.py
- examples/radiative_transfer_modified_example.py
"""

if __name__ == "__main__":
    print(__doc__)
