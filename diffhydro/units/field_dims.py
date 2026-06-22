"""Canonical mapping from field names to physical dimensions."""

FIELD_DIMS = {
    "rho": "density",
    "vx": "velocity",
    "vy": "velocity",
    "vz": "velocity",
    "p": "pressure",
    "Etot": "energy_density",
    # Radiative transfer fields
    "E_gamma": "energy_density",
    "Fx_gamma": "radiation_flux",
    "Fy_gamma": "radiation_flux",
    "Fz_gamma": "radiation_flux",
}

