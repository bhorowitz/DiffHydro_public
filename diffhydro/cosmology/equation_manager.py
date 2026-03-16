"""Supercomoving wrapper around an existing EquationManager."""

from __future__ import annotations

from . import conversions


class SuperComovingEquationManager:
    """Wrapper that reinterprets hydro variables in supercomoving units."""

    _LOCAL_ATTRS = {"_equation_manager", "enforce_gamma_53"}

    def __init__(self, equation_manager, enforce_gamma_53: bool = True):
        object.__setattr__(self, "_equation_manager", equation_manager)
        object.__setattr__(self, "enforce_gamma_53", bool(enforce_gamma_53))
        if enforce_gamma_53:
            self.set_gamma(5.0 / 3.0)

    @property
    def base(self):
        return self._equation_manager

    @property
    def gamma(self):
        return self._equation_manager.gamma

    @gamma.setter
    def gamma(self, value):
        self.set_gamma(value)

    def set_gamma(self, value):
        self._equation_manager.gamma = float(value)
        if hasattr(self._equation_manager, "R"):
            g = self._equation_manager.gamma
            self._equation_manager.cp = g / (g - 1.0) * self._equation_manager.R

    def get_conservatives_from_primitives(self, primitives):
        return self._equation_manager.get_conservatives_from_primitives(primitives)

    def get_primitives_from_conservatives(self, conservatives):
        return self._equation_manager.get_primitives_from_conservatives(conservatives)

    def physical_primitives_to_code(self, rho_phys, vx_pec, vy_pec, vz_pec, p_phys, a):
        return conversions.primitives_phys_to_code(rho_phys, vx_pec, vy_pec, vz_pec, p_phys, a)

    def code_primitives_to_physical(self, primitives_code, a):
        return conversions.primitives_code_to_phys(primitives_code, a)

    def physical_primitives_to_conservatives(self, rho_phys, vx_pec, vy_pec, vz_pec, p_phys, a):
        return conversions.conservatives_phys_to_code(
            self._equation_manager,
            rho_phys,
            vx_pec,
            vy_pec,
            vz_pec,
            p_phys,
            a,
        )

    def conservatives_to_physical_primitives(self, conservatives_code, a):
        return conversions.conservatives_code_to_phys(self._equation_manager, conservatives_code, a)

    def __getattr__(self, name):
        return getattr(self._equation_manager, name)

    def __setattr__(self, name, value):
        if name in self._LOCAL_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self._equation_manager, name, value)

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__.keys()) + dir(self._equation_manager)))
