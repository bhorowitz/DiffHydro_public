"""Physical <-> supercomoving conversion utilities."""

from __future__ import annotations

import jax.numpy as jnp


def _safe_scale_factor(a):
    return jnp.maximum(jnp.asarray(a), 1.0e-12)


def density_phys_to_code(rho_phys, a):
    a = _safe_scale_factor(a)
    return jnp.asarray(rho_phys) * a**3


def density_code_to_phys(rho_tilde, a):
    a = _safe_scale_factor(a)
    return jnp.asarray(rho_tilde) / a**3


def velocity_phys_to_code(v_pec, a):
    a = _safe_scale_factor(a)
    return jnp.asarray(v_pec) * a


def velocity_code_to_phys(v_tilde, a):
    a = _safe_scale_factor(a)
    return jnp.asarray(v_tilde) / a


def pressure_phys_to_code(p_phys, a):
    a = _safe_scale_factor(a)
    return jnp.asarray(p_phys) * a**5


def pressure_code_to_phys(p_tilde, a):
    a = _safe_scale_factor(a)
    return jnp.asarray(p_tilde) / a**5


def primitives_phys_to_code(rho_phys, vx_pec, vy_pec, vz_pec, p_phys, a):
    """Pack supercomoving primitives [rho_tilde, v_tilde, p_tilde]."""
    v_tilde_x = velocity_phys_to_code(vx_pec, a)
    v_tilde_y = velocity_phys_to_code(vy_pec, a)
    v_tilde_z = velocity_phys_to_code(vz_pec, a)
    return jnp.stack(
        [
            density_phys_to_code(rho_phys, a),
            v_tilde_x,
            v_tilde_y,
            v_tilde_z,
            pressure_phys_to_code(p_phys, a),
        ],
        axis=0,
    )


def primitives_code_to_phys(primitives_code, a):
    """Unpack physical primitives (rho_phys, vx_pec, vy_pec, vz_pec, p_phys)."""
    rho_tilde = primitives_code[0]
    vtx = primitives_code[1]
    vty = primitives_code[2]
    vtz = primitives_code[3]
    p_tilde = primitives_code[4]
    return (
        density_code_to_phys(rho_tilde, a),
        velocity_code_to_phys(vtx, a),
        velocity_code_to_phys(vty, a),
        velocity_code_to_phys(vtz, a),
        pressure_code_to_phys(p_tilde, a),
    )


def conservatives_phys_to_code(eq_manager, rho_phys, vx_pec, vy_pec, vz_pec, p_phys, a):
    """Convert physical primitives directly to conservative code-state U."""
    primitives_code = primitives_phys_to_code(rho_phys, vx_pec, vy_pec, vz_pec, p_phys, a)
    return eq_manager.get_conservatives_from_primitives(primitives_code)


def conservatives_code_to_phys(eq_manager, conservatives_code, a):
    """Convert conservative code-state U to physical primitives."""
    primitives_code = eq_manager.get_primitives_from_conservatives(conservatives_code)
    return primitives_code_to_phys(primitives_code, a)
