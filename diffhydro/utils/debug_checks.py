import os
from dataclasses import is_dataclass, asdict

import jax
import jax.numpy as jnp


DEBUG_CHECKS_ENABLED = os.environ.get("DIFFHYDRO_DEBUG_CHECKS", "0").lower() in ("1", "true", "yes", "on")


def _is_float_array(x):
    dtype = getattr(x, "dtype", None)
    if dtype is None:
        return False
    return jnp.issubdtype(dtype, jnp.floating) or jnp.issubdtype(dtype, jnp.complexfloating)


def _safe_name(prefix, key):
    if prefix is None or prefix == "":
        return str(key)
    return f"{prefix}.{key}"


def _check_finite(name, arr, raise_on_nan: bool = False):
    """
    Vérifie seulement si toutes les valeurs sont finies.
    Retourne True si tout est fini, sinon False.
    """
    try:
        finite = jnp.all(jnp.isfinite(arr))
    except Exception:
        finite = False

    if DEBUG_CHECKS_ENABLED == True:
        jax.debug.print("{} finite={}", name, finite)

    if raise_on_nan:
        try:
            any_nonfinite = jnp.any(~jnp.isfinite(arr))
        except Exception:
            any_nonfinite = True
        if any_nonfinite:
            raise FloatingPointError(f"Non-finite values in {name}")

    return finite


def _check_float_status(
    name,
    arr,
    raise_on_error: bool = False,
    high_threshold_multiplier: float = 0.9,
    low_threshold_multiplier: float = 10.0,
):
    """
    Détecte:
      - nan / inf / non-finies
      - proximité de l'overflow
      - subnormaux
      - proximité de l'underflow normal

    Compatible arrays float/complex JAX.

    Paramètres
    ----------
    high_threshold_multiplier : float
        Doit être < 1.0 pour détecter une proximité de la borne haute.
        Ex: 0.9 -> alerte si |x| >= 0.9 * finfo.max
    low_threshold_multiplier : float
        Facteur > 1.0 autour de tiny pour alerter sur near-underflow.
        Ex: 10.0 -> alerte si 0 < |x| < 10 * tiny
    """
    try:
        dtype = getattr(arr, "dtype", None)
        if dtype is None:
            return {
                "checked": False,
                "reason": "no dtype",
            }

        if not _is_float_array(arr):
            return {
                "checked": False,
                "reason": f"non-floating dtype {dtype}",
            }

        real_dtype = arr.real.dtype if jnp.issubdtype(dtype, jnp.complexfloating) else dtype
        info = jnp.finfo(real_dtype)

        abs_arr = jnp.abs(arr)

        any_nan = jnp.any(jnp.isnan(arr))
        any_inf = jnp.any(jnp.isinf(arr))
        any_nonfinite = jnp.any(~jnp.isfinite(arr))

        max_abs = jnp.max(abs_arr)

        inf_like = jnp.full_like(abs_arr, jnp.inf, dtype=abs_arr.dtype)
        min_abs_nonzero = jnp.min(jnp.where(abs_arr > 0, abs_arr, inf_like))

        max_val = info.max
        tiny = info.tiny
        smallest_subnormal = getattr(info, "smallest_subnormal", 0.0)

        any_near_overflow = jnp.any(abs_arr >= max_val * high_threshold_multiplier)

        any_subnormal = jnp.any((abs_arr > 0) & (abs_arr < tiny))
        any_near_underflow = jnp.any((abs_arr > 0) & (abs_arr < tiny * low_threshold_multiplier))

        zero_count = jnp.count_nonzero(arr == 0)

        overflow = any_inf | any_nan
        underflow_risk = any_subnormal | any_near_underflow

    except Exception as e:
        if DEBUG_CHECKS_ENABLED == True:
            print(f"{name}: check failed with exception {e}")
        return {
            "checked": False,
            "reason": str(e),
        }

    if DEBUG_CHECKS_ENABLED == True:
        jax.debug.print(
            """
    ================ {name} ================
    dtype={dtype}
    nan={nan} inf={inf} nonfinite={nonfinite}
    near_overflow={near_overflow}
    subnormal={subnormal} near_underflow={near_underflow}
    max_abs={max_abs}
    min_abs_nonzero={min_abs_nonzero}
    zero_count={zero_count}
    tiny={tiny}
    smallest_subnormal={smallest_subnormal}
    max={max_val}
    ========================================
    """,
            name=name,
            dtype=real_dtype,
            nan=any_nan,
            inf=any_inf,
            nonfinite=any_nonfinite,
            near_overflow=any_near_overflow,
            subnormal=any_subnormal,
            near_underflow=any_near_underflow,
            max_abs=max_abs,
            min_abs_nonzero=min_abs_nonzero,
            zero_count=zero_count,
            tiny=tiny,
            smallest_subnormal=smallest_subnormal,
            max_val=max_val,
            ordered=True,
        )

    if raise_on_error and (overflow | underflow_risk):
        raise FloatingPointError(f"Floating-point issue detected in {name}")

    return {
        "checked": True,
        "dtype": real_dtype,
        "any_nan": any_nan,
        "any_inf": any_inf,
        "any_nonfinite": any_nonfinite,
        "any_near_overflow": any_near_overflow,
        "any_subnormal": any_subnormal,
        "any_near_underflow": any_near_underflow,
        "max_abs": max_abs,
        "min_abs_nonzero": min_abs_nonzero,
        "zero_count": zero_count,
        "tiny": tiny,
        "smallest_subnormal": smallest_subnormal,
        "max": max_val,
    }


def _check_overflow(
    name,
    arr,
    raise_on_overflow: bool = False,
    threshold_multiplier: float = 0.9,
):
    """
    Alias compatible avec l'import existant dans radiative_transfer.py.
    Détecte surtout:
      - nan / inf
      - proximité de l'overflow
    """
    out = _check_float_status(
        name=name,
        arr=arr,
        raise_on_error=False,
        high_threshold_multiplier=threshold_multiplier,
        low_threshold_multiplier=10.0,
    )

    if not out.get("checked", False):
        return False

    overflow = out["any_nan"] | out["any_inf"] | out["any_nonfinite"] | out["any_near_overflow"]

    if raise_on_overflow and overflow:
        raise OverflowError(f"Overflow or non-finite values in {name}")

    return overflow


def _walk_and_check(
    obj,
    prefix="",
    raise_on_error: bool = False,
    high_threshold_multiplier: float = 0.9,
    low_threshold_multiplier: float = 10.0,
    results=None,
):
    """
    Parcourt récursivement:
      - arrays float/complex
      - dict
      - list / tuple
      - dataclass
      - objets avec __dict__

    Retourne un dict {nom_variable: résultat_check}.
    """
    if results is None:
        results = {}

    if obj is None:
        return results

    if _is_float_array(obj):
        results[prefix or "root"] = _check_float_status(
            name=prefix or "root",
            arr=obj,
            raise_on_error=raise_on_error,
            high_threshold_multiplier=high_threshold_multiplier,
            low_threshold_multiplier=low_threshold_multiplier,
        )
        return results

    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_and_check(
                v,
                prefix=_safe_name(prefix, k),
                raise_on_error=raise_on_error,
                high_threshold_multiplier=high_threshold_multiplier,
                low_threshold_multiplier=low_threshold_multiplier,
                results=results,
            )
        return results

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk_and_check(
                v,
                prefix=_safe_name(prefix, i),
                raise_on_error=raise_on_error,
                high_threshold_multiplier=high_threshold_multiplier,
                low_threshold_multiplier=low_threshold_multiplier,
                results=results,
            )
        return results

    if is_dataclass(obj):
        return _walk_and_check(
            asdict(obj),
            prefix=prefix,
            raise_on_error=raise_on_error,
            high_threshold_multiplier=high_threshold_multiplier,
            low_threshold_multiplier=low_threshold_multiplier,
            results=results,
        )

    if hasattr(obj, "__dict__"):
        return _walk_and_check(
            vars(obj),
            prefix=prefix,
            raise_on_error=raise_on_error,
            high_threshold_multiplier=high_threshold_multiplier,
            low_threshold_multiplier=low_threshold_multiplier,
            results=results,
        )

    return results


def _check_all_float_variables(
    raise_on_error: bool = False,
    high_threshold_multiplier: float = 0.9,
    low_threshold_multiplier: float = 10.0,
    **named_objects,
):
    """
    Fonction pratique pour tester plusieurs variables d'un coup.

    Exemple:
        _check_all_float_variables(
            sol=sol,
            primitives=primitives,
            E=E_gamma,
            Fx=Fx,
            Fy=Fy,
            Fz=Fz,
        )
    """
    results = {}
    for name, obj in named_objects.items():
        _walk_and_check(
            obj,
            prefix=name,
            raise_on_error=raise_on_error,
            high_threshold_multiplier=high_threshold_multiplier,
            low_threshold_multiplier=low_threshold_multiplier,
            results=results,
        )
    return results