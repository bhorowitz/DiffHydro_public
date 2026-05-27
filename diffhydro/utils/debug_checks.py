import os
import jax
import jax.numpy as jnp

# Enable debug prints when the environment variable DIFFHYDRO_DEBUG_CHECKS is set
# to 1, true, yes, or on. Default is disabled.
DEBUG_CHECKS_ENABLED = os.environ.get("DIFFHYDRO_DEBUG_CHECKS", "0").lower() in ("1", "true", "yes", "on")

def _check_finite(name, arr, raise_on_nan: bool = False):
    """Debug helper: print finite/min/max and optionally raise on non-finite.

    Returns
    -------
    bool
        True if all entries are finite.
    """
    # Try to compute a simple finite mask; avoid expensive/reduction ops
    # that may fail on complex pytrees or tracers. Print only the boolean.
    try:
        finite = jnp.all(jnp.isfinite(arr))
    except Exception:
        # If arr is not a plain array (pytree/etc), conservatively assume False
        finite = False

    # Use jax.debug.print so this works under JIT/pjit; print only the flag.
    if DEBUG_CHECKS_ENABLED:
        jax.debug.print("{} finite={}", name, finite)
    if raise_on_nan:
        try:
            any_nonfinite = jnp.any(~jnp.isfinite(arr))
        except Exception:
            any_nonfinite = True
        if any_nonfinite:
            raise FloatingPointError(f"Non-finite values in {name}")
    return finite
