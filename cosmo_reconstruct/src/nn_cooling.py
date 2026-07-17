"""Neural-network cooling-rate surrogate (drop-in for NyxCoolingRateInterpolator).

Why: the tabulated cooling curve has razor-thin high-curvature regions whose
exact adjoint is chaos-amplified through the 128-step rollout (|grad U|~2.6e9,
leapfrog NaN at the first step -- see HANDOFF.md section 8f'). A small tanh MLP
fit to the SAME rate map is C-infinity with bounded, training-controllable
curvature, so the potential built on it is conservative under AD (pair with
cooling_integrator="exponential_exact") AND integrable by leapfrog. The NN
posterior is a surrogate model in exactly the same sense as the exponential
integrator (validate observable-space bias against the extrapolation targets).

The evaluator mirrors NyxCoolingRateInterpolator.evaluate:
    heat, cool, net = evaluate(rho_cgs, temp_k, z)
Inputs are mapped to the table's natural coordinates (log10 delta, log10 T,
z-embedding) with the same delta = rho/rho_mean(z) conversion, and outputs are
10**MLP in the same cgs units as the table. Trained by
cosmo_reconstruct/_train_nn_cooling.py; weights live in an npz.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diffhydro.cosmology.nyx_cooling_table import rho_b_mean_cgs_jax


def mlp_apply(params, x):
    """x: (..., n_in) -> (..., n_out); tanh MLP, linear head."""
    h = x
    n_layers = len(params["weights"])
    for i, (W, b) in enumerate(zip(params["weights"], params["biases"])):
        h = h @ W + b
        if i < n_layers - 1:
            h = jnp.tanh(h)
    return h


class NNCoolingEvaluator:
    """Drop-in for NyxCoolingRateInterpolator backed by a tanh MLP."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        n_layers = int(data["n_layers"])
        self.params = {
            "weights": [jnp.asarray(data[f"W{i}"], jnp.float32) for i in range(n_layers)],
            "biases": [jnp.asarray(data[f"b{i}"], jnp.float32) for i in range(n_layers)],
        }
        # input normalization: x_norm = (x - mu) / sigma over (logdelta, logT, zfeat)
        self.in_mu = jnp.asarray(data["in_mu"], jnp.float32)
        self.in_sigma = jnp.asarray(data["in_sigma"], jnp.float32)
        # output de-normalization for (log10 heat, log10 cool)
        self.out_mu = jnp.asarray(data["out_mu"], jnp.float32)
        self.out_sigma = jnp.asarray(data["out_sigma"], jnp.float32)
        self.z_clip = (float(data["z_min"]), float(data["z_max"]))
        self.h = float(data["h"])
        self.omega_b = float(data["omega_b"])
        self.log_floor = float(data.get("log_floor", -40.0))
        # NyxTabulatedCoolingForce sets this attribute; it is meaningless for the
        # NN (already C-infinity) but must exist and be assignable.
        self.smooth_interp = False

    def evaluate(self, rho_cgs, temp_k, z):
        z_scalar = jnp.asarray(z, dtype=jnp.float32)
        rho_mean = rho_b_mean_cgs_jax(z_scalar, h=self.h, omega_b=self.omega_b)
        delta = jnp.maximum(jnp.asarray(rho_cgs, jnp.float32) / jnp.maximum(rho_mean, 1.0e-30), 1.0e-30)
        logdelta = jnp.log10(delta)
        logt = jnp.log10(jnp.maximum(jnp.asarray(temp_k, jnp.float32), 1.0e-30))
        zc = jnp.clip(z_scalar, self.z_clip[0], self.z_clip[1])
        zfeat = jnp.log10(1.0 + zc) * jnp.ones_like(logdelta)

        x = jnp.stack([logdelta, logt, zfeat], axis=-1)
        x = (x - self.in_mu) / self.in_sigma
        y = mlp_apply(self.params, x) * self.out_sigma + self.out_mu
        # Upper cap: the table's log-rates top out at ~-17; NN extrapolation in
        # unseen corners must never reach 10^y overflow (f32 inf -> NaN in the
        # 2nd-order exponential_exact gradient). -10 is 5+ dex above physical.
        log_heat = jnp.clip(y[..., 0], self.log_floor, -10.0)
        log_cool = jnp.clip(y[..., 1], self.log_floor, -10.0)
        heat = 10.0 ** log_heat
        cool = 10.0 ** log_cool
        return heat, cool, heat - cool
