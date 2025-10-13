import jax.numpy as jnp
from diffhydro.amr_ops import restrict_conserve, prolong_bilinear

def test_restrict_prolong_round_trip():
    C, Hc, Wc, r = 3, 8, 8, 2
    Uc = jnp.arange(C*Hc*Wc, dtype=jnp.float32).reshape(C,Hc,Wc)
    Uf = prolong_bilinear(Uc, r)
    Uc_rt = restrict_conserve(Uf, r)
    # Linear prolongation+conservative restriction should preserve means
    assert jnp.allclose(Uc.mean(), Uc_rt.mean(), rtol=1e-6, atol=1e-6)