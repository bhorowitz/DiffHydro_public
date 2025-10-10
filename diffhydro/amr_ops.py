# diffhydro/amr_ops.py
import jax.numpy as jnp
import jax

def restrict_conserve(U_f, r):
    # U_f: [C, Hf, Wf]; Hf, Wf divisible by r
    C, Hf, Wf = U_f.shape
    return U_f.reshape(C, Hf//r, r, Wf//r, r).mean(axis=(2,4))

def prolong_bilinear(U_c, r):
    # separable repeat; you can swap for jax.image.resize linear
    return jnp.repeat(jnp.repeat(U_c, r, axis=1), r, axis=2)

def reflux_add(coarse_face_flux, fine_face_flux, r, face_map):
    # coarse_face_flux: [C, Hc, Wc] on a chosen face direction
    # fine_face_flux:   [C, Hf, Wf]
    # face_map: indices telling which coarse face each fine face contributes to
    corr = jnp.zeros_like(coarse_face_flux)
    # sum fine faces over r-by-r patch onto the coarse face
    corr = corr.at[face_map].add(fine_face_flux.reshape(fine_face_flux.shape[0], -1))
    return coarse_face_flux + corr