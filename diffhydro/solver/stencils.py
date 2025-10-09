import jax
from typing import Tuple, Union
from jax import Array 
import jax.numpy as jnp

class CentralSixthOrderReconstruction():
    """CentralSixthOrderReconstruction 

    6th order stencil for reconstruction at the cell face
                       x
    |      |     |     |     |     |     |
    | i-2  | i-1 |  i  | i+1 | i+2 | i+3 | 
    |      |     |     |     |     |     |
    """

    
    def __init__(
            self,
            **kwargs
            ) -> None:
        pass
       # self.array_slices([range(-3, 3, 1)])

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            **kwargs
        ) -> Array:
        # shift buffers so that positive shifts walk towards lower indices
        f_im2 = jnp.roll(buffer, 2, axis=axis)
        f_im1 = jnp.roll(buffer, 1, axis=axis)
        f_i = buffer
        f_ip1 = jnp.roll(buffer, -1, axis=axis)
        f_ip2 = jnp.roll(buffer, -2, axis=axis)
        f_ip3 = jnp.roll(buffer, -3, axis=axis)

        cell_state_xi = (1.0 / 256.0) * (
            3.0 * f_im2 \
            - 25.0 * f_im1 \
            + 150.0 * f_i \
            + 150.0 * f_ip1 \
            - 25.0 * f_ip2 \
            + 3.0 * f_ip3)
        return cell_state_xi

    def derivative_xi(
            self,
            buffer: Array,
            axis: int,
            **kwargs
            ) -> Array:
        dxi=1.0
        # sixth-order centered derivative, aligned with mesh orientation
        f_im3 = jnp.roll(buffer, 3, axis=axis)
        f_im2 = jnp.roll(buffer, 2, axis=axis)
        f_im1 = jnp.roll(buffer, 1, axis=axis)
        f_ip1 = jnp.roll(buffer, -1, axis=axis)
        f_ip2 = jnp.roll(buffer, -2, axis=axis)
        f_ip3 = jnp.roll(buffer, -3, axis=axis)

        deriv_xi = 1.0 / (60.0 * dxi) * (
            f_im3 \
            - 9.0 * f_im2 \
            + 45.0 * f_im1 \
            - 45.0 * f_ip1 \
            + 9.0 * f_ip2 \
            - f_ip3)
        return deriv_xi