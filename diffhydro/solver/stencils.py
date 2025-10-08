import jax
from typing import Tuple, Union
from jax import Array 


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
        #maybe wrong direction?
        s_0 = jnp.roll(buffer,-2, axis=axis)
        s_1 = jnp.roll(buffer,-1, axis=axis)
        s_2 = jnp.roll(buffer,0, axis=axis)
        s_3 = jnp.roll(buffer,1, axis=axis)
        s_4 = jnp.roll(buffer,2, axis=axis)
        s_5 = jnp.roll(buffer,3, axis=axis)

        cell_state_xi = (1.0 / 256.0) * (
            3.0 * s_0 \
            - 25.0 * s_1 \
            + 150.0 * s_2 \
            + 150.0 * s_3 \
            - 25.0 * s_4 \
            + 3.0 * s_5)
        return cell_state_xi

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array, #maybe put this elsewhere...
            axis: int,
            **kwargs
            ) -> Array:
        
        #maybe wrong direction?
        s_0 = jnp.roll(buffer,-2, axis=axis)
        s_1 = jnp.roll(buffer,-1, axis=axis)
        s_2 = jnp.roll(buffer,0, axis=axis)
        s_3 = jnp.roll(buffer,1, axis=axis)
        s_4 = jnp.roll(buffer,2, axis=axis)
        s_5 = jnp.roll(buffer,3, axis=axis)

        deriv_xi = 1.0 / (60.0 * dxi) * (
            - s_0 \
            + 9.0 * s_1 \
            - 45.0 * s_2 \
            + 45.0 * s_3 \
            - 9.0 * s_4 \
            + s_5)
        return deriv_xi
        
