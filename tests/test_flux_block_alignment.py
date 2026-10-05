import jax.numpy as jnp

import diffhydro as dh
from diffhydro.equationmanager_radiative_transf_no_chat_copy import (
    EquationManager as RTEquationManager,
)


def test_fluxes_can_accumulate_across_separate_state_blocks():
    shape = (4, 4, 4)
    hydro_eq = dh.EquationManager(gamma=5.0 / 3.0, mesh_shape=shape, eps=1e-20)
    rt_eq = RTEquationManager(light_speed=1.0, mesh_shape=shape, eps=1e-20, debug=False)

    hydro_solver = dh.LaxFriedrichs(
        equation_manager=hydro_eq,
        signal_speed=dh.signal_speed_Rusanov,
    )
    rt_solver = dh.LaxFriedrichs_Radiative_transfer(
        equation_manager=rt_eq,
        signal_speed=dh.signal_speed_Rusanov,
    )

    hydro_flux = dh.ConvectiveFlux(
        hydro_eq,
        hydro_solver,
        dh.PLM(limiter="VANLEER"),
        dx=1.0,
        state_slice=slice(0, 5),
    )
    rt_flux = dh.ConvectiveFlux_Radiative_transfer(
        rt_eq,
        rt_solver,
        dh.PLM(limiter="VANLEER"),
        dx=1.0,
        state_slice=slice(5, 10),
    )

    sim = dh.hydro(fluxes=[hydro_flux, rt_flux], dx=1.0, pmesh_shape=(1, 1, 1))

    sol = jnp.zeros((10, *shape), dtype=jnp.float32)
    sol = sol.at[0].set(1.0)
    sol = sol.at[4].set(1.0)
    sol = sol.at[5].set(1.0)
    sol = sol.at[9].set(0.1)

    total_flux = sim.flux(sol, 1, None)

    assert total_flux.shape == sol.shape
