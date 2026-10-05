#Depending on your system/cluster, you may want to specify which GPU you want to use
import os, sys
sys.path.append("../../")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #on CPU you can leave it as '' 

import jax 
jax.config.update('jax_disable_jit', False) #turn this True for better debugging, much slower

import jax.numpy as jnp #all of the numpy-like functionalities

#core diffhydro
import diffhydro as dh
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
import matplotlib.pyplot as plt
jax.config.update('jax_debug_nans', True)
jax.config.update("jax_disable_jit", True)
# above: original version; below: same info as for turbulence
#load athena ICs
from diffhydro.utils.io import athinput,athdf
from diffhydro.equationmanager_radiative_transf_no_chat import EquationManager as EquationManager_RT

athena_outputs_loc = "../../data/athena_comparison/"

ic_filename = "Blast.out2.00000.athdf"


ICs = athdf(athena_outputs_loc+ic_filename)

# print("ICs loaded, keys are: ", ICs.keys(),ICs)
#setting up my dh ICs to match the athena file
#will probably package this up into a single common func
sol = jnp.zeros((5,100,100,100))
sol = sol.at[0].set(ICs["dens"]) # density
sol = sol.at[-1].set(ICs["Etot"]) # total energy

print(sol.shape)
plt.imshow(ICs["Etot"][50])
# plt.imshow(ICs["Etot"][50])
print("ICs loaded, keys are: ", ICs.keys())

# #equation manager deals with lots of the system configuration and basic gas EOS physics
# import importlib
# import diffhydro.equationmanager_radiative_transf_no_chat as eq_rt_mod
# import diffhydro.fluxes as fluxes_mod
# import diffhydro.solver.riemann_solver as rs_mod

# # Force reload to avoid stale module state in notebook kernels
# importlib.reload(eq_rt_mod)
# importlib.reload(fluxes_mod)
# importlib.reload(rs_mod)
# EquationManager_RT = eq_rt_mod.EquationManager

size_shape = 100

eq = dh.equationmanager.EquationManager()
eq_test = EquationManager_RT(light_speed=2,mesh_shape=(size_shape, size_shape, size_shape)) # radiative transfer version; light_speed=40 seemed to suppress the shock wave
print(eq)
print(eq_test)
#need to specify a signal speed for wave prop
ss = dh.signal_speed_Rusanov
print(ss)

#Lots of solvers are available, HLLC is probably the "fanciest"; good for both shocks and instabilities
solver = dh.LaxFriedrichs(equation_manager=eq,signal_speed=ss)
solver_test = dh.LaxFriedrichs_Radiative_transfer(equation_manager=eq_test,signal_speed=ss)#_Radiative_transfer


#Specifying the flux terms and flux reconstruction methods. MUSCL3 is probably the most well tested
# cf = dh.ConvectiveFlux(eq,solver,dh.MUSCL3(limiter="VANLEER")) # does not work with Ben's version, but the chat version fixed it
# current version is from the chat, the other one is commented out
# cf_test = dh.ConvectiveFlux(eq_test,solver_test,dh.MUSCL3(limiter="VANLEER"))

#specify the total simulation setup 
cf = dh.ConvectiveFlux(eq,solver,dh.PLM(limiter="MC"))#limiter="VANLEER" best MC KOREN
cf_test = dh.ConvectiveFlux_Radiative_transfer(eq_test,solver_test,dh.PPM_CW(limiter="VANLEER"))#limiter="VANLEER" ici diffuse mieux moins de valeurs extremes 
#MINMOD is more beautiful on figures 
hydrosim = dh.hydro(n_super_step=1000,fluxes=[cf])

from diffhydro.physics.radiative_transfer import StellarRadiationForce
stellar_force = StellarRadiationForce(escape_fraction=0.1, dx=1.0, injection_mode="stromgren", 
                                      stromgren_rate=1e-3, injection_momentum=True,injection_geometry="2D"
                                      , eq=eq_test, debug=False, momentum_only=False)
hydrosim_test = dh.hydro(n_super_step=1000,fluxes=[cf_test],forces=[stellar_force], debug_fixed_dt=1e-6)
print(hydrosim)
import copy as cp
import numpy as np
import matplotlib.pyplot as plt

# Disk of stars in the z=50 plane
# center_x, center_y, center_z = 50, 50, 50
# radius_cells = 10

# # All cells (ix, iy) within the disk of radius radius_cells
# offsets = [(dx, dy)
#            for dx in range(-radius_cells, radius_cells + 1)
#            for dy in range(-radius_cells, radius_cells + 1)
#            if dx**2 + dy**2 <= radius_cells**2]

# n_stars = len(offsets)
# x = jnp.array([center_x + dx for dx, dy in offsets], dtype=jnp.int32)
# y = jnp.array([center_y + dy for dx, dy in offsets], dtype=jnp.int32)
# z = jnp.full((n_stars,), center_z, dtype=jnp.int32)
# star_positions = jnp.stack([x, y, z], axis=1)

# print(f"Number of stars in the disk: {n_stars}")

# params = {
#     "star_masses": jnp.full((n_stars,), 1.0),
#     "star_ages": jnp.full((n_stars,), 0.1),
#     "star_metallicities": jnp.full((n_stars,), 0.02),
#     "star_positions": star_positions,
# }
# For testing with a single star at the center of the box
params = {
    "star_masses": jnp.array([10]),
    "star_ages": jnp.array([0.1]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions": jnp.array([[0, size_shape // 2, size_shape // 2]], dtype=jnp.int32),
}

# For testing with two stars, one at (50, 50, 50) and another at (56, 50, 50) in 2D injection
# params = {
#     "star_masses": jnp.array([10,10]),
#     "star_ages": jnp.array([0.1,0.1]),
#     "star_metallicities": jnp.array([0.02,0.02]),
#     "star_positions": jnp.array([[0, 54, 50],[0, , 50]], dtype=jnp.int32), # update to follow Enrico's example
    
# }

# Visual check of initial star positions
# sol_test = jnp.zeros((4, size_shape, size_shape, size_shape))
# sol_test = sol_test.at[0, x, y, z].add(1.0)

# fig, ax = plt.subplots(figsize=(5, 5))
# # sol_test shape: (4, nx, ny, nz) -> slice at z=center_z: sol_test[0, :, :, center_z]
# im = ax.imshow(np.array(sol_test[0, :, :, center_z]), origin="upper", cmap="hot")
# ax.set_title(f"Initial star positions (z={center_z}, n={n_stars})")
# plt.colorbar(im, ax=ax)
# plt.tight_layout()
# plt.show()

# Reset to zero for clean simulation
sol_test = jnp.zeros((4, size_shape, size_shape, size_shape))

# Run standard hydro with 5-variable state (rho, momx, momy, momz, Etot)
# field, parametre, temps_final_simulation, dt_historique, nombre_de_pas = hydrosim.evolve_till_time(cp.deepcopy(sol), params, 18.6)

field_test, parametre_test, temps_final_simulation_test, dt_historique_test, nombre_de_pas_test_test = hydrosim_test.evolve_till_time(cp.deepcopy(sol_test), params, 18.6)
# field_test, parametre_test, dthistoriquetest = hydrosim_test.evolve_with_callbacks(cp.deepcopy(sol_test), params)
iz_slice = size_shape // 2
ix_center, iy_center, iz_center = size_shape // 2, size_shape // 2, size_shape // 2

# Time tracking of field 0 (E_gamma) with snapshots at fixed scale
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
# print(field_test)
plt.imshow(field_test[0, :,:,iz_center], origin="lower", cmap="hot")
plt.title(f"field_test[0] at center point ({ix_center}, {iy_center}, {iz_center}) after evolve_till_time")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()
plt.imshow(np.log(field_test[0, :,:,iz_center]), origin="lower", cmap="hot")
plt.title(f"field_test[0] at center point ({ix_center}, {iy_center}, {iz_center}) after evolve_till_time")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()
