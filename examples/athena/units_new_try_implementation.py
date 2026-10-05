#Depending on your system/cluster, you may want to specify which GPU you want to use
import os, sys ,importlib
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
# jax.config.update('jax_debug_nans', True)
# jax.config.update("jax_disable_jit", True)
os.environ['DIFFHYDRO_DEBUG_CHECKS'] = 'False' 
import diffhydro.utils.debug_checks as dc
importlib.reload(dc)
# above: original version; below: same info as for turbulence
#load athena ICs
from diffhydro.utils.io import athinput,athdf
from diffhydro.equationmanager_radiative_transf_no_chat import EquationManager as EquationManager_RT
from diffhydro.physics.radiative_transfer import StellarRadiationForce
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

size_shape = 256 #250

eq = dh.equationmanager.EquationManager()
eq_test = EquationManager_RT(light_speed=3e10,mesh_shape=(size_shape, size_shape, size_shape),debug=False) # radiative transfer version; light_speed=40 seemed to suppress the shock wave
print(eq)
print(eq_test)
#need to specify a signal speed for wave prop
ss = dh.signal_speed_Rusanov
print(ss)

#Lots of solvers are available, HLLC is probably the "fanciest"; good for both shocks and instabilities
solver = dh.LaxFriedrichs(equation_manager=eq,signal_speed=ss)
solver_test = dh.LaxFriedrichs_Radiative_transfer(equation_manager=eq_test,signal_speed=ss)#_Radiative_transfer HLL_Radiative_transfer_Local


#Specifying the flux terms and flux reconstruction methods. MUSCL3 is probably the most well tested
# cf = dh.ConvectiveFlux(eq,solver,dh.MUSCL3(limiter="VANLEER")) # does not work with Ben's version, but the chat version fixed it
# current version is from the chat, the other one is commented out
# cf_test = dh.ConvectiveFlux(eq_test,solver_test,dh.MUSCL3(limiter="VANLEER"))

#specify the total simulation setup 
cf = dh.ConvectiveFlux(eq,solver,dh.PLM(limiter="MC"))#limiter="VANLEER" best MC KOREN

cf_test = dh.ConvectiveFlux_Radiative_transfer(eq_test,solver_test,dh.PLM(limiter="MINMOD"))#limiter="VANLEER" ici diffuse mieux moins de valeurs extremes 
#MINMOD is more beautiful on figures 
hydrosim = dh.hydro(n_super_step=1000,fluxes=[cf])

stellar_force = StellarRadiationForce(escape_fraction=0.1, dx=1.0, injection_mode="stromgren", 
                                      stromgren_rate=10e10, injection_momentum=False,injection_geometry="3D"
                                      , eq=eq_test, debug=False, momentum_only=False,beam_axis=0,
                                        beam_sign=1,
                                        beam_length_cells=1,
                                        beam_reduced_flux=1,
                                        beam_momentum_scaling="legacy_c2_source2")#valeur origine 0.95legacy_c2_source2
hydrosim_test = dh.hydro(n_super_step=6000,fluxes=[cf_test],forces=[stellar_force])#,debug_fixed_dt=1e-6
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
    "star_positions": jnp.array([[size_shape // 2, size_shape // 2, size_shape // 2]], dtype=jnp.int32),
}#size_shape // 2  ,[100, size_shape // 2, size_shape // 2]]

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
# injection_density = 100  # Adjust this value as needed
# iz_slice = size_shape // 2
# ix_center, iy_center, iz_center = size_shape // 2, size_shape // 2, size_shape // 2
# sol_test = sol_test.at[0,:,:,:].set(injection_density) # density
# vmin = sol_test[0, :, :, iz_center].min()
# vmax = sol_test[0, :, :, iz_center].max()

# fig, ax = plt.subplots()
# im = ax.imshow(
#     sol_test[0, :, :, iz_center],
#     origin="lower",
#     cmap="hot",
#     vmin=vmax,   # inversé : vmin prend la valeur max
#     vmax=vmin,   # inversé : vmax prend la valeur min
# )
# ax.set_title(f"sol_test[0] at center point ({ix_center}, {iy_center}, {iz_center}), injection {injection_density} ")
# ax.set_xlabel("y")
# ax.set_ylabel("x")

# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label("Photon number density", fontsize=12)   # <-- titre de la colorbar

# plt.tight_layout()
# plt.show()
# print(  jnp.max(sol_test[0])  )



# Étoile en (50, 50, 50) → injecter flux selon +x
# ix, iy, iz = 0, 50, 50
# beam_length = 10      # cellules en aval selon +x
# f_reduced   = 0.99   # |F| = f * c * E  (reste < c*E, admissible M1)
# c           = 2.0    # lightspeed ton unité

# weights = jnp.exp(-0.5 * (jnp.arange(beam_length) / 3.0)**2)
# weights = weights / weights.sum()

# E_inject = 10  # amplitude à ajuster selon ton problème

# for s in range(beam_length):
#     ixs = ix + s
#     if ixs < 100:
#         dE = E_inject * weights[s]
#         sol_test = sol_test.at[0, ixs, iy, iz].add(dE)
#         sol_test = sol_test.at[1, ixs, iy, iz].add(f_reduced * c * dE)  # Fgammax
# Run standard hydro with 5-variable state (rho, momx, momy, momz, Etot)
# field, parametre, temps_final_simulation, dt_historique, nombre_de_pas = hydrosim.evolve_till_time(cp.deepcopy(sol), params, 18.6)

field_test, parametre_test, temps_final_simulation_test, dt_historique_test, nombre_de_pas_test_test = hydrosim_test.evolve_till_time(cp.deepcopy(sol_test), params,5e-11 )#37.2
# field_test, parametre_test, dthistoriquetest = hydrosim_test.evolve_with_callbacks(cp.deepcopy(sol_test), params)
# print(jnp.max(field_test[0]), jnp.max(field_test[1]), jnp.max(field_test[2]), jnp.max(field_test[3]))
# print(jnp.mean(field_test[0]), jnp.mean(field_test[1]), jnp.mean(field_test[2]), jnp.mean(field_test[3]))
# print(jnp.sum(field_test[0]), jnp.sum(field_test[1]), jnp.sum(field_test[2]), jnp.sum(field_test[3]))
iz_slice = size_shape // 2
ix_center, iy_center, iz_center = size_shape // 2, size_shape // 2, size_shape // 2

# Time tracking of field 0 (E_gamma) with snapshots at fixed scale
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
# print(field_test)
import matplotlib.colors as mcolors

vmin = field_test[0, :, :, iz_center].min()
vmax = field_test[0, :, :, iz_center].max()

fig, ax = plt.subplots()
im = ax.imshow(
    field_test[0, :, :, iz_center],
    origin="lower",
    cmap="hot",
    vmin=vmax,   # inversé : vmin prend la valeur max
    vmax=vmin,   # inversé : vmax prend la valeur min
)
# ax.set_title(f"field_test[0] at center point ({ix_center}, {iy_center}, {iz_center}), injection {injection_density} ")
ax.set_xlabel("y")
ax.set_ylabel("x")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Photon number density", fontsize=12)   # <-- titre de la colorbar

plt.tight_layout()
plt.show()


plt.imshow(field_test[0, :,:,iz_center], origin="lower", cmap="hot")
# plt.title(f"field_test[0] at center point ({ix_center}, {iy_center}, {iz_center}), injection {injection_density} ")
plt.xlabel("y")
plt.ylabel("x")
plt.xlim(ix_center - 5, ix_center + 5)
plt.ylim(iy_center - 5, iy_center + 5)
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(
    np.log(field_test[0, :, :, iz_center]),
    origin="lower",
    cmap="hot",
    vmin=-20,   # inversé : vmin prend la valeur max
)
# ax.set_title(f"field_test[0] at center point ({ix_center}, {iy_center}, {iz_center}), injection {injection_density} ")
ax.set_xlabel("y")
ax.set_ylabel("x")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(" Log photon number density", fontsize=12)   # <-- titre de la colorbar

plt.tight_layout()
plt.show()

plt.imshow(np.log(field_test[0, :,:,iz_center]), origin="lower", cmap="hot", vmin=-20)
# plt.title(f"field_test[0] at center point ({ix_center}, {iy_center}, {iz_center}), injection {injection_density} ")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.xlim(ix_center - 5, ix_center + 5)
plt.ylim(iy_center - 5, iy_center + 5)
plt.show()
mask = (
    (field_test[0, :, :, iz_center] > 1e-10)
    & (field_test[0, :, :, iz_center] < 1e-1)
)

field_test_masked = jnp.where(mask, field_test[0, :, :, iz_center], 0.0)
plt.imshow(field_test_masked, origin="lower", cmap="hot")
# plt.title(f"Masked field_test[0] at center point ({ix_center}, {iy_center}, {iz_center}), injection {injection_density} ")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()
print(temps_final_simulation_test, dt_historique_test, nombre_de_pas_test_test)
