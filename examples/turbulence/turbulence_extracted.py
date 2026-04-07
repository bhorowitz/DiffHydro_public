# Extracted code cells from turbulence.ipynb

# --- cell 1 ---
!nvidia-smi

# --- cell 2 ---
# --- cell 2 --- TOUJOURS EN PREMIER avant tout import jax
import os, sys
sys.path.append("../../")

# GPU config — DOIT être avant import jax
os.environ['CUDA_DEVICE_ORDER']              = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']           = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']  = 'false'   # ← clé manquante
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'     # limite à 70% VRAM
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']    = 'platform'
os.environ['XLA_FLAGS'] = '--xla_gpu_fft_plan_cache_capacity=0'

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_disable_jit', False)
jax.config.update('jax_debug_nans', False)

print("Backend :", jax.default_backend())
print("Devices :", jax.devices())

# --- cell 3 ---
import diffhydro as dh


eq = dh.equationmanager.EquationManager()
eq.mesh_shape=[150,150,150]

# --- cell 4 ---
# U = dh.turbulence.init_turbulent_velocity(eq, 100, 1.0, 1.0,kmax=10)
U = dh.turbulence.init_turbulent_velocity_cpu(eq, 100, 1.0, 1.0, kmax=10)

# --- cell 5 ---
%pylab inline
imshow(U[3][30])

# --- cell 6 ---
#np.save("turb_100x100x100",U)

# --- cell 7 ---
ss = dh.signal_speed_Rusanov

solver = dh.HLLC(equation_manager=eq,signal_speed=ss)

cf = dh.ConvectiveFlux(eq,solver,dh.MUSCL3(limiter="VANLEER"),positivity=False)
#ct = dh.mhd.ConstrainedTransportFlux(eq, solver, dh.MUSCL3(limiter="MINMOD"), positivity=False)

hydro = dh.hydro(n_super_step=1000, fluxes=[cf],forces=[],use_mol=True, integrator="SSPRK3") 

# --- cell 8 ---
params = {}

import jax.numpy as jnp

print("ρ  — min:", jnp.min(U[0]),  "max:", jnp.max(U[0]))
print("vx — min:", jnp.min(U[1]),  "max:", jnp.max(U[1]))
print("vy — min:", jnp.min(U[2]),  "max:", jnp.max(U[2]))
print("vz — min:", jnp.min(U[3]),  "max:", jnp.max(U[3]))
print("E  — min:", jnp.min(U[-1]), "max:", jnp.max(U[-1]))
print("NaN dans U?", jnp.any(jnp.isnan(U)))


output = hydro.evolve(U,params)


rho = U[0]
vx, vy, vz = U[1]/rho, U[2]/rho, U[3]/rho
KE = 0.5 * rho * (vx**2 + vy**2 + vz**2)
E_thermal = U[-1] - KE
print("E_thermal min:", jnp.min(E_thermal))
print("Cellules P<0:", jnp.sum(E_thermal < 0))

# --- cell 9 ---
imshow(output[0][0][30])

# --- cell 10 ---


# --- cell 11 ---
TF = dh.turbulence.TurbulentForce(eq)

# --- cell 12 ---
hydro = dh.hydro(n_super_step=1000, fluxes=[cf],forces=[TF],use_mol=True, integrator="SSPRK3") 

# --- cell 13 ---
params = {}
output = hydro.evolve(U,params)

# --- cell 14 ---
imshow(output[0][0][30]) #cool!

# --- cell 15 ---


# --- cell 16 ---
eq.n_cons = 5
solver = dh.HLLC(equation_manager=eq,signal_speed=ss)
cf = dh.ConvectiveFlux(eq,solver,dh.MUSCL3(limiter="VANLEER"),positivity=False)


# --- cell 17 ---
cf.flux_shapes

# --- cell 18 ---

TF_MC = dh.turbulence.TurbulentForce_MCSTATE(eq)
hydro_MC = dh.hydro(n_super_step=1000, fluxes=[cf],forces=[TF_MC],use_mol=True, integrator="SSPRK3") 

# --- cell 19 ---
import jax.numpy as jnp
#a_start = jnp.zeros((1,100,100,100))#

# --- cell 20 ---
#new_state = jnp.concat([U,a_start],axis=0)

# --- cell 21 ---
nx, ny, nz = eq.mesh_shape if len(eq.mesh_shape)==3 else (*eq.mesh_shape, 1)
accel_k0 = jnp.zeros((3, nx, ny, nz), dtype=jnp.complex64)

accel_k0 = jnp.zeros((3, nx, ny, nz), dtype=jnp.complex64)
params = {
    "turb_seed": jnp.int32(12345),              # keep as JAX scalar
    "turb_key": jax.random.PRNGKey(12345),     # <-- ADD THIS
    "accel_k_state": accel_k0,
}

output = hydro_MC.evolve(U,params)

# --- cell 22 ---
imshow(output[0][0].sum(axis=0)) #cool!

# --- cell 23 ---

