"""Core corrected Stromgren test runner."""
import os, sys, math, copy as cp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import numpy as np
import diffhydro as dh
from diffhydro.units import CodeUnits
from diffhydro.units.registry import UnitParser
from diffhydro.equationmanager_radiative_transf_no_chat_copy import EquationManager as EquationManager_RT
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.physics.fraction_xHII import HydrogenPhotoChemistryForce
from diffhydro.physics import hydrogen_chemistry as hchem

jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
N = 25
ULEN = "4.7536191406e22 cm"
BOXPHYS = "4.7536191406e22 cm"
UVEL = "2.99792458e7 cm/s"       # reduced light speed = 10^-3 c
SRC = 5e48
TPHYS = "1.92915e16 s"
EPS = 1e-30
NSTEP = 3000
MAXDT = None
N_H_CGS = 1.0e-3
T_AMBIENT_K = 1.0e4
MAKE_GIFS = False
GIF_FRAMES = 100
up = UnitParser()

def parse_quantity(text, expected_dim):
    return up.parse(text, expected_dim=expected_dim)

def sanitize_tag(text):
    return text.replace(" ", "").replace("/", "p").replace("^", "")

# -----------------------------------------------------------------------------
# Units and physical setup
# -----------------------------------------------------------------------------
size_shape = N
ulen_q = parse_quantity(ULEN, "length")
unit_length_phys_cgs = float(ulen_q.cgs_value)
unit_length_str = f"{ulen_q.value:g}{ulen_q.unit}"

box_q = parse_quantity(BOXPHYS, "length")
box_width_phys_cgs = float(box_q.cgs_value)
box_width_code = box_width_phys_cgs / unit_length_phys_cgs
box_width_str = f"{box_q.value:g}{box_q.unit}"
axis_unit_name = box_q.unit
axis_unit_scale = up.unit_factor_to_cgs(axis_unit_name, expected_dim="length")

dx_code = box_width_code / size_shape
dx_phys_cgs = dx_code * unit_length_phys_cgs
cell_volume_code = dx_code**3
cell_volume_cm3 = dx_phys_cgs**3

uvel_q = parse_quantity(UVEL, "velocity")
unit_velocity_phys = float(uvel_q.cgs_value)
unit_velocity_str = f"{uvel_q.value:g}{uvel_q.unit}"

c_cgs = 2.99792458e10
c_red_cgs = 1.0e-3 * c_cgs
if not np.isclose(unit_velocity_phys, c_red_cgs, rtol=1e-12):
    raise ValueError(f"UVEL must equal reduced light speed {c_red_cgs:.8e} cm/s")

source_rate_phys = float(SRC)
tphys_q = parse_quantity(TPHYS, "time")
t_phys = float(tphys_q.cgs_value)
time_axis_unit = tphys_q.unit
time_axis_scale = up.unit_factor_to_cgs(time_axis_unit, expected_dim="time")

rho_ambient_cgs = N_H_CGS * 1.6726219e-24
mass_unit_phys_cgs = rho_ambient_cgs * unit_length_phys_cgs**3

cu = CodeUnits.from_config(
    {
        "length": f"{unit_length_phys_cgs} cm",
        "mass": f"{mass_unit_phys_cgs} g",
        "velocity": f"{c_red_cgs} cm/s",
    },
    {"gamma": 5.0 / 3.0, "mu": 1.0},
)

light_speed_code = 1.0
time_code = t_phys / cu.T_cgs
source_rate_code = source_rate_phys * cu.T_cgs

cfl_code = 0.4
dt_cfl = cfl_code / (3.0 * light_speed_code / dx_code)
max_dt = MAXDT if MAXDT is not None else dt_cfl
n_steps_est = int(math.ceil(time_code / dt_cfl))
n_super_step = NSTEP

run_tag = (
    f"N{size_shape}_ulen{sanitize_tag(unit_length_str)}"
    f"_box{sanitize_tag(box_width_str)}"
    f"_uvel{sanitize_tag(unit_velocity_str)}"
    f"_src{source_rate_phys:.2e}phs_t{sanitize_tag(tphys_q.value.__str__() + tphys_q.unit)}"
)

print("Backend:", jax.default_backend(), jax.devices())
print("=" * 70)
print(f" ULEN = {unit_length_phys_cgs:.6e} cm")
print(f" BOX = {box_width_phys_cgs:.6e} cm")
print(f" UVEL = {unit_velocity_phys:.6e} cm/s")
print(f" dx_code = {dx_code:.6e}")
print(f" dx_phys_cgs = {dx_phys_cgs:.6e} cm")
print(f" L_cgs = {cu.L_cgs:.6e} cm")
print(f" V_cgs = {cu.V_cgs:.6e} cm/s")
print(f" T_cgs = {cu.T_cgs:.6e} s")
print(f" Temp_cgs = {cu.Temp_cgs:.6e} K")
print(f" time_code = {time_code:.6e}")
print(f" source_rate_code = {source_rate_code:.6e}")
print(f" dt_cfl = {dt_cfl:.6e}")
print(f" max_dt = {max_dt:.6e}")
print(f" estimated n_steps = {n_steps_est} (n_super_step = {n_super_step})")
print("=" * 70)

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
eps_code = float(EPS)
eq_test = EquationManager_RT(
    light_speed=light_speed_code,
    mesh_shape=(size_shape, size_shape, size_shape),
    eps=eps_code,
    debug=False,
)
eq_test_hydro = dh.EquationManager(
    gamma=5.0 / 3.0,
    n_cons=6,
    passive_names=("x_HII",),
    mesh_shape=(size_shape, size_shape, size_shape),
    eps=eps_code,
)
assert abs(cfl_code - eq_test.cfl) < 1e-12

source_density_per_step = source_rate_code * dt_cfl / cell_volume_code
print(f" eps_code = {eps_code:.3e}")
print(f" source/step = {source_density_per_step:.6e} [ph/vol code]")

class StateBlockFlux:
    def __init__(self, base_flux, state_slice):
        self.base_flux = base_flux
        self.state_slice = state_slice
        self.dx_o = base_flux.dx_o

    def flux(self, sol, ax, params, flux):
        local = self.base_flux.flux(sol[self.state_slice], ax, params, flux)
        return jnp.zeros_like(sol).at[self.state_slice].set(local)

    def timestep(self, sol):
        return self.base_flux.timestep(sol[self.state_slice])

rt_flux = StateBlockFlux(
    dh.ConvectiveFlux_Radiative_transfer(
        eq_test,
        dh.HLL_Radiative_transfer_Local(eq_test, dh.signal_speed_Rusanov),
        dh.PLM(limiter="VANLEER"),
        dx=dx_code,
    ),
    slice(0, eq_test.n_cons),
)
hydro_flux = StateBlockFlux(
    dh.ConvectiveFlux(
        eq_test_hydro,
        dh.LaxFriedrichs(eq_test_hydro, dh.signal_speed_Rusanov),
        dh.PLM(limiter="VANLEER"),
        dx=dx_code,
    ),
    slice(eq_test.n_cons, eq_test.n_cons + eq_test_hydro.n_cons),
)

stellar_force = StellarRadiationForce(
    dx=dx_code,
    injection_mode="stromgren",
    stromgren_rate=source_rate_code,
    injection_momentum=False,
    injection_geometry="radial_3D",
    gaussian_star=True,
    beam_momentum_scaling="legacy_c2_source2",
    eq=eq_test,
    hydro_eq=eq_test_hydro,
    cu=cu,
    chemistry=False,
)
chem_force = HydrogenPhotoChemistryForce(
    stellar_force,
    case="B",
    collisional=False,
    max_frac=0.9,
    include_heating=False,
    include_cooling=False,
)

hydrosim_test = dh.hydro(
    n_super_step=n_super_step,
    fluxes=[hydro_flux, rt_flux],
    forces=[stellar_force, chem_force],
    dx=dx_code,
    max_dt=max_dt,
)
assert hydrosim_test.dx_o == rt_flux.dx_o == stellar_force.dx

# -----------------------------------------------------------------------------
# Initial condition and run
# -----------------------------------------------------------------------------
params = {
    "star_masses": jnp.array([1.0]),
    "star_ages": jnp.array([0.1]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions": jnp.array([[size_shape // 2] * 3], dtype=jnp.int32),
}

n_total_fields = eq_test.n_cons + eq_test_hydro.n_cons
sol_test = jnp.zeros(
    (n_total_fields, size_shape, size_shape, size_shape), dtype=jnp.float64
)
center = size_shape // 2
idx_rho_local = eq_test.n_cons + eq_test_hydro.mass_ids
idx_p_local = eq_test.n_cons + eq_test_hydro.energy_ids

mH_cgs = 1.6726219e-24
kB_cgs = 1.380649e-16
rho_ambient_code = rho_ambient_cgs / cu.rho_cgs
p_ambient_cgs = rho_ambient_cgs * kB_cgs * T_AMBIENT_K / (1.0 * mH_cgs)
p_ambient_code = p_ambient_cgs / cu.P_cgs

sol_test = sol_test.at[idx_rho_local].set(rho_ambient_code)
sol_test = sol_test.at[idx_p_local].set(p_ambient_code)
sol_test = sol_test.at[0, center, center, center].set(1e-20)
sol_test = sol_test.at[9].set(0.0)

print(f"Running to t = {t_phys:.6e} s = {time_code:.6e} code")
field_test, params_final, _, dt_hist, n_steps = hydrosim_test.evolve_till_time(
    cp.deepcopy(sol_test), params, time_code
)
field_test = field_test.at[0].set(jnp.maximum(field_test[0], 0.0))


dt_hist = np.asarray(dt_hist)
dt_sum = float(dt_hist[dt_hist > 0].sum())
E3d = np.asarray(field_test[0], dtype=np.float64)
E_cell = E3d * cell_volume_code

print("Done.")
print(f" steps = {int(n_steps)}")
print(f" sum(dt) = {dt_sum:.6e} code = {dt_sum * cu.T_cgs:.6e} s")
print(f" target = {time_code:.6e} code")
print(f" E_gamma min/max/negative = {E3d.min():.6e} / {E3d.max():.6e} / {int(np.sum(E3d < 0.0))}")

# photons_in_box = float(E_cell.sum())
# photons_expect = float(source_rate_code * dt_sum)
# print(f" photons in box = {photons_in_box:.6e}")
# print(f" expected = {photons_expect:.6e}")
# print(f" source conservation ratio = {photons_in_box / max(photons_expect, 1e-300):.6e}")

photons_in_box = float(E_cell.sum())
photons_expected = float(source_rate_code * dt_sum)

print(f" photons in box = {photons_in_box:.6e}")
print(f" photons injected = {photons_expected:.6e}")
print(
    f" radiation-only ratio = "
    f"{photons_in_box / max(photons_expected, 1e-300):.6e}"
)
print(
    " NOTE: photons absorbed by chemistry are not included "
    "in photons_in_box."
)

xHII_abs_idx = getattr(stellar_force, "idx_xHII", None)
if xHII_abs_idx is None or field_test.shape[0] <= xHII_abs_idx:
    raise RuntimeError("x_HII field is unavailable")
xHII_3d = np.asarray(field_test[xHII_abs_idx], dtype=np.float64)
print(f" x_HII min/max/mean = {xHII_3d.min():.6e} / {xHII_3d.max():.6e} / {xHII_3d.mean():.6e}")
print(f" x_HII out of bounds = {int(np.sum((xHII_3d < -1e-9) | (xHII_3d > 1.0 + 1e-9)))}")



# -----------------------------------------------------------------------------
# Analytic Stromgren radius and interpolated numerical front
# -----------------------------------------------------------------------------
alpha_B_cgs = float(hchem.alpha_B_HII_cgs(T_AMBIENT_K))
t_rec_cgs = 1.0 / (alpha_B_cgs * N_H_CGS)
R_stromgren_cgs = (3.0 * source_rate_phys / (4.0 * np.pi * alpha_B_cgs * N_H_CGS**2)) ** (1.0 / 3.0)
R_I_t_cgs = R_stromgren_cgs * (-np.expm1(-t_phys / t_rec_cgs)) ** (1.0 / 3.0)

print("=" * 70)
print(f" alpha_B = {alpha_B_cgs:.6e} cm^3 s^-1")
print(f" R_stromgren = {R_stromgren_cgs:.6e} cm")
print(f" R_I(t) = {R_I_t_cgs:.6e} cm")

xx, yy, zz = np.meshgrid(
    np.arange(size_shape) - center,
    np.arange(size_shape) - center,
    np.arange(size_shape) - center,
    indexing="ij",
)
r_int = np.round(np.sqrt(xx**2 + yy**2 + zz**2)).astype(int)
r_vals = np.arange(r_int.max() + 1)
x_shell = np.array([
    np.mean(xHII_3d[r_int == r]) if np.any(r_int == r) else np.nan
    for r in r_vals
])

above = np.where(x_shell >= 0.5)[0]
if above.size == 0:
    print(" simulated interpolated front = NOT REACHED")
else:
    i1 = above[-1]
    if i1 >= len(r_vals) - 1:
        r_front_cells = float(r_vals[i1])
    else:
        r1, r2 = float(r_vals[i1]), float(r_vals[i1 + 1])
        x1, x2 = float(x_shell[i1]), float(x_shell[i1 + 1])
        r_front_cells = r1 if abs(x2 - x1) < 1e-14 else r1 + (0.5 - x1) * (r2 - r1) / (x2 - x1)
    r_front_cgs = r_front_cells * dx_phys_cgs
    print(f" simulated interpolated front = {r_front_cells:.6f} cells = {r_front_cgs:.6e} cm")
    print(f" ratio simulated / analytic = {r_front_cgs / R_I_t_cgs:.6f}")

# rho_cgs_3d = np.asarray(field_test[idx_rho_local], dtype=np.float64) * cu.rho_cgs
# nH_3d = rho_cgs_3d / 1.6726219e-24
rho_code_3d = np.asarray(field_test[idx_rho_local], dtype=np.float64)
rho_cgs_3d = rho_code_3d * cu.rho_cgs

nH_3d = rho_cgs_3d / 1.6726219e-24
ionized_atoms = float(np.sum(nH_3d * xHII_3d) * cell_volume_cm3)


print(f" ionized H atoms = {ionized_atoms:.6e}")
print(
    f" injected photons / ionized atoms = "
    f"{photons_expected / max(ionized_atoms, 1e-300):.6e}"
)
