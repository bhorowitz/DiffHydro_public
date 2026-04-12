"""
stage0_conventions.py
=====================
Stage 0: Convention freeze and IC sanity checks.

Answers the question: "Do my code units and conventions make sense before
any physics runs?"

Four simple tests (no evolution):
  1. Uniform gas box    – check min/max/mean of each field, histograms
  2. Single DM particle – check CIC deposit shape and PM gravity field
  3. Static DM-only NFW halo – check enclosed mass, COM, momentum
  4. Static gas sphere  – Gaussian density, check radial profiles

Outputs (written to merger/outputs/stage0/):
  conventions.yaml          – explicit table of code conventions
  scalars.txt               – min/max/mean of every IC field
  test1_uniform_histograms.png
  test2_single_particle.png
  test3_dm_halo_profiles.png
  test4_gas_sphere_profiles.png

Usage:
  conda run -n jax-gpu python merger/stage0_conventions.py
"""

import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy import units as us

# Make merger/ importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp

# Limit to 1 GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from merger.physical_pm_force import (
    PhysicalDMGravityForce,
    nfw_density, nfw_enclosed_mass,
    _cic_paint, G_PHYS,
)
from merger.halo_reference import HaloReference
import cluster_generator as cg

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(_HERE, "outputs", "stage0")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
Z = 0.295
COS = FlatLambdaCDM(H0=70.0, Om0=0.3)
RHOC = COS.critical_density(Z).to(us.M_sun / us.kpc**3).value   # Msun/kpc^3

M200 = 5.0e14   # Msun
CONC = 3.5
R200 = cg.find_overdensity_radius(M200, 200.0, z=Z)              # kpc
A_SCALE = R200 / CONC                                             # NFW scale radius kpc
RHOS = 200.0 * RHOC * CONC**3 / ((np.log(1.0 + CONC) - CONC / (1.0 + CONC)) * 3)

N_GRID = 64
L_BOX  = 4.0 * R200   # box for single-halo: ~6000 kpc
DX     = L_BOX / N_GRID
CENTER = np.array([L_BOX / 2.0] * 3)  # box center in kpc

N_PAR_STAGE0 = 4096   # small for Stage 0 checks

G = G_PHYS  # kpc^3 Msun^-1 Myr^-2

# Minimum gas temperature / pressure for background gas [code units = Msun kpc^-1 Myr^-2]
GAMMA = 5.0 / 3.0
# Background ICM-like temperature: ~1 keV ≈ 1.16e7 K
# cs ≈ sqrt(gamma * kB * T / (mu * mH)) with mu=0.59 (fully ionised plasma)
# ≈ sqrt(1.67 * 1.38e-23 * 1.16e7 / (0.59 * 1.67e-27)) m/s ≈ 1100 km/s ≈ 1.12 kpc/Myr
# p = rho * cs^2 / gamma → internal energy per volume = p/(gamma-1)
CS_BG   = 0.1          # kpc/Myr  (cold background, just to keep solver stable)
RHO_BG  = 1.0e-6 * RHOC  # Msun/kpc^3 (very low background)
P_BG    = RHO_BG * CS_BG**2 / GAMMA


# ===========================================================================
# Step 0: Write conventions YAML
# ===========================================================================

def write_conventions():
    conv = {
        "stage": 0,
        "description": "Physical-coordinate (non-cosmological) single-halo tests",
        "length_unit": "kpc",
        "mass_unit": "Msun",
        "time_unit": "Myr",
        "velocity_unit": "kpc/Myr  (1 km/s = 1.022e-3 kpc/Myr)",
        "G": f"{G_PHYS:.4e} kpc^3 Msun^-1 Myr^-2",
        "coordinate_type": "proper (physical) Cartesian",
        "density_variable": "physical mass density rho in Msun/kpc^3",
        "velocity_variable": "proper peculiar velocity in kpc/Myr",
        "pressure_variable": "thermal pressure in Msun kpc^-1 Myr^-2",
        "energy_variable": "total energy density E in Msun kpc^-1 Myr^-2",
        "gas_conservative_vars": {
            "U[0]": "rho  [Msun/kpc^3]",
            "U[1]": "rho*vx  [Msun kpc^-2 Myr^-1]",
            "U[2]": "rho*vy  [Msun kpc^-2 Myr^-1]",
            "U[3]": "rho*vz  [Msun kpc^-2 Myr^-1]",
            "U[4]": "E_total = rho*(0.5*v^2 + e_internal)  [Msun kpc^-1 Myr^-2]",
        },
        "particle_variables": {
            "pos": "3D position in kpc in [0, L_box)",
            "vel": "3D peculiar velocity in kpc/Myr",
            "m_par": "particle mass in Msun (equal-mass particles)",
        },
        "gravity": "periodic PM via FFT Poisson, explicit G, subtract_mean=True",
        "comoving": False,
        "scale_factor": "a = 1 (fixed, non-cosmological)",
        "redshift_interpretation": (
            f"z={Z} used only to compute rho_c(z) for overdensity radii; "
            "IC placed in fixed physical box, no Hubble flow."
        ),
        "box": {
            "N_grid": N_GRID,
            "L_box_kpc": float(f"{L_BOX:.1f}"),
            "dx_kpc": float(f"{DX:.3f}"),
            "center_kpc": list(CENTER.astype(float)),
        },
        "halo": {
            "M200_Msun": M200,
            "conc": CONC,
            "r200_kpc": float(f"{R200:.1f}"),
            "a_scale_kpc": float(f"{A_SCALE:.1f}"),
            "rho_s_Msun_per_kpc3": float(f"{RHOS:.4e}"),
            "redshift_for_rho_c": Z,
        },
    }
    path = os.path.join(OUT_DIR, "conventions.yaml")
    with open(path, "w") as f:
        yaml.dump(conv, f, default_flow_style=False, sort_keys=False)
    print(f"[Stage 0] Wrote {path}")
    return conv


# ===========================================================================
# Test 1: Uniform gas box
# ===========================================================================

def test1_uniform_gas():
    print("\n[Stage 0 / Test 1] Uniform gas box")
    N = N_GRID

    x = (np.arange(N) + 0.5) * DX  # kpc
    X, Y, Z_ = np.meshgrid(x, x, x, indexing="ij")

    rho = np.full((N, N, N), RHO_BG, dtype=np.float32)   # Msun/kpc^3
    vx  = np.zeros((N, N, N), dtype=np.float32)
    vy  = np.zeros_like(vx)
    vz  = np.zeros_like(vx)
    p   = np.full((N, N, N), P_BG, dtype=np.float32)
    e_int = p / ((GAMMA - 1.0) * rho)
    E   = rho * (0.5 * (vx**2 + vy**2 + vz**2) + e_int)

    fields = {"rho": rho, "vx": vx, "vy": vy, "vz": vz, "p": p, "E": E}
    scalar_rows = []

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Test 1: Uniform gas box — IC field histograms", fontsize=12)
    for ax, (name, arr) in zip(axes.flat, fields.items()):
        flat = arr.ravel()
        ax.hist(flat, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        mn, mx, me = flat.min(), flat.max(), flat.mean()
        ax.axvline(me, color="red", lw=1.5, linestyle="--", label=f"mean={me:.3e}")
        ax.legend(fontsize=7)
        scalar_rows.append(f"  {name:6s}: min={mn:.4e}  max={mx:.4e}  mean={me:.4e}")
        print(f"    {name}: min={mn:.4e}  max={mx:.4e}  mean={me:.4e}")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "test1_uniform_histograms.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  → {out}")
    return scalar_rows


# ===========================================================================
# Test 2: Single DM particle
# ===========================================================================

def test2_single_dm_particle():
    print("\n[Stage 0 / Test 2] Single DM particle — CIC deposit + PM gravity")
    N = N_GRID
    L = L_BOX

    # Place a single massive particle at box center
    m_single = M200   # Msun, use full halo mass in one particle
    pos = np.array([[L / 2.0, L / 2.0, L / 2.0]], dtype=np.float32)   # (1, 3)

    # CIC deposit
    pos_grid = (pos / DX) % N
    mesh = jnp.zeros((N, N, N), dtype=jnp.float32)
    rho_mesh = _cic_paint(mesh, jnp.array(pos_grid), m_single)
    rho_kpc = np.array(rho_mesh) / DX**3  # Msun/kpc^3

    # PM gravity
    force = PhysicalDMGravityForce(N, L, G=G, subtract_mean=True)
    ax, ay, az = force.compute_accel_field(
        jnp.array(pos, dtype=jnp.float32), m_single
    )
    ax = np.array(ax)
    ay = np.array(ay)
    az = np.array(az)
    amag = np.sqrt(ax**2 + ay**2 + az**2)  # kpc/Myr^2

    # Analytic radial acceleration from point mass (for comparison)
    x_cells = (np.arange(N) + 0.5) * DX
    X3, Y3, Z3 = np.meshgrid(x_cells, x_cells, x_cells, indexing="ij")
    r3 = np.sqrt((X3 - L / 2) ** 2 + (Y3 - L / 2) ** 2 + (Z3 - L / 2) ** 2)
    r3 = np.maximum(r3, 0.5 * DX)
    a_analytic = G * m_single / r3**2   # kpc/Myr^2

    # Ratio PM/analytic on a radial slice
    r_flat = r3.ravel()
    amag_flat = amag.ravel()
    a_anal_flat = a_analytic.ravel()

    # Sort by radius for plotting
    idx_sort = np.argsort(r_flat)
    r_sorted = r_flat[idx_sort]
    amag_sorted = amag_flat[idx_sort]
    aanal_sorted = a_anal_flat[idx_sort]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Test 2: Single DM particle", fontsize=11)

    # Density slice (mid-z plane)
    im = axes[0].imshow(
        np.log10(rho_kpc[:, :, N // 2] + 1e-30),
        origin="lower", cmap="hot",
        extent=[0, L, 0, L],
    )
    axes[0].set_title(r"log$_{10}$(CIC $\rho$) [Msun/kpc$^3$]  z-mid slice")
    axes[0].set_xlabel("x [kpc]"); axes[0].set_ylabel("y [kpc]")
    plt.colorbar(im, ax=axes[0])

    # Acceleration magnitude slice
    im2 = axes[1].imshow(
        np.log10(amag[:, :, N // 2] + 1e-30),
        origin="lower", cmap="viridis",
        extent=[0, L, 0, L],
    )
    axes[1].set_title(r"log$_{10}$(|$\mathbf{a}$|) [kpc/Myr$^2$]  z-mid slice")
    axes[1].set_xlabel("x [kpc]"); axes[1].set_ylabel("y [kpc]")
    plt.colorbar(im2, ax=axes[1])

    # Radial profile: PM vs analytic (1/r^2)
    # Bin by radius
    r_bins = np.logspace(np.log10(DX), np.log10(L / 2), 40)
    r_cen = 0.5 * (r_bins[:-1] + r_bins[1:])
    amag_bin, _ = np.histogram(r_sorted, bins=r_bins, weights=amag_sorted)
    aanal_bin, _ = np.histogram(r_sorted, bins=r_bins, weights=aanal_sorted)
    count_bin, _ = np.histogram(r_sorted, bins=r_bins)
    mask = count_bin > 0
    amag_mean = amag_bin[mask] / count_bin[mask]
    aanal_mean = aanal_bin[mask] / count_bin[mask]
    r_plot = r_cen[mask]

    axes[2].loglog(r_plot, amag_mean, "o-", ms=4, label="PM gravity")
    axes[2].loglog(r_plot, aanal_mean, "k--", lw=2, label="GM/r² analytic")
    axes[2].set_xlabel("r [kpc]"); axes[2].set_ylabel(r"$|a|$ [kpc/Myr²]")
    axes[2].set_title("Radial acceleration: PM vs analytic")
    axes[2].legend()
    axes[2].axvline(DX, color="gray", linestyle=":", label=f"dx={DX:.0f} kpc")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "test2_single_particle.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  → {out}")

    # Scalar summary
    rows = [
        f"  total_deposited_mass: {np.sum(np.array(rho_mesh)):.4e} Msun (input {m_single:.4e})",
        f"  peak_rho_cell: {rho_kpc.max():.4e} Msun/kpc^3",
        f"  peak_accel: {amag.max():.4e} kpc/Myr^2",
        f"  accel_ratio_pm_over_analytic (median, r>2*dx): "
        f"{np.median(amag_flat[r_flat > 2*DX] / a_anal_flat[r_flat > 2*DX]):.3f}",
    ]
    for r in rows:
        print(r)
    return rows


# ===========================================================================
# Test 3: Static DM-only NFW halo
# ===========================================================================

def test3_dm_halo():
    print(f"\n[Stage 0 / Test 3] Static DM-only NFW halo (N_par={N_PAR_STAGE0})")

    ref = HaloReference(z=Z, m200_msun=M200, conc=CONC, r_grid_max=3.0 * R200)
    hse1 = ref.hse

    # Generate DM particles (centered at origin)
    np.random.seed(42)
    particles = hse1.generate_dm_particles(N_PAR_STAGE0, r_max=2.5*R200)
    pos_raw = np.array(particles[("dm", "particle_position")])  # kpc, centred at 0
    vel_raw = np.array(particles[("dm", "particle_velocity")])  # kpc/Myr
    m_par   = float(np.array(particles[("dm", "particle_mass")])[0])

    # Shift to box center
    pos = pos_raw + CENTER[None, :]  # kpc, in [0, L_box)

    # --- Metrics ---
    r_par = np.sqrt(np.sum((pos - CENTER[None, :])**2, axis=1))  # radii from halo center

    # Total mass
    M_total_par = N_PAR_STAGE0 * m_par
    M200_total = nfw_enclosed_mass(R200, RHOS, A_SCALE)
    M200_dm = float(ref.mass_dm_at([R200])[0])
    M_rmax_dm = float(ref.mass_dm_at([2.5 * R200])[0])
    print(f"  Particle total mass: {M_total_par:.4e} Msun")
    print(f"  DM target M(<r200>): {M200_dm:.4e} Msun")
    print(f"  DM target M(<r_max>): {M_rmax_dm:.4e} Msun")
    print(f"  Ratio to DM M(<r_max>): {M_total_par / M_rmax_dm:.4f}")

    # COM
    com = np.mean(pos, axis=0)
    com_offset_kpc = np.linalg.norm(com - CENTER)
    com_offset_frac = com_offset_kpc / R200
    print(f"  COM offset: {com_offset_kpc:.1f} kpc = {com_offset_frac:.4f} r200")

    # Net momentum
    p_net = np.sum(vel_raw * m_par, axis=0)
    v_com = p_net / M_total_par
    print(f"  Net v_COM: ({v_com[0]:.4e}, {v_com[1]:.4e}, {v_com[2]:.4e}) kpc/Myr")

    # Particles outside r_max
    r_max_test = 2.5 * R200
    n_outside = np.sum(r_par > r_max_test)
    print(f"  Particles outside r_max={r_max_test:.0f} kpc: {n_outside}/{N_PAR_STAGE0}")

    # Enclosed mass profile
    r_bins = np.logspace(np.log10(0.01 * R200), np.log10(3.0 * R200), 50)
    r_cen  = 0.5 * (r_bins[:-1] + r_bins[1:])
    M_enc_par = np.array([N_PAR_STAGE0 * m_par * np.mean(r_par < r) for r in r_cen])
    M_enc_dm = ref.mass_dm_at(r_cen)
    M_enc_total = nfw_enclosed_mass(r_cen, RHOS, A_SCALE)

    # Radial number density
    dr = np.diff(r_bins)
    counts, _ = np.histogram(r_par, bins=r_bins)
    vol_shells = 4.0 / 3.0 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    n_density = counts / vol_shells   # par/kpc^3

    # --- Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Test 3: Static NFW DM halo  (N_par={N_PAR_STAGE0})", fontsize=12)

    # Enclosed mass
    ax = axes[0, 0]
    ax.loglog(r_cen, M_enc_par / M200, "o-", ms=3, label="Particles")
    ax.loglog(r_cen, M_enc_dm / M200, "k--", lw=2, label="DM target")
    ax.loglog(r_cen, M_enc_total / M200, color="gray", ls=":", lw=1.5, label="Total NFW")
    ax.axvline(R200, color="gray", ls=":", label=f"r200={R200:.0f} kpc")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel("M(<r) / M200")
    ax.set_title("Enclosed mass"); ax.legend()

    # Fractional error in enclosed mass
    ax = axes[0, 1]
    good = M_enc_dm > 0
    frac_err = (M_enc_par[good] - M_enc_dm[good]) / M_enc_dm[good]
    ax.semilogx(r_cen[good], frac_err, "o-", ms=3)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(0.05, color="gray", ls="--", label="±5%")
    ax.axhline(-0.05, color="gray", ls="--")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel("ΔM/M")
    ax.set_title("Fractional mass error"); ax.legend()

    # Radial number density
    ax = axes[0, 2]
    ax.loglog(r_cen, n_density + 1e-30, "o-", ms=3)
    ax.set_xlabel("r [kpc]"); ax.set_ylabel("n [par/kpc³]")
    ax.set_title("Particle radial number density")

    # Velocity histogram (each component)
    ax = axes[1, 0]
    for comp, label in zip(vel_raw.T, ["vx", "vy", "vz"]):
        ax.hist(comp, bins=60, histtype="step", label=label)
    ax.set_xlabel("velocity [kpc/Myr]"); ax.set_ylabel("count")
    ax.set_title("Velocity components"); ax.legend()

    # 2D projection of particle positions
    ax = axes[1, 1]
    ax.scatter(pos[:, 0], pos[:, 1], s=0.5, alpha=0.3, c="steelblue")
    ax.set_xlabel("x [kpc]"); ax.set_ylabel("y [kpc]")
    ax.set_title("Particle positions (xy projection)")
    circle = plt.Circle(CENTER[:2], R200, color="red", fill=False, lw=1.5, label="r200")
    ax.add_patch(circle)
    ax.set_aspect("equal"); ax.legend()

    # Density profile from particles vs NFW
    ax = axes[1, 2]
    rho_par = counts * m_par / vol_shells  # Msun/kpc^3
    ax.loglog(r_cen, rho_par + 1e-30, "o-", ms=3, label="Particles")
    rho_dm = ref.rho_dm_at(r_cen)
    rho_nfw = nfw_density(r_cen, RHOS, A_SCALE)
    ax.loglog(r_cen, rho_dm, "k--", lw=2, label="DM target")
    ax.loglog(r_cen, rho_nfw, color="gray", ls=":", lw=1.5, label="Total NFW")
    ax.axvline(A_SCALE, color="gray", ls=":", label=f"a={A_SCALE:.0f} kpc")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$\rho$ [Msun/kpc³]")
    ax.set_title("Density profile"); ax.legend()

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "test3_dm_halo_profiles.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  → {out}")

    scalar_rows = [
        f"  M_total_particles: {M_total_par:.4e} Msun",
        f"  M200_total_analytic: {M200_total:.4e} Msun",
        f"  M200_dm_target: {M200_dm:.4e} Msun",
        f"  M_rmax_dm_target: {M_rmax_dm:.4e} Msun",
        f"  mass_ratio_to_dm_rmax: {M_total_par / M_rmax_dm:.4f}",
        f"  COM_offset_kpc: {com_offset_kpc:.2f}",
        f"  COM_offset_over_r200: {com_offset_frac:.4f}",
        f"  v_COM_kpc_per_Myr: ({v_com[0]:.3e}, {v_com[1]:.3e}, {v_com[2]:.3e})",
        f"  n_particles_outside_rmax: {n_outside}",
    ]
    return scalar_rows


# ===========================================================================
# Test 4: Static gas sphere (Gaussian density profile)
# ===========================================================================

def test4_gas_sphere():
    print("\n[Stage 0 / Test 4] Static gas sphere (Gaussian density)")
    N = N_GRID

    x = (np.arange(N) + 0.5) * DX  # kpc
    X3, Y3, Z3 = np.meshgrid(x, x, x, indexing="ij")
    r3 = np.sqrt((X3 - CENTER[0])**2 + (Y3 - CENTER[1])**2 + (Z3 - CENTER[2])**2)

    sigma = 0.3 * R200   # width of Gaussian ~400 kpc
    rho_peak = 1.0e-2 * RHOC  # peak density Msun/kpc^3

    rho = RHO_BG + rho_peak * np.exp(-r3**2 / (2.0 * sigma**2))
    rho = rho.astype(np.float32)

    # Isothermal temperature: T ~ const → p = cs² * rho
    cs = 1.0  # kpc/Myr  (warm cluster gas)
    p  = rho * cs**2 / GAMMA
    e_int = p / ((GAMMA - 1.0) * rho)
    E   = rho * e_int   # zero velocity

    fields = {"rho": rho, "p": p, "e_int": e_int, "E": E}

    # Radial profiles
    r_bins = np.logspace(np.log10(DX), np.log10(L_BOX / 2), 50)
    r_cen  = 0.5 * (r_bins[:-1] + r_bins[1:])
    r_flat = r3.ravel()
    idx_sort = np.argsort(r_flat)
    r_sorted = r_flat[idx_sort]

    def radial_mean(arr):
        vals, _ = np.histogram(r_sorted, bins=r_bins, weights=arr.ravel()[idx_sort])
        cnts, _ = np.histogram(r_sorted, bins=r_bins)
        out = np.full_like(r_cen, np.nan, dtype=np.float64)
        mask = cnts > 0
        out[mask] = vals[mask] / cnts[mask]
        return out

    rho_prof = radial_mean(rho)
    p_prof   = radial_mean(p)
    T_prof   = radial_mean(p / rho)   # ∝ kT/m (kpc²/Myr²)

    rho_analytic = RHO_BG + rho_peak * np.exp(-r_cen**2 / (2.0 * sigma**2))
    p_analytic   = rho_analytic * cs**2 / GAMMA

    # Scalar summary
    scalar_rows = []
    for name, arr in fields.items():
        flat = arr.ravel()
        scalar_rows.append(
            f"  gas_{name}: min={flat.min():.4e}  max={flat.max():.4e}  mean={flat.mean():.4e}"
        )
        print(f"    {name}: min={flat.min():.4e}  max={flat.max():.4e}  mean={flat.mean():.4e}")

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Test 4: Static gas sphere — IC profiles", fontsize=12)

    # Density slice
    ax = axes[0, 0]
    im = ax.imshow(
        np.log10(rho[:, :, N // 2] + 1e-30),
        origin="lower", cmap="hot",
        extent=[0, L_BOX, 0, L_BOX],
    )
    ax.set_title(r"$\log_{10}(\rho)$ z-mid slice"); ax.set_xlabel("x [kpc]"); ax.set_ylabel("y [kpc]")
    plt.colorbar(im, ax=ax, label="Msun/kpc³")

    # Pressure slice
    ax = axes[0, 1]
    im = ax.imshow(
        np.log10(p[:, :, N // 2] + 1e-30),
        origin="lower", cmap="plasma",
        extent=[0, L_BOX, 0, L_BOX],
    )
    ax.set_title(r"$\log_{10}(p)$ z-mid slice"); ax.set_xlabel("x [kpc]"); ax.set_ylabel("y [kpc]")
    plt.colorbar(im, ax=ax)

    # Density profile
    ax = axes[0, 2]
    ax.semilogy(r_cen, rho_prof, "o-", ms=3, label="grid profile")
    ax.semilogy(r_cen, rho_analytic, "k--", lw=2, label="analytic Gaussian")
    ax.axvline(sigma, color="gray", ls=":", label=f"σ={sigma:.0f} kpc")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$\rho$ [Msun/kpc³]")
    ax.set_title("Density radial profile"); ax.legend()

    # Pressure profile
    ax = axes[1, 0]
    ax.semilogy(r_cen, p_prof, "o-", ms=3, label="grid")
    ax.semilogy(r_cen, p_analytic, "k--", lw=2, label="analytic")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel("p [Msun kpc⁻¹ Myr⁻²]")
    ax.set_title("Pressure radial profile"); ax.legend()

    # T proxy (p/rho)
    ax = axes[1, 1]
    ax.plot(r_cen, T_prof, "o-", ms=3)
    ax.axhline(cs**2 / GAMMA, color="k", ls="--", label=f"expected {cs**2/GAMMA:.4f} kpc²/Myr²")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$p/\rho$ [kpc²/Myr²]")
    ax.set_title("Temperature proxy p/ρ"); ax.legend()

    # Histogram of rho
    ax = axes[1, 2]
    ax.hist(rho.ravel(), bins=80, color="steelblue", log=True)
    ax.set_xlabel(r"$\rho$ [Msun/kpc³]"); ax.set_ylabel("count (log)")
    ax.set_title("Density histogram")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "test4_gas_sphere_profiles.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  → {out}")

    return scalar_rows


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("Stage 0: Convention freeze and IC sanity checks")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    print(f"N_grid={N_GRID}, L_box={L_BOX:.0f} kpc, dx={DX:.1f} kpc")
    print(f"M200={M200:.2e} Msun, r200={R200:.0f} kpc")

    conv = write_conventions()

    all_rows = ["=" * 60, "Stage 0 Scalar Diagnostics", "=" * 60, ""]

    all_rows += ["--- Test 1: Uniform gas box ---"]
    all_rows += test1_uniform_gas()

    all_rows += ["", "--- Test 2: Single DM particle ---"]
    all_rows += test2_single_dm_particle()

    all_rows += ["", "--- Test 3: Static NFW DM halo ---"]
    all_rows += test3_dm_halo()

    all_rows += ["", "--- Test 4: Static gas sphere ---"]
    all_rows += test4_gas_sphere()

    # Write scalar summary
    scalar_path = os.path.join(OUT_DIR, "scalars.txt")
    with open(scalar_path, "w") as f:
        f.write("\n".join(all_rows) + "\n")
    print(f"\n[Stage 0] Wrote {scalar_path}")

    print("\n[Stage 0] DONE. Files written:")
    for fname in sorted(os.listdir(OUT_DIR)):
        print(f"  merger/outputs/stage0/{fname}")


if __name__ == "__main__":
    main()
