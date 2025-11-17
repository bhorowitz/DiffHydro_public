
import numpy as np
import jax
import jax.numpy as jnp

import diffhydro as dh
from diffhydro.physics.turbulence import init_turbulent_velocity,TurbulentForce  # :contentReference[oaicite:1]{index=1}

def test_turbulence_ic_zero_net_momentum_and_reasonable_mach():
    shape = (32, 32, 32)
    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = list(shape)
    eq.box_size = (1.0, 1.0, 1.0)

    rho0 = 1.0
    p0 = 1.0
    U = init_turbulent_velocity(
        eq,
        Lbox=1.0,
        rho0=rho0,
        p0=p0,
        target_M=1.0,
        kmin=1,
        kmax=3,
    )

    # 5 = (rho, mx, my, mz, E)
    assert U.shape == (5,) + shape

    # ---- bulk momentum sanity: v_bulk << u_rms ----
    rho = np.asarray(U[0])
    mom = np.asarray(U[1:4])  # mx, my, mz

    total_mom = mom.sum(axis=(1, 2, 3))
    total_mass = rho.sum()

    v_bulk = total_mom / total_mass                      # 3-vector
    vel = mom / rho                                      # (3, nx, ny, nz)
    u_rms = np.sqrt((vel**2).sum(axis=0).mean())         # scalar RMS speed

    # Require bulk drift to be tiny compared to turbulent speeds
    assert np.linalg.norm(v_bulk) < 1e-3 * u_rms, (
        f"bulk speed too large: |v_bulk|={np.linalg.norm(v_bulk)}, u_rms={u_rms}"
    )

    # ---- Mach number sanity, as before ----
    W = eq.get_primitives_from_conservatives(U)
    vel_prim = np.asarray(W[eq.vel_ids])                # (3, ...)
    rho_prim = np.asarray(W[eq.mass_ids])
    P_prim = np.asarray(W[eq.energy_ids])               # pressure for primitives

    cs = np.asarray(eq.get_speed_of_sound(P_prim, rho_prim))
    speed = np.sqrt((vel_prim**2).sum(axis=0))
    M_rms = np.sqrt((speed**2).mean()) / np.sqrt((cs**2).mean())

    assert 0.3 < M_rms < 5.0, f"RMS Mach looks weird: {M_rms}"

def test_turbulent_force_band_limited():

    shape = (16, 16, 16)
    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = list(shape)
    eq.box_size = (1.0, 1.0, 1.0)

    tf = TurbulentForce(eq, kmin=2.0, kmax=4.0, rms_accel=1.0, tau_corr=0.5)

    _ = tf._step_accel_k(0.1)  # advances self.accel_k

    accel_k = np.asarray(tf.accel_k)
    K = np.asarray(tf.K)
    kmin_abs = tf.kmin * (2*np.pi / tf.Lx)
    kmax_abs = tf.kmax * (2*np.pi / tf.Lx)
    band = (K >= kmin_abs) & (K <= kmax_abs)

    # Outside band, everything should be numerically tiny
    outside = np.logical_not(band)
    max_inside = np.max(np.abs(accel_k[:, band]))
    max_outside = np.max(np.abs(accel_k[:, outside]))
    assert max_outside < 1e-6 * max_inside

def test_turbulent_force_solenoidal_projection():
    import diffhydro as dh
    from diffhydro.physics.turbulence import TurbulentForce
    import numpy as np

    shape = (16, 16, 16)
    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = list(shape)
    eq.box_size = (1.0, 1.0, 1.0)

    tf = TurbulentForce(eq, solenoidal_fraction=1.0)

    _ = tf._step_accel_k(0.1)
    ak = np.asarray(tf.accel_k)       # shape (3, nx, ny, nz)
    KX, KY, KZ = map(np.asarray, (tf.KX, tf.KY, tf.KZ))
    nonzero = np.asarray(tf.nonzero_mask)

    dot = KX*ak[0] + KY*ak[1] + KZ*ak[2]
    dot = dot[nonzero]

    mag2 = (ak[0]**2 + ak[1]**2 + ak[2]**2)[nonzero]

    # RMS of k·a divided by RMS(|a|)
    rel = np.sqrt(np.mean(np.abs(dot)**2)) / np.sqrt(np.mean(mag2))
    assert rel < 1e-2, f"Solenoidal forcing has too much divergence: rel={rel}"

def _corr(a, b):
    import numpy as np
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_turbulent_force_OU_time_correlation():

    shape = (8, 8, 8)
    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = list(shape)
    eq.box_size = (1.0, 1.0, 1.0)

    tau = 1.0
    tf = TurbulentForce(eq, tau_corr=tau, rms_accel=1.0, seed=0)

    # --- small dt regime ---
    dt_small = 0.01
    for _ in range(50):  # warm up
        _ = tf._step_accel_k(dt_small)
    a1 = tf._step_accel_k(dt_small)
    a2 = tf._step_accel_k(dt_small)
    c_small = _corr(a1, a2)

    # --- reset and use large dt ---
    tf.accel_k = jnp.zeros_like(tf.accel_k)
    dt_large = 1.0
    for _ in range(10):
        _ = tf._step_accel_k(dt_large)
    b1 = tf._step_accel_k(dt_large)
    b2 = tf._step_accel_k(dt_large)
    c_large = _corr(b1, b2)

    assert c_small > 0.7, f"Small-dt OU correlation too weak: {c_small}"
    assert c_large < 0.7, f"Large-dt OU correlation not reduced: {c_large}"
    assert c_small > c_large + 0.1

def test_turbulent_force_no_net_momentum_injection():

    shape = (16, 16, 16)
    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = list(shape)
    eq.box_size = (1.0, 1.0, 1.0)

    # Simple uniform state
    rho0, p0 = 1.0, 1.0
    U = jnp.zeros((5,) + shape)
    U = U.at[0].set(rho0)
    U = U.at[-1].set(p0 / (eq.gamma - 1.0))  # zero velocity, all thermal

    tf = TurbulentForce(eq, rms_accel=1.0)
    U_new = tf.force(0, U, {}, 0.1)

    dU = np.asarray(U_new - U)
    d_mom = dU[1:4]  # mx,my,mz
    total = d_mom.reshape(3, -1).sum(axis=1)

    # Net momentum is tiny vs. typical cell momentum kick
    rms_cell = float(np.sqrt(np.mean(d_mom**2)))
    assert np.linalg.norm(total) < 1e-3 * rms_cell * np.prod(shape)



def test_driven_turbulence_preserves_small_bulk_drift():
    shape = (16, 16, 16)

    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = list(shape)
    eq.box_size = (1.0, 1.0, 1.0)

    # Euler hydro with HLLC
    ss = dh.signal_speed_Rusanov
    solver = dh.HLLC(equation_manager=eq, signal_speed=ss)
    cf = dh.ConvectiveFlux(eq, solver, dh.MUSCL3(limiter="VANLEER"))

    # Solenoidal turbulent forcing
    force = TurbulentForce(
        eq,
        kmin=1.0,
        kmax=3.0,
        solenoidal_fraction=1.0,
        tau_corr=0.5,
        rms_accel=1.0,
        seed=123,
    )

    hydrosim = dh.hydro(
        n_super_step=40,
        fluxes=[cf],
        forces=[force],
        use_mol=True,
        max_dt=0.1,
    )

    # Uniform static IC: rho=1, p=1, u=0
    rho0, p0 = 1.0, 1.0
    U0 = jnp.zeros((5,) + shape)
    U0 = U0.at[0].set(rho0)
    E_th = p0 / (eq.gamma - 1.0)
    U0 = U0.at[-1].set(E_th)

    final_fields, _ = hydrosim.evolve(U0, {})
    U_final = jnp.asarray(final_fields)

    # Go to primitives
    W = eq.get_primitives_from_conservatives(U_final)
    rho = np.asarray(W[eq.mass_ids])
    vx  = np.asarray(W[eq.vel_ids[0]])
    vy  = np.asarray(W[eq.vel_ids[1]])
    vz  = np.asarray(W[eq.vel_ids[2]])

    # Bulk velocity (mass-weighted)
    total_mass = rho.sum()
    v_bulk = np.array([
        (rho * vx).sum() / total_mass,
        (rho * vy).sum() / total_mass,
        (rho * vz).sum() / total_mass,
    ])
    v_bulk_mag = float(np.linalg.norm(v_bulk))

    # RMS turbulent speed
    speed2 = vx**2 + vy**2 + vz**2
    u_rms = float(np.sqrt(speed2.mean()))

    # Require bulk drift to be small compared to turbulent motions
    # (5% is pretty strict; you can loosen/tighten after seeing numbers)
    assert u_rms > 0.0, "u_rms is zero; forcing didn't inject any motion?"
    assert v_bulk_mag < 0.05 * u_rms, (
        f"Bulk drift too large relative to turbulence: "
        f"|v_bulk|={v_bulk_mag}, u_rms={u_rms}"
    )
