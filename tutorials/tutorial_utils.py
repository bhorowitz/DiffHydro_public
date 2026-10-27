"""Small helpers shared by the tutorial notebooks.

Each helper is derived and explained in the notebook that first needs it:
  exact Riemann solver     -> 03_euler_1d
  Adam / gradient clipping -> 02_autodiff_through_time
  1D Euler solver          -> 03_euler_1d
  Gaussian random fields   -> 05_decaying_turbulence_2d
  diffhydro helpers        -> 04_diffhydro_intro / 05_decaying_turbulence_2d
Later notebooks simply import them from here.
"""
import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Exact Riemann solver for the 1D Euler equations (Toro, ch. 4)
# ---------------------------------------------------------------------------
def _f_and_df(p, rhoK, pK, cK, gamma):
    """Toro's pressure function f_K(p) and its derivative for one side K."""
    if p > pK:  # shock
        A = 2.0 / ((gamma + 1.0) * rhoK)
        B = (gamma - 1.0) / (gamma + 1.0) * pK
        f = (p - pK) * np.sqrt(A / (p + B))
        df = np.sqrt(A / (B + p)) * (1.0 - 0.5 * (p - pK) / (B + p))
    else:  # rarefaction
        f = 2.0 * cK / (gamma - 1.0) * ((p / pK) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        df = 1.0 / (rhoK * cK) * (p / pK) ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, df


def riemann_star(rhoL, uL, pL, rhoR, uR, pR, gamma=1.4, tol=1e-12):
    """Star-region pressure and velocity (p*, u*) from Newton iteration on f_L + f_R + du = 0."""
    cL, cR = np.sqrt(gamma * pL / rhoL), np.sqrt(gamma * pR / rhoR)
    p = max(tol, 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (cL + cR))
    for _ in range(100):
        fL, dfL = _f_and_df(p, rhoL, pL, cL, gamma)
        fR, dfR = _f_and_df(p, rhoR, pR, cR, gamma)
        p_new = max(tol, p - (fL + fR + uR - uL) / (dfL + dfR))
        if abs(p_new - p) < tol * 0.5 * (p_new + p):
            p = p_new
            break
        p = p_new
    fL, _ = _f_and_df(p, rhoL, pL, cL, gamma)
    fR, _ = _f_and_df(p, rhoR, pR, cR, gamma)
    return p, 0.5 * (uL + uR) + 0.5 * (fR - fL)


def exact_riemann(rhoL, uL, pL, rhoR, uR, pR, x, t, gamma=1.4, x0=0.5):
    """Exact solution (rho, u, p) of a Riemann problem at positions x and time t > 0."""
    g = gamma
    cL, cR = np.sqrt(g * pL / rhoL), np.sqrt(g * pR / rhoR)
    ps, us = riemann_star(rhoL, uL, pL, rhoR, uR, pR, g)
    gm, gp = (g - 1.0) / (g + 1.0), 2.0 / (g + 1.0)
    out = np.zeros((3, np.size(x)))
    for i, S in enumerate((np.atleast_1d(x) - x0) / t):
        if S <= us:  # left of the contact
            if ps > pL:  # left shock
                SL = uL - cL * np.sqrt((g + 1) / (2 * g) * ps / pL + (g - 1) / (2 * g))
                W = (rhoL, uL, pL) if S <= SL else (rhoL * (ps / pL + gm) / (gm * ps / pL + 1), us, ps)
            else:  # left rarefaction
                SHL, STL = uL - cL, us - cL * (ps / pL) ** ((g - 1) / (2 * g))
                if S <= SHL:
                    W = (rhoL, uL, pL)
                elif S > STL:
                    W = (rhoL * (ps / pL) ** (1 / g), us, ps)
                else:
                    fac = gp + gm / cL * (uL - S)
                    W = (rhoL * fac ** (2 / (g - 1)), gp * (cL + 0.5 * (g - 1) * uL + S),
                         pL * fac ** (2 * g / (g - 1)))
        else:  # right of the contact
            if ps > pR:  # right shock
                SR = uR + cR * np.sqrt((g + 1) / (2 * g) * ps / pR + (g - 1) / (2 * g))
                W = (rhoR, uR, pR) if S >= SR else (rhoR * (ps / pR + gm) / (gm * ps / pR + 1), us, ps)
            else:  # right rarefaction
                SHR, STR = uR + cR, us + cR * (ps / pR) ** ((g - 1) / (2 * g))
                if S >= SHR:
                    W = (rhoR, uR, pR)
                elif S <= STR:
                    W = (rhoR * (ps / pR) ** (1 / g), us, ps)
                else:
                    fac = gp - gm / cR * (uR - S)
                    W = (rhoR * fac ** (2 / (g - 1)), gp * (-cR + 0.5 * (g - 1) * uR + S),
                         pR * fac ** (2 * g / (g - 1)))
        out[:, i] = W
    return out[0], out[1], out[2]


# ---------------------------------------------------------------------------
# Optimisation: a minimal Adam on pytrees (no optax needed)
# ---------------------------------------------------------------------------
def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": 0}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam update. Returns (new_params, new_state)."""
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    mhat_scale, vhat_scale = 1.0 / (1 - b1 ** t), 1.0 / (1 - b2 ** t)
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * (m * mhat_scale) / (jnp.sqrt(v * vhat_scale) + eps), params, m, v)
    return params, {"m": m, "v": v, "t": t}


def global_norm(tree):
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(tree)))


def clip_by_global_norm(grads, max_norm):
    scale = jnp.minimum(1.0, max_norm / (global_norm(grads) + 1e-30))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


# ---------------------------------------------------------------------------
# 2D Fourier tools: wavenumber grids, Gaussian random fields, power spectra
# ---------------------------------------------------------------------------
def k_grid_2d(n, boxsize=1.0):
    """Physical wavenumbers (kx, ky) on the fft2 grid of an n x n periodic box."""
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=boxsize / n)
    return np.meshgrid(k1, k1, indexing="ij")


def gaussian_random_field_2d(n, pk, seed=0, boxsize=1.0):
    """Unit-variance Gaussian random field with power spectrum shape pk(k) (k=0 removed)."""
    kx, ky = k_grid_2d(n, boxsize)
    k = np.sqrt(kx ** 2 + ky ** 2)
    amp = np.where(k > 0, np.sqrt(pk(np.where(k > 0, k, 1.0))), 0.0)
    amp /= np.sqrt(np.mean(amp ** 2))
    w = np.random.default_rng(seed).standard_normal((n, n))
    return np.fft.ifft2(np.fft.fft2(w) * amp).real


def power_spectrum_2d(field, boxsize=1.0):
    """Azimuthally averaged power spectrum. Returns (k, P(k)) in integer-k shells."""
    field = np.asarray(field)
    n = field.shape[0]
    fk = np.fft.fft2(field - field.mean()) / n ** 2
    kx, ky = k_grid_2d(n, boxsize)
    kint = np.rint(np.sqrt(kx ** 2 + ky ** 2) * boxsize / (2 * np.pi)).astype(int)
    pk = np.bincount(kint.ravel(), weights=(np.abs(fk) ** 2).ravel())
    nk = np.bincount(kint.ravel())
    kk = np.arange(len(pk))
    good = (nk > 0) & (kk > 0) & (kk <= n // 2)
    return kk[good] * 2 * np.pi / boxsize, pk[good] / nk[good]


# ---------------------------------------------------------------------------
# The hand-written 1D Euler solver of notebook 03 (HLL/HLLC/Rusanov + MUSCL + SSP-RK2,
# outflow boundaries). W = (rho, u, p) primitives, U = (rho, rho u, E) conservatives.
# ---------------------------------------------------------------------------
def prim_to_cons(W, gamma):
    rho, u, p = W
    return jnp.stack([rho, rho * u, p / (gamma - 1) + 0.5 * rho * u ** 2])

def cons_to_prim(U, gamma):
    rho = U[0]; u = U[1] / rho
    return jnp.stack([rho, u, (gamma - 1) * (U[2] - 0.5 * rho * u ** 2)])

def phys_flux(W, gamma):
    rho, u, p = W
    E = p / (gamma - 1) + 0.5 * rho * u ** 2
    return jnp.stack([rho * u, rho * u ** 2 + p, (E + p) * u])

def rusanov(WL, WR, gamma):
    cL, cR = jnp.sqrt(gamma * WL[2] / WL[0]), jnp.sqrt(gamma * WR[2] / WR[0])
    s = jnp.maximum(jnp.abs(WL[1]) + cL, jnp.abs(WR[1]) + cR)
    return 0.5 * (phys_flux(WL, gamma) + phys_flux(WR, gamma)) - 0.5 * s * (prim_to_cons(WR, gamma) - prim_to_cons(WL, gamma))

def hll(WL, WR, gamma):
    cL, cR = jnp.sqrt(gamma * WL[2] / WL[0]), jnp.sqrt(gamma * WR[2] / WR[0])
    SL = jnp.minimum(WL[1] - cL, WR[1] - cR); SR = jnp.maximum(WL[1] + cL, WR[1] + cR)
    FL, FR = phys_flux(WL, gamma), phys_flux(WR, gamma)
    UL, UR = prim_to_cons(WL, gamma), prim_to_cons(WR, gamma)
    F_hll = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
    return jnp.where(SL >= 0, FL, jnp.where(SR <= 0, FR, F_hll))

def hllc(WL, WR, gamma):
    rL, uL, pL = WL; rR, uR, pR = WR
    cL, cR = jnp.sqrt(gamma * pL / rL), jnp.sqrt(gamma * pR / rR)
    SL = jnp.minimum(uL - cL, uR - cR); SR = jnp.maximum(uL + cL, uR + cR)
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))
    FL, FR = phys_flux(WL, gamma), phys_flux(WR, gamma)
    UL, UR = prim_to_cons(WL, gamma), prim_to_cons(WR, gamma)
    def star(U, r, u, p, S):
        fac = r * (S - u) / (S - Ss)
        return fac * jnp.stack([jnp.ones_like(r), Ss, U[2] / r + (Ss - u) * (Ss + p / (r * (S - u)))])
    FsL = FL + SL * (star(UL, rL, uL, pL, SL) - UL)
    FsR = FR + SR * (star(UR, rR, uR, pR, SR) - UR)
    return jnp.where(SL >= 0, FL, jnp.where(Ss >= 0, FsL, jnp.where(SR > 0, FsR, FR)))

def minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)

def mc(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.minimum(2 * jnp.abs(a), 2 * jnp.abs(b)), 0.5 * jnp.abs(a + b)), 0.0)

def no_slope(a, b):
    return jnp.zeros_like(a)

def rhs(U, dx, gamma, riemann, limiter):
    W = cons_to_prim(U, gamma)
    Wg = jnp.pad(W, ((0, 0), (2, 2)), mode="edge")       # outflow boundaries: 2 ghost cells
    Wc = Wg[:, 1:-1]                                     # cells -1 .. N (one ghost each side)
    slope = limiter(Wc - Wg[:, :-2], Wg[:, 2:] - Wc)
    WL = Wc[:, :-1] + 0.5 * slope[:, :-1]                # left state at faces -1/2 .. N-1/2
    WR = Wc[:, 1:] - 0.5 * slope[:, 1:]
    F = riemann(WL, WR, gamma)
    return -(F[:, 1:] - F[:, :-1]) / dx

def ssprk2(U, dt, dx, gamma, riemann, limiter):
    U1 = U + dt * rhs(U, dx, gamma, riemann, limiter)
    return 0.5 * U + 0.5 * (U1 + dt * rhs(U1, dx, gamma, riemann, limiter))

def max_speed(U, gamma):
    W = cons_to_prim(U, gamma)
    return jnp.max(jnp.abs(W[1]) + jnp.sqrt(gamma * W[2] / W[0]))

def solve(W0, dx, t_end, gamma=1.4, riemann=hllc, limiter=mc, cfl=0.4):
    step = jax.jit(lambda U, dt: ssprk2(U, dt, dx, gamma, riemann, limiter))
    U, t = prim_to_cons(jnp.asarray(W0), gamma), 0.0
    while t < t_end - 1e-12:
        dt = min(cfl * dx / float(max_speed(U, gamma)), t_end - t)
        U = step(U, dt); t += dt
    return cons_to_prim(U, gamma)


# ---------------------------------------------------------------------------
# 2D diffhydro helpers (notebooks 04-06)
# ---------------------------------------------------------------------------
def make_hydro(nx, ny=1, box=1.0, gamma=5.0 / 3.0, riemann=None, recon=None, integrator="RK2",
               cfl=0.4, max_steps=20000):
    """EquationManager + periodic HLLC/MUSCL3 diffhydro solver on an nx x ny box of side `box`."""
    import diffhydro as dh
    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = [nx, ny, 1]
    eq.gamma, eq.cfl = gamma, cfl
    riemann = riemann if riemann is not None else dh.HLLC
    flux = dh.ConvectiveFlux(eq, riemann(equation_manager=eq, signal_speed=dh.signal_speed_Rusanov),
                             recon if recon is not None else dh.MUSCL3(limiter="VANLEER"),
                             ignore_singleton_axes=True)
    flux.dx_o = box / nx
    sim = dh.hydro(n_super_step=max_steps, fluxes=[flux], forces=[], max_dt=1.0,
                   use_mol=True, integrator=integrator)
    return eq, sim


def make_turbulent_ic(n, mach=1.0, kcut=8.0, pslope=-2.0, solenoidal_fraction=0.5, gamma=5.0 / 3.0):
    """Differentiable map w (2, n, n) -> primitive state W (5, n, n, 1) of a turbulent 2D gas.

    Uniform density 1 and pressure 1/gamma (so the sound speed is 1). The velocity is the gradient of a
    potential (compressive part) plus the curl of a stream function (solenoidal part), both Gaussian
    random fields with power spectrum P(k) ~ k^pslope exp(-(k/kcut)^2) built from the white-noise
    channels w[0] and w[1]. Unit-variance w gives an rms Mach number `mach`. All k-space kernels are
    constants with the k=0 mode removed, so the map is linear in w.
    """
    kx, ky = k_grid_2d(n)
    k = np.sqrt(kx ** 2 + ky ** 2)
    kint = k / (2 * np.pi)
    nz = k > 0
    amp = np.where(nz, np.where(nz, kint, 1.0) ** (pslope / 2) * np.exp(-0.5 * (kint / kcut) ** 2), 0.0)
    inv_k = np.where(nz, 1.0 / np.where(nz, k, 1.0), 0.0)
    gx, gy = 1j * kx * inv_k * amp, 1j * ky * inv_k * amp          # unit vector i k/|k| times amplitude
    norm = np.sqrt(np.mean(np.abs(gx) ** 2 + np.abs(gy) ** 2))
    gx, gy = jnp.asarray((gx / norm).astype(np.complex64)), jnp.asarray((gy / norm).astype(np.complex64))
    fc, fs = np.sqrt(1.0 - solenoidal_fraction), np.sqrt(solenoidal_fraction)
    p0 = 1.0 / gamma

    def ic(w):
        wc, ws = jnp.fft.fft2(w[0]), jnp.fft.fft2(w[1])
        vx = mach * jnp.fft.ifft2(fc * gx * wc + fs * gy * ws).real     # grad(phi) + curl(psi z)
        vy = mach * jnp.fft.ifft2(fc * gy * wc - fs * gx * ws).real
        one = jnp.ones((n, n), vx.dtype)
        return jnp.stack([one, vx, vy, 0.0 * vx, p0 * one])[..., None]

    return ic


def div_curl_2d(vx, vy, box=1.0):
    """Spectral divergence and vorticity of a periodic 2D velocity field."""
    kx, ky = k_grid_2d(np.shape(vx)[0], box)
    fx, fy = np.fft.fft2(vx), np.fft.fft2(vy)
    return np.fft.ifft2(1j * (kx * fx + ky * fy)).real, np.fft.ifft2(1j * (kx * fy - ky * fx)).real
