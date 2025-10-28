import jax
import jax.numpy as jnp


def init_turbulent_velocity(eq, Lbox, rho0, p0,
                            kmin=1, kmax=3, solenoidal_frac=1.0,
                            pslope=-2.0, target_M=1.0, seed=123):
    nx, ny, nz = eq.mesh_shape
    # k-grid (fundamental k0 = 2π/L)
    k0 = 2.0 * jnp.pi / Lbox
    kx = k0 * jnp.fft.fftfreq(nx) * nx
    ky = k0 * jnp.fft.fftfreq(ny) * ny
    kz = k0 * jnp.fft.fftfreq(nz) * nz
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K = jnp.sqrt(jnp.maximum(K2, 1e-30))

    # band-pass mask
    kmag = jnp.sqrt((KX/k0)**2 + (KY/k0)**2 + (KZ/k0)**2)
    band = (kmag >= kmin) & (kmag <= kmax)

    # random complex field with Hermitian symmetry
    key = jax.random.PRNGKey(seed)
    def rand_complex(k):
        a = jax.random.normal(k, (nx, ny, nz))
        b = jax.random.normal(jax.random.split(k)[0], (nx, ny, nz))
        return a + 1j*b
    g1 = rand_complex(key); key = jax.random.split(key)[0]
    g2 = rand_complex(key); key = jax.random.split(key)[0]
    g3 = rand_complex(key)

    G = jnp.stack([g1, g2, g3], axis=0)

    # amplitude spectrum ~ k^{pslope}
    Amp = (K**(0.5*pslope)) * band
    Amp = Amp / jnp.sqrt(jnp.mean(Amp**2) + 1e-30)  # stabilize scaling

    # project to solenoidal/compressive mix
    # tensors per k: P = I - kk^T/k^2, C = kk^T/k^2
    kk_over_k2 = jnp.stack([KX, KY, KZ], 0) / jnp.maximum(K, 1e-30)
    C = jnp.einsum("i...,j...->ij...", kk_over_k2, kk_over_k2)
    I = jnp.eye(3)[:, :, None, None, None]
    P = I - C
    zeta = solenoidal_frac
    Proj = zeta * P + (1.0 - zeta) * C

    Uhat = jnp.einsum("ij...,j...->i...", Proj, G) * Amp

    # enforce Hermitian symmetry for real IFFT (simple way: zero Nyquist, rely on ifftn real)
    u = jnp.fft.ifftn(Uhat, axes=(1,2,3)).real  # shape (3, nx, ny, nz)

    # normalize to target Mach
    rho = rho0 * jnp.ones((nx, ny, nz))
    p = p0 * jnp.ones_like(rho)
    cs = jnp.sqrt(eq.gamma * p / rho)
    urms = jnp.sqrt(jnp.mean(jnp.sum(u**2, axis=0)))
    alpha = (target_M * jnp.mean(cs)) / (urms + 1e-30)
    v = alpha * u

    # fill conservatives (Euler: U=[rho, rho*u, rho*v, rho*w, E])
    U = jnp.zeros((5, nx, ny, nz))
    U = U.at[0].set(rho)
    U = U.at[1].set(rho * v[0]); U = U.at[2].set(rho * v[1]); U = U.at[3].set(rho * v[2])
    E_th = p / (eq.gamma - 1.0)
    E_kin = 0.5 * rho * (v**2).sum(axis=0)
    U = U.at[4].set(E_th + E_kin)
    return U


class TurbulentForce:
    """NaN-safe Ornstein–Uhlenbeck turbulent driver (Athena-style)."""

    def __init__(
        self,
        equation_manager,
        *,
        kmin=1.0, kmax=3.0,
        solenoidal_fraction=1.0,
        tau_corr=0.5,
        rms_accel=1.0,
        seed=12345,
    ):
        self.eq = equation_manager
        self.kmin, self.kmax = float(kmin), float(kmax)
        self.sol_frac = float(solenoidal_fraction)
        self.tau = float(tau_corr)
        self.a_rms_target = float(rms_accel)
        self.key = jax.random.PRNGKey(int(seed))

        shape = tuple(self.eq.mesh_shape)
        if len(shape) == 2: nx, ny, nz = shape[0], shape[1], 1
        else: nx, ny, nz = shape
        self.nx, self.ny, self.nz = nx, ny, nz
        Lx, Ly, Lz = getattr(self.eq, "box_size", (1.0, 1.0, 1.0))
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)

        kx = 2*jnp.pi*jnp.fft.fftfreq(nx, d=self.Lx/nx)
        ky = 2*jnp.pi*jnp.fft.fftfreq(ny, d=self.Ly/ny)
        kz = 2*jnp.pi*jnp.fft.fftfreq(nz, d=self.Lz/nz) if nz>1 else jnp.array([0.0])
        self.KX, self.KY, self.KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.K = jnp.sqrt(self.K2)
        self.nonzero_mask = self.K2 > 0.0

        self.accel_k = jnp.zeros((3, nx, ny, nz), dtype=jnp.complex64)
        self.i_rho = self.eq.mass_ids
        self.i_mom = self.eq.vel_ids
        self.i_E   = self.eq.energy_ids

    # -------------------------------------------------------------
    def _safe_div(self, num, denom):
        denom = jnp.where(denom == 0, 1.0, denom)
        return num / denom

    def _step_accel_k(self, dt):
        self.key, sub = jax.random.split(self.key)
        noise = jax.random.normal(sub, self.accel_k.real.shape, dtype=jnp.float32)

        decay = jnp.exp(-jnp.clip(dt/self.tau, 0.0, 50.0))
        drive = jnp.sqrt(jnp.maximum(1.0 - decay**2, 0.0))
        self.accel_k = decay*self.accel_k + (drive*noise).astype(self.accel_k.dtype)

        kmin_abs = self.kmin*(2*jnp.pi/self.Lx)
        kmax_abs = self.kmax*(2*jnp.pi/self.Lx)
        band = (self.K >= kmin_abs) & (self.K <= kmax_abs)
        self.accel_k = jnp.where(band, self.accel_k, 0.0)

        # projection
        kx, ky, kz = self.KX, self.KY, self.KZ
        dot = kx*self.accel_k[0] + ky*self.accel_k[1] + kz*self.accel_k[2]
        invK2 = jnp.where(self.nonzero_mask, 1.0/self.K2, 0.0)

        proj_divfree = jnp.stack([
            self.accel_k[0] - kx*dot*invK2,
            self.accel_k[1] - ky*dot*invK2,
            self.accel_k[2] - kz*dot*invK2,
        ], axis=0)
        proj_comp = jnp.stack([
            kx*dot*invK2, ky*dot*invK2, kz*dot*invK2
        ], axis=0)
        self.accel_k = self.sol_frac*proj_divfree + (1.0-self.sol_frac)*proj_comp
        self.accel_k = jnp.where(self.nonzero_mask, self.accel_k, 0.0)  # kill k=0

        a = jnp.fft.ifftn(self.accel_k, axes=(1,2,3)).real
        rms = jnp.sqrt(jnp.maximum(jnp.mean(a**2), 1e-30))
        a = a * (self.a_rms_target / rms)
        a = jnp.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return a

    # -------------------------------------------------------------
    def timestep(self, U):
        return 1e10  # defer to CFL limiter

    def force(self, i_step, U, params, dt):
        dt = jnp.maximum(dt, 0.0)
        a = self._step_accel_k(dt)

        W = self.eq.get_primitives_from_conservatives(U)
        rho = jnp.maximum(W[self.i_rho], 1e-30)
        u, v, w = W[self.i_mom[0]], W[self.i_mom[1]], W[self.i_mom[2]]

        d_mx = rho*a[0]*dt
        d_my = rho*a[1]*dt
        d_mz = rho*a[2]*dt

        U = U.at[self.i_mom[0]].add(jnp.nan_to_num(d_mx))
        U = U.at[self.i_mom[1]].add(jnp.nan_to_num(d_my))
        U = U.at[self.i_mom[2]].add(jnp.nan_to_num(d_mz))

        work = rho*(u*a[0] + v*a[1] + w*a[2])*dt
        U = U.at[self.i_E].add(jnp.nan_to_num(work))
        return U
