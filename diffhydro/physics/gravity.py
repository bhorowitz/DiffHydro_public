
import jax
import jax.numpy as jnp

class FFTSelfGravityForce:
    def __init__(self, equation_manager, G=1.0, subtract_mean=True, eps=1e-20,
                 cfl_ff=0.5, cfl_acc=0.5, dx=1.0):
        self.eq = equation_manager
        self.G = G
        self.subtract_mean = subtract_mean
        self.eps = eps
        self.cfl_ff = cfl_ff
        self.cfl_acc = cfl_acc
        self.dx = dx
        self.i_rho = self.eq.mass_ids
        self.i_mx, self.i_my, self.i_mz = self.eq.vel_ids
        self.i_E   = self.eq.energy_ids

    def _accel_from_density(self, rho):
        # build k^2 and solve Poisson like in force()
        ks = [jnp.fft.fftfreq(n, d=1.0) * (2.0 * jnp.pi) for n in rho.shape]
        if rho.ndim == 1:
            kx = ks[0]
            k2 = kx**2
        elif rho.ndim == 2:
            kx, ky = jnp.meshgrid(ks[0], ks[1], indexing="ij")
            k2 = kx*kx + ky*ky
        else:
            kx, ky, kz = jnp.meshgrid(ks[0], ks[1], ks[2], indexing="ij")
            k2 = kx*kx + ky*ky + kz*kz

        rho_src = rho - (jnp.mean(rho) if self.subtract_mean else 0.0)
        rho_hat = jnp.fft.fftn(rho_src)
        denom = jnp.where(k2 == 0.0, 1.0, k2)
        phi_hat = -4.0 * jnp.pi * self.G * rho_hat / denom
        phi_hat = jnp.where(k2 == 0.0, 0.0 + 0.0j, phi_hat)

        # a = -∇Φ via spectral (ik Φ̂), same as in force()
        if rho.ndim == 1:
            ax = -jnp.fft.ifftn(1j * (ks[0]) * phi_hat).real
            ay = az = 0.0
        elif rho.ndim == 2:
            ax = -jnp.fft.ifftn(1j * kx * phi_hat).real
            ay = -jnp.fft.ifftn(1j * ky * phi_hat).real
            az = 0.0
        else:
            ax = -jnp.fft.ifftn(1j * kx * phi_hat).real
            ay = -jnp.fft.ifftn(1j * ky * phi_hat).real
            az = -jnp.fft.ifftn(1j * kz * phi_hat).real
        return ax, ay, az

    def timestep(self, U):
        #not really sure about units here, definitely needs some double checking...
        rho = jnp.maximum(U[self.i_rho], self.eps)

        # 1) free-fall (use densest cell): t_ff = sqrt(3π/(32 G ρ_max))
        rho_max = jnp.max(rho)
        t_ff = self.cfl_ff * jnp.sqrt(3.0 * jnp.pi / (32.0 * self.G * rho_max + self.eps))

        # 2) acceleration limit: 0.5 * |a|max * dt^2 < cfl_acc * dx  =>  dt_acc
        ax, ay, az = self._accel_from_density(rho)
        a_mag_max = jnp.max(jnp.sqrt(ax*ax + (ay if isinstance(ay,float) else ay)**2
                                     + (az if isinstance(az,float) else az)**2))
        dt_acc = self.cfl_acc * jnp.sqrt(2.0 * self.dx / (a_mag_max + self.eps))

        return jnp.minimum(t_ff, dt_acc)
        
    def force(self, i, U, params, dt):
        """Apply a gravity kick over dt to CONS state U and return updated U."""
        dt = jnp.maximum(jnp.asarray(dt), 0.0)

        # Grid/shape: U shape is (vars, nx[, ny[, nz]])
        rho = U[self.i_rho]
        spatial_shape = rho.shape
        dim = rho.ndim

        # Optionally remove mean density (periodic Poisson)
        rho_src = rho - (jnp.mean(rho) if self.subtract_mean else 0.0)

        # k-grid (assume unit cell size; if tracking dx elsewhere, multiply k by dx)
        ks = [jnp.fft.fftfreq(n, d=1.0) * (2.0 * jnp.pi) for n in spatial_shape]
        if dim == 1:
            kx = ks[0][:, None] * 0 + ks[0]  # keep broadcastable shapes simple
            k2 = (ks[0] ** 2)
        elif dim == 2:
            kx, ky = jnp.meshgrid(ks[0], ks[1], indexing="ij")
            k2 = kx * kx + ky * ky
        else:
            kx, ky, kz = jnp.meshgrid(ks[0], ks[1], ks[2], indexing="ij")
            k2 = kx * kx + ky * ky + kz * kz

        # Poisson solve in k-space: Φ̂ = -4πG ρ̂ / k²  (set k=0 mode to 0)
        rho_hat = jnp.fft.fftn(rho_src)
        denom = jnp.where(k2 == 0.0, 1.0, k2)
        phi_hat = -4.0 * jnp.pi * self.G * rho_hat / denom
        phi_hat = jnp.where(k2 == 0.0, 0.0 + 0.0j, phi_hat)  # zero out mean mode
        phi = jnp.fft.ifftn(phi_hat).real

        # Acceleration a = -∇Φ (central difference via spectral is exact: i k Φ̂)
        if dim == 1:
            ax_hat = (1j) * kx * phi_hat
            ax = -jnp.fft.ifftn(ax_hat).real
            ay = az = 0.0
        elif dim == 2:
            ax_hat = (1j) * kx * phi_hat
            ay_hat = (1j) * ky * phi_hat
            ax = -jnp.fft.ifftn(ax_hat).real
            ay = -jnp.fft.ifftn(ay_hat).real
            az = 0.0
        else:
            ax_hat = (1j) * kx * phi_hat
            ay_hat = (1j) * ky * phi_hat
            az_hat = (1j) * kz * phi_hat
            ax = -jnp.fft.ifftn(ax_hat).real
            ay = -jnp.fft.ifftn(ay_hat).real
            az = -jnp.fft.ifftn(az_hat).real

        # Momentum update: m += ρ a dt
        U_new = U
        U_new = U_new.at[self.i_mx].add(rho * ax * dt)
        if dim >= 2:
            U_new = U_new.at[self.i_my].add(rho * ay * dt)
        if dim >= 3:
            U_new = U_new.at[self.i_mz].add(rho * az * dt)

        # Energy update: E += ρ (v · a) dt  with v = m / ρ
        rho_safe = jnp.maximum(rho, self.eps)
        ux = U[self.i_mx] / rho_safe
        uy = U[self.i_my] / rho_safe if dim >= 2 else 0.0
        uz = U[self.i_mz] / rho_safe if dim >= 3 else 0.0
        power = rho * (ux * ax + uy * ay + uz * az)
        U_new = U_new.at[self.i_E].add(power * dt)

        return U_new