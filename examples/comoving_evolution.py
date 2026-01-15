"""Small example: evolve a Gaussian-random density field with FFT self-gravity.

Produces two PNGs in `examples/outputs/`: one for non-comoving evolution and
one for comoving evolution (scale factor updates). This script is CPU-friendly
and uses small grids (16^3) so it runs quickly in the jax2 environment.
"""
from __future__ import annotations
import os
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
import warnings

# Optional cosmology / PM libraries
try:
    import jax_cosmo as jc
except Exception:
    jc = None

from diffhydro.equationmanager import EquationManager
from diffhydro.physics.gravity import FFTSelfGravityForce, gravity_accel_rfft
from diffhydro import comoving

try:
    import jaxpm
except Exception:
    jaxpm = None


def make_gaussian_field(mesh_shape=(64, 64, 64), seed=0, smooth_sigma=2.0):
    rng = jax.random.PRNGKey(seed)
    phases = jax.random.normal(rng, mesh_shape)
    # Simple smoothing in Fourier space to give a correlated Gaussian field
    fhat = jnp.fft.rfftn(phases)
    # build k-magnitude grid
    kx = jnp.fft.fftfreq(mesh_shape[0])
    ky = jnp.fft.fftfreq(mesh_shape[1])
    kz = jnp.fft.rfftfreq(mesh_shape[2])
    kxg, kyg, kzg = jnp.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kxg**2 + kyg**2 + kzg**2
    filter_r = jnp.exp(-0.5 * (k2 * (smooth_sigma**2)))
    fhat = fhat * filter_r
    field = jnp.fft.irfftn(fhat, s=mesh_shape)
    # normalize to zero mean and unit variance
    field = (field - jnp.mean(field)) / (jnp.std(field) + 1e-12)
    return field


def make_U_from_delta(delta, background=1.0):
    nx, ny, nz = delta.shape
    nvars = 5
    U = jnp.zeros((nvars, nx, ny, nz), dtype=jnp.float32)
    rho = background + 0.1 * delta  # small perturbation amplitude
    U = U.at[0].set(rho.astype(jnp.float32))
    return U


def evolve_with_force(U0, force, nsteps=100, dt=0.05, params=None, comoving_flag=False):
    U = U0
    params_local = dict(params) if params is not None else {}
    a = float(params_local.get('a', 1.0))
    for step in range(nsteps):
        U, _ = force.force(step, U, params_local, dt)
        if comoving_flag:
            # advance scale factor
            a = float(comoving.step_a(a, dt, params_local.get('cosmo', None)))
            params_local['a'] = a
    return U, params_local


def plot_central_slice(rho, outpath, title=""):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    nz = rho.shape[2]
    k = nz // 2
    sl = onp.asarray(rho[:, :, k])
    plt.figure(figsize=(4, 4))
    plt.imshow(sl, origin='lower', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    outdir = "examples/outputs"
    os.makedirs(outdir, exist_ok=True)

    mesh_shape = (64, 64, 64)
    eq = EquationManager()
    eq.mesh_shape = list(mesh_shape)
    box_size = (1.0, 1.0, 1.0)

    # Create FFT gravity force object
    force = FFTSelfGravityForce(eq, G=1.0, subtract_mean=True)

    # Initial density field: prefer cosmological linear field if jax_cosmo is available
    if jc is not None:
        cosmo = jc.Planck15(Omega_c=0.25, sigma8=0.8)
        k = jnp.logspace(-4.0, 1.0, 128)
        pk = jc.power.linear_matter_power(cosmo, k)
        pk_fn = lambda x: jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

        # build k-magnitude grid for rFFT shape
        nx, ny, nz = mesh_shape
        kx = jnp.fft.fftfreq(nx, d=box_size[0]/nx) * (2.0 * jnp.pi)
        ky = jnp.fft.fftfreq(ny, d=box_size[1]/ny) * (2.0 * jnp.pi)
        kz = jnp.fft.rfftfreq(nz, d=box_size[2]/nz) * (2.0 * jnp.pi)
        kxg, kyg, kzg = jnp.meshgrid(kx, ky, kz, indexing='ij')
        kmesh = jnp.sqrt(kxg**2 + kyg**2 + kzg**2)

        pkmesh = pk_fn(kmesh)

        # Random phases and build Gaussian linear field with target power
        rng = jax.random.PRNGKey(0)
        phases = jax.random.normal(rng, mesh_shape)
        fhat = jnp.fft.rfftn(phases)
        fhat = fhat * jnp.sqrt(jnp.maximum(pkmesh, 0.0))
        field = jnp.fft.irfftn(fhat, s=mesh_shape)
        # normalize to zero mean and unit variance, then scale to small overdensity
        field = (field - jnp.mean(field)) / (jnp.std(field) + 1e-12)
        delta = 0.01 * field
        U0 = make_U_from_delta(delta, background=1.0)
    else:
        warnings.warn('jax_cosmo not available; falling back to Gaussian ICs')
        delta = make_gaussian_field(mesh_shape=mesh_shape, seed=1, smooth_sigma=2.0)
        U0 = make_U_from_delta(delta, background=1.0)

    # Build particle initial conditions: one particle per cell on regular grid
    nx, ny, nz = mesh_shape
    Np = nx * ny * nz
    # particle positions in box coordinates [0,box)
    ix = jnp.arange(nx)
    iy = jnp.arange(ny)
    iz = jnp.arange(nz)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing='ij')
    qx = (gx + 0.5) / nx * box_size[0]
    qy = (gy + 0.5) / ny * box_size[1]
    qz = (gz + 0.5) / nz * box_size[2]
    particle_positions = jnp.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=-1)
    particle_velocities = jnp.zeros_like(particle_positions)
    # mass per particle consistent with mean density in U0
    mean_rho = float(jnp.mean(U0[0]))
    box_vol = box_size[0] * box_size[1] * box_size[2]
    mass_per_particle = mean_rho * box_vol / float(Np)

    # Now use the real hydro runtime with convective fluxes so advection is
    # handled by the code's own flux/reconstruction/solver machinery.
    from diffhydro.hydro_core import hydro as Hydro
    from diffhydro.solver.signal_speeds import signal_speed_Rusanov
    from diffhydro.solver.riemann_solver import LaxFriedrichs
    from diffhydro.solver.recon import MUSCL3
    from diffhydro.fluxes import ConvectiveFlux

    # Set up equation manager, solver, reconstructor and convective flux
    eq.cfl = 0.4
    ss = signal_speed_Rusanov
    grav = force  # already created FFTSelfGravityForce
    solver = LaxFriedrichs(eq, ss)
    cf = ConvectiveFlux(eq, solver, MUSCL3(limiter="VANLEER"), positivity=True)

    # Hydro object shared between runs (we will create two separate instances)
    total_steps = 60
    # use chunk=1 so we can couple particles and grid each step
    chunk = 1
    dt = 0.05

    def deposit_NGP(positions, mass_per_particle):
        """Deposit particles to grid using Cloud-In-Cell (CIC) scheme.

        positions: (Np,3) in box coordinates [0,box)
        mass_per_particle: scalar mass per particle
        Returns grid mass density (mass per cell divided by cell volume)
        """
        # Prefer jaxpm.painting.cic_paint(mesh, positions) when available
        if jaxpm is not None and hasattr(jaxpm, 'painting') and hasattr(jaxpm.painting, 'cic_paint'):
            try:
                import jax.numpy as _jnp
                mesh = _jnp.zeros(mesh_shape, dtype=_jnp.float32)
                # cic_paint returns the painted mesh (counts); convert counts -> mass density
                painted = jaxpm.painting.cic_paint(mesh, positions)
                cell_vol = box_vol / (mesh_shape[0] * mesh_shape[1] * mesh_shape[2])
                return painted * (mass_per_particle / cell_vol)
            except Exception:
                warnings.warn('jaxpm.painting.cic_paint exists but calling it failed; falling back to local CIC')

        # Local CIC fallback (same behavior as previous inline implementation)
        nx, ny, nz = mesh_shape
        Np = positions.shape[0]
        # convert to cell-space coordinates
        cx = positions[:, 0] / box_size[0] * nx
        cy = positions[:, 1] / box_size[1] * ny
        cz = positions[:, 2] / box_size[2] * nz

        ix = jnp.floor(cx).astype(int) % nx
        iy = jnp.floor(cy).astype(int) % ny
        iz = jnp.floor(cz).astype(int) % nz

        fx = cx - jnp.floor(cx)
        fy = cy - jnp.floor(cy)
        fz = cz - jnp.floor(cz)

        # neighbor offsets (0 or 1) along each axis
        offs = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]])

        rho = jnp.zeros(mesh_shape, dtype=jnp.float32)
        cell_vol = box_vol / (nx * ny * nz)

        # weights for the 8 corners
        w000 = (1-fx)*(1-fy)*(1-fz)
        w100 = (fx)*(1-fy)*(1-fz)
        w010 = (1-fx)*(fy)*(1-fz)
        w001 = (1-fx)*(1-fy)*(fz)
        w110 = (fx)*(fy)*(1-fz)
        w101 = (fx)*(1-fy)*(fz)
        w011 = (1-fx)*(fy)*(fz)
        w111 = (fx)*(fy)*(fz)

        weights = jnp.stack([w000, w100, w010, w001, w110, w101, w011, w111], axis=1)

        # compute neighbor indices and scatter-add
        for n in range(8):
            dx, dy, dz = offs[n]
            ixn = (ix + dx) % nx
            iyn = (iy + dy) % ny
            izn = (iz + dz) % nz
            # flatten linear indices for scatter
            rho = rho.at[(ixn, iyn, izn)].add(mass_per_particle * weights[:, n] / cell_vol)

        return rho


    def interp_acc_NGP(positions, ax, ay, az):
        nx, ny, nz = mesh_shape
        # Prefer jaxpm.painting.cic_read(mesh, positions) when available
        if jaxpm is not None and hasattr(jaxpm, 'painting') and hasattr(jaxpm.painting, 'cic_read'):
            try:
                a_px = jaxpm.painting.cic_read(ax, positions)
                a_py = jaxpm.painting.cic_read(ay, positions)
                a_pz = jaxpm.painting.cic_read(az, positions)
                return jnp.stack([a_px, a_py, a_pz], axis=-1)
            except Exception:
                warnings.warn('jaxpm.painting.cic_read exists but calling it failed; falling back to local NGP interp')

        # Local NGP fallback: nearest-grid interpolation
        ix = jnp.floor(positions[:, 0] / box_size[0] * nx).astype(int) % nx
        iy = jnp.floor(positions[:, 1] / box_size[1] * ny).astype(int) % ny
        iz = jnp.floor(positions[:, 2] / box_size[2] * nz).astype(int) % nz
        a_px = ax[(ix, iy, iz)]
        a_py = ay[(ix, iy, iz)]
        a_pz = az[(ix, iy, iz)]
        return jnp.stack([a_px, a_py, a_pz], axis=-1)


    class DMParticleForce:
        """Simple PM NGP particle force that stores particle state in params.

        It expects params to contain 'particles_pos', 'particles_vel', and
        'mass_per_particle'. It deposits particle mass with NGP, computes
        grid acceleration with gravity_accel_rfft, interpolates accel back
        to particles with NGP, and performs a leapfrog-like kick-drift-kick
        update. It writes back 'particles_pos', 'particles_vel', and
        'particle_rho' to params so downstream forces (e.g., grid gravity)
        can use the particle density.
        """
        def __init__(self, grav_force, mesh_shape, box_size):
            self.grav = grav_force
            self.mesh_shape = mesh_shape
            self.box_size = box_size

        def timestep(self, U):
            # cheap timestep estimate: let gravity force set dt; return large value if unknown
            return 1e6

        def force(self, i, U, params, dt):
            # read particle state
            p_pos = params.get('particles_pos', None)
            p_vel = params.get('particles_vel', None)
            mass_per_particle = params.get('mass_per_particle', None)
            if p_pos is None or p_vel is None or mass_per_particle is None:
                # nothing to do
                return U, params

            # deposit particles onto mesh (NGP)
            rho_particles = deposit_NGP(p_pos, mass_per_particle)

            # compute grid accel from particle density via same FFT Poisson
            a_param = params.get('a', 1.0)
            ax_g, ay_g, az_g = gravity_accel_rfft(rho_particles, self.grav.kx_r, self.grav.ky_r, self.grav.kz_r, self.grav.k2_r, self.grav.G, self.grav.subtract_mean, a_param)

            # half-kick
            a_p = interp_acc_NGP(p_pos, ax_g, ay_g, az_g)
            p_vel = p_vel + 0.5 * a_p * dt
            # drift
            p_pos = jnp.mod(p_pos + p_vel * dt, jnp.array(self.box_size))
            # deposit at new positions and compute accel
            rho_particles = deposit_NGP(p_pos, mass_per_particle)
            ax_g, ay_g, az_g = gravity_accel_rfft(rho_particles, self.grav.kx_r, self.grav.ky_r, self.grav.kz_r, self.grav.k2_r, self.grav.G, self.grav.subtract_mean, a_param)
            a_p = interp_acc_NGP(p_pos, ax_g, ay_g, az_g)
            p_vel = p_vel + 0.5 * a_p * dt

            # write back into params (create shallow copy)
            params = {**params, 'particles_pos': p_pos, 'particles_vel': p_vel, 'particle_rho': rho_particles}
            return U, params


    def run_hydro_record(comoving_flag: bool, cosmo: dict | None = None, with_particles: bool = True):
        h = Hydro(n_super_step=chunk, fluxes=[cf], forces=[grav], use_mol=True,
                  integrator="RK2", pmesh_shape=(1, 1, 1))

        fields = U0
        params = {'a': 1.0, 'comoving': bool(comoving_flag)}
        if cosmo is not None:
            params['cosmo'] = cosmo
        # put particle state into params so forces can access and persist them
        params['particles_pos'] = jnp.array(particle_positions)
        params['particles_vel'] = jnp.array(particle_velocities)
        params['mass_per_particle'] = float(mass_per_particle)

        n_chunks = total_steps // chunk
        rms_list = []
        mean_list = []
        total_E_list = []
        kin_E_list = []
        int_E_list = []

        # particle state for this run (copy master arrays so multiple runs don't share)
        p_pos = jnp.array(particle_positions)
        p_vel = jnp.array(particle_velocities)

        for c in range(n_chunks):
            # call hydro evolve for one super-step (chunk==1) so forcing runs each step
            fields, params = h.evolve(fields, params)
            rho = jnp.asarray(fields[0])
            mean = float(jnp.mean(rho))
            rms = float(jnp.sqrt(jnp.mean(((rho - mean) / (mean + 1e-12)) ** 2)))
            mean_list.append(mean)
            rms_list.append(rms)

            # Compute energy diagnostics from conservatives/primitives
            # fields are conservatives: [rho, rho*ux, rho*uy, rho*uz, Etot]
            prim = eq.get_primitives_from_conservatives(fields)
            p = prim[eq.energy_ids]
            u = prim[eq.vel_ids[0]]
            v = prim[eq.vel_ids[1]]
            w = prim[eq.vel_ids[2]]
            rho_arr = fields[0]

            kin_density = 0.5 * rho_arr * (u * u + v * v + w * w)
            int_density = p / (eq.gamma - 1.0)
            tot_density = kin_density + int_density

            # integrate over grid (dx=1 unit cells)
            total_E_list.append(float(jnp.sum(tot_density)))
            kin_E_list.append(float(jnp.sum(kin_density)))
            int_E_list.append(float(jnp.sum(int_density)))

            # read back particle state from params (may have been updated by DM force earlier)
            p_pos = params.get('particles_pos', p_pos)
            p_vel = params.get('particles_vel', p_vel)

        return fields, params, onp.array(rms_list), onp.array(mean_list), onp.array(total_E_list), onp.array(kin_E_list), onp.array(int_E_list)


    fields_nc, params_nc, rms_nc, mean_nc, totalE_nc, kinE_nc, intE_nc = run_hydro_record(False)

    # Several comoving cosmologies to compare (dicts consumed by comoving.hubble)
    cosmos = [
        {'H0': 1.0, 'Omega_m': 0.2},
        {'H0': 1.0, 'Omega_m': 0.3},
        {'H0': 1.0, 'Omega_m': 0.9},
    ]

    comoving_results = []
    for cos in cosmos:
        fields_c, params_c, rms_c, mean_c, totalE_c, kinE_c, intE_c = run_hydro_record(True, cosmo=cos)
        comoving_results.append((fields_c, params_c, rms_c, mean_c, totalE_c, kinE_c, intE_c, cos))
        plot_central_slice(fields_c[0], os.path.join(outdir, f"rho_c_Om{cos['Omega_m']}.png"), title=f'comoving Omega_m={cos["Omega_m"]} (final)')

    # Unpack results for plotting
    fields_c, params_c, rms_c0, mean_c0, totalE_c0, kinE_c0, intE_c0, cos0 = comoving_results[0]
    fields_c1, params_c1, rms_c1, mean_c1, totalE_c1, kinE_c1, intE_c1, cos1 = comoving_results[1]
    fields_c2, params_c2, rms_c2, mean_c2, totalE_c2, kinE_c2, intE_c2, cos2 = comoving_results[2]

    # Save final central slices
    plot_central_slice(fields_nc[0], os.path.join(outdir, 'rho_nc_slice.png'), title='Non-comoving (final)')
    plot_central_slice(fields_c[0], os.path.join(outdir, 'rho_c_slice.png'), title='Comoving (final)')

    # Plot RMS growth comparison
    times = onp.arange(len(rms_nc)) * (chunk * dt)
    plt.figure(figsize=(5, 3))
    plt.plot(times, rms_nc, label='non-comoving')
    plt.plot(times, rms_c0, label=f'comoving Omega_m={cos0["Omega_m"]}')
    plt.plot(times, rms_c1, label=f'comoving Omega_m={cos1["Omega_m"]}')
    plt.plot(times, rms_c2, label=f'comoving Omega_m={cos2["Omega_m"]}')
    plt.xlabel('time')
    plt.ylabel('rms(delta)')
    plt.legend()
    plt.tight_layout()
    rms_path = os.path.join(outdir, 'rms_vs_time.png')
    plt.savefig(rms_path, dpi=150)
    plt.close()

    # Plot mean density comparison
    plt.figure(figsize=(5, 3))
    plt.plot(times, mean_nc, label='non-comoving')
    plt.plot(times, mean_c0, label=f'comoving Omega_m={cos0["Omega_m"]}')
    plt.plot(times, mean_c1, label=f'comoving Omega_m={cos1["Omega_m"]}')
    plt.plot(times, mean_c2, label=f'comoving Omega_m={cos2["Omega_m"]}')
    plt.xlabel('time')
    plt.ylabel('mean density')
    plt.legend()
    plt.tight_layout()
    mean_path = os.path.join(outdir, 'mean_vs_time.png')
    plt.savefig(mean_path, dpi=150)
    plt.close()

    # Plot total energy evolution (normalized to initial non-comoving total)
    plt.figure(figsize=(6, 3))
    E0 = totalE_nc[0] if len(totalE_nc) > 0 else 1.0
    plt.plot(times, totalE_nc / E0, label='non-comoving')
    plt.plot(times, totalE_c0 / E0, label=f'comoving Om={cos0["Omega_m"]}')
    plt.plot(times, totalE_c1 / E0, label=f'comoving Om={cos1["Omega_m"]}')
    plt.plot(times, totalE_c2 / E0, label=f'comoving Om={cos2["Omega_m"]}')
    plt.xlabel('time')
    plt.ylabel('total E (normalized)')
    plt.legend()
    plt.tight_layout()
    energy_path = os.path.join(outdir, 'energy_vs_time.png')
    plt.savefig(energy_path, dpi=150)
    plt.close()

    # Print simple energy summary
    print('Energy summary (final/initial):')
    print(f' non-comoving: {totalE_nc[-1]:.6g} / {totalE_nc[0]:.6g} = {totalE_nc[-1]/totalE_nc[0]:.6g}')
    print(f' comoving Om={cos0["Omega_m"]}: {totalE_c0[-1]:.6g} / {totalE_c0[0]:.6g} = {totalE_c0[-1]/totalE_c0[0]:.6g}')
    print(f' comoving Om={cos1["Omega_m"]}: {totalE_c1[-1]:.6g} / {totalE_c1[0]:.6g} = {totalE_c1[-1]/totalE_c1[0]:.6g}')
    print(f' comoving Om={cos2["Omega_m"]}: {totalE_c2[-1]:.6g} / {totalE_c2[0]:.6g} = {totalE_c2[-1]/totalE_c2[0]:.6g}')

    print('Wrote:', os.path.join(outdir, 'rho_nc_slice.png'))
    print('Wrote:', os.path.join(outdir, 'rho_c_slice.png'))
    print('Wrote:', rms_path)
    print('Wrote:', mean_path)


if __name__ == '__main__':
    main()
