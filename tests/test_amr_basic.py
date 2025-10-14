
# test_amr_basic.py
# Pytest suite distilled from amr_basic_test.ipynb
# Focus: tile roundtrip, periodic halo plumbing, and ghost-freeze RK2 step.
#
# Run with:  pytest -q -k amr_basic

import os
import importlib
import pytest

# Soft import JAX; if missing, mark tests xfail to avoid confusion in CI environments.
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np  = pytest.importorskip("numpy")

# Try the package first; fall back to a local module named hydro_core
try:
    import diffhydro.hydro_core as dh_core  # package layout
except Exception:
    try:
        import hydro_core as dh_core        # local module in PYTHONPATH
    except Exception as e:
        pytest.skip(f"Neither diffhydro.hydro_core nor hydro_core importable: {e}",
                    allow_module_level=True)

# ----------------------------- Utilities -----------------------------

class DummyHydro:
    """Minimal hydro stub: flux(U) = U and dx_o = 1.
    Only used to test stencil & halo plumbing; not a physical flux."""
    dx_o = 1.0
    def flux(self, U, ax, params):
        return U

def _pattern_tiles(Ny=2, Nx=3, C=1, T=4):
    # Distinct rows/cols to catch off-by-one and face selection
    yy = jnp.arange(T)[:, None]
    xx = jnp.arange(T)[None, :]
    base = yy * 10 + xx                           # [T,T] rows grow by 10, cols by 1
    tiles = []
    for i in range(Ny):
        row = []
        for j in range(Nx):
            val = (i * 1000 + j * 100)
            row.append(jnp.stack([base + val], axis=0))  # [C,T,T]
        tiles.append(jnp.stack(row, axis=0))
    return jnp.stack(tiles, axis=0)  # [Ny,Nx,C,T,T]

def _make_halos_y_local(tiles, h):
    """Periodic y halos for tiles with shape [Ny,Nx,C,T,T].
    Returns bottom, top halos with shapes [Ny,Nx,C,h,T]."""
    below_nb = jnp.roll(tiles, -1, axis=0)  # neighbor below (i+1)
    B = below_nb[..., :h, :]                # TOP h rows of the below neighbor

    above_nb = jnp.roll(tiles,  1, axis=0)  # neighbor above (i-1)
    Tt = above_nb[..., -h:, :]              # BOTTOM h rows of the above neighbor
    return B, Tt

def build_Ubc_from_blocks(top, interior, bot, left, right):
    """Assembles a tile with frozen ghost rings.
    top: [C,h,T], interior: [C,T,T], bot: [C,h,T]
    left/right: [C,T+2h,h]  -> returns U_bc: [C, T+2h, T+2h]"""
    center = jnp.concatenate([top, interior, bot], axis=1)   # [C, T+2h, T]
    U_bc   = jnp.concatenate([left, center, right], axis=2)  # [C, T+2h, T+2h]
    return U_bc

def ref_rk2_periodic(U, dt, ax):
    """Reference RK2 on a *global* periodic canvas.
    Matches the algebra used by the per-tile step."""
    ra = 2 if int(ax) == 1 else 1   # sweep->array axis mapping
    fu1  = U
    rhs1 = fu1 - jnp.roll(fu1, 1, axis=ra)
    U1   = U - (dt / 2.0) * rhs1
    fu2  = U1
    rhs2 = fu2 - jnp.roll(fu2, 1, axis=ra)
    U2   = U1 - dt * rhs2
    # gentle density floor on channel 0 to mirror production code
    U2 = U2.at[0].set(jnp.maximum(U2[0], 1e-12))
    return U2

# ------------------------------ Tests --------------------------------

@pytest.mark.parametrize("C,H,W,T", [(3, 16, 12, 4), (1, 8, 8, 4)])
def test_extract_assemble_tiles_roundtrip(C, H, W, T):
    cfg = dh_core.build_level0_config(H, W, T, r=2)
    U   = jax.random.normal(jax.random.PRNGKey(0), (C, H, W))
    tiles = dh_core.extract_tiles(U, cfg)
    U2    = dh_core.assemble_from_tiles(tiles, cfg)
    np.testing.assert_allclose(np.asarray(U2), np.asarray(U), rtol=0, atol=0)

def test_make_halos_x_periodic():
    Ny, Nx, C, T = 2, 3, 1, 4
    tiles = _pattern_tiles(Ny, Nx, C, T)
    h = 2
    # The module has an internal helper _make_halos_x; use it if present
    if not hasattr(dh_core, "_make_halos_x"):
        pytest.skip("_make_halos_x not available in dh_core")
    L, R = dh_core._make_halos_x(tiles, h)
    # For each (i,j): left halo must be rightmost h cols of left neighbor (j-1)
    for i in range(Ny):
        for j in range(Nx):
            jm = (j - 1) % Nx
            jp = (j + 1) % Nx
            expected_L = tiles[i, jm, :, :, -h:]
            expected_R = tiles[i, jp, :, :, :h]
            np.testing.assert_allclose(np.asarray(L[i, j]), np.asarray(expected_L))
            np.testing.assert_allclose(np.asarray(R[i, j]), np.asarray(expected_R))

def test_make_halos_y_periodic():
    Ny, Nx, C, T = 3, 2, 1, 4
    tiles = _pattern_tiles(Ny, Nx, C, T)
    h = 1
    B, Tt = _make_halos_y_local(tiles, h)
    for i in range(Ny):
        for j in range(Nx):
            ip = (i + 1) % Ny   # neighbor below
            im = (i - 1) % Ny   # neighbor above
            expected_B  = tiles[ip, j, :, :h, :]
            expected_Tt = tiles[im, j, :, -h:, :]
            np.testing.assert_allclose(np.asarray(B[i, j]),  np.asarray(expected_B))
            np.testing.assert_allclose(np.asarray(Tt[i, j]), np.asarray(expected_Tt))

@pytest.mark.parametrize("T,h", [(6,2), (8,1)])
def test_solve_step_freeze_ghosts_preserves_ghosts(T, h):
    # Construct a padded tile with distinct ghost rings
    C = 2
    interior = jnp.zeros((C, T, T))
    top = jnp.full((C, h, T),  111.0)
    bot = jnp.full((C, h, T),  222.0)
    lef = jnp.full((C, T + 2*h, h), 333.0)
    rig = jnp.full((C, T + 2*h, h), 444.0)
    U_bc = build_Ubc_from_blocks(top, interior, bot, lef, rig)

    hydro = DummyHydro()
    U2 = dh_core._solve_step_freeze_ghosts(hydro, U_bc, dx_o=1.0, dt=0.1, ax=1, params={}, halo_w=h)
    # side halos should be unchanged
    np.testing.assert_allclose(np.asarray(U2[:, :h, :]),   np.asarray(U_bc[:, :h, :]))
    np.testing.assert_allclose(np.asarray(U2[:, -h:, :]),  np.asarray(U_bc[:, -h:, :]))
    np.testing.assert_allclose(np.asarray(U2[:, :, :h]),   np.asarray(U_bc[:, :, :h]))
    np.testing.assert_allclose(np.asarray(U2[:, :, -h:]),  np.asarray(U_bc[:, :, -h:]))
"""
@pytest.mark.parametrize("C,H,W,T,ax", [(1,32,32,8,1), (2,16,24,4,2)])
def test_tile_rk2_matches_global_reference(C, H, W, T, ax):
    # This exercises extract_tiles -> per-tile step with halo exchange -> assemble_from_tiles,
    # compared to a single global RK2 on the whole canvas using the same algebra.
    cfg = dh_core.build_level0_config(H, W, T, r=2)
    U   = jax.random.normal(jax.random.PRNGKey(123), (C, H, W))
    tiles = dh_core.extract_tiles(U, cfg)

    # Use the module's step implementation if available; else fall back to calling
    # the internal freeze-and-step on each tile with exchanged halos.
    step_fun = getattr(dh_core, "step_tiles_with_halo", None)
    dt = 0.1
    hydro = DummyHydro()

    if step_fun is not None:
        tiles_next = step_fun(hydro, tiles, dt, ax=ax, halo_w=1, dx_o=1.0, params={})
    else:
        # Manual reference using the same algebra as _solve_step_freeze_ghosts
        # 1) compute periodic halos
        if not hasattr(dh_core, "_make_halos_x"):
            pytest.skip("_make_halos_x not available; cannot construct manual tile step.")
        L, R = dh_core._make_halos_x(tiles, h=1) if int(ax) == 1 else (None, None)
        B, Tt = _make_halos_y_local(tiles, h=1) if int(ax) == 2 else (None, None)
        # 2) assemble padded tiles per direction
        def pad_tile(tile, l=None, r=None, b=None, t=None):
            Cc, Tt_, _ = tile.shape
            cols = [l if l is not None else jnp.zeros((Cc,Tt_,1)),
                    tile,
                    r if r is not None else jnp.zeros((Cc,Tt_,1))]
            tmp = jnp.concatenate(cols, axis=2)
            rows = [t if t is not None else jnp.zeros((Cc,1,tmp.shape[2])),
                    tmp,
                    b if b is not None else jnp.zeros((Cc,1,tmp.shape[2]))]
            return jnp.concatenate(rows, axis=1)
        Ny, Nx = tiles.shape[:2]
        padded = []
        for i in range(Ny):
            row = []
            for j in range(Nx):
                if int(ax) == 1:
                    row.append(pad_tile(tiles[i,j],
                                        l=L[i,j], r=R[i,j],
                                        t=None,  b=None))
                else:
                    row.append(pad_tile(tiles[i,j],
                                        l=None, r=None,
                                        t=Tt[i,j], b=B[i,j]))
            padded.append(jnp.stack(row, axis=0))
        padded = jnp.stack(padded, axis=0)  # [Ny,Nx,C,T+2h,T+2h]
        # 3) freeze-and-step via the module helper (keeps ghosts fixed)
        stepped = jax.vmap(jax.vmap(
            lambda Ubc: dh_core._solve_step_freeze_ghosts(hydro, Ubc, dx_o=1.0, dt=dt, ax=ax, params={}, halo_w=1),
            in_axes=0), in_axes=0)(padded)
        # 4) crop interior back
        tiles_next = stepped[..., 1:-1, 1:-1]

    U_tiles = dh_core.assemble_from_tiles(tiles_next, cfg)
    U_ref   = ref_rk2_periodic(U, dt, ax)  # global periodic reference
    np.testing.assert_allclose(np.asarray(U_tiles), np.asarray(U_ref), rtol=1e-6, atol=1e-6)
"""