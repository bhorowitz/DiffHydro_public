from jax import Array 
from functools import partial
from typing import List
import jax.numpy as np
import numpy as numpy
from diffhydro import NoBoundary, NoForcing
import jax
from jax import jit
import jax.numpy as jnp

from .amr import AMRHierarchy, AMRBlock, AMRLevel
from .amr_ops import restrict_conserve, prolong_bilinear, reflux_add


from .amr import AMRHierarchy, AMRLevel, AMRBlock
from .amr_ops import restrict_conserve, prolong_bilinear
import jax
import jax.numpy as jnp


def pad_to_tile(U, tile_h, tile_w):
    # U: [C, H, W]
    C, H, W = U.shape
    ph = tile_h - H
    pw = tile_w - W
    ph = jnp.maximum(ph, 0); pw = jnp.maximum(pw, 0)
    # pad on the max edges; you can choose symmetric if you prefer
    U_pad = jnp.pad(U, ((0,0), (0, int(ph)), (0, int(pw))))
    mask  = jnp.pad(jnp.ones((1, H, W), U.dtype), ((0,0),(0,int(ph)),(0,int(pw))))
    return U_pad, mask, (H, W)  # keep inner size for write-back

def crop_from_tile(U_pad, inner_hw):
    H, W = inner_hw
    return U_pad[..., :H, :W]



def advance_L0_without_touching_solver(hydro, convective_flux, L0, dt, params):
    U_batch = jnp.stack([b.U for b in L0.blocks], 0)  # [B,C,H,W]
    # For each sweep in your splitting scheme:
    faces_accum = {"x": [], "y": []}
    U = U_batch
    for scheme in hydro.splitting_schemes:
        for ax in scheme:
            U_next, fbdry = step_tiles_and_recompute(hydro, convective_flux, U, dt/len(scheme)*2, ax, params)
            U = U_next
            if ax == 1: faces_accum["x"].append(fbdry)
            elif ax == 2: faces_accum["y"].append(fbdry)

    new_blocks = tuple(AMRBlock(U[i], L0.blocks[i].mask, L0.blocks[i].origin, L0.blocks[i].dx)
                       for i in range(U.shape[0]))
    return AMRLevel(L0.ratio, new_blocks), faces_accum
def restrict_state_from_L1_to_L0(L0, L1, ratio):
    updated = []
    for cb, fb in zip(L0.blocks, L1.blocks):
        Uc_from_fine = restrict_conserve(fb.U, ratio)
        updated.append(AMRBlock(Uc_from_fine, cb.mask, cb.origin, cb.dx))
    return AMRLevel(L0.ratio, tuple(updated))

def reflux_faces_onto_L0(L0, faces_L0, faces_L1_all, ratio, dt, dx, dy):
    # You’ll build index maps once (fine→coarse), then do scatter-adds.
    # Pseudocode; implement geometry with your tile layout.
    corrected = []
    for bi, cb in enumerate(L0.blocks):
        Uc = cb.U

        # Example for X faces:
        # coarse_left  : [C, H]
        # coarse_right : [C, H]
        coarse_left  = faces_L0["x"][0]["left"][:, :]   # pick one sweep or sum them
        coarse_right = faces_L0["x"][0]["right"][:, :]

        # Sum fine faces across subcycles and across r fine faces that align with a coarse face:
        fine_left_sum  = 0.0
        fine_right_sum = 0.0
        for step_faces in faces_L1_all:
            # step_faces["x"][k]["left"] has shape [B,C,H_f] for some k; pick bi, sum over r
            fL = step_faces["x"][0]["left"][:, :]   # adapt indexing to your stored structure
            fR = step_faces["x"][0]["right"][:, :]
            # Reduce fine H_f -> H by summing each group of r rows (or cols) that align to a coarse face:
            # fL_grouped = fL.reshape(C, H, ratio).sum(axis=-1)
            # same for fR_grouped
            fine_left_sum  = fine_left_sum  + fL.reshape(fL.shape[0], -1, ratio).sum(-1)
            fine_right_sum = fine_right_sum + fR.reshape(fR.shape[0], -1, ratio).sum(-1)

        dF_left  = fine_left_sum  - coarse_left
        dF_right = fine_right_sum - coarse_right

        # Convert flux mismatch to conservative updates on Uc’s border cells:
        # (signs depend on your flux convention; below is illustrative)
        Uc = Uc.at[:, :, 0].add(+ (dt/dx) * dF_left)
        Uc = Uc.at[:, :, -1].add(- (dt/dx) * dF_right)

        # Repeat similarly for Y faces with dy and top/bottom

        corrected.append(AMRBlock(Uc, cb.mask, cb.origin, cb.dx))
    return AMRLevel(L0.ratio, tuple(corrected))
from .amr_ops import prolong_bilinear, restrict_conserve

def subcycle_L1_without_touching_solver(hydro, convective_flux, L1, L0_parent, dt, ratio, params):
    # 1) Prolong L0 → L1 for BC/initialization
    UcB = jnp.stack([b.U for b in L0_parent.blocks], 0)
    Uf_init = jax.vmap(lambda Uc: prolong_bilinear(Uc, ratio))(UcB)
    L1_blocks = tuple(AMRBlock(Uf_init[i], L1.blocks[i].mask, L1.blocks[i].origin, L1.blocks[i].dx)
                      for i in range(len(L1.blocks)))

    # 2) Subcycle r times; accumulate fine boundary face fluxes each sub-step
    dt_f = dt / ratio
    faces_fine_all = []
    U = jnp.stack([b.U for b in L1_blocks], 0)
    for _ in range(ratio):
        faces_step = {"x": [], "y": []}
        for scheme in hydro.splitting_schemes:
            for ax in scheme:
                U_next, fbdry = step_tiles_and_recompute(hydro, convective_flux, U, dt_f/len(scheme)*2, ax, params)
                U = U_next
                if ax == 1: faces_step["x"].append(fbdry)
                elif ax == 2: faces_step["y"].append(fbdry)
        faces_fine_all.append(faces_step)

    L1_new = AMRLevel(L1.ratio, tuple(
        AMRBlock(U[i], L1.blocks[i].mask, L1.blocks[i].origin, L1.blocks[i].dx)
        for i in range(U.shape[0])
    ))
    return L1_new, faces_fine_all

def step_tile_and_recompute_faces(hydro_obj, convective_flux, U, dt, ax, params):
    """
    1) Use your original update (no changes to solve_step).
    2) Recompute boundary face fluxes for refluxing.
    """
    # Your unchanged path:
    U_bc  = hydro_obj.boundary.impose(U, ax)
    U_out = hydro_obj.solve_step(U_bc, dt, int(ax), params)

    # Recompute boundary faces from *pre-update* state (consistent across levels)
    fbdry = boundary_face_fluxes(convective_flux, hydro_obj.boundary, U_bc, ax, params)
    return U_out, fbdry

def impose_tile_bcs(boundary, U, ax):
    # Use your existing boundary.impose per axis; no changes needed.
    return boundary.impose(U, ax)

def flux_on_tile(convective_flux, U, ax, params):
    """
    Call your existing ConvectiveFlux.flux on the tile for axis `ax`.
    It should return face-aligned fluxes; if it returns cell-centered,
    adapt the slicing below to grab the faces you need.
    """
    return convective_flux.flux(U, ax, params)

def boundary_face_fluxes(convective_flux, boundary, U, ax, params):
    """
    Recompute ONLY the *outer* boundary-face fluxes along `ax`.
    Returns a dict with 'left' & 'right' for ax==x, and 'bottom' & 'top' for ax==y.
    (3D: also 'front' & 'back' for ax==z.)
    """
    # 1) Make sure halos are valid for the requested axis.
    U_bc = impose_tile_bcs(boundary, U, ax)

    # 2) Compute fluxes on the whole tile (cheap: dense, JIT’d)
    F = flux_on_tile(convective_flux, U_bc, ax, params)
    # Convention: for X-sweep, F has faces along x (width-1 or width+1); likewise for Y.

    if ax == 1:  # x-direction sweep
        # Grab left/right-most *faces* across all rows (and channels).
        # Depending on your flux tensor shape:
        # - If F.shape == [C, H, W-1], faces lie between cells: indices 0 and -1
        # - If F is padded to W or has halos, adjust slice (+1/-2 accordingly).
        left  = F[..., :, 0]    # [C, H]
        right = F[..., :, -1]   # [C, H]
        return {"ax": "x", "left": left, "right": right}

    elif ax == 2:  # y-direction sweep
        bottom = F[..., 0, :]   # [C, W]
        top    = F[..., -1, :]  # [C, W]
        return {"ax": "y", "bottom": bottom, "top": top}

    else:
        # If you have 3D: ax == 3 → z direction.
        # Example:
        # front = F[..., 0, :, :]   # [C, H, W]
        # back  = F[..., -1, :, :]
        # return {"ax": "z", "front": front, "back": back}
        raise NotImplementedError("Only 2D shown here; add the z case analogously.")

def rasterize_hierarchy(hierarchy, base_shape, base_ratio=1):
    """Return depth_map (max level) and per-level masks at coarse resolution."""
    import numpy as np
    Hc, Wc = base_shape[-2], base_shape[-1]
    depth = np.zeros((Hc, Wc), dtype=np.int8)
    level_masks = []
    # total refinement factor from level 0 to level ℓ is (ratio^ℓ) if fixed
    total_ratio = 1
    for ℓ, L in enumerate(hierarchy.levels):
        if ℓ == 0:
            total_ratio = 1
            # Level 0 covers everything by definition
            level_masks.append(np.ones((Hc, Wc), dtype=bool))
            continue
        total_ratio *= L.ratio
        mℓ = np.zeros((Hc, Wc), dtype=bool)
        for b in L.blocks:
            # b.origin is in level-0 (coarse) cell indices in our scaffold
            y0, x0 = b.origin
            h, w = b.U.shape[-2], b.U.shape[-1]
            # Map fine tile extent back to coarse cells
            hc = h // total_ratio
            wc = w // total_ratio
            mℓ[y0:y0+hc, x0:x0+wc] = True
            depth[y0:y0+hc, x0:x0+wc] = np.maximum(depth[y0:y0+hc, x0:x0+wc], ℓ)
        level_masks.append(mℓ)
    return depth, level_masks

def _advance_level(h: AMRHierarchy, ell: int, hydro_obj, params, dt):
    # Batch blocks to keep kernels dense
    blocks = h.levels[ell].blocks
    U = jnp.stack([b.U for b in blocks], 0)     # [B,C,H,W]
    # One Strang sweep over axes using your existing kernels:
    def solve_per_block(Ub):
        state = (Ub, params)
        # reuse your sweep_stack inner logic on a single tile
        for scheme in hydro_obj.splitting_schemes:
            for ax in scheme:
                Ub = hydro_obj.boundary.impose(Ub, ax)
                Ub = hydro_obj.solve_step(Ub, dt/len(scheme)*2, int(ax), params)
        return Ub
    U_new = jax.vmap(solve_per_block)(U)
    # write back
    new_blocks = []
    for i, b in enumerate(blocks):
        new_blocks.append(type(b)(U_new[i], b.mask, b.origin, b.dx))
    h.levels[ell] = type(h.levels[ell])(h.levels[ell].ratio, new_blocks)
    return h

def advance_hierarchy(h, hydro_obj, params, dt_coarse):
    # level 0 (coarse)
    h = _advance_level(h, 0, hydro_obj, params, dt_coarse)
    # finer levels
    for ell in range(1, len(h.levels)):
        Lc, Lf = h.levels[ell-1], h.levels[ell]
        r = Lf.ratio
        dt_f = dt_coarse / r

        # Prolongate coarse state into fine blocks (BCs/init)
        # (Assumes blocks are aligned; if not, slice the coarse parent region)
        # Example for 1:1 covering — adapt to your block layout
        for i, fb in enumerate(Lf.blocks):
            # simple demo: prolong a matching coarse patch
            # (Replace with a proper parent->child slice using fb.origin)
            Uc_patch = Lc.blocks[i].U
            Lf.blocks[i].U = prolong_bilinear(Uc_patch, r)

        # Subcycle fine
        for _ in range(r):
            h = _advance_level(h, ell, hydro_obj, params, dt_f)

        # Sync back to coarse (restrict + reflux)
        # (Sketch: apply restriction on state and a reflux on stored face fluxes)
        for i, cb in enumerate(Lc.blocks):
            Uf = Lf.blocks[i].U
            Uc_corr = restrict_conserve(Uf, r)
            # conservative overwrite or weighted blend; reflux handles fluxes
            Lc.blocks[i].U = Uc_corr
    return h



@jax.tree_util.register_pytree_node_class
class hydro:
    #TO DO, pretty up this area...
    def __init__(self,
                 n_super_step = 600,
                 max_dt = 0.5, 
                 boundary = NoBoundary,
                 snapshots = False,
                splitting_schemes=[[3,1,2,2,1,3],[1,2,3,3,2,1],[2,3,1,1,3,2]], #cyclic permutations
                fluxes = None, #convection, conduction
                forces = [NoForcing()], #gravity, etc.
                maxjit=False,
                use_amr: bool=False,
                 adapt_interval: int=20,
                 refine_ratio: int=2
                ):
        #parameters that are held constant per run (i.e. probably don't want to take derivatives with respect to...)
   #     self.init_dt = init_dt # tiny starting timestep to smooth out anything too sharp
        self.splitting_schemes = splitting_schemes #strang splitting for x,y,z sweeps
        self.max_dt = max_dt
        self.boundary = boundary
        #supersteps, each superstep has len(splitting_schemes) time steps
        self.n_super_step = n_super_step
        self.snapshots = snapshots #poorly names/
        self.outputs = []
        self.fluxes = fluxes
        self.forces = forces
        self.maxjit = maxjit
        self.dx_o = 1.0
        self.timescale = jnp.zeros(self.n_super_step)
        self.use_amr = use_amr
        self.adapt_interval = adapt_interval
        self.refine_ratio = refine_ratio
    
    def timestep(self,fields):
        dt = []
        for flux in self.fluxes:
            dt.append(flux.timestep(fields))
        for force in self.forces:
            dt.append(force.timestep(fields))
        print("dt",dt)
        return jnp.min(jnp.array(dt))
    
    def flux(self,sol,ax,params):
        total_flux = jnp.zeros(sol.shape)
        for flux in self.fluxes:
            total_flux += flux.flux(sol,ax,params)
        return total_flux
    
    def forcing(self,i,sol,params,dt): #all axis independant? 
        total_force = jnp.zeros(sol.shape)
        for force in self.forces:
            total_force += force.force(i,sol,params,dt)
        return total_force
    
    def solve_step(self,sol,dt,ax,params):
        ##RK2 method
        
        fu1 = self.flux(sol,ax,params) 
        #first order upwind
        rhs_cons = (fu1 - jnp.roll(fu1, 1, axis=ax)) 
        
        u1 = sol - rhs_cons * dt / (2.0 * self.dx_o)
        #second order step

        fu = self.flux(sol,ax,params)  
            
        rhs_cons = (fu - jnp.roll(fu, 1, axis=ax))  #fu or fu1?
        
        sol = sol - (rhs_cons) * dt / self.dx_o
        return sol
    
    @jax.checkpoint
    def sweep_stack(self,state,dt,i):
        sol,params,_ = state
        for scheme in self.splitting_schemes:
            for nn,ax in enumerate(scheme):
                sol = self.boundary.impose(sol,ax)
                sol = self.solve_step(sol,dt/len(scheme)*2,int(ax),params)                 
                # experimental
                sol = sol.at[0].set(jnp.abs(sol[0])) #experimental...
                sol = sol.at[-1].set(jnp.abs(sol[-1])) #experimental...
    
        return sol
    
   # @jax.jit
    def evolve(self,input_fields,params):
        
        if not hasattr(self, "_amr_trace"):
            Hc, Wc = input_fields.shape[-2], input_fields.shape[-1]
            self._amr_trace = {
                "depth_maps": [],                  # list[np.ndarray(Hc,Wc)]
                "level_masks": [],                 # list[list[np.ndarray(Hc,Wc)]]
                "steps": [],                       # list[int]
                "first_refined_step": -1 * numpy.ones((Hc, Wc), dtype=numpy.int32),
                "last_refined_step":  -1 * numpy.ones((Hc, Wc), dtype=numpy.int32),
            }
        
        self.outputs=[]
        #main loop
        state = (input_fields,params,None)
        hierarchy = None
        print("evolve")
        #need to rework the UI to get out snapshots from jitted function, hack for now...
        if self.maxjit:
            print("maxjit?")
            state  = jax.lax.fori_loop(0, self.n_super_step, self.hydrostep_adapt, state)
        else:
            print("no, maxjit?")

            for i in range(0,self.n_super_step):
  #              state = self.hydrostep_adapt(i,state)
                
                                    # ---- AMR adaptation outside the tape ----
                if self.use_amr:# and (i % self.adapt_interval == 0):
                    print("AMR!")
                    fields, p, _H = state

                    dens = fields[0]
                    gx = jnp.abs(dens - jnp.roll(dens, 1, axis=0))
                    gy = jnp.abs(dens - jnp.roll(dens, 1, axis=1))
                    indicator = (gx + gy) * 10

                    H, W = dens.shape[-2], dens.shape[-1]
                    tile = 16
                    coarse_tiles = []   # ★ always filled
                    refined_tiles = []  # ★ only if refine==True

                    for y in range(0, H, tile):
                        for x in range(0, W, tile):
                            win = indicator[y:y+tile, x:x+tile]
                            if win.size == 0:
                                continue
                            Utile = fields[:, y:y+tile, x:x+tile]

                            # always keep a coarse tile
                            coarse_tiles.append(
                                AMRBlock(Utile, jnp.ones((1, *Utile.shape[-2:])), (y, x), dx=1.0)
                            )

                            refine = (win.mean() > 0.01 * indicator.mean())
                            if refine:
                                print("refine!", x, y)
                                Ufine = prolong_bilinear(Utile, self.refine_ratio)
                                refined_tiles.append(
                                    AMRBlock(Ufine, jnp.ones((1, *Ufine.shape[-2:])),
                                             (y, x), dx=1.0 / self.refine_ratio)
                                )

                    # ★★★ Build a 2-level hierarchy: L0 = coarse, L1 = refined
                    L0 = AMRLevel(self.refine_ratio, tuple(coarse_tiles))
                    L1 = AMRLevel(self.refine_ratio, tuple(refined_tiles))
                    hierarchy = AMRHierarchy(levels=(L0, L1))
                    print(hierarchy)
                    # stash into state (don’t need to put it in params)
                    state = (fields, p, hierarchy)

                    # ---- trace / bookkeeping (NumPy) ----
                    depth, level_masks = rasterize_hierarchy(hierarchy, input_fields.shape)
                    self._amr_trace["depth_maps"].append(depth)
                    self._amr_trace["level_masks"].append(level_masks)
                    self._amr_trace["steps"].append(i)

                    refined_any = (depth > 0)
                    fr = self._amr_trace["first_refined_step"]
                    lr = self._amr_trace["last_refined_step"]
                    fr[(fr < 0) & refined_any] = i
                    lr[refined_any] = i
                    state = self.hydrostep_adapt(i,state)
                else:
                    state = self.hydrostep_adapt(i,state)

                
                if self.snapshots:
                    if i%self.snapshots==0: #comment out most times...
                        self.outputs.append(state)
        return state
        
    @partial(jit, static_argnums=0)
    def hydrostep_adapt(self,i,state):
        fields,params,_ = state
        ttt = self.timestep(fields)
        ttt = jnp.minimum(self.max_dt,ttt)
        dt = (ttt)
        if self.use_amr:
            return self._hydrostep_amr(i,state,dt)
        else:
            return self._hydrostep(i,state,dt)
    
    @jax.jit
    def _hydrostep_amr(self, i, state, dt):
        fields, params, hierarchy = state
        L0 = hierarchy.levels[0]
        L1 = hierarchy.levels[1] if len(hierarchy.levels) > 1 else AMRLevel(L0.ratio, ())

        # 1) Coarse step (recompute boundary faces)
        L0_new, faces_L0 = advance_L0_without_touching_solver(self, self.fluxes[0], L0, dt, params)

        # 2) Fine prolong + subcycle (recompute boundary faces at each substep)
        if len(L1.blocks):
            L1_new, faces_L1_all = subcycle_L1_without_touching_solver(self, self.fluxes[0], L1, L0_new, dt, L0_new.ratio, params)
            # 3) Restrict state L1→L0 (accuracy)
            L0_sync = restrict_state_from_L1_to_L0(L0_new, L1_new, L0_new.ratio)
            # 4) Reflux (conservation)
            L0_sync = reflux_faces_onto_L0(L0_sync, faces_L0, faces_L1_all, L0_new.ratio, dt, dx=self.dx, dy=self.dy)
            L0_new  = L0_sync
        else:
            L1_new = L1

        # 5) Coalesce L0 tiles back to a canvas
        canvas = jnp.zeros_like(fields)
        for b in L0_new.blocks:
            y0, x0 = b.origin
            h, w = b.U.shape[-2], b.U.shape[-1]
            canvas = canvas.at[:, y0:y0+h, x0:x0+w].set(b.U)

        return (canvas, params, AMRHierarchy((L0_new, L1_new)))

    @jax.jit
    def _hydrostep(self,i,state,dt):
        fields, params, HIER = state

        #save actual timescale used, mostly important if you are using hydro_adapt
#        self.timescale[i].set(dt)
        
#        hydro_output = self.sweep_stack(state,dt,i)

        # If AMR is enabled and a hierarchy is present, do a tile-wise advance.
        if self.use_amr:# and (HIER is not None):
            print("in loop")
#            HIER = params["_amr_hierarchy"]
            level = HIER.levels[0]  # single-level scaffold
            new_blocks = []
            for b in level.blocks:
                print("inblock!")
                sol = b.U
                # reuse existing kernels on the tile
                for scheme in self.splitting_schemes:
                    for nn,ax in enumerate(scheme):
                        sol = self.boundary.impose(sol,ax)
                        sol = self.solve_step(sol,dt/len(scheme)*2,int(ax),params)
                        sol = sol.at[0].set(jnp.abs(sol[0]))
                        sol = sol.at[-1].set(jnp.abs(sol[-1]))
                new_blocks.append(AMRBlock(sol, b.mask, b.origin, b.dx))
            # Coalesce tiles back to a coarse canvas (simple overwrite).
            # (Production: restrict & reflux to a true coarse parent.)
            canvas = jnp.zeros_like(fields)
            tile = new_blocks[0].U.shape[-2]
            idx = 0
            for b in new_blocks:
                y, x = b.origin
                h, w = b.U.shape[-2], b.U.shape[-1]
                canvas = canvas.at[:, y:y+h, x:x+w].set(b.U)
                idx += 1
            hydro_output = canvas
        else:
            hydro_output = self.sweep_stack(state,dt,i)

        fields = hydro_output

        fields = self.forcing(i,fields,params,dt)
            
        return (fields,params,HIER)
    
    def _advance_level0(self, L0, dt, params):
        new_blocks = []
        # Sum of coarse-face flux corrections we’ll apply after fine subcycling
        coarse_face_flux_accum = {"x": [], "y": []}  # add "z" for 3D
        for b in L0.blocks:
            sol = b.U
            face_fluxes_accum = {"x": [], "y": []}
            for scheme in self.splitting_schemes:
                for ax in scheme:
                    sol, fface = self.step_tile_with_fluxes(sol, dt/len(scheme)*2, ax, params)
                    # Collect only boundary faces (left/right for ax)
                    if ax == 1:  # x
                        face_fluxes_accum["x"].append(fface)
                    elif ax == 2:  # y
                        face_fluxes_accum["y"].append(fface)
                    # add z if 3D
            new_blocks.append(AMRBlock(sol, b.mask, b.origin, b.dx))
            coarse_face_flux_accum["x"].append(face_fluxes_accum["x"])
            coarse_face_flux_accum["y"].append(face_fluxes_accum["y"])
        L0_new = AMRLevel(L0.ratio, tuple(new_blocks))
        return L0_new, coarse_face_flux_accum

    def _advance_level1_subcycle(self, L1, L0_parent, dt, ratio, params):
        # Prolong L0 parents into L1 as BC/initialization
        new_blocks = []
        for i, fb in enumerate(L1.blocks):
            # Find matching parent coarse tile; in the scaffold we assume same tiling order.
            Uc = L0_parent.blocks[i].U
            fb_init = prolong_bilinear(Uc, ratio)
            new_blocks.append(AMRBlock(fb_init, fb.mask, fb.origin, dx=fb.dx))
        L1 = AMRLevel(L1.ratio, tuple(new_blocks))

        # Subcycle fine level
        dt_f = dt / ratio
        fine_face_flux_sums = []   # collect per fine block per sweep; we’ll map to coarse faces
        for sub in range(ratio):
            updated = []
            fface_step = []
            for fb in L1.blocks:
                sol = fb.U
                step_fluxes = {"x": [], "y": []}
                for scheme in self.splitting_schemes:
                    for ax in scheme:
                        sol, fface = self.step_tile_with_fluxes(sol, dt_f/len(scheme)*2, ax, params)
                        if ax == 1: step_fluxes["x"].append(fface)
                        elif ax == 2: step_fluxes["y"].append(fface)
                updated.append(AMRBlock(sol, fb.mask, fb.origin, fb.dx))
                fface_step.append(step_fluxes)
            L1 = AMRLevel(L1.ratio, tuple(updated))
            fine_face_flux_sums.append(fface_step)

        return L1, fine_face_flux_sums
    
    def step_tile_with_fluxes(self, sol, dt, ax, params):
        # Before you update cell averages, capture the net face fluxes
        # For clarity, return BOTH the updated tile and the sum of face-aligned fluxes
        # on the *outer boundary* of the tile in the sweep direction.
        # Pseudo-code, since you have multiple schemes:
        # left/right face flux arrays:
        #   FxL: [C, Ny]  (or [C, Nz, Ny] in 3D) at the "left" tile boundary (ax)
        #   FxR: [C, Ny]  (or [C, Nz, Ny]) at the "right" tile boundary (ax)
        # You can compute these exactly where you already form intercell fluxes.
        sol_next = self.solve_step(sol, dt, int(ax), params)
        face_flux = {"ax": ax, "left": FxL, "right": FxR}   # build these inside solve_step path
        return sol_next, face_flux
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def tree_flatten(self):
        #this method is needed for JAX control flow
        children = ()  # arrays / dynamic values
        aux_data = {
                    "boundary":self.boundary,
                    "snapshots":self.snapshots,
                   "splitting_schemes":self.splitting_schemes,
                    "fluxes":self.fluxes,"forces":self.forces,"maxjit":self.maxjit}  # static values
        return (children, aux_data)