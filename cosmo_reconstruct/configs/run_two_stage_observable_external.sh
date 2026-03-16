#!/usr/bin/env bash
set -euo pipefail

# Workflow 1:
# 1) Build target mgas/temp from either a Gadget/DiffHydro NPZ or a Nyx plotfile
# 2) Generate LyA-flux or X-ray proxy target cube + central slice PNG
# 3) Run two-stage reconstruction: painted model -> full hydro

source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu

cd /home/ben.horowitz/DiffHydro_public

GPU=${GPU:-3}
OBSERVABLE=${OBSERVABLE:-lya_flux}          # lya_flux | xray_proxy
SOURCE_KIND=${SOURCE_KIND:-gadget_npz}      # gadget_npz | nyx_plotfile
MESH_N=${MESH_N:-128}
BOX_SIZE_MPC_H=${BOX_SIZE_MPC_H:-25.0}
Z_INIT=${Z_INIT:-127.0}
Z_TARGET=${Z_TARGET:-2.0}

# External source inputs
SOURCE_FIELDS_NPZ=${SOURCE_FIELDS_NPZ:-diffhydro_gadgetic_n128_z127to2_hllcd/snapshots/fields_ic_final.npz}
NYX_PLOTFILE=${NYX_PLOTFILE:-Nyx/Exec/LyA/gadget_ic_runs/n128_z2/plt00379}

# Shared physics
H=${H:-0.6711}
OMEGA_M=${OMEGA_M:-0.3}
OMEGA_B=${OMEGA_B:-0.045}
RECON_IC_POWER_SUPPRESSION=${RECON_IC_POWER_SUPPRESSION:-1.0}
TREECOOL_FILE=${TREECOOL_FILE:-diffhydro/nyx_eos/TREECOOL_middle}
LYA_EOS_PATH=${LYA_EOS_PATH:-diffhydro/nyx_eos}
XRAY_COOLING_TABLE_NPZ=${XRAY_COOLING_TABLE_NPZ:-diffhydro_gadgetic_n128_z127to2_hll/nyx_cooling_table.npz}

# Reconstruction settings
STAGE1_ITERS=${STAGE1_ITERS:-250}
STAGE2_ITERS=${STAGE2_ITERS:-500}
STAGE1_ADAM_LR=${STAGE1_ADAM_LR:-0.003}

OUT_ROOT=${OUT_ROOT:-cosmo_reconstruct/outputs/workflow1_external_${SOURCE_KIND}_${OBSERVABLE}}
TARGET_DIR="${OUT_ROOT}/target"
STAGE1_OUT="${OUT_ROOT}/stage1_simple"
STAGE2_OUT="${OUT_ROOT}/stage2_full_hydro"
mkdir -p "${TARGET_DIR}" "${STAGE1_OUT}" "${STAGE2_OUT}"

TARGET_MGAS="${TARGET_DIR}/target_mgas.npy"
TARGET_TEMP="${TARGET_DIR}/target_temp.npy"
TARGET_OBS_3D="${TARGET_DIR}/target_${OBSERVABLE}_3d.npy"
TARGET_OBS_SLICE_PNG="${TARGET_DIR}/target_${OBSERVABLE}_slice_xy.png"

# -----------------------------------------------------------------------------
# Step 1: Build pseudo-CV0 mgas/temp targets from external source
# -----------------------------------------------------------------------------
SOURCE_KIND="${SOURCE_KIND}" \
SOURCE_FIELDS_NPZ="${SOURCE_FIELDS_NPZ}" \
NYX_PLOTFILE="${NYX_PLOTFILE}" \
TARGET_MGAS="${TARGET_MGAS}" \
TARGET_TEMP="${TARGET_TEMP}" \
MESH_N="${MESH_N}" \
python - <<'PY'
import os
from pathlib import Path
import numpy as np

kind = os.environ["SOURCE_KIND"].strip().lower()
out_mgas = Path(os.environ["TARGET_MGAS"])
out_temp = Path(os.environ["TARGET_TEMP"])
mesh_n = int(os.environ["MESH_N"])
out_mgas.parent.mkdir(parents=True, exist_ok=True)

if kind == "gadget_npz":
    src = Path(os.environ["SOURCE_FIELDS_NPZ"]).resolve()
    d = np.load(src)
    if "gas_dh_final" not in d or "temp_dh_final" not in d:
        raise KeyError(f"{src} must contain gas_dh_final and temp_dh_final")
    mgas = np.asarray(d["gas_dh_final"], dtype=np.float32)
    temp = np.asarray(d["temp_dh_final"], dtype=np.float32)
elif kind == "nyx_plotfile":
    import yt

    pf = Path(os.environ["NYX_PLOTFILE"]).resolve()
    if not (pf / "Header").exists():
        raise FileNotFoundError(f"Missing Nyx plotfile Header in {pf}")
    ds = yt.load(str(pf), hint="NyxDataset")
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    rho = np.asarray(cg[("boxlib", "density")], dtype=np.float32)
    temp = np.asarray(cg[("boxlib", "Temp")], dtype=np.float32)
    mgas = rho / max(float(np.mean(rho)), 1.0e-30)
else:
    raise ValueError(f"Unknown SOURCE_KIND={kind}")

if mgas.shape != (mesh_n, mesh_n, mesh_n) or temp.shape != (mesh_n, mesh_n, mesh_n):
    raise ValueError(
        f"Target shape mismatch: mgas={mgas.shape}, temp={temp.shape}, expected={(mesh_n, mesh_n, mesh_n)}"
    )

mgas = np.asarray(mgas, dtype=np.float32)
temp = np.maximum(np.asarray(temp, dtype=np.float32), 1.0)
np.save(out_mgas, mgas[None, ...])
np.save(out_temp, temp[None, ...])
print(f"[ok] wrote {out_mgas}")
print(f"[ok] wrote {out_temp}")
PY

# -----------------------------------------------------------------------------
# Step 2: Generate target observable cube + central xy slice
# -----------------------------------------------------------------------------
OBSERVABLE="${OBSERVABLE}" \
H="${H}" OMEGA_M="${OMEGA_M}" OMEGA_B="${OMEGA_B}" \
Z_TARGET="${Z_TARGET}" BOX_SIZE_MPC_H="${BOX_SIZE_MPC_H}" \
TREECOOL_FILE="${TREECOOL_FILE}" LYA_EOS_PATH="${LYA_EOS_PATH}" \
XRAY_COOLING_TABLE_NPZ="${XRAY_COOLING_TABLE_NPZ}" \
TARGET_MGAS="${TARGET_MGAS}" TARGET_TEMP="${TARGET_TEMP}" \
TARGET_OBS_3D="${TARGET_OBS_3D}" TARGET_OBS_SLICE_PNG="${TARGET_OBS_SLICE_PNG}" \
python - <<'PY'
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from cosmo_reconstruct.src.observable_utils import make_observable_mapper

obs = os.environ["OBSERVABLE"]
mgas = np.load(os.environ["TARGET_MGAS"], mmap_mode="r")[0].astype(np.float32)
temp = np.load(os.environ["TARGET_TEMP"], mmap_mode="r")[0].astype(np.float32)

mapper, _ = make_observable_mapper(
    observable=obs,
    projection="3d",
    los_axis=2,
    z_target=float(os.environ["Z_TARGET"]),
    h=float(os.environ["H"]),
    omega_m=float(os.environ["OMEGA_M"]),
    omega_b=float(os.environ["OMEGA_B"]),
    box_size_mpc_h=float(os.environ["BOX_SIZE_MPC_H"]),
    lya_num_integ_pixels=20,
    lya_delta_floor=1.0e-8,
    lya_temp_floor_k=1.0,
    lya_eos_grid_size=96,
    lya_treecool_file=os.environ["TREECOOL_FILE"],
    lya_logdelta_min=-3.0,
    lya_logdelta_max=4.0,
    lya_logt_min=0.0,
    lya_logt_max=8.0,
    lya_eos_path=os.environ["LYA_EOS_PATH"],
    xray_cooling_table_npz=os.environ["XRAY_COOLING_TABLE_NPZ"],
    xray_temp_floor_k=1.0,
    xray_proxy_kind="cool",
)

obs3d = np.asarray(mapper(jnp.asarray(mgas), jnp.asarray(temp), None), dtype=np.float32)
np.save(os.environ["TARGET_OBS_3D"], obs3d)

iz = obs3d.shape[2] // 2
sl = obs3d[:, :, iz]
if obs.lower() == "lya_flux":
    show = sl
    title = f"Target LyA flux (xy, z={iz})"
else:
    show = np.log10(np.clip(sl, 1.0e-40, None))
    title = f"Target X-ray proxy log10 (xy, z={iz})"

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
im = ax.imshow(show, origin="lower", cmap="magma")
ax.set_title(title)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(os.environ["TARGET_OBS_SLICE_PNG"], dpi=180)
plt.close(fig)

print(f"[ok] wrote {os.environ['TARGET_OBS_3D']}")
print(f"[ok] wrote {os.environ['TARGET_OBS_SLICE_PNG']}")
PY

# -----------------------------------------------------------------------------
# Step 3A: Stage-1 reconstruction (painted DM->gas model)
# -----------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU} python cosmo_reconstruct/run_optimize_cv0_density.py \
  --mesh-n ${MESH_N} \
  --box-size-mpc-h ${BOX_SIZE_MPC_H} \
  --z-init ${Z_INIT} \
  --z-target ${Z_TARGET} \
  --kdk-steps 64 \
  --checkpoint-every 4 \
  --h ${H} --omega-m ${OMEGA_M} --omega-b ${OMEGA_B} \
  --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
  --cv0-mgas-path "${TARGET_MGAS}" \
  --cv0-temp-path "${TARGET_TEMP}" \
  --observable ${OBSERVABLE} \
  --observable-projection 3d \
  --target-source cv0 \
  --compare-space log \
  --noise-sigma 0.05 \
  --prior-weight 1.0 \
  --optimizer adam \
  --adam-lr ${STAGE1_ADAM_LR} \
  --n-iters ${STAGE1_ITERS} \
  --save-every 25 \
  --xray-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}" \
  --lya-treecool-file "${TREECOOL_FILE}" \
  --lya-eos-path "${LYA_EOS_PATH}" \
  --output-dir "${STAGE1_OUT}"

# -----------------------------------------------------------------------------
# Step 3B: Stage-2 reconstruction (full hydro refinement)
# -----------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU} python cosmo_reconstruct/run_optimize_cv0_density_full_hydro.py \
  --mesh-n ${MESH_N} \
  --box-size-mpc-h ${BOX_SIZE_MPC_H} \
  --z-init ${Z_INIT} \
  --z-target ${Z_TARGET} \
  --hydro-steps 128 \
  --solver hll \
  --dual-energy \
  --temperature0 20000 \
  --rho-unit-cgs 1.6e-24 \
  --enable-cooling \
  --cooling-model nyx_table \
  --cooling-stop-gradient \
  --nyx-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}" \
  --nyx-cooling-z-nodes '2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100' \
  --h ${H} --omega-m ${OMEGA_M} --omega-b ${OMEGA_B} \
  --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
  --cv0-mgas-path "${TARGET_MGAS}" \
  --cv0-temp-path "${TARGET_TEMP}" \
  --observable ${OBSERVABLE} \
  --observable-projection 3d \
  --target-source cv0 \
  --compare-space log \
  --noise-sigma 0.05 \
  --prior-weight 1.0 \
  --optimizer lbfgs \
  --lbfgs-linesearch zoom \
  --lbfgs-lr 0.5 \
  --lbfgs-memory 8 \
  --n-iters ${STAGE2_ITERS} \
  --save-every 20 \
  --xray-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}" \
  --lya-treecool-file "${TREECOOL_FILE}" \
  --lya-eos-path "${LYA_EOS_PATH}" \
  --init-white-noise-npy "${STAGE1_OUT}/optimized_white_noise.npy" \
  --output-dir "${STAGE2_OUT}"

printf "\n[done] Workflow 1 complete.\n"
printf "Target observable slice: %s\n" "${TARGET_OBS_SLICE_PNG}"
printf "Stage-1 outputs: %s\n" "${STAGE1_OUT}"
printf "Stage-2 outputs: %s\n" "${STAGE2_OUT}"
