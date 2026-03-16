#!/usr/bin/env bash
set -euo pipefail

# Workflow 2:
# 1) Generate a self-consistent Full-Hydro target from a fixed white-noise IC
# 2) Build LyA flux or X-ray proxy target cube + central slice PNG
# 3) Run two-stage reconstruction (painted model -> full hydro) against that target

source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu

cd /home/ben.horowitz/DiffHydro_public

GPU=${GPU:-3}
OBSERVABLE=${OBSERVABLE:-lya_flux}          # lya_flux | xray_proxy
COMPARE_SPACE=${COMPARE_SPACE:-linear}           # log | linear
MESH_N=${MESH_N:-64}
BOX_SIZE_MPC_H=${BOX_SIZE_MPC_H:-25.0}
Z_INIT=${Z_INIT:-127.0}
Z_TARGET=${Z_TARGET:-2.50}
H=${H:-0.6711}
OMEGA_M=${OMEGA_M:-0.3}
OMEGA_B=${OMEGA_B:-0.045}
RECON_IC_POWER_SUPPRESSION=${RECON_IC_POWER_SUPPRESSION:-0.150}

HYDRO_STEPS_TARGET=${HYDRO_STEPS_TARGET:-128}
TARGET_SEED=${TARGET_SEED:-31415}
TARGET_SCALE=${TARGET_SCALE:-1.0}
TARGET_NOISE_SIGMA=${TARGET_NOISE_SIGMA:-0.15}
TARGET_NOISE_SEED=${TARGET_NOISE_SEED:-27182}
STATE_FLOOR=${STATE_FLOOR:-2e-8}

TREECOOL_FILE=${TREECOOL_FILE:-diffhydro/nyx_eos/TREECOOL_middle}
LYA_EOS_PATH=${LYA_EOS_PATH:-diffhydro/nyx_eos}
LYA_SKEWER_BATCH_SIZE=${LYA_SKEWER_BATCH_SIZE:-128}
LYA_NUM_INTEG_PIXELS=${LYA_NUM_INTEG_PIXELS:-20}
XRAY_COOLING_TABLE_NPZ=${XRAY_COOLING_TABLE_NPZ:-diffhydro_gadgetic_n128_z127to2_hll/nyx_cooling_table.npz}

STAGE1_ITERS=${STAGE1_ITERS:-0}
STAGE2_ITERS=${STAGE2_ITERS:-1000}
STAGE1_ADAM_LR=${STAGE1_ADAM_LR:-0.001}
STAGE2_OPTIMIZER=${STAGE2_OPTIMIZER:-adam}
STAGE2_ADAM_LR=${STAGE2_ADAM_LR:-0.005}
STAGE2_LBFGS_LR=${STAGE2_LBFGS_LR:-0.1}
STAGE2_LBFGS_MEMORY=${STAGE2_LBFGS_MEMORY:-8}
STAGE2_LBFGS_LINESEARCH=${STAGE2_LBFGS_LINESEARCH:-zoom}   # zoom | none
STAGE2_LBFGS_MAX_LINESEARCH_STEPS=${STAGE2_LBFGS_MAX_LINESEARCH_STEPS:-20}
STAGE2_INIT_FROM=${STAGE2_INIT_FROM:-iter0}      # final | iter0
STAGE2_CHECKPOINT_EVERY=${STAGE2_CHECKPOINT_EVERY:-8}

OUT_ROOT=${OUT_ROOT:-cosmo_reconstruct/outputs/n64_ss_${OBSERVABLE}}
TARGET_DIR="${OUT_ROOT}/target"
TARGET_FORWARD_OUT="${TARGET_DIR}/full_hydro_target_forward"
STAGE1_OUT="${OUT_ROOT}/stage1_simple"
STAGE2_OUT="${OUT_ROOT}/stage2_full_hydro"
mkdir -p "${TARGET_DIR}" "${STAGE1_OUT}" "${STAGE2_OUT}"

TARGET_WHITE_NOISE="${TARGET_DIR}/target_white_noise.npy"
TARGET_MGAS="${TARGET_DIR}/target_mgas.npy"
TARGET_TEMP="${TARGET_DIR}/target_temp.npy"
TARGET_OBS_CLEAN_3D="${TARGET_DIR}/target_${OBSERVABLE}_clean_3d.npy"
TARGET_OBS_NOISY_3D="${TARGET_DIR}/target_${OBSERVABLE}_noisy_3d.npy"
TARGET_OBS_CLEAN_SLICE_PNG="${TARGET_DIR}/target_${OBSERVABLE}_clean_slice_xy.png"
TARGET_OBS_NOISY_SLICE_PNG="${TARGET_DIR}/target_${OBSERVABLE}_noisy_slice_xy.png"

if [[ "${STAGE2_INIT_FROM}" == "final" ]]; then
  STAGE2_INIT_WHITE_NOISE="${STAGE1_OUT}/optimized_white_noise.npy"
elif [[ "${STAGE2_INIT_FROM}" == "iter0" ]]; then
  STAGE2_INIT_WHITE_NOISE="${STAGE1_OUT}/checkpoints/iter_000000/optimized_white_noise.npy"
else
  echo "[error] STAGE2_INIT_FROM must be 'final' or 'iter0' (got: ${STAGE2_INIT_FROM})" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Step 1: Build deterministic target white-noise IC
# -----------------------------------------------------------------------------
TARGET_WHITE_NOISE="${TARGET_WHITE_NOISE}" TARGET_SEED="${TARGET_SEED}" TARGET_SCALE="${TARGET_SCALE}" MESH_N="${MESH_N}" \
python - <<'PY'
import os
from pathlib import Path
import numpy as np

out = Path(os.environ["TARGET_WHITE_NOISE"])
seed = int(os.environ["TARGET_SEED"])
scale = float(os.environ["TARGET_SCALE"])
n = int(os.environ["MESH_N"])

rng = np.random.default_rng(seed)
wn = (scale * rng.standard_normal(size=(n, n, n))).astype(np.float32)
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, wn)
print(f"[ok] wrote {out}")
PY

# -----------------------------------------------------------------------------
# Step 2: Run full hydro forward model from target IC and write snapshots
# -----------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU} python cosmo_reconstruct/run_generate_full_hydro_animation_snapshots.py \
  --gpu ${GPU} \
  --init-white-noise "${TARGET_WHITE_NOISE}" \
  --mesh-n ${MESH_N} \
  --box-size-mpc-h ${BOX_SIZE_MPC_H} \
  --z-init ${Z_INIT} \
  --z-target ${Z_TARGET} \
  --hydro-steps ${HYDRO_STEPS_TARGET} \
  --solver hll \
  --dual-energy \
  --temperature0 20000 \
  --rho-unit-cgs 1.6e-24 \
  --enable-cooling \
  --cooling-model nyx_table \
  --cooling-stop-gradient \
  --h ${H} --omega-m ${OMEGA_M} --omega-b ${OMEGA_B} \
  --nyx-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}" \
  --nyx-cooling-z-nodes '0,1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100' \
  --save-every-steps ${HYDRO_STEPS_TARGET} \
  --output-dir "${TARGET_FORWARD_OUT}"\
  --state-floor ${STATE_FLOOR} \
  --hydro-temp-floor-k 20 \
  --pressure-floor 1e-7 

# -----------------------------------------------------------------------------
# Step 3: Convert final full-hydro snapshot -> pseudo-CV0 mgas/temp and observable
# -----------------------------------------------------------------------------
TARGET_FORWARD_OUT="${TARGET_FORWARD_OUT}" \
TARGET_MGAS="${TARGET_MGAS}" TARGET_TEMP="${TARGET_TEMP}" \
TARGET_OBS_CLEAN_3D="${TARGET_OBS_CLEAN_3D}" TARGET_OBS_NOISY_3D="${TARGET_OBS_NOISY_3D}" \
TARGET_OBS_CLEAN_SLICE_PNG="${TARGET_OBS_CLEAN_SLICE_PNG}" TARGET_OBS_NOISY_SLICE_PNG="${TARGET_OBS_NOISY_SLICE_PNG}" \
OBSERVABLE="${OBSERVABLE}" \
COMPARE_SPACE="${COMPARE_SPACE}" TARGET_NOISE_SIGMA="${TARGET_NOISE_SIGMA}" TARGET_NOISE_SEED="${TARGET_NOISE_SEED}" \
H="${H}" OMEGA_M="${OMEGA_M}" OMEGA_B="${OMEGA_B}" \
Z_TARGET="${Z_TARGET}" BOX_SIZE_MPC_H="${BOX_SIZE_MPC_H}" \
TREECOOL_FILE="${TREECOOL_FILE}" LYA_EOS_PATH="${LYA_EOS_PATH}" \
XRAY_COOLING_TABLE_NPZ="${XRAY_COOLING_TABLE_NPZ}" \
python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from cosmo_reconstruct.src.observable_utils import make_observable_mapper

forward_out = Path(os.environ["TARGET_FORWARD_OUT"])
manifest = json.loads((forward_out / "snapshot_manifest.json").read_text())
if not manifest.get("snapshots"):
    raise RuntimeError(f"No snapshots listed in {forward_out / 'snapshot_manifest.json'}")
last = manifest["snapshots"][-1]
snap = forward_out / "snapshots" / last["filename"]
d = np.load(snap)

mgas = np.asarray(d["gas_density_norm"], dtype=np.float32)
temp = np.asarray(d["gas_temperature_kelvin"], dtype=np.float32)

out_mgas = Path(os.environ["TARGET_MGAS"])
out_temp = Path(os.environ["TARGET_TEMP"])
out_mgas.parent.mkdir(parents=True, exist_ok=True)
np.save(out_mgas, mgas[None, ...])
np.save(out_temp, temp[None, ...])

mapper, _ = make_observable_mapper(
    observable=os.environ["OBSERVABLE"],
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
np.save(os.environ["TARGET_OBS_CLEAN_3D"], obs3d)

compare_space = os.environ["COMPARE_SPACE"].lower()
noise_sigma = float(os.environ["TARGET_NOISE_SIGMA"])
noise_seed = int(os.environ["TARGET_NOISE_SEED"])
rng = np.random.default_rng(noise_seed)
if noise_sigma > 0.0:
    if compare_space == "log":
        noisy = np.log(np.clip(obs3d, 1.0e-20, None)) + noise_sigma * rng.standard_normal(size=obs3d.shape)
        obs3d_noisy = np.exp(noisy).astype(np.float32)
    elif compare_space == "linear":
        obs3d_noisy = (obs3d + noise_sigma * rng.standard_normal(size=obs3d.shape)).astype(np.float32)
#        obs3d_noisy = np.maximum(obs3d_noisy, 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported COMPARE_SPACE={compare_space}")
else:
    obs3d_noisy = np.asarray(obs3d, dtype=np.float32)
np.save(os.environ["TARGET_OBS_NOISY_3D"], obs3d_noisy)

def _plot_slice(cube: np.ndarray, out_png: str, title_prefix: str) -> None:
    iz = cube.shape[2] // 2
    sl = cube[:, :, iz]
    if os.environ["OBSERVABLE"].lower() == "lya_flux":
        show = sl
        title = f"{title_prefix} LyA flux (xy, z={iz})"
    else:
        show = np.log10(np.clip(sl, 1.0e-40, None))
        title = f"{title_prefix} X-ray proxy log10 (xy, z={iz})"
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(show, origin="lower", cmap="magma")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

_plot_slice(obs3d, os.environ["TARGET_OBS_CLEAN_SLICE_PNG"], "Self-consistent clean target")
_plot_slice(obs3d_noisy, os.environ["TARGET_OBS_NOISY_SLICE_PNG"], "Self-consistent noisy target")

print(f"[ok] final snapshot: {snap}")
print(f"[ok] wrote {out_mgas}")
print(f"[ok] wrote {out_temp}")
print(f"[ok] wrote {os.environ['TARGET_OBS_CLEAN_3D']}")
print(f"[ok] wrote {os.environ['TARGET_OBS_NOISY_3D']}")
print(f"[ok] wrote {os.environ['TARGET_OBS_CLEAN_SLICE_PNG']}")
print(f"[ok] wrote {os.environ['TARGET_OBS_NOISY_SLICE_PNG']}")
print(f"[ok] noise model: compare_space={compare_space}, sigma={noise_sigma:.6e}, seed={noise_seed}")

mg_std = float(np.std(mgas))
temp_std = float(np.std(temp))
if (not np.isfinite(mg_std)) or (not np.isfinite(temp_std)) or mg_std < 1.0e-4 or temp_std < 1.0e-2:
    raise RuntimeError(
        "Generated target appears numerically collapsed/flat "
        f"(mgas std={mg_std:.3e}, temp std={temp_std:.3e}). "
        "Try a higher z-target (e.g. z=2), smaller time steps, or different solver settings."
    )
PY

# -----------------------------------------------------------------------------
# Step 4A: Stage-1 reconstruction (painted DM->gas model)
# -----------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU} python cosmo_reconstruct/run_optimize_cv0_density.py \
  --gpu ${GPU} \
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
  --target-observable-npy "${TARGET_OBS_NOISY_3D}" \
  --compare-space ${COMPARE_SPACE} \
  --noise-sigma ${TARGET_NOISE_SIGMA} \
  --prior-weight 1.0 \
  --optimizer adam \
  --adam-lr ${STAGE1_ADAM_LR} \
  --n-iters ${STAGE1_ITERS} \
  --save-every 5 \
  --xray-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}" \
  --lya-treecool-file "${TREECOOL_FILE}" \
  --lya-eos-path "${LYA_EOS_PATH}" \
  --lya-num-integ-pixels ${LYA_NUM_INTEG_PIXELS} \
  --lya-skewer-batch-size ${LYA_SKEWER_BATCH_SIZE} \
  --output-dir "${STAGE1_OUT}"

# -----------------------------------------------------------------------------
# Step 4B: Stage-2 reconstruction (full hydro refinement)
# -----------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU} python cosmo_reconstruct/run_optimize_cv0_density_full_hydro.py \
  --gpu ${GPU} \
  --mesh-n ${MESH_N} \
  --box-size-mpc-h ${BOX_SIZE_MPC_H} \
  --z-init ${Z_INIT} \
  --z-target ${Z_TARGET} \
  --hydro-steps ${HYDRO_STEPS_TARGET}  \
  --solver hll \
  --temperature0 20000 \
  --rho-unit-cgs 1.6e-24 \
  --enable-cooling \
  --dual-energy \
  --cooling-model nyx_table \
  --cooling-stop-gradient \
  --checkpoint-every ${STAGE2_CHECKPOINT_EVERY} \
  --nyx-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}" \
  --nyx-cooling-z-nodes '0,1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100' \
  --state-floor ${STATE_FLOOR} \
  --h ${H} --omega-m ${OMEGA_M} --omega-b ${OMEGA_B} \
  --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
  --cv0-mgas-path "${TARGET_MGAS}" \
  --cv0-temp-path "${TARGET_TEMP}" \
  --observable ${OBSERVABLE} \
  --observable-projection 3d \
  --target-source cv0 \
  --target-observable-npy "${TARGET_OBS_NOISY_3D}" \
  --compare-space ${COMPARE_SPACE} \
  --noise-sigma ${TARGET_NOISE_SIGMA} \
  --prior-weight 1.0 \
  --optimizer ${STAGE2_OPTIMIZER} \
  --adam-lr ${STAGE2_ADAM_LR} \
  --lbfgs-linesearch ${STAGE2_LBFGS_LINESEARCH} \
  --lbfgs-max-linesearch-steps ${STAGE2_LBFGS_MAX_LINESEARCH_STEPS} \
  --lbfgs-lr ${STAGE2_LBFGS_LR} \
  --lbfgs-memory ${STAGE2_LBFGS_MEMORY} \
  --n-iters ${STAGE2_ITERS} \
  --save-every 10 \
  --xray-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}" \
  --lya-treecool-file "${TREECOOL_FILE}" \
  --lya-eos-path "${LYA_EOS_PATH}" \
  --lya-num-integ-pixels ${LYA_NUM_INTEG_PIXELS} \
  --lya-skewer-batch-size ${LYA_SKEWER_BATCH_SIZE} \
  --init-white-noise-npy "${STAGE2_INIT_WHITE_NOISE}" \
  --output-dir "${STAGE2_OUT}" \
  --hydro-temp-floor-k 20 \
  --pressure-floor 1e-7 

printf "\n[done] Workflow 2 complete.\n"
printf "Target observable clean slice: %s\n" "${TARGET_OBS_CLEAN_SLICE_PNG}"
printf "Target observable noisy slice: %s\n" "${TARGET_OBS_NOISY_SLICE_PNG}"
printf "Target observable clean cube: %s\n" "${TARGET_OBS_CLEAN_3D}"
printf "Target observable noisy cube: %s\n" "${TARGET_OBS_NOISY_3D}"
printf "Target forward snapshots: %s\n" "${TARGET_FORWARD_OUT}/snapshots"
printf "Stage-1 outputs: %s\n" "${STAGE1_OUT}"
printf "Stage-2 outputs: %s\n" "${STAGE2_OUT}"
