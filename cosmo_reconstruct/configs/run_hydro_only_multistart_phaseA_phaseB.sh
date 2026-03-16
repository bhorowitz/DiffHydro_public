#!/usr/bin/env bash
set -euo pipefail

# Hydro-only two-phase reconstruction with multistart:
# 1) Generate a self-consistent Full-Hydro target from fixed white-noise IC
# 2) Build observable target cube (LyA flux or X-ray proxy)
# 3) Phase A: run multiple ADAM starts from random white-noise ICs
# 4) Select best Phase-A run by final loss
# 5) Phase B: run L-BFGS from selected Phase-A white noise

source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu

cd /home/ben.horowitz/DiffHydro_public

GPU=${GPU:-3}
OBSERVABLE=${OBSERVABLE:-lya_flux}               # lya_flux | xray_proxy
COMPARE_SPACE=${COMPARE_SPACE:-linear}            # log | linear
MESH_N=${MESH_N:-64}
BOX_SIZE_MPC_H=${BOX_SIZE_MPC_H:-25.0}
Z_INIT=${Z_INIT:-127.0}
Z_TARGET=${Z_TARGET:-2.50}
H=${H:-0.6711}
OMEGA_M=${OMEGA_M:-0.3}
OMEGA_B=${OMEGA_B:-0.045}
RECON_IC_POWER_SUPPRESSION=${RECON_IC_POWER_SUPPRESSION:-1.0}

HYDRO_STEPS=${HYDRO_STEPS:-128}
TARGET_SEED=${TARGET_SEED:-31415}
TARGET_SCALE=${TARGET_SCALE:-1.0}
TARGET_NOISE_SIGMA=${TARGET_NOISE_SIGMA:-0.15}
TARGET_NOISE_SEED=${TARGET_NOISE_SEED:-27182}

STATE_FLOOR=${STATE_FLOOR:-2e-8}
PRESSURE_FLOOR=${PRESSURE_FLOOR:-1e-7}
HYDRO_TEMP_FLOOR_K=${HYDRO_TEMP_FLOOR_K:-20}

TREECOOL_FILE=${TREECOOL_FILE:-diffhydro/nyx_eos/TREECOOL_middle}
LYA_EOS_PATH=${LYA_EOS_PATH:-diffhydro/nyx_eos}
LYA_SKEWER_BATCH_SIZE=${LYA_SKEWER_BATCH_SIZE:-128}
LYA_NUM_INTEG_PIXELS=${LYA_NUM_INTEG_PIXELS:-20}
XRAY_COOLING_TABLE_NPZ=${XRAY_COOLING_TABLE_NPZ:-diffhydro_gadgetic_n128_z127to2_hll/nyx_cooling_table.npz}

PHASEA_SEEDS=${PHASEA_SEEDS:-0,1,2,3}
PHASEA_ITERS=${PHASEA_ITERS:-250}
PHASEA_ADAM_LR=${PHASEA_ADAM_LR:-3e-4}
PHASEA_NOISE_SIGMA=${PHASEA_NOISE_SIGMA:-0.20}
PHASEA_SAVE_EVERY=${PHASEA_SAVE_EVERY:-25}
PHASEA_INIT_RANDOM_SCALE=${PHASEA_INIT_RANDOM_SCALE:-1.0}
PHASEA_MAX_ROLLBACKS=${PHASEA_MAX_ROLLBACKS:-20}
PHASEA_ROLLBACK_LOSS_FACTOR=${PHASEA_ROLLBACK_LOSS_FACTOR:-2.5}
PHASEA_GRAD_CLIP_NORM=${PHASEA_GRAD_CLIP_NORM:-1.0}

PHASEB_ITERS=${PHASEB_ITERS:-600}
PHASEB_NOISE_SIGMA=${PHASEB_NOISE_SIGMA:-0.10}
PHASEB_LBFGS_LR=${PHASEB_LBFGS_LR:-0.05}
PHASEB_LBFGS_MEMORY=${PHASEB_LBFGS_MEMORY:-12}
PHASEB_LBFGS_LINESEARCH=${PHASEB_LBFGS_LINESEARCH:-zoom}    # zoom | none
PHASEB_LBFGS_MAX_LINESEARCH_STEPS=${PHASEB_LBFGS_MAX_LINESEARCH_STEPS:-20}
PHASEB_SAVE_EVERY=${PHASEB_SAVE_EVERY:-10}
PHASEB_MAX_ROLLBACKS=${PHASEB_MAX_ROLLBACKS:-20}
PHASEB_ROLLBACK_LOSS_FACTOR=${PHASEB_ROLLBACK_LOSS_FACTOR:-2.5}

OUT_ROOT=${OUT_ROOT:-cosmo_reconstruct/outputs/hydro_only_multistart_${OBSERVABLE}}
TARGET_DIR="${OUT_ROOT}/target"
TARGET_FORWARD_OUT="${TARGET_DIR}/full_hydro_target_forward"
PHASEA_ROOT="${OUT_ROOT}/phaseA_multistart"
PHASEB_OUT="${OUT_ROOT}/phaseB_lbfgs"
mkdir -p "${TARGET_DIR}" "${PHASEA_ROOT}" "${PHASEB_OUT}"

TARGET_WHITE_NOISE="${TARGET_DIR}/target_white_noise.npy"
TARGET_MGAS="${TARGET_DIR}/target_mgas.npy"
TARGET_TEMP="${TARGET_DIR}/target_temp.npy"
TARGET_OBS_CLEAN_3D="${TARGET_DIR}/target_${OBSERVABLE}_clean_3d.npy"
TARGET_OBS_NOISY_3D="${TARGET_DIR}/target_${OBSERVABLE}_noisy_3d.npy"
TARGET_OBS_CLEAN_SLICE_PNG="${TARGET_DIR}/target_${OBSERVABLE}_clean_slice_xy.png"
TARGET_OBS_NOISY_SLICE_PNG="${TARGET_DIR}/target_${OBSERVABLE}_noisy_slice_xy.png"

printf "[info] output root: %s\n" "${OUT_ROOT}"
printf "[info] phase-A seeds: %s\n" "${PHASEA_SEEDS}"

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
  --hydro-steps ${HYDRO_STEPS} \
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
  --save-every-steps ${HYDRO_STEPS} \
  --output-dir "${TARGET_FORWARD_OUT}" \
  --state-floor ${STATE_FLOOR} \
  --hydro-temp-floor-k ${HYDRO_TEMP_FLOOR_K} \
  --pressure-floor ${PRESSURE_FLOOR}

# -----------------------------------------------------------------------------
# Step 3: Convert final snapshot -> target mgas/temp + target observable
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
LYA_NUM_INTEG_PIXELS="${LYA_NUM_INTEG_PIXELS}" LYA_SKEWER_BATCH_SIZE="${LYA_SKEWER_BATCH_SIZE}" \
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
    lya_num_integ_pixels=int(os.environ["LYA_NUM_INTEG_PIXELS"]),
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
    lya_skewer_batch_size=int(os.environ["LYA_SKEWER_BATCH_SIZE"]),
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
print(f"[ok] noise model: compare_space={compare_space}, sigma={noise_sigma:.6e}, seed={noise_seed}")
PY

# -----------------------------------------------------------------------------
# Shared argument block for full-hydro optimizer
# -----------------------------------------------------------------------------
COMMON_OPT_ARGS=(
  --gpu "${GPU}"
  --mesh-n "${MESH_N}"
  --box-size-mpc-h "${BOX_SIZE_MPC_H}"
  --z-init "${Z_INIT}"
  --z-target "${Z_TARGET}"
  --hydro-steps "${HYDRO_STEPS}"
  --solver hll
  --temperature0 20000
  --rho-unit-cgs 1.6e-24
  --enable-cooling
  --dual-energy
  --cooling-model nyx_table
  --cooling-stop-gradient
  --checkpoint-every 8
  --nyx-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}"
  --nyx-cooling-z-nodes '0,1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100'
  --state-floor "${STATE_FLOOR}"
  --pressure-floor "${PRESSURE_FLOOR}"
  --hydro-temp-floor-k "${HYDRO_TEMP_FLOOR_K}"
  --h "${H}" --omega-m "${OMEGA_M}" --omega-b "${OMEGA_B}"
  --ic-power-suppression "${RECON_IC_POWER_SUPPRESSION}"
  --cv0-mgas-path "${TARGET_MGAS}"
  --cv0-temp-path "${TARGET_TEMP}"
  --observable "${OBSERVABLE}"
  --observable-projection 3d
  --target-source cv0
  --target-observable-npy "${TARGET_OBS_NOISY_3D}"
  --compare-space "${COMPARE_SPACE}"
  --prior-weight 1.0
  --xray-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}"
  --lya-treecool-file "${TREECOOL_FILE}"
  --lya-eos-path "${LYA_EOS_PATH}"
  --lya-num-integ-pixels "${LYA_NUM_INTEG_PIXELS}"
  --lya-skewer-batch-size "${LYA_SKEWER_BATCH_SIZE}"
)

# -----------------------------------------------------------------------------
# Step 4: Phase A multistart (ADAM from random white noise)
# -----------------------------------------------------------------------------
IFS=',' read -r -a PHASEA_SEED_ARR <<< "${PHASEA_SEEDS}"
if [[ ${#PHASEA_SEED_ARR[@]} -eq 0 ]]; then
  echo "[error] PHASEA_SEEDS produced an empty seed list." >&2
  exit 2
fi

for raw_seed in "${PHASEA_SEED_ARR[@]}"; do
  seed="$(echo "${raw_seed}" | tr -d '[:space:]')"
  if [[ -z "${seed}" ]]; then
    continue
  fi

  RUN_OUT="${PHASEA_ROOT}/seed_${seed}"
  mkdir -p "${RUN_OUT}"
  printf "[phaseA] seed=%s out=%s\n" "${seed}" "${RUN_OUT}"

  if CUDA_VISIBLE_DEVICES=${GPU} python cosmo_reconstruct/run_optimize_cv0_density_full_hydro.py \
    "${COMMON_OPT_ARGS[@]}" \
    --noise-sigma "${PHASEA_NOISE_SIGMA}" \
    --optimizer adam \
    --adam-lr "${PHASEA_ADAM_LR}" \
    --grad-clip-norm "${PHASEA_GRAD_CLIP_NORM}" \
    --max-rollbacks "${PHASEA_MAX_ROLLBACKS}" \
    --rollback-loss-factor "${PHASEA_ROLLBACK_LOSS_FACTOR}" \
    --n-iters "${PHASEA_ITERS}" \
    --save-every "${PHASEA_SAVE_EVERY}" \
    --seed "${seed}" \
    --init-random-scale "${PHASEA_INIT_RANDOM_SCALE}" \
    --output-dir "${RUN_OUT}"; then
    printf "[phaseA] seed=%s complete\n" "${seed}"
  else
    printf "[phaseA] seed=%s failed, skipping for selection\n" "${seed}" >&2
  fi
done

# -----------------------------------------------------------------------------
# Step 5: Select best Phase-A run and launch Phase B (L-BFGS)
# -----------------------------------------------------------------------------
BEST_PHASEA_WN="$(
PHASEA_ROOT="${PHASEA_ROOT}" python - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["PHASEA_ROOT"])
candidates = []
for run_dir in sorted(root.glob("seed_*")):
    stats_path = run_dir / "optimize_stats.json"
    wn_path = run_dir / "optimized_white_noise.npy"
    if not stats_path.exists() or not wn_path.exists():
        continue
    try:
        stats = json.loads(stats_path.read_text())
    except Exception:
        continue
    loss_final = stats.get("loss", {}).get("final_total", None)
    n_iters_completed = stats.get("run", {}).get("n_iters_completed", -1)
    stop_reason = stats.get("run", {}).get("stop_reason", None)
    try:
        loss_val = float(loss_final)
        n_iter_val = int(n_iters_completed)
    except Exception:
        continue
    if isinstance(stop_reason, str) and ("non-finite gradients" in stop_reason.lower()):
        continue
    if not math.isfinite(loss_val):
        continue
    candidates.append(
        {
            "run_dir": str(run_dir),
            "loss_final": loss_val,
            "n_iters_completed": n_iter_val,
            "stop_reason": stop_reason,
            "white_noise": str(wn_path),
            "stats_path": str(stats_path),
        }
    )

if not candidates:
    raise SystemExit("No successful Phase-A runs with finite final loss were found.")

candidates.sort(key=lambda x: (x["loss_final"], -x["n_iters_completed"]))
best = candidates[0]

selection = {
    "selection_metric": "min(loss.final_total), tiebreak=max(run.n_iters_completed)",
    "best": best,
    "candidates": candidates,
}
(root / "phaseA_selection.json").write_text(json.dumps(selection, indent=2))
print(best["white_noise"])
PY
)"

printf "[phaseA] selected best init: %s\n" "${BEST_PHASEA_WN}"
printf "[phaseA] selection summary: %s\n" "${PHASEA_ROOT}/phaseA_selection.json"

CUDA_VISIBLE_DEVICES=${GPU} python cosmo_reconstruct/run_optimize_cv0_density_full_hydro.py \
  "${COMMON_OPT_ARGS[@]}" \
  --noise-sigma "${PHASEB_NOISE_SIGMA}" \
  --optimizer lbfgs \
  --lbfgs-linesearch "${PHASEB_LBFGS_LINESEARCH}" \
  --lbfgs-max-linesearch-steps "${PHASEB_LBFGS_MAX_LINESEARCH_STEPS}" \
  --lbfgs-lr "${PHASEB_LBFGS_LR}" \
  --lbfgs-memory "${PHASEB_LBFGS_MEMORY}" \
  --max-rollbacks "${PHASEB_MAX_ROLLBACKS}" \
  --rollback-loss-factor "${PHASEB_ROLLBACK_LOSS_FACTOR}" \
  --n-iters "${PHASEB_ITERS}" \
  --save-every "${PHASEB_SAVE_EVERY}" \
  --init-white-noise-npy "${BEST_PHASEA_WN}" \
  --output-dir "${PHASEB_OUT}"

printf "\n[done] Hydro-only multistart Phase A/B workflow complete.\n"
printf "Target observable clean cube: %s\n" "${TARGET_OBS_CLEAN_3D}"
printf "Target observable noisy cube: %s\n" "${TARGET_OBS_NOISY_3D}"
printf "Target forward snapshots: %s\n" "${TARGET_FORWARD_OUT}/snapshots"
printf "Phase-A outputs: %s\n" "${PHASEA_ROOT}"
printf "Phase-B outputs: %s\n" "${PHASEB_OUT}"
