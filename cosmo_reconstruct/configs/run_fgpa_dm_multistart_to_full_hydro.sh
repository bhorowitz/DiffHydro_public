#!/usr/bin/env bash
set -euo pipefail

# Hybrid LyA reconstruction workflow:
# 1) Generate a self-consistent full-hydro target observable (LyA flux).
# 2) Fit FGPA parameters A,B from target flux vs pure-JaxPM DM overdensity:
#       tau = A * (1 + delta_dm)^B, flux = exp(-tau)
# 3) Phase A (multistart): optimize white noise with pure JaxPM + FGPA (ADAM).
# 4) Select best Phase-A run by final loss.
# 5) Phase B: initialize full-hydro optimizer from best Phase-A white noise (L-BFGS).

source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu

cd /home/ben.horowitz/DiffHydro_public

GPU=${GPU:-3}
MESH_N=${MESH_N:-64}
BOX_SIZE_MPC_H=${BOX_SIZE_MPC_H:-25.0}
Z_INIT=${Z_INIT:-127.0}
Z_TARGET=${Z_TARGET:-2.50}
H=${H:-0.6711}
OMEGA_M=${OMEGA_M:-0.3}
OMEGA_B=${OMEGA_B:-0.045}
RECON_IC_POWER_SUPPRESSION=${RECON_IC_POWER_SUPPRESSION:-1.0}

TARGET_SEED=${TARGET_SEED:-31415}
TARGET_SCALE=${TARGET_SCALE:-1.0}
TARGET_NOISE_SIGMA=${TARGET_NOISE_SIGMA:-0.15}
TARGET_NOISE_SEED=${TARGET_NOISE_SEED:-27182}
TARGET_COMPARE_SPACE=${TARGET_COMPARE_SPACE:-linear}   # log | linear

HYDRO_STEPS=${HYDRO_STEPS:-128}
STATE_FLOOR=${STATE_FLOOR:-2e-8}
PRESSURE_FLOOR=${PRESSURE_FLOOR:-1e-7}
HYDRO_TEMP_FLOOR_K=${HYDRO_TEMP_FLOOR_K:-20}

KDK_STEPS_STAGEA=${KDK_STEPS_STAGEA:-64}
KDK_CHECKPOINT_EVERY_STAGEA=${KDK_CHECKPOINT_EVERY_STAGEA:-4}

TREECOOL_FILE=${TREECOOL_FILE:-diffhydro/nyx_eos/TREECOOL_middle}
LYA_EOS_PATH=${LYA_EOS_PATH:-diffhydro/nyx_eos}
LYA_SKEWER_BATCH_SIZE=${LYA_SKEWER_BATCH_SIZE:-128}
LYA_NUM_INTEG_PIXELS=${LYA_NUM_INTEG_PIXELS:-20}
XRAY_COOLING_TABLE_NPZ=${XRAY_COOLING_TABLE_NPZ:-diffhydro_gadgetic_n128_z127to2_hll/nyx_cooling_table.npz}

FGPA_FIT_SOURCE=${FGPA_FIT_SOURCE:-clean}              # clean | noisy
FGPA_RHO_FLOOR=${FGPA_RHO_FLOOR:-1e-6}
FGPA_FLUX_CLIP_MIN=${FGPA_FLUX_CLIP_MIN:-1e-4}
FGPA_FLUX_CLIP_MAX=${FGPA_FLUX_CLIP_MAX:-0.999}
FGPA_A_MIN=${FGPA_A_MIN:-1e-6}
FGPA_A_MAX=${FGPA_A_MAX:-1e4}
FGPA_B_MIN=${FGPA_B_MIN:-0.2}
FGPA_B_MAX=${FGPA_B_MAX:-4.0}

PHASEA_SEEDS=${PHASEA_SEEDS:-0,1,2,3}
PHASEA_ITERS=${PHASEA_ITERS:-250}
PHASEA_ADAM_LR=${PHASEA_ADAM_LR:-3e-4}
PHASEA_NOISE_SIGMA=${PHASEA_NOISE_SIGMA:-0.20}
PHASEA_COMPARE_SPACE=${PHASEA_COMPARE_SPACE:-linear}
PHASEA_PRIOR_WEIGHT=${PHASEA_PRIOR_WEIGHT:-1.0}
PHASEA_SAVE_EVERY=${PHASEA_SAVE_EVERY:-25}
PHASEA_INIT_RANDOM_SCALE=${PHASEA_INIT_RANDOM_SCALE:-1.0}
PHASEA_GRAD_CLIP_NORM=${PHASEA_GRAD_CLIP_NORM:-1.0}

PHASEB_ITERS=${PHASEB_ITERS:-600}
PHASEB_NOISE_SIGMA=${PHASEB_NOISE_SIGMA:-0.10}
PHASEB_COMPARE_SPACE=${PHASEB_COMPARE_SPACE:-linear}
PHASEB_PRIOR_WEIGHT=${PHASEB_PRIOR_WEIGHT:-1.0}
PHASEB_LBFGS_LR=${PHASEB_LBFGS_LR:-0.05}
PHASEB_LBFGS_MEMORY=${PHASEB_LBFGS_MEMORY:-12}
PHASEB_LBFGS_LINESEARCH=${PHASEB_LBFGS_LINESEARCH:-zoom}   # zoom | none
PHASEB_LBFGS_MAX_LINESEARCH_STEPS=${PHASEB_LBFGS_MAX_LINESEARCH_STEPS:-20}
PHASEB_SAVE_EVERY=${PHASEB_SAVE_EVERY:-10}
PHASEB_MAX_ROLLBACKS=${PHASEB_MAX_ROLLBACKS:-20}
PHASEB_ROLLBACK_LOSS_FACTOR=${PHASEB_ROLLBACK_LOSS_FACTOR:-2.5}
PHASEB_TEMP_PS_LOSS_WEIGHT=${PHASEB_TEMP_PS_LOSS_WEIGHT:-0.0}
PHASEB_TEMP_PS_LOSS_NBINS=${PHASEB_TEMP_PS_LOSS_NBINS:-32}
PHASEB_TEMP_PS_LOSS_SPACE=${PHASEB_TEMP_PS_LOSS_SPACE:-log}

OUT_ROOT=${OUT_ROOT:-cosmo_reconstruct/outputs/hybrid_fgpa_dm_to_full_hydro_lya_flux}
TARGET_DIR="${OUT_ROOT}/target"
TARGET_FORWARD_OUT="${TARGET_DIR}/full_hydro_target_forward"
PHASEA_ROOT="${OUT_ROOT}/phaseA_fgpa_dm_multistart"
PHASEB_OUT="${OUT_ROOT}/phaseB_full_hydro"
mkdir -p "${TARGET_DIR}" "${PHASEA_ROOT}" "${PHASEB_OUT}"

TARGET_WHITE_NOISE="${TARGET_DIR}/target_white_noise.npy"
TARGET_MGAS="${TARGET_DIR}/target_mgas.npy"
TARGET_TEMP="${TARGET_DIR}/target_temp.npy"
TARGET_OBS_CLEAN_3D="${TARGET_DIR}/target_lya_flux_clean_3d.npy"
TARGET_OBS_NOISY_3D="${TARGET_DIR}/target_lya_flux_noisy_3d.npy"
TARGET_OBS_CLEAN_SLICE_PNG="${TARGET_DIR}/target_lya_flux_clean_slice_xy.png"
TARGET_OBS_NOISY_SLICE_PNG="${TARGET_DIR}/target_lya_flux_noisy_slice_xy.png"
FGPA_FIT_JSON="${TARGET_DIR}/fgpa_fit.json"

printf "[info] output root: %s\n" "${OUT_ROOT}"
printf "[info] Phase-A seeds: %s\n" "${PHASEA_SEEDS}"

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
# Step 3: Convert final full-hydro snapshot -> target mgas/temp/observable
# -----------------------------------------------------------------------------
TARGET_FORWARD_OUT="${TARGET_FORWARD_OUT}" \
TARGET_MGAS="${TARGET_MGAS}" TARGET_TEMP="${TARGET_TEMP}" \
TARGET_OBS_CLEAN_3D="${TARGET_OBS_CLEAN_3D}" TARGET_OBS_NOISY_3D="${TARGET_OBS_NOISY_3D}" \
TARGET_OBS_CLEAN_SLICE_PNG="${TARGET_OBS_CLEAN_SLICE_PNG}" TARGET_OBS_NOISY_SLICE_PNG="${TARGET_OBS_NOISY_SLICE_PNG}" \
TARGET_COMPARE_SPACE="${TARGET_COMPARE_SPACE}" TARGET_NOISE_SIGMA="${TARGET_NOISE_SIGMA}" TARGET_NOISE_SEED="${TARGET_NOISE_SEED}" \
H="${H}" OMEGA_M="${OMEGA_M}" OMEGA_B="${OMEGA_B}" Z_TARGET="${Z_TARGET}" BOX_SIZE_MPC_H="${BOX_SIZE_MPC_H}" \
TREECOOL_FILE="${TREECOOL_FILE}" LYA_EOS_PATH="${LYA_EOS_PATH}" XRAY_COOLING_TABLE_NPZ="${XRAY_COOLING_TABLE_NPZ}" \
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

np.save(Path(os.environ["TARGET_MGAS"]), mgas[None, ...])
np.save(Path(os.environ["TARGET_TEMP"]), temp[None, ...])

mapper, _ = make_observable_mapper(
    observable="lya_flux",
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

compare_space = os.environ["TARGET_COMPARE_SPACE"].lower()
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
        raise ValueError(f"Unsupported TARGET_COMPARE_SPACE={compare_space}")
else:
    obs3d_noisy = np.asarray(obs3d, dtype=np.float32)
np.save(os.environ["TARGET_OBS_NOISY_3D"], obs3d_noisy)

def _plot_slice(cube: np.ndarray, out_png: str, title: str) -> None:
    iz = cube.shape[2] // 2
    sl = cube[:, :, iz]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(sl, origin="lower", cmap="magma")
    ax.set_title(f"{title} (xy, z={iz})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

_plot_slice(obs3d, os.environ["TARGET_OBS_CLEAN_SLICE_PNG"], "Self-consistent clean LyA flux")
_plot_slice(obs3d_noisy, os.environ["TARGET_OBS_NOISY_SLICE_PNG"], "Self-consistent noisy LyA flux")

print(f"[ok] final snapshot: {snap}")
print(f"[ok] wrote {os.environ['TARGET_MGAS']}")
print(f"[ok] wrote {os.environ['TARGET_TEMP']}")
print(f"[ok] wrote {os.environ['TARGET_OBS_CLEAN_3D']}")
print(f"[ok] wrote {os.environ['TARGET_OBS_NOISY_3D']}")
print(f"[ok] target noise: compare_space={compare_space}, sigma={noise_sigma:.6e}, seed={noise_seed}")
PY

# -----------------------------------------------------------------------------
# Step 4: Fit FGPA A,B from target flux and pure-JaxPM DM field
# -----------------------------------------------------------------------------
FGPA_TARGET_OBS="${TARGET_OBS_CLEAN_3D}"
if [[ "${FGPA_FIT_SOURCE}" == "noisy" ]]; then
  FGPA_TARGET_OBS="${TARGET_OBS_NOISY_3D}"
elif [[ "${FGPA_FIT_SOURCE}" != "clean" ]]; then
  echo "[error] FGPA_FIT_SOURCE must be 'clean' or 'noisy' (got '${FGPA_FIT_SOURCE}')" >&2
  exit 2
fi

read -r FGPA_TAU_A FGPA_TAU_B < <(
TARGET_WHITE_NOISE="${TARGET_WHITE_NOISE}" \
FGPA_TARGET_OBS="${FGPA_TARGET_OBS}" \
FGPA_FIT_JSON="${FGPA_FIT_JSON}" \
MESH_N="${MESH_N}" BOX_SIZE_MPC_H="${BOX_SIZE_MPC_H}" Z_INIT="${Z_INIT}" Z_TARGET="${Z_TARGET}" \
KDK_STEPS_STAGEA="${KDK_STEPS_STAGEA}" KDK_CHECKPOINT_EVERY_STAGEA="${KDK_CHECKPOINT_EVERY_STAGEA}" \
H="${H}" OMEGA_M="${OMEGA_M}" OMEGA_B="${OMEGA_B}" \
FGPA_RHO_FLOOR="${FGPA_RHO_FLOOR}" \
FGPA_FLUX_CLIP_MIN="${FGPA_FLUX_CLIP_MIN}" FGPA_FLUX_CLIP_MAX="${FGPA_FLUX_CLIP_MAX}" \
FGPA_A_MIN="${FGPA_A_MIN}" FGPA_A_MAX="${FGPA_A_MAX}" FGPA_B_MIN="${FGPA_B_MIN}" FGPA_B_MAX="${FGPA_B_MAX}" \
python - <<'PY'
import json
import os
import numpy as np
import jax.numpy as jnp
import jaxpm.pm as jpm

from cosmo_reconstruct.src.forward_model import (
    ForwardModelConfig,
    a_from_z,
    build_cosmology,
    integrate_kdk,
    make_lattice_positions,
    make_pk_sqrt,
    paint_density,
    prime_growth_cache,
    white_noise_to_init_mesh,
)

wn = np.asarray(np.load(os.environ["TARGET_WHITE_NOISE"]), dtype=np.float32)
flux = np.asarray(np.load(os.environ["FGPA_TARGET_OBS"]), dtype=np.float64)

cfg = ForwardModelConfig(
    mesh_n=int(os.environ["MESH_N"]),
    box_size_mpc_h=float(os.environ["BOX_SIZE_MPC_H"]),
    z_init=float(os.environ["Z_INIT"]),
    z_target=float(os.environ["Z_TARGET"]),
    kdk_steps=int(os.environ["KDK_STEPS_STAGEA"]),
    omega_m=float(os.environ["OMEGA_M"]),
    omega_b=float(os.environ["OMEGA_B"]),
    h=float(os.environ["H"]),
    checkpoint=True,
    checkpoint_every=max(1, int(os.environ["KDK_CHECKPOINT_EVERY_STAGEA"])),
)

cosmo = build_cosmology(cfg)
prime_growth_cache(cosmo, a_from_z(cfg.z_init))
grid = make_lattice_positions(cfg.mesh_n)
pk_sqrt = make_pk_sqrt(cosmo, cfg)

wn_j = jnp.asarray(wn, dtype=jnp.float32)
init_mesh = white_noise_to_init_mesh(wn_j, pk_sqrt)
dx, p, _ = jpm.lpt(cosmo, init_mesh, grid, a_from_z(cfg.z_init), order=cfg.lpt_order)
pos0 = jnp.mod(grid + dx, jnp.asarray(float(cfg.mesh_n), dtype=jnp.float32))
posf, _ = integrate_kdk(pos0, p, cosmo, cfg)
rho_dm = paint_density(posf, cfg.mesh_n)
rho_dm = rho_dm / (jnp.mean(rho_dm) + 1.0e-8)
rho_dm_np = np.asarray(rho_dm, dtype=np.float64)

rho_floor = float(os.environ["FGPA_RHO_FLOOR"])
fmin = float(os.environ["FGPA_FLUX_CLIP_MIN"])
fmax = float(os.environ["FGPA_FLUX_CLIP_MAX"])

f = np.clip(flux.ravel(), fmin, fmax)
tau = -np.log(np.clip(f, 1.0e-30, None))
x = np.log(np.clip(rho_dm_np.ravel(), rho_floor, None))
y = np.log(np.clip(tau, 1.0e-30, None))

mask = np.isfinite(x) & np.isfinite(y)
if np.count_nonzero(mask) < 1024:
    raise RuntimeError("Insufficient finite samples for FGPA fit.")
x = x[mask]
y = y[mask]
xm = float(np.mean(x))
ym = float(np.mean(y))
xd = x - xm
yd = y - ym
b = float(np.dot(xd, yd) / max(np.dot(xd, xd), 1.0e-30))
a = float(np.exp(ym - b * xm))

a = float(np.clip(a, float(os.environ["FGPA_A_MIN"]), float(os.environ["FGPA_A_MAX"])))
b = float(np.clip(b, float(os.environ["FGPA_B_MIN"]), float(os.environ["FGPA_B_MAX"])))

tau_fit = a * np.power(np.clip(rho_dm_np, rho_floor, None), b)
flux_fit = np.exp(-tau_fit)
mse = float(np.mean((flux_fit - flux) ** 2))

payload = {
    "fgpa_tau_a": a,
    "fgpa_tau_b": b,
    "rho_floor": rho_floor,
    "fit_source_path": os.environ["FGPA_TARGET_OBS"],
    "fit_flux_clip_min": fmin,
    "fit_flux_clip_max": fmax,
    "fit_mse": mse,
}
with open(os.environ["FGPA_FIT_JSON"], "w", encoding="utf-8") as fjson:
    json.dump(payload, fjson, indent=2, sort_keys=True)

print(f"{a:.12g} {b:.12g}")
PY
)

printf "[fgpa] fitted A=%s B=%s\n" "${FGPA_TAU_A}" "${FGPA_TAU_B}"
printf "[fgpa] fit summary: %s\n" "${FGPA_FIT_JSON}"

# -----------------------------------------------------------------------------
# Step 5: Phase A multistart (pure JaxPM DM + FGPA, ADAM)
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

  if CUDA_VISIBLE_DEVICES=${GPU} python cosmo_reconstruct/run_optimize_cv0_density.py \
    --gpu ${GPU} \
    --mesh-n ${MESH_N} \
    --box-size-mpc-h ${BOX_SIZE_MPC_H} \
    --z-init ${Z_INIT} \
    --z-target ${Z_TARGET} \
    --kdk-steps ${KDK_STEPS_STAGEA} \
    --checkpoint-every ${KDK_CHECKPOINT_EVERY_STAGEA} \
    --h ${H} --omega-m ${OMEGA_M} --omega-b ${OMEGA_B} \
    --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
    --cv0-mgas-path "${TARGET_MGAS}" \
    --cv0-temp-path "${TARGET_TEMP}" \
    --observable fgpa_flux_dm \
    --fgpa-tau-a ${FGPA_TAU_A} \
    --fgpa-tau-b ${FGPA_TAU_B} \
    --fgpa-rho-floor ${FGPA_RHO_FLOOR} \
    --observable-projection 3d \
    --target-source self_consistent \
    --self-target-white-noise-npy "${TARGET_WHITE_NOISE}" \
    --target-observable-npy "${TARGET_OBS_NOISY_3D}" \
    --compare-space ${PHASEA_COMPARE_SPACE} \
    --noise-sigma ${PHASEA_NOISE_SIGMA} \
    --prior-weight ${PHASEA_PRIOR_WEIGHT} \
    --optimizer adam \
    --adam-lr ${PHASEA_ADAM_LR} \
    --grad-clip-norm ${PHASEA_GRAD_CLIP_NORM} \
    --n-iters ${PHASEA_ITERS} \
    --save-every ${PHASEA_SAVE_EVERY} \
    --seed ${seed} \
    --init-random-scale ${PHASEA_INIT_RANDOM_SCALE} \
    --lya-treecool-file "${TREECOOL_FILE}" \
    --lya-eos-path "${LYA_EOS_PATH}" \
    --lya-num-integ-pixels ${LYA_NUM_INTEG_PIXELS} \
    --lya-skewer-batch-size ${LYA_SKEWER_BATCH_SIZE} \
    --xray-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}" \
    --output-dir "${RUN_OUT}"; then
    printf "[phaseA] seed=%s complete\n" "${seed}"
  else
    printf "[phaseA] seed=%s failed, skipping for selection\n" "${seed}" >&2
  fi
done

# -----------------------------------------------------------------------------
# Step 6: Select best Phase-A run
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
    stop_reason = stats.get("run", {}).get("stop_reason", None)
    if isinstance(stop_reason, str) and ("non-finite gradients" in stop_reason.lower()):
        continue
    loss_final = stats.get("loss", {}).get("final_total", None)
    n_iters_completed = stats.get("run", {}).get("n_iters_completed", -1)
    try:
        loss_val = float(loss_final)
        n_iter_val = int(n_iters_completed)
    except Exception:
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
payload = {
    "selection_metric": "min(loss.final_total), tiebreak=max(run.n_iters_completed)",
    "best": best,
    "candidates": candidates,
}
(root / "phaseA_selection.json").write_text(json.dumps(payload, indent=2))
print(best["white_noise"])
PY
)"

printf "[phaseA] selected best init: %s\n" "${BEST_PHASEA_WN}"
printf "[phaseA] selection summary: %s\n" "${PHASEA_ROOT}/phaseA_selection.json"

# -----------------------------------------------------------------------------
# Step 7: Phase B (full hydro from best Phase-A init)
# -----------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU} python cosmo_reconstruct/run_optimize_cv0_density_full_hydro.py \
  --gpu ${GPU} \
  --mesh-n ${MESH_N} \
  --box-size-mpc-h ${BOX_SIZE_MPC_H} \
  --z-init ${Z_INIT} \
  --z-target ${Z_TARGET} \
  --hydro-steps ${HYDRO_STEPS} \
  --solver hll \
  --temperature0 20000 \
  --rho-unit-cgs 1.6e-24 \
  --enable-cooling \
  --dual-energy \
  --cooling-model nyx_table \
  --cooling-stop-gradient \
  --checkpoint-every 8 \
  --nyx-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}" \
  --nyx-cooling-z-nodes '0,1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,60,100' \
  --state-floor ${STATE_FLOOR} \
  --pressure-floor ${PRESSURE_FLOOR} \
  --hydro-temp-floor-k ${HYDRO_TEMP_FLOOR_K} \
  --h ${H} --omega-m ${OMEGA_M} --omega-b ${OMEGA_B} \
  --ic-power-suppression ${RECON_IC_POWER_SUPPRESSION} \
  --cv0-mgas-path "${TARGET_MGAS}" \
  --cv0-temp-path "${TARGET_TEMP}" \
  --observable lya_flux \
  --observable-projection 3d \
  --target-source cv0 \
  --target-observable-npy "${TARGET_OBS_NOISY_3D}" \
  --compare-space ${PHASEB_COMPARE_SPACE} \
  --noise-sigma ${PHASEB_NOISE_SIGMA} \
  --prior-weight ${PHASEB_PRIOR_WEIGHT} \
  --optimizer lbfgs \
  --lbfgs-linesearch ${PHASEB_LBFGS_LINESEARCH} \
  --lbfgs-max-linesearch-steps ${PHASEB_LBFGS_MAX_LINESEARCH_STEPS} \
  --lbfgs-lr ${PHASEB_LBFGS_LR} \
  --lbfgs-memory ${PHASEB_LBFGS_MEMORY} \
  --max-rollbacks ${PHASEB_MAX_ROLLBACKS} \
  --rollback-loss-factor ${PHASEB_ROLLBACK_LOSS_FACTOR} \
  --temp-ps-loss-weight ${PHASEB_TEMP_PS_LOSS_WEIGHT} \
  --temp-ps-loss-nbins ${PHASEB_TEMP_PS_LOSS_NBINS} \
  --temp-ps-loss-space ${PHASEB_TEMP_PS_LOSS_SPACE} \
  --n-iters ${PHASEB_ITERS} \
  --save-every ${PHASEB_SAVE_EVERY} \
  --xray-cooling-table-npz "${XRAY_COOLING_TABLE_NPZ}" \
  --lya-treecool-file "${TREECOOL_FILE}" \
  --lya-eos-path "${LYA_EOS_PATH}" \
  --lya-num-integ-pixels ${LYA_NUM_INTEG_PIXELS} \
  --lya-skewer-batch-size ${LYA_SKEWER_BATCH_SIZE} \
  --init-white-noise-npy "${BEST_PHASEA_WN}" \
  --output-dir "${PHASEB_OUT}"

printf "\n[done] FGPA-DM -> full-hydro hybrid workflow complete.\n"
printf "Target clean flux: %s\n" "${TARGET_OBS_CLEAN_3D}"
printf "Target noisy flux: %s\n" "${TARGET_OBS_NOISY_3D}"
printf "FGPA fit file: %s\n" "${FGPA_FIT_JSON}"
printf "Phase-A outputs: %s\n" "${PHASEA_ROOT}"
printf "Phase-B outputs: %s\n" "${PHASEB_OUT}"
