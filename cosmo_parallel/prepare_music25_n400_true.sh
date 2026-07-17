#!/bin/bash
set -euo pipefail

ROOT="/home/ben.horowitz/DiffHydro_public"
SRC_GENERIC="$ROOT/Nyx/Exec/LyA/music25_ics/music25_n512_true_generic.hdf5"
GENERIC="$ROOT/Nyx/Exec/LyA/music25_ics/music25_n400_true_generic.hdf5"
NYX_BIN="$ROOT/Nyx/Exec/LyA/music25_ics/music25_n400_true.nyx"

set +u
source /home/ben.horowitz/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu
set -u

if [[ ! -f "$SRC_GENERIC" ]]; then
  echo "[error] missing true MUSIC source generic: $SRC_GENERIC" >&2
  exit 1
fi

python "$ROOT/cosmo_parallel/resample_music_generic.py" \
  --input-h5 "$SRC_GENERIC" \
  --output-h5 "$GENERIC" \
  --output-n 400

python "$ROOT/music_generic_to_nyx_binary.py" \
  --music-h5 "$GENERIC" \
  --output-nyx "$NYX_BIN" \
  --box-size-mpch 25.0 \
  --h 0.71 \
  --omega-m 0.27 \
  --omega-b 0.045 \
  --mass-convention total_matter \
  --dtype f8

echo "[done] prepared $GENERIC and $NYX_BIN"
