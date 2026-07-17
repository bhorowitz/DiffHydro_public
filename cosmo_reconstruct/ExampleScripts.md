# Workflow 1: Gadget/DiffHydro NPZ -> LyA target -> two-stage reconstruction
OBSERVABLE=lya_flux \
SOURCE_KIND=gadget_npz \
SOURCE_FIELDS_NPZ=diffhydro_gadgetic_n128_z127to2_hllcd/snapshots/fields_ic_final.npz \
GPU=3 \
bash cosmo_reconstruct/configs/run_two_stage_observable_external.sh

# Workflow 1: Nyx plotfile -> X-ray target -> two-stage reconstruction
OBSERVABLE=xray_proxy \
SOURCE_KIND=nyx_plotfile \
NYX_PLOTFILE=Nyx/Exec/LyA/gadget_ic_runs/n128_z2/plt00379 \
GPU=3 \
bash cosmo_reconstruct/configs/run_two_stage_observable_external.sh

# Workflow 2: Full-Hydro self-consistent target -> LyA -> two-stage reconstruction
OBSERVABLE=lya_flux \
TARGET_SEED=31415 \
GPU=3 \
bash cosmo_reconstruct/configs/run_two_stage_observable_self_consistent_full_hydro.sh

# Workflow 2: Full-Hydro self-consistent target -> X-ray -> two-stage reconstruction
OBSERVABLE=xray_proxy \
TARGET_SEED=31415 \
GPU=2 \
bash cosmo_reconstruct/configs/run_two_stage_observable_self_consistent_full_hydro.sh
