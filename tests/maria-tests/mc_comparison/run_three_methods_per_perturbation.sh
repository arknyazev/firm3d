#!/bin/bash
#SBATCH --job-name=mc_comparison
#SBATCH --account=m4505
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3          # one task per method; leaves GPU 3 idle
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm_%j.log
#
# Runs the three MC-comparison workflows (forward MC, uniform-s IS,
# backward-informed IS) in parallel on a single 4-GPU node — one process per
# GPU.  GPU 3 is intentionally left unused (only 3 methods).
#
#   GPU 0 : forward_mc_perturbed.py
#   GPU 1 : uniform_s_is_perturbed.py
#   GPU 2 : backward_informed_is_perturbed.py
#
# All three methods share the same perturbation_id (default 57) so the
# perturbed field, fusion pool, and therefore the target Q are identical
# across methods.  Outputs are written under a single timestamped root so
# the three methods' metrics can be compared side-by-side.
#
# Usage (interactive / after `sbatch`):
#     mkdir -p output
#     sbatch run_three_methods_per_perturbation.sh
#
# Override defaults via env vars, e.g.:
#     PERT_ID=57 N_SAMPLES=50000 N_POOL=100000 sbatch run_three_methods_per_perturbation.sh
#
# Optional: run for several perturbation ids in one submission with an
# array job, e.g. `#SBATCH --array=0,57,1,2,3` and use
# PERT_ID=$SLURM_ARRAY_TASK_ID below.

conda activate firm3d-maria
set -u

# ── Configurable parameters (env overrides) ────────────────────────────────
PERT_ID=${PERT_ID:-57}
N_SAMPLES=${N_SAMPLES:-100000}
N_POOL=${N_POOL:-1000000}
N_PILOT=${N_PILOT:-1000000}
S_SCORE_NBINS=${S_SCORE_NBINS:-40}
ALPHA_MIX=${ALPHA_MIX:-0.05}
SEED=${SEED:-57}
OUT_ROOT=${OUT_ROOT:-/pscratch/sd/m/mariagar/projects/mc_proj/results/mc_comparison}

# Trajectory polylines for Paraview
SAVE_TRAJECTORIES=${SAVE_TRAJECTORIES:-1}
N_TRAJECTORY=${N_TRAJECTORY:-50}
N_SNAPSHOTS=${N_SNAPSHOTS:-100}

TRAJ_FLAGS=()
if [[ "${SAVE_TRAJECTORIES}" != "0" ]]; then
    TRAJ_FLAGS=(--save_trajectories
                --n_trajectory "${N_TRAJECTORY}"
                --n_snapshots  "${N_SNAPSHOTS}")
fi

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUT_DIR="${OUT_ROOT}/${TIMESTAMP}_pert${PERT_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p output

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${THIS_DIR}"

echo "============================================================"
echo " MC comparison — perturbation_id=${PERT_ID}"
echo "   N_SAMPLES      = ${N_SAMPLES}"
echo "   N_POOL         = ${N_POOL}"
echo "   N_PILOT        = ${N_PILOT}"
echo "   S_SCORE_NBINS  = ${S_SCORE_NBINS}"
echo "   ALPHA_MIX      = ${ALPHA_MIX}"
echo "   SEED           = ${SEED}"
echo "   OUT_DIR        = ${OUT_DIR}"
echo "   SAVE_TRAJ      = ${SAVE_TRAJECTORIES} (N_TRAJ=${N_TRAJECTORY}, N_SNAP=${N_SNAPSHOTS})"
echo "============================================================"

# ── Launch three methods, one per GPU ──────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python forward_mc_perturbed.py \
    --perturbation_id "${PERT_ID}" \
    --n_samples       "${N_SAMPLES}" \
    --n_pool          "${N_POOL}" \
    --seed            "${SEED}" \
    --out_dir         "${OUT_DIR}/forward_mc" \
    "${TRAJ_FLAGS[@]}" \
    > "${LOG_DIR}/forward_mc.log" 2>&1 &
PID_FWD=$!

CUDA_VISIBLE_DEVICES=1 python uniform_s_is_perturbed.py \
    --perturbation_id "${PERT_ID}" \
    --n_samples       "${N_SAMPLES}" \
    --n_pool          "${N_POOL}" \
    --s_score_nbins   "${S_SCORE_NBINS}" \
    --seed            "${SEED}" \
    --out_dir         "${OUT_DIR}/uniform_s_is" \
    "${TRAJ_FLAGS[@]}" \
    > "${LOG_DIR}/uniform_s_is.log" 2>&1 &
PID_UNIF=$!

CUDA_VISIBLE_DEVICES=2 python backward_informed_is_perturbed.py \
    --perturbation_id "${PERT_ID}" \
    --n_samples       "${N_SAMPLES}" \
    --n_pool          "${N_POOL}" \
    --n_pilot         "${N_PILOT}" \
    --s_score_nbins   "${S_SCORE_NBINS}" \
    --alpha_mix       "${ALPHA_MIX}" \
    --seed            "${SEED}" \
    --out_dir         "${OUT_DIR}/backward_informed_is" \
    "${TRAJ_FLAGS[@]}" \
    > "${LOG_DIR}/backward_informed_is.log" 2>&1 &
PID_BACK=$!

# Wait for all three; remember exit statuses.
EXIT_FWD=0; EXIT_UNIF=0; EXIT_BACK=0
wait $PID_FWD  || EXIT_FWD=$?
wait $PID_UNIF || EXIT_UNIF=$?
wait $PID_BACK || EXIT_BACK=$?

echo "============================================================"
echo " forward_mc            exit=${EXIT_FWD}   log=${LOG_DIR}/forward_mc.log"
echo " uniform_s_is          exit=${EXIT_UNIF}  log=${LOG_DIR}/uniform_s_is.log"
echo " backward_informed_is  exit=${EXIT_BACK}  log=${LOG_DIR}/backward_informed_is.log"
echo " Outputs at: ${OUT_DIR}"
echo "============================================================"

# Non-zero exit if any method failed, so SLURM marks the job accordingly.
if (( EXIT_FWD != 0 || EXIT_UNIF != 0 || EXIT_BACK != 0 )); then
    exit 1
fi
