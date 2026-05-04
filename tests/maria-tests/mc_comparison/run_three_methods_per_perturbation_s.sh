#!/bin/bash
#SBATCH --job-name=mc_comparison
#SBATCH --account=m4680
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 3 estimators + (optional) trajectory_viz
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --output=/pscratch/sd/m/mariagar/projects/mc_proj/results/mc_comparison/slurm_logs/slurm_%j.log

#
# Runs the three MC-comparison workflows (forward MC, uniform-s IS,
# backward-informed IS) in parallel on a single 4-GPU node — one process per GPU.
#
#   GPU 0 : forward_mc_perturbed.py
#   GPU 1 : uniform_s_is_perturbed.py
#   GPU 2 : backward_informed_is_perturbed.py
#   GPU 3 : trajectory tracing with trajectory_viz.py.
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
PERT_ID=${PERT_ID:-0}   # no perturbation run
N_SAMPLES=${N_SAMPLES:-50000}
N_POOL=${N_POOL:-1000000}
N_PILOT=${N_PILOT:-1000000}
S_SCORE_NBINS=${S_SCORE_NBINS:-50}
ALPHA_MIX=${ALPHA_MIX:-0.05}
SEED=${SEED:-57}
OUT_ROOT=${OUT_ROOT:-/pscratch/sd/m/mariagar/projects/mc_proj/results/mc_comparison}

# Backward-informed IS: coordinate used for the 1-D score histogram.
# 'sd' = signed distance to VMEC LCFS (always defined, no Boozer inversion).
# 's'  = Boozer s (original spec; requires cylindrical_to_boozer success).
# Set SCORE_COORDINATE=s to restore the original Boozer-s behaviour.
SCORE_COORDINATE=${SCORE_COORDINATE:-s}

# Backward pilot tmax = BACKWARD_TMAX_FACTOR * ln(H_fusion/H_low) * tau_s
# Energy-stop fires first in practice, so this is just headroom.
BACKWARD_TMAX_FACTOR=${BACKWARD_TMAX_FACTOR:-2.0}

# CPU multiprocessing for the cylindrical->Boozer conversion in
# backward_informed_mc_s.py.  Each worker rebuilds its own boozer_field
# upfront in parallel, then handles a slice of the backward birth points.
# Default 16: with --ntasks-per-node=4 the four estimator processes share
# 64 CPU cores on a Perlmutter GPU node, so 16 workers per estimator is
# the natural fit.  Set to 1 to fall back to the old sequential path.
N_BOOZER_WORKERS=${N_BOOZER_WORKERS:-16}

# Trajectory polylines are handled by the separate `trajectory_viz.py`
# script so that deterministic trajectories aren't re-traced 3x.  When
# ENABLE_VIZ=1 (default), we launch it on GPU 3 in parallel with the three
# estimator runs — GPU 3 is otherwise idle.
ENABLE_VIZ=${ENABLE_VIZ:-1}
VIZ_INDICES=${VIZ_INDICES:-}              # empty → trajectory_viz uses its baked defaults
VIZ_TMAX=${VIZ_TMAX:-1e-3}
VIZ_N_SNAPSHOTS=${VIZ_N_SNAPSHOTS:-1000}

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
echo "   SCORE_COORD    = ${SCORE_COORDINATE}  (backward-IS score axis)"
echo "   BWD_TMAX_FAC   = ${BACKWARD_TMAX_FACTOR}"
echo "   N_BWD_WORKERS  = ${N_BOOZER_WORKERS}  (CPU procs for Boozer conv.)"
echo "   ENABLE_VIZ     = ${ENABLE_VIZ}  (GPU 3, trajectory_viz.py)"
echo "============================================================"

# ── Launch three methods, one per GPU ──────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python forward_mc_perturbed.py \
    --perturbation_id "${PERT_ID}" \
    --n_samples       "${N_SAMPLES}" \
    --n_pool          "${N_POOL}" \
    --seed            "${SEED}" \
    --out_dir         "${OUT_DIR}/forward_mc" \
    > "${LOG_DIR}/forward_mc.log" 2>&1 &
PID_FWD=$!

CUDA_VISIBLE_DEVICES=1 python uniform_s_is_perturbed.py \
    --perturbation_id "${PERT_ID}" \
    --n_samples       "${N_SAMPLES}" \
    --n_pool          "${N_POOL}" \
    --s_score_nbins   "${S_SCORE_NBINS}" \
    --seed            "${SEED}" \
    --out_dir         "${OUT_DIR}/uniform_s_is" \
    > "${LOG_DIR}/uniform_s_is.log" 2>&1 &
PID_UNIF=$!

CUDA_VISIBLE_DEVICES=2 python backward_informed_mc_s.py \
    --perturbation_id       "${PERT_ID}" \
    --n_samples             "${N_SAMPLES}" \
    --n_pool                "${N_POOL}" \
    --n_pilot               "${N_PILOT}" \
    --s_score_nbins         "${S_SCORE_NBINS}" \
    --alpha_mix             "${ALPHA_MIX}" \
    --seed                  "${SEED}" \
    --score_coordinate      "${SCORE_COORDINATE}" \
    --backward_tmax_factor  "${BACKWARD_TMAX_FACTOR}" \
    --n_boozer_workers      "${N_BOOZER_WORKERS}" \
    --out_dir               "${OUT_DIR}/backward_informed_is" \
    > "${LOG_DIR}/backward_informed_is.log" 2>&1 &
PID_BACK=$!

# ── Optional 4th GPU: high-res trajectory visualisation ────────────────────
# Runs in parallel with the three estimators on the otherwise-idle GPU 3.
# Uses trajectory_viz.py's baked-in default pool indices unless VIZ_INDICES
# is non-empty.
PID_VIZ=""
if [[ "${ENABLE_VIZ}" != "0" ]]; then
    VIZ_ARGS=(
        --perturbation_id "${PERT_ID}"
        --n_pool          "${N_POOL}"
        --tmax_trajectory "${VIZ_TMAX}"
        --n_snapshots     "${VIZ_N_SNAPSHOTS}"
        --out_dir         "${OUT_DIR}/trajectory_viz"
    )
    if [[ -n "${VIZ_INDICES}" ]]; then
        VIZ_ARGS+=(--viz_indices "${VIZ_INDICES}")
    fi
    CUDA_VISIBLE_DEVICES=3 python trajectory_viz.py "${VIZ_ARGS[@]}" \
        > "${LOG_DIR}/trajectory_viz.log" 2>&1 &
    PID_VIZ=$!
fi

# Wait for all; remember exit statuses.  The viz job is informational —
# its failure does not fail the overall job.
EXIT_FWD=0; EXIT_UNIF=0; EXIT_BACK=0; EXIT_VIZ=0
wait $PID_FWD  || EXIT_FWD=$?
wait $PID_UNIF || EXIT_UNIF=$?
wait $PID_BACK || EXIT_BACK=$?
if [[ -n "${PID_VIZ}" ]]; then
    wait $PID_VIZ || EXIT_VIZ=$?
fi

echo "============================================================"
echo " forward_mc            exit=${EXIT_FWD}   log=${LOG_DIR}/forward_mc.log"
echo " uniform_s_is          exit=${EXIT_UNIF}  log=${LOG_DIR}/uniform_s_is.log"
echo " backward_informed_is  exit=${EXIT_BACK}  log=${LOG_DIR}/backward_informed_is.log"
if [[ -n "${PID_VIZ}" ]]; then
    echo " trajectory_viz        exit=${EXIT_VIZ}  log=${LOG_DIR}/trajectory_viz.log"
fi
echo " Outputs at: ${OUT_DIR}"
echo "============================================================"

# Non-zero exit only if an *estimator* method failed.  A viz failure is
# noted in the log but doesn't mark the job as failed.
if (( EXIT_FWD != 0 || EXIT_UNIF != 0 || EXIT_BACK != 0 )); then
    exit 1
fi
