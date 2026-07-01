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
# All three methods share the same perturbation_id (default 57) so the
# perturbed field, fusion pool, and therefore the target Q are identical
# across methods.
#
# Usage:
#     mkdir -p output
#     sbatch run_three_methods_per_perturbation.sh
#
# Override defaults via env vars, e.g.:
#     PERT_ID=57 N_SAMPLES=50000 N_POOL=100000 sbatch run_three_methods_per_perturbation.sh

conda activate firm3d-maria     # CHANGE THIS to your environment on Perlmutter
set -u

# ── Configurable parameters (env overrides) ────────────────────────────────
PERT_ID=${PERT_ID:-0}
N_SAMPLES=${N_SAMPLES:-50000}
N_POOL=${N_POOL:-1000000}
N_PILOT=${N_PILOT:-1000000}
S_SCORE_NBINS=${S_SCORE_NBINS:-50}
ALPHA_MIX=${ALPHA_MIX:-0.05}
SEED=${SEED:-57}
OUT_ROOT=${OUT_ROOT:-/pscratch/sd/m/mariagar/projects/mc_proj/results/mc_comparison}

# Backward-informed IS: coordinate used for the 1-D score histogram.
# 'sd' = signed distance to VMEC LCFS
# 's'  = Boozer s
SCORE_COORDINATE=${SCORE_COORDINATE:-s}

# Backward pilot tmax = BACKWARD_TMAX_FACTOR * ln(H_fusion/H_low) * tau_s
# giving a lot of headroom
BACKWARD_TMAX_FACTOR=${BACKWARD_TMAX_FACTOR:-2.0}

# CPU multiprocessing for the cylindrical->Boozer conversion in
# backward_informed_mc_s.py
N_BOOZER_WORKERS=${N_BOOZER_WORKERS:-32}

ENABLE_VIZ=${ENABLE_VIZ:-1}
VIZ_INDICES=${VIZ_INDICES:-}              # if empty, trajectory_viz uses its own defaults
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

# Wait for all and remember exit statuses
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

# Non-zero exit only if an estimator (not viz) method failed
if (( EXIT_FWD != 0 || EXIT_UNIF != 0 || EXIT_BACK != 0 )); then
    exit 1
fi