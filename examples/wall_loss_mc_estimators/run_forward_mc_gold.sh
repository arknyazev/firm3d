#!/bin/bash
#SBATCH --job-name=fwd_mc_gold
#SBATCH --account=m4680
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one FWD shard per GPU
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --output=output/slurm_gold_%j.log
#
# High-precision "gold-standard" FWD estimate of Q, built by sharding
# forward_mc_perturbed.py across the 4 GPUs of a single node.
#
# Usage:
#     mkdir -p output
#     N_TOTAL=100000000 sbatch run_forward_mc_gold.sh
#
# Output layout:
#     <OUT_ROOT>/<timestamp>_gold_pert<PERT_ID>/
#         shard_0/ .. shard_3/   per-GPU forward_mc_perturbed outputs
#         combined/              combined Y_all.npy, metrics_combined.csv
#         logs/                  per-shard + combine logs

conda activate firm3d-maria     # CHANGE THIS to your environment on Perlmutter
set -u

# ── Configurable parameters (env overrides) ────────────────────────────────
PERT_ID=${PERT_ID:-0}
N_TOTAL=${N_TOTAL:-100000000}
N_POOL=${N_POOL:-1000000}
SEED_BASE=${SEED_BASE:-57}        # shard k uses SEED_BASE + k
OUT_ROOT=${OUT_ROOT:-/pscratch/sd/m/mariagar/projects/mc_proj/results/mc_comparison}

# Forward-tracer physics
TMAX_FORWARD=${TMAX_FORWARD:-1e-2}
TOL=${TOL:-1e-9}
NE0=${NE0:-1e21}
TE0_EV=${TE0_EV:-100}
COULOMB_LOG=${COULOMB_LOG:-17}

N_PER_SHARD=$(( N_TOTAL / 4 ))
if (( N_PER_SHARD * 4 != N_TOTAL )); then
    echo "WARNING: N_TOTAL=${N_TOTAL} not divisible by 4; will run "
    echo "         4 x ${N_PER_SHARD} = $(( N_PER_SHARD * 4 )) samples."
fi

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUT_DIR="${OUT_ROOT}/${TIMESTAMP}_gold_pert${PERT_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p output

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${THIS_DIR}"

echo "============================================================"
echo " FWD gold-standard — perturbation_id=${PERT_ID}"
echo "   N_TOTAL      = ${N_TOTAL}"
echo "   N_PER_SHARD  = ${N_PER_SHARD}"
echo "   N_POOL       = ${N_POOL}"
echo "   SEED_BASE    = ${SEED_BASE}   (shard k uses seed = SEED_BASE + k)"
echo "   TMAX_FORWARD = ${TMAX_FORWARD}"
echo "   OUT_DIR      = ${OUT_DIR}"
echo "============================================================"

# ── Launch 4 shards, one per GPU ───────────────────────────────────────────
PIDS=()
for SHARD in 0 1 2 3; do
    SEED=$(( SEED_BASE + SHARD ))
    CUDA_VISIBLE_DEVICES=${SHARD} python forward_mc_perturbed.py \
        --perturbation_id "${PERT_ID}" \
        --n_samples       "${N_PER_SHARD}" \
        --n_pool          "${N_POOL}" \
        --seed            "${SEED}" \
        --tmax_forward    "${TMAX_FORWARD}" \
        --tol             "${TOL}" \
        --ne0             "${NE0}" \
        --Te0_ev          "${TE0_EV}" \
        --coulomb_log     "${COULOMB_LOG}" \
        --out_dir         "${OUT_DIR}/shard_${SHARD}" \
        > "${LOG_DIR}/shard_${SHARD}.log" 2>&1 &
    PIDS+=($!)
done

# Wait for all shards; record exit codes
EXITS=()
for i in "${!PIDS[@]}"; do
    EXIT=0
    wait "${PIDS[$i]}" || EXIT=$?
    EXITS+=("$EXIT")
done

echo "============================================================"
ANY_FAIL=0
for i in 0 1 2 3; do
    echo " shard_${i}: exit=${EXITS[$i]}  log=${LOG_DIR}/shard_${i}.log"
    if (( EXITS[i] != 0 )); then ANY_FAIL=1; fi
done

if (( ANY_FAIL )); then
    echo "One or more shards failed — skipping combine step."
    exit 1
fi

# ── Combine ────────────────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo " Combining shards..."
python combine_forward_mc.py --run_dir "${OUT_DIR}" \
    > "${LOG_DIR}/combine.log" 2>&1
COMBINE_EXIT=$?
echo " combine exit=${COMBINE_EXIT}  log=${LOG_DIR}/combine.log"
echo " Outputs at: ${OUT_DIR}"
echo "============================================================"

if (( COMBINE_EXIT != 0 )); then
    exit 1
fi

# Echo the combined metrics for visibility in the log
if [[ -f "${OUT_DIR}/combined/metrics_combined.csv" ]]; then
    echo "Combined metrics:"
    column -s, -t < "${OUT_DIR}/combined/metrics_combined.csv"
fi