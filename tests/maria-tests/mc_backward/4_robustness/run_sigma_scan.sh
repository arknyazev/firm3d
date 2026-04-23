#!/bin/bash
#SBATCH --job-name=sigma_scan
#SBATCH --account=m4505
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
#
# Run ``3_trace_perturbed_drag.py`` for 4 sigma values in parallel on 1
# node (one GPU per sigma).  The Python script itself knows how to build
# its output path from ``--sigma`` and ``--out_base``, so this wrapper
# just passes CLI args — no sed on a temp copy, no pre-creation of an
# output directory.
#
# Output layout (created by the Python script):
#     /pscratch/.../results/robustness/<sigma>/<timestamp>/
#         bn_stats_NNNN.npy
#         initial_boozer_NNNN.npy   final_time_NNNN.npy   ...
#         loss_summary_NNNN.npy
#
# SLURM stdout goes to ``sigma_scan_<jobid>.log`` in the submission dir
# (the ``%x_%j.log`` directive) — nothing to ``mkdir`` up front.
#
# Usage on Perlmutter:
#     cd tests/maria-tests/mc_backward/4_robustness
#     sbatch run_sigma_scan.sh
#
# To change the perturbation seed or the sigma list:
#     PERT_ID=12 sbatch run_sigma_scan.sh
#     SIGMAS="1e-3 5e-3 1e-2 5e-2" sbatch run_sigma_scan.sh

conda activate firm3d-maria
set -u

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${THIS_DIR}"

PERT_ID=${PERT_ID:-57}
SIGMAS_STR=${SIGMAS:-"1e-2 5e-2 1e-3 5e-3"}
read -r -a SIGMAS <<< "${SIGMAS_STR}"

OUT_BASE=${OUT_BASE:-/pscratch/sd/m/mariagar/projects/mc_proj/results/robustness}

echo "================================================================"
echo " sigma scan with drag — perturbation_id=${PERT_ID}"
echo "   sigmas      : ${SIGMAS[*]}"
echo "   out_base    : ${OUT_BASE}"
echo "   per-sigma logs are streamed into ${OUT_BASE}/<sigma>/<timestamp>/"
echo "   SLURM stdout: ${SLURM_JOB_NAME:-sigma_scan}_${SLURM_JOB_ID:-local}.log"
echo "================================================================"

PIDS=()
SIGMA_LOGS=()
for i in 0 1 2 3; do
    if (( i >= ${#SIGMAS[@]} )); then
        break
    fi
    SIGMA=${SIGMAS[$i]}
    # Mirror the Python script's OUT_DIR construction so we can
    # stream per-run stdout into the same directory.
    RUN_STAMP=$(date +%Y-%m-%d_%H-%M-%S)
    OUT_DIR="${OUT_BASE}/${SIGMA}/${RUN_STAMP}"
    mkdir -p "${OUT_DIR}"
    LOG="${OUT_DIR}/stdout.log"

    CUDA_VISIBLE_DEVICES=$i python 3_trace_perturbed_drag.py \
        --perturbation_id "${PERT_ID}" \
        --sigma           "${SIGMA}" \
        --out_base        "${OUT_BASE}" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
    SIGMA_LOGS+=("${SIGMA}::${LOG}")
    echo "  launched sigma=${SIGMA} on GPU ${i}  pid=${PIDS[-1]}  log=${LOG}"
    # 1-second stagger so each process gets a unique timestamp for its own
    # internal OUT_DIR construction inside Python (it calls datetime.now()
    # independently, and the timestamp granularity is 1s).
    sleep 1
done

EXITS=()
for PID in "${PIDS[@]}"; do
    EXIT=0
    wait "${PID}" || EXIT=$?
    EXITS+=("${EXIT}")
done

echo ""
echo "================================================================"
echo " Per-sigma loss fractions"
echo "================================================================"
for i in "${!SIGMA_LOGS[@]}"; do
    ENTRY=${SIGMA_LOGS[$i]}
    SIGMA=${ENTRY%%::*}
    LOG=${ENTRY##*::}
    # The Python script prints the summary line:
    #   Perturbation   57 — sigma=1e-2  — Lost: 1949/49990 (3.899%)
    LOSS_LINE=$(grep -E "Lost:" "${LOG}" | tail -n 1 || true)
    if [[ -z "${LOSS_LINE}" ]]; then
        LOSS_LINE="(no 'Lost:' line found — check ${LOG})"
    fi
    echo "  sigma=${SIGMA}  exit=${EXITS[$i]}  ${LOSS_LINE}"
done
echo "================================================================"
echo " Full per-sigma logs: in the per-run ${OUT_BASE}/<sigma>/<timestamp>/ dirs"
echo "================================================================"

for EXIT in "${EXITS[@]}"; do
    if (( EXIT != 0 )); then exit 1; fi
done
