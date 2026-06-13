#!/bin/bash
#SBATCH --job-name=bwd_tau_sweep
#SBATCH --account=m4680
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --output=output/bwd_tau_sweep_%j.log

conda activate firm3d-maria
set -u

PERT_ID=${PERT_ID:-57}
N_SAMPLES=${N_SAMPLES:-50000}
N_POOL=${N_POOL:-1000000}
N_PILOT=${N_PILOT:-100000}
S_SCORE_NBINS=${S_SCORE_NBINS:-50}
ALPHA_MIX=${ALPHA_MIX:-0.05}
SEED=${SEED:-57}
OUT_ROOT=${OUT_ROOT:-/pscratch/sd/m/mariagar/projects/mc_proj/results/mc_comparison}

SCORE_COORDINATE=${SCORE_COORDINATE:-s}
BACKWARD_TMAX_FACTOR=${BACKWARD_TMAX_FACTOR:-1.0}
N_BOOZER_WORKERS=${N_BOOZER_WORKERS:-32}
TMAX_FORWARD=${TMAX_FORWARD:-1e-2}
TOL=${TOL:-1e-9}
TE0_EV=${TE0_EV:-100}
COULOMB_LOG=${COULOMB_LOG:-17}
H_LOW_MEV=${H_LOW_MEV:-1.0}

FACTORS=(1 5 10 20)
NE0S=(1e21 2e20 1e20 5e19)

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUT_DIR="${OUT_ROOT}/${TIMESTAMP}_bwd_tau_sweep_pert${PERT_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}" output

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${THIS_DIR}"

echo "Backward-informed tau_s sweep: ${OUT_DIR}"

PIDS=()
LABELS=()

for GPU in 0 1 2 3; do
    FACTOR=${FACTORS[$GPU]}
    NE0=${NE0S[$GPU]}
    LABEL="tau_x${FACTOR}"

    echo "Launching ${LABEL}: ne0=${NE0}, Te0_ev=${TE0_EV}, coulomb_log=${COULOMB_LOG}"

    CUDA_VISIBLE_DEVICES=${GPU} python backward_informed_mc_s.py \
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
        --tmax_forward          "${TMAX_FORWARD}" \
        --tol                   "${TOL}" \
        --ne0                   "${NE0}" \
        --Te0_ev                "${TE0_EV}" \
        --coulomb_log           "${COULOMB_LOG}" \
        --H_low_MeV             "${H_LOW_MEV}" \
        --out_dir               "${OUT_DIR}/${LABEL}" \
        > "${LOG_DIR}/${LABEL}.log" 2>&1 &

    PIDS+=($!)
    LABELS+=("${LABEL}")
done

ANY_FAIL=0
for i in "${!PIDS[@]}"; do
    EXIT=0
    wait "${PIDS[$i]}" || EXIT=$?
    echo "${LABELS[$i]} exit=${EXIT} log=${LOG_DIR}/${LABELS[$i]}.log"
    if (( EXIT != 0 )); then ANY_FAIL=1; fi
done

echo "Outputs at: ${OUT_DIR}"
exit "${ANY_FAIL}"