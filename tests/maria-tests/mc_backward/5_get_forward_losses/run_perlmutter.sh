#!/bin/bash
#SBATCH --job-name=fwd_losses
#SBATCH --account=m4680
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm_%j.log
#
# Runs 4 independent 1M-particle traces simultaneously (one per GPU).
#
# To accumulate more statistics, resubmit:
#   sbatch run_perlmutter.sh
#
# Each resubmission adds lost particles to output/.

conda activate mc

# One timestamp per job so all 4 GPU outputs are grouped together
RUN_TS=$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python run_gpu.py \
        --gpu_id "$i" \
        --run_tag "${RUN_TS}_gpu${i}" &
done

wait
