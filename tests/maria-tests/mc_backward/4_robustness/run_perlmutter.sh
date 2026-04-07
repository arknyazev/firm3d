#!/bin/bash
#SBATCH --job-name=robustness
#SBATCH --account=m4505
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --time=00:15:00
#SBATCH --output=output/slurm_%A_%a.log
#SBATCH --array=0-5
# ^^^ Each array job uses all 4 GPUs on the node to run 4 perturbation IDs
#     in parallel.  Layout:
#
#   Array job  GPU 0        GPU 1        GPU 2        GPU 3
#   ---------  -----------  -----------  -----------  -----------
#       0      ID  0 (base) ID  1        ID  2        ID  3
#       1      ID  4        ID  5        ID  6        ID  7
#       2      ID  8        ID  9        ID  10       ID  11
#       3      ID  12       ID  13       ID  14       ID  15
#       4      ID  16       ID  17       ID  18       ID  19
#       5      ID  20       ID  21       ID  22       ID  23
#
#   6 nodes × 4 GPUs = 24 runs total (1 baseline + 23 perturbations).
#   Adjust --array=0-N to change ensemble size (covers 4*(N+1) runs).
#
# Submit with:
#   mkdir -p output
#   sbatch run_perlmutter.sh

conda activate mc

BASE=$(( SLURM_ARRAY_TASK_ID * 4 ))

for i in 0 1 2 3; do
    PID=$(( BASE + i ))
    CUDA_VISIBLE_DEVICES=$i python 1_trace_perturbed.py --perturbation_id "$PID" &
done

# Wait for all 4 GPU jobs on this node to finish before SLURM releases it
wait
