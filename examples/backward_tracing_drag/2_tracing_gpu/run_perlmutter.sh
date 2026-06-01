#!/bin/bash
#SBATCH --job-name=gpu_tracing
#SBATCH --account=m4505
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm_%j.log

conda activate mc
python tracing_gpu.py
