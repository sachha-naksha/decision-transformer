#!/bin/bash
#SBATCH --job-name=dt-gym
#SBATCH --account=bhdw-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x-%A_%a.out

# One array task per (env, dataset); submit from the gym/ directory.
ENVS=(hopper hopper hopper halfcheetah halfcheetah halfcheetah walker2d walker2d walker2d)
DATASETS=(medium medium-replay expert medium medium-replay expert medium medium-replay expert)
ENV=${ENVS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL_TYPE=${MODEL_TYPE:-dt}

export PYTHONNOUSERSITE=1
export PYTHONWARNINGS=ignore
PY=/projects/bhdw/asachan/.conda/envs/dt-gym/bin/python

echo "host=$(hostname) partition=$SLURM_JOB_PARTITION env=$ENV dataset=$DATASET model=$MODEL_TYPE"
nvidia-smi -L
$PY -u experiment.py --env $ENV --dataset $DATASET --model_type $MODEL_TYPE
