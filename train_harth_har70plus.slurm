#!/bin/bash
# HARTH -> HAR70plus smoke test job.
#
# Example:
#   EPOCHS_PRETRAIN=20 EPOCHS_FINETUNE=50 sbatch train_harth_har70plus.slurm

#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --job-name=harth_har70plus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=logs/harth_har70plus_%j.out
#SBATCH --error=logs/harth_har70plus_%j.err

set -euo pipefail

mkdir -p logs
source setup_idun_environment.sh
bash run_harth_har70plus.sh
