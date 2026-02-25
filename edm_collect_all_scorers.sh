#!/bin/bash
### Delta(AI) Cluster ###
#SBATCH --account=beyz-delta-gpu
#SBATCH --partition=gpuA100x4
### Job options ###
#SBATCH --output=slurm_logs/slurm-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%A_%a.out
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=0-8:00
### GPU options ###
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=verbose,closest

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "This script is meant to be run via sbatch (usage: sbatch edm_collect_all_scorers.sh)." >&2
    exit 1
fi

set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
conda activate diffusion-tts

cd /scratch/beyz/mshen2/diffusion-tts
mkdir -p slurm_logs logs/search_stats outputs
mkdir -p /scratch/beyz/mshen2/diffusion-tts/.cache

# Keep cache off home quota.
export XDG_CACHE_HOME=/scratch/beyz/mshen2/diffusion-tts/.cache
export HF_HOME=/scratch/beyz/mshen2/diffusion-tts/.cache/huggingface
export IMAGENET_CLASSIFIER_CACHE_DIR=/scratch/beyz/mshen2/diffusion-tts/.cache/imagenet_classifier

for scorer in imagenet brightness compressibility; do
  python main.py \
    --backend edm \
    --scorer "$scorer" \
    --method eps_greedy \
    --N 4 \
    --K 20 \
    --eps 0.4 \
    --lambda_ 0.15 \
    --seed 0 \
    --log_search_data \
    --search_log_dir logs/search_stats \
    --search_log_prefix "edm_${scorer}_epsgreedy"
done
