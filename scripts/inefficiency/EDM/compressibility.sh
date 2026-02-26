#!/bin/bash
### Delta(AI) Cluster ###
#SBATCH --account=beyz-delta-gpu
#SBATCH --partition=gpuA100x4

### Job options ###
#SBATCH --output=slurm_logs/slurm-%A.out
#SBATCH --error=slurm_logs/slurm-%A.out
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=0-8:00

### GPU options ###
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=verbose,closest

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Run with: sbatch scripts/inefficiency/EDM/compressibility.sh" >&2
  exit 1
fi

# Keep cache off home quota.
export HOME=/scratch/beyz/mshen2
export XDG_CACHE_HOME=/scratch/beyz/mshen2/diffusion-tts/.cache
export HF_HOME=/scratch/beyz/mshen2/diffusion-tts/.cache/huggingface
export IMAGENET_CLASSIFIER_CACHE_DIR=/scratch/beyz/mshen2/diffusion-tts/.cache/imagenet_classifier
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$IMAGENET_CLASSIFIER_CACHE_DIR"

echo "HOME redirected to: $HOME"

if ! command -v conda >/dev/null 2>&1; then
  source "/u/mshen2/miniconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
conda activate diffusion-tts

cd /scratch/beyz/mshen2/diffusion-tts
mkdir -p slurm_logs

# -------- Experiment config --------
SEED=0
CLASS_INDICES="0,1,2,3,4,5,6,7,8,9"
IFS=',' read -r -a CLASS_ARRAY <<< "${CLASS_INDICES}"
N_CLASSES="${#CLASS_ARRAY[@]}"
BACKEND="edm"
SCORER="compressibility"
METHOD="eps_greedy"
N=4
K=20
EPS=0.4
LAMBDA=0.15

EXPERIMENT_NAME="inefficiency"
MODEL_NAME="EDM"
REWARD_NAME="COMPRESSIBILITY"

RUN_TAG="${METHOD}_c${N_CLASSES}_seed${SEED}_N${N}_K${K}_eps${EPS}_lam${LAMBDA}"
LOG_DIR="logs/search_stats/${EXPERIMENT_NAME}/${MODEL_NAME}/${REWARD_NAME}/${RUN_TAG}"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}/${MODEL_NAME}/${REWARD_NAME}/${RUN_TAG}"
CLASS_FILE="${OUTPUT_DIR}/class_indices.txt"
OUTPUT_STEM="${OUTPUT_DIR}/images/${BACKEND}_${SCORER}_${METHOD}.png"

mkdir -p "$LOG_DIR" "${OUTPUT_DIR}/images"
echo "${CLASS_INDICES}" > "${CLASS_FILE}"

echo "[run] experiment=${EXPERIMENT_NAME} model=${MODEL_NAME} reward=${REWARD_NAME}"
echo "[run] run_tag=${RUN_TAG}"
echo "[run] backend=${BACKEND} scorer=${SCORER} method=${METHOD}"
echo "[run] hyperparams: N=${N} K=${K} eps=${EPS} lambda=${LAMBDA} seed=${SEED}"
echo "[run] class indices: ${CLASS_INDICES}"
echo "[run] logs -> ${LOG_DIR}"
echo "[run] outputs -> ${OUTPUT_DIR}"

python main.py \
  --backend "${BACKEND}" \
  --scorer "${SCORER}" \
  --method "${METHOD}" \
  --class_indices "${CLASS_INDICES}" \
  --output "${OUTPUT_STEM}" \
  --N "${N}" \
  --K "${K}" \
  --eps "${EPS}" \
  --lambda_ "${LAMBDA}" \
  --seed "${SEED}" \
  --log_search_data \
  --search_log_dir "${LOG_DIR}" \
  --search_log_prefix "${RUN_TAG}"

echo "Done."
echo "Logs: ${LOG_DIR}"
echo "Outputs: ${OUTPUT_DIR}"
