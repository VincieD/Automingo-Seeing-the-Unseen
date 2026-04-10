#!/bin/bash
#PBS -q iti@pbs-m1.metacentrum.cz
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb:cl_alfrid=True
#PBS -N Automingo_Qwen3VL_Sweep
#PBS -j oe

set -euo pipefail

export WANDB_ENTITY="ovalach"
export WANDB_PROJECT="Automingo-Qwen3VL"
# Existing sweep id
SWEEP_ID="TBD"

PROJECT_ROOT="TBD"
LOG_DIR="TBD"
AGENT_LABEL="qwen3vl_automingo_sweep_${PBS_JOBID:-local}"
RUN_LOG_DIR="${LOG_DIR}/${AGENT_LABEL}"

mkdir -p "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/finetune_sweep.log") 2>&1

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RUN_LOG_DIR=${RUN_LOG_DIR}"

cd "${PROJECT_ROOT}"

module load mambaforge
conda activate YOUR_ENVIROMENT

source .env
unset WANDB_NAME

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

# Sweep run settings
DATASET="ibarcelo/Automingo_dataset"
TRAIN_SPLIT="train"
EVAL_SPLIT="validation"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="Qwen3-VL-8B-Instruct-Automingo-sweep"
DATASET_FORMAT="automingo"
IMAGE_PAYLOAD_MODE="multi"
PRECISION="fp16"

# Default setup
LEARNING_RATE="2e-4"
WARMUP_RATIO="0.03"
WEIGHT_DECAY="0.01"
LORA_R="32"
LORA_ALPHA="32"
LORA_DROPOUT="0.0"
PER_DEVICE_TRAIN_BATCH_SIZE="4"
GRADIENT_ACCUMULATION_STEPS="8"
MAX_STEPS="200"
LOGGING_STEPS="5"
EVAL_STEPS="20"
SEED="42"

MIN_PIXELS="0"
MAX_PIXELS="401408"     # 512*28*28 ~ 848*480 cost
export MIN_PIXELS MAX_PIXELS

LOG_EVAL_SAMPLES="0"
export LOG_EVAL_SAMPLES

EVAL_LIMIT="256"
export EVAL_LIMIT

MAP_BATCH_SIZE="8"
MAP_WRITER_BATCH_SIZE="8"
export MAP_BATCH_SIZE MAP_WRITER_BATCH_SIZE

DEBUG_MODE="0"   # 1 enables 10/2/2 schedule
PUSH_TO_HUB="0"  # 1 enables hub upload
AGENT_COUNT="24"  # runs handled by this job

export DATASET TRAIN_SPLIT EVAL_SPLIT MODEL_NAME OUTPUT_DIR DATASET_FORMAT IMAGE_PAYLOAD_MODE PRECISION
export LEARNING_RATE WARMUP_RATIO WEIGHT_DECAY LORA_R LORA_ALPHA LORA_DROPOUT
export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS MAX_STEPS LOGGING_STEPS EVAL_STEPS SEED
export DEBUG_MODE PUSH_TO_HUB

if [[ -z "${SWEEP_ID}" ]]; then
  echo "ERROR: SWEEP_ID is empty. Create a sweep once and paste its id here." >&2
  echo "Example: python finetune_sweep.py --sweep --project \"${WANDB_PROJECT}\" --entity \"${WANDB_ENTITY}\"" >&2
  exit 1
fi

echo "Using sweep id: ${SWEEP_ID}"
echo "Launching wandb agent (count=${AGENT_COUNT})..."
wandb agent --count "${AGENT_COUNT}" "${SWEEP_ID}"
