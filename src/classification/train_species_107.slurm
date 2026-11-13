#!/bin/bash
#SBATCH --job-name=species107
#SBATCH --output=species107_%j.out
#SBATCH --error=species107_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-${CODE_ROOT}}"
SOURCE_DATASET="${SOURCE_DATASET:-${WORK_ROOT}/data_spp_seed}"
FILTERED_DATASET="${FILTERED_DATASET:-${WORK_ROOT}/data_spp_107}"
PROJECT="${PROJECT:-${WORK_ROOT}/runs_species_107}"
RUN_NAME="${RUN_NAME:-species_107}"
MODEL="${MODEL:-yolo11l-cls.pt}"
PYTHON="${PYTHON:-python}"

if [[ -n "${CONDA_SETUP:-}" && -n "${CONDA_ENV:-}" ]]; then
    source "${CONDA_SETUP}"
    conda activate "${CONDA_ENV}"
fi

cd "${WORK_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"${PYTHON}" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

if [[ -e "${FILTERED_DATASET}" ]]; then
    "${PYTHON}" "${CODE_ROOT}/src/classification/filter_species_split_107.py" \
        --validate-existing "${FILTERED_DATASET}"
else
    "${PYTHON}" "${CODE_ROOT}/src/classification/filter_species_split_107.py" \
        --source "${SOURCE_DATASET}" \
        --output "${FILTERED_DATASET}" \
        --mode hardlink
fi

"${PYTHON}" "${CODE_ROOT}/src/classification/train_species_107.py" \
    --data "${FILTERED_DATASET}" \
    --taxonomy "${CODE_ROOT}/notebooks/species_ecology.csv" \
    --model "${MODEL}" \
    --project "${PROJECT}" \
    --name "${RUN_NAME}" \
    --device 0 \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --batch 32 \
    --imgsz 224 \
    --epochs 150 \
    --patience 30

if [[ -n "${CONDA_SETUP:-}" && -n "${CONDA_ENV:-}" ]]; then
    conda deactivate
fi
