#!/bin/bash
#SBATCH --job-name=species107
#SBATCH --output=species107_%j.out
#SBATCH --error=species107_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1

set -Eeuo pipefail

source /etc/miniconda3/etc/profile.d/conda.sh
conda activate /usr/local/share/jupyter/kernels/torch2.2.0

ROOT=/home/larnaud/MARBEC_classification
CODE_ROOT="${ROOT}/Species_recognition_code"
SOURCE_DATASET="${ROOT}/data_spp_seed"
FILTERED_DATASET="${ROOT}/data_spp_107_seed40_clean"
PROJECT="${ROOT}/runs_species_107"
RUN_NAME="${RUN_NAME:-species_107_seed40_slurm}"
MODEL="${ROOT}/yolo11l-cls.pt"

cd "${ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

if [[ -e "${FILTERED_DATASET}" ]]; then
    python "${CODE_ROOT}/src/classification/filter_species_split_107.py" \
        --validate-existing "${FILTERED_DATASET}"
else
    python "${CODE_ROOT}/src/classification/filter_species_split_107.py" \
        --source "${SOURCE_DATASET}" \
        --output "${FILTERED_DATASET}" \
        --mode hardlink
fi

python "${CODE_ROOT}/src/classification/train_species_107.py" \
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

conda deactivate
