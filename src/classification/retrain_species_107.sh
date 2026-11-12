#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_DATASET="${SOURCE_DATASET:-${REPOSITORY_ROOT}/data_spp_seed}"
FILTERED_DATASET="${FILTERED_DATASET:-${REPOSITORY_ROOT}/data_spp_107_seed40}"
MODEL="${MODEL:-yolo11l-cls.pt}"
PROJECT="${PROJECT:-${REPOSITORY_ROOT}/runs_species_107}"
RUN_NAME="${RUN_NAME:-species_107_seed40}"
TAXONOMY="${TAXONOMY:-${REPOSITORY_ROOT}/notebooks/species_ecology.csv}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-30}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-224}"
DEVICE="${DEVICE:-0}"

if [[ ! -f "${TAXONOMY}" && -f "${SCRIPT_DIR}/species_ecology.csv" ]]; then
    TAXONOMY="${SCRIPT_DIR}/species_ecology.csv"
fi

RUN_DIR="${PROJECT}/${RUN_NAME}"
LOG_DIR="${PROJECT}/logs"

if [[ ! -d "${SOURCE_DATASET}/train" || ! -d "${SOURCE_DATASET}/val" ]]; then
    echo "Dataset not found: ${SOURCE_DATASET}" >&2
    exit 1
fi

if [[ ! -f "${TAXONOMY}" ]]; then
    echo "Taxonomy file not found: ${TAXONOMY}" >&2
    exit 1
fi

if [[ -e "${RUN_DIR}" ]]; then
    echo "Training run already exists: ${RUN_DIR}" >&2
    echo "Choose another RUN_NAME; an existing run will not be overwritten." >&2
    exit 1
fi

if [[ -e "${FILTERED_DATASET}" ]]; then
    "${PYTHON}" "${SCRIPT_DIR}/filter_species_split_107.py" \
        --validate-existing "${FILTERED_DATASET}"
else
    "${PYTHON}" "${SCRIPT_DIR}/filter_species_split_107.py" \
        --source "${SOURCE_DATASET}" \
        --output "${FILTERED_DATASET}" \
        --mode symlink
fi

mkdir -p "${LOG_DIR}"

yolo classify train \
    model="${MODEL}" \
    data="${FILTERED_DATASET}" \
    epochs="${EPOCHS}" \
    patience="${PATIENCE}" \
    batch="${BATCH}" \
    imgsz="${IMGSZ}" \
    device="${DEVICE}" \
    workers=8 \
    project="${PROJECT}" \
    name="${RUN_NAME}" \
    exist_ok=False \
    pretrained=True \
    optimizer=auto \
    seed=0 \
    deterministic=True \
    dropout=0.2 \
    cos_lr=False \
    lr0=0.0001 \
    lrf=0.001 \
    momentum=0.937 \
    weight_decay=0.01 \
    warmup_epochs=3.0 \
    translate=0.5 \
    scale=0.5 \
    fliplr=0.5 \
    mixup=0.0 \
    auto_augment=randaugment \
    erasing=0.2 \
    val=True \
    plots=True \
    2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"

"${PYTHON}" "${SCRIPT_DIR}/evaluate_species_classification.py" \
    --model "${RUN_DIR}/weights/best.pt" \
    --data "${FILTERED_DATASET}" \
    --taxonomy "${TAXONOMY}" \
    --output "${RUN_DIR}/publication_evaluation" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}"

cp "${FILTERED_DATASET}/split_manifest.csv" "${RUN_DIR}/publication_evaluation/"
cp "${FILTERED_DATASET}/class_counts.csv" "${RUN_DIR}/publication_evaluation/"
cp "${FILTERED_DATASET}/split_report.json" "${RUN_DIR}/publication_evaluation/"
sha256sum "${RUN_DIR}/weights/best.pt" > "${RUN_DIR}/publication_evaluation/best.pt.sha256"

echo "Training completed: ${RUN_DIR}"
echo "Metrics and figures: ${RUN_DIR}/publication_evaluation"
