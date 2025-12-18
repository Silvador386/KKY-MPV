#!/bin/bash
# Schedule execution of many runs with variable batch sizes
# Run from root folder with: bash scripts/schedule.sh

DATA_ROOT='/scratch.ssd/silvador386/job_14662737.pbs-m1/datasets'
DATASET_SHORTCUT='ft-fewshot'
OUTPUT_DIR="./logs/embeddings"

DEFAULT_BATCH_SIZE=32
DEFAULT_USE_F16_MODEL="false"
DEFAULT_USE_VIEWS="true"
NUM_WORKERS=6

# -----------------------------
# Model definitions
# -----------------------------

declare -A MODEL_ID
declare -A SOURCE
declare -A IMAGE_SIZE
declare -A MEAN
declare -A STD
declare -A EVAL_RESIZE
declare -A BATCH_SIZE_MODEL
declare -A USE_F16_MODEL
declare -A USE_VIEWS

# ----- Predefine models -----
# DINOv3
MODEL_ID["DINOv3_7B_16"]="facebook/dinov3-vit7b16-pretrain-lvd1689m"
SOURCE["DINOv3_7B_16"]="hf"
IMAGE_SIZE["DINOv3_7B_16"]=512
MEAN["DINOv3_7B_16"]="0.485 0.456 0.406"
STD["DINOv3_7B_16"]="0.229 0.224 0.225"
EVAL_RESIZE["DINOv3_7B_16"]="false"
USE_F16_MODEL["DINOv3_7B_16"]="true"

# BVRA-ViT-b16
MODEL_ID["BVRA-ViT-b16"]="hf-hub:BVRA/vit_base_patch16_224.in1k_ft_fungitastic_224"
SOURCE["BVRA-ViT-b16"]="timm"
IMAGE_SIZE["BVRA-ViT-b16"]=224  # Should be 448?
MEAN["BVRA-ViT-b16"]="0.5 0.5 0.5"
STD["BVRA-ViT-b16"]="0.5 0.5 0.5"
EVAL_RESIZE["BVRA-ViT-b16"]="false"

# BVRA-Swin
MODEL_ID["BVRA-Swin-p4w12"]="hf-hub:BVRA/swin_base_patch4_window12_384.in1k_ft_fungitastic_384"
SOURCE["BVRA-Swin-p4w12"]="timm"
IMAGE_SIZE["BVRA-Swin-p4w12"]=224  # Should be 448?
MEAN["BVRA-Swin-p4w12"]="0.5 0.5 0.5"
STD["BVRA-Swin-p4w12"]="0.5 0.5 0.5"
EVAL_RESIZE["BVRA-Swin-p4w12"]="false"

# -----------------------------
# Run All Models
# -----------------------------

for model in "${!MODEL_ID[@]}"; do
    arch="${MODEL_ID[$model]}"
    extractor="${SOURCE[$model]}"
    size="${IMAGE_SIZE[$model]}"
    mean="${MEAN[$model]}"
    std="${STD[$model]}"
    eval_resize="${EVAL_RESIZE[$model]}"

    # Use model-specific batch size if defined, otherwise default
    batch_size="${BATCH_SIZE_MODEL[$model]:-$DEFAULT_BATCH_SIZE}"
    use_f16="${USE_F16_MODEL[$model]:-$DEFAULT_USE_F16_MODEL}"
    use_view="${USE_VIEWS[$model]:-$DEFAULT_USE_VIEWS}"

    echo "-------------------------------------------------------"
    echo "[+] Dataset: $DATASET_SHORTCUT"
    echo "[+] Running model: $model"
    echo "    HF/Timm/HUB id: $arch"
    echo "    Extractor: $extractor"
    echo "    Size: $size"
    echo "    Mean: $mean"
    echo "    Std: $std"
    echo "    Eval Resize: $eval_resize"
    echo "    Batch Size: $batch_size"
    echo "    Num workers: $NUM_WORKERS"
    echo "    F16: $use_f16"
    echo "    Views Transforms: $use_view"
    echo "-------------------------------------------------------"

    PYTHONPATH=. python src/scripts/generate_embeddings.py \
        --catalog "$DATASET_SHORTCUT" \
        --dataset-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR/$DATASET_SHORTCUT" \
        --architecture "$arch" \
        --extractor "$extractor" \
        --image-size "$size" \
        --eval-resize $eval_resize \
        --mean $mean \
        --std $std \
        --batch-size $batch_size \
        --num-workers $NUM_WORKERS \
        --use-f16 $use_f16 \
        --use-views $use_view
done
