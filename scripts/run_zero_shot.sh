#!/bin/bash
# Script to run zero-shot evaluation on TextVQA

# Default parameters
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
SPLIT="validation"
BATCH_SIZE=8
OUTPUT_DIR="./results/zero_shot"
MAX_SAMPLES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="--max_samples $2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Running Zero-Shot Evaluation"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Split: $SPLIT"
echo "Batch Size: $BATCH_SIZE"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

# Run evaluation
python evaluation/zero_shot_eval.py \
    --model_name "$MODEL_NAME" \
    --split "$SPLIT" \
    --batch_size $BATCH_SIZE \
    --output_dir "$OUTPUT_DIR" \
    --use_hf_direct \
    $MAX_SAMPLES

echo ""
echo "=========================================="
echo "Zero-shot evaluation completed!"
echo "=========================================="
