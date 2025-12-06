#!/bin/bash
# Script to run fine-tuning on TextVQA

# Default parameters
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="./checkpoints/qwen_textvqa"
LEARNING_RATE=5e-5
NUM_EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUM=4
LORA_R=32
LORA_ALPHA=64
WARMUP_STEPS=500
USE_WANDB=""
WANDB_PROJECT="textvqa-qwen"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Fine-tuning Qwen2.5-VL-3B on TextVQA"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUM"
echo "LoRA Rank: $LORA_R"
echo "LoRA Alpha: $LORA_ALPHA"
echo "Warmup Steps: $WARMUP_STEPS"
echo "=========================================="

# Run training
python training/train.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --warmup_steps $WARMUP_STEPS \
    --use_hf_direct \
    $USE_WANDB \
    --wandb_project "$WANDB_PROJECT"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
