#!/bin/bash
# Run Inference-Time Ablation Studies
# No retraining needed - uses existing checkpoint!

set -e  # Exit on error

# Configuration
MODEL_PATH="${1:-checkpoints/qwen_max_accuracy/checkpoint-best}"
DATASET="${2:-lmms-lab/textvqa}"
SPLIT="${3:-validation}"
OUTPUT_DIR="${4:-results/inference_ablations}"
MAX_SAMPLES="${5:-}"  # Optional: leave empty for full dataset

echo "=============================================="
echo "INFERENCE-TIME ABLATION STUDIES"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET ($SPLIT)"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if model checkpoint exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found at $MODEL_PATH"
    echo "Please train the model first using:"
    echo "  python training/train.py --config configs/train_max_accuracy.yaml"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run ablation
run_ablation() {
    local ablation_type=$1
    local estimated_time=$2
    
    echo ""
    echo "================================================"
    echo "Running $ablation_type ablations..."
    echo "Estimated time: $estimated_time"
    echo "================================================"
    
    if [ -z "$MAX_SAMPLES" ]; then
        python ablation/inference_ablations.py \
            --model_path "$MODEL_PATH" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --ablation_type "$ablation_type" \
            --output_dir "$OUTPUT_DIR"
    else
        python ablation/inference_ablations.py \
            --model_path "$MODEL_PATH" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --ablation_type "$ablation_type" \
            --output_dir "$OUTPUT_DIR" \
            --max_samples "$MAX_SAMPLES"
    fi
    
    echo "✓ Completed $ablation_type ablations"
}

# Menu for selecting which ablations to run
echo "Select ablation studies to run:"
echo "  1) All ablations (~7-8 hours)"
echo "  2) Priority only: OCR + Prompt + Ensemble (~4 hours)"
echo "  3) OCR only (~2 hours) [RECOMMENDED]"
echo "  4) Prompt only (~1.5 hours)"
echo "  5) Generation params only (~1.5 hours)"
echo "  6) Post-processing only (~45 min)"
echo "  7) Ensemble only (~1.5 hours)"
echo "  8) Custom selection"
echo ""

read -p "Enter choice (1-8) [default: 3]: " choice
choice=${choice:-3}

case $choice in
    1)
        echo "Running ALL ablations..."
        run_ablation "generation" "1.5 hours"
        run_ablation "prompting" "1.5 hours"
        run_ablation "ocr" "2 hours"
        run_ablation "postprocessing" "45 min"
        run_ablation "ensemble" "1.5 hours"
        ;;
    2)
        echo "Running PRIORITY ablations..."
        run_ablation "ocr" "2 hours"
        run_ablation "prompting" "1.5 hours"
        run_ablation "ensemble" "1.5 hours"
        ;;
    3)
        echo "Running OCR ablations (most important!)..."
        run_ablation "ocr" "2 hours"
        ;;
    4)
        echo "Running Prompt ablations..."
        run_ablation "prompting" "1.5 hours"
        ;;
    5)
        echo "Running Generation parameter ablations..."
        run_ablation "generation" "1.5 hours"
        ;;
    6)
        echo "Running Post-processing ablations..."
        run_ablation "postprocessing" "45 min"
        ;;
    7)
        echo "Running Ensemble ablations..."
        run_ablation "ensemble" "1.5 hours"
        ;;
    8)
        echo "Custom selection..."
        read -p "Run generation? (y/n): " gen
        read -p "Run prompting? (y/n): " prompt
        read -p "Run OCR? (y/n): " ocr
        read -p "Run post-processing? (y/n): " post
        read -p "Run ensemble? (y/n): " ensemble
        
        [ "$gen" = "y" ] && run_ablation "generation" "1.5 hours"
        [ "$prompt" = "y" ] && run_ablation "prompting" "1.5 hours"
        [ "$ocr" = "y" ] && run_ablation "ocr" "2 hours"
        [ "$post" = "y" ] && run_ablation "postprocessing" "45 min"
        [ "$ensemble" = "y" ] && run_ablation "ensemble" "1.5 hours"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "ALL SELECTED ABLATIONS COMPLETED!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  - JSON files: $OUTPUT_DIR/*.json"
echo "  - Summary tables printed above"
echo ""
echo "Next steps:"
echo "  1. Analyze results in JSON files"
echo "  2. Generate visualizations"
echo "  3. Write up findings for your report"
echo ""
echo "Expected accuracy improvements:"
echo "  - Baseline (fine-tuned): 75-78%"
echo "  - + OCR: 80-83% (+5-8%)"
echo "  - + Best prompting: 81-84% (+1-2%)"
echo "  - + Ensemble: 82-85% (+2-3%)"
echo "  - Final best: 82-87%"
echo ""

# Generate quick summary
if command -v python &> /dev/null; then
    echo "Generating summary report..."
    python -c "
import json
import os
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
json_files = list(output_dir.glob('*_ablations.json'))

if json_files:
    print('\n' + '='*80)
    print('QUICK SUMMARY OF RESULTS')
    print('='*80)
    
    all_results = {}
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_results.update(data)
    
    # Find best configuration
    best_acc = 0
    best_config = None
    
    for name, result in all_results.items():
        acc = result['metrics']['accuracy']
        if acc > best_acc:
            best_acc = acc
            best_config = name
    
    print(f'\nBest configuration: {best_config}')
    print(f'Best accuracy: {best_acc:.2%}')
    print('\nTop 5 configurations:')
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]['metrics']['accuracy'], 
                          reverse=True)
    for i, (name, result) in enumerate(sorted_results[:5], 1):
        acc = result['metrics']['accuracy']
        print(f'  {i}. {name}: {acc:.2%}')
    print('='*80)
"
fi

echo ""
echo "Done! 🎉"
