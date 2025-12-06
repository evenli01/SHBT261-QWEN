#!/bin/bash
#SBATCH --job-name=textvqa_qwen        # Job name
#SBATCH --partition=gpu                 # Partition name
#SBATCH --gres=gpu:H200:1              # Request 1 H200 GPU
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --mem=64G                      # Memory
#SBATCH --time=48:00:00                # Time limit (48 hours)
#SBATCH --output=logs/job_%j.out       # Standard output log
#SBATCH --error=logs/job_%j.err        # Standard error log
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your.email@domain.com  # Where to send mail

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust based on your cluster configuration)
# module load cuda/12.1
# module load python/3.10
# module load gcc/11.2.0

# Activate conda/virtual environment
# source activate textvqa_env
# OR
# source /path/to/venv/bin/activate

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/path/to/huggingface/cache  # Adjust to your cache directory

# Parse command line arguments for task selection
TASK=${1:-"training"}  # Default to training
SHIFT_ARGS=1

case $TASK in
    zero_shot)
        echo ""
        echo "=========================================="
        echo "Running Zero-Shot Evaluation"
        echo "=========================================="
        python evaluation/zero_shot_eval.py \
            --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
            --split validation \
            --batch_size 16 \
            --output_dir ./results/zero_shot \
            --use_hf_direct
        ;;
    
    training)
        echo ""
        echo "=========================================="
        echo "Running Training"
        echo "=========================================="
        python training/train.py \
            --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
            --output_dir ./checkpoints/qwen_textvqa \
            --learning_rate 5e-5 \
            --num_epochs 3 \
            --batch_size 4 \
            --gradient_accumulation_steps 4 \
            --lora_r 32 \
            --lora_alpha 64 \
            --warmup_steps 500 \
            --use_hf_direct \
            --use_wandb \
            --wandb_project textvqa-qwen
        ;;
    
    ablation)
        echo ""
        echo "=========================================="
        echo "Running Ablation Studies"
        echo "=========================================="
        python ablation/ablation_experiments.py \
            --output_dir ./results/ablation \
            --checkpoint_dir ./checkpoints/ablation \
            --use_hf_direct \
            --use_wandb \
            --wandb_project textvqa-ablation
        ;;
    
    ablation_quick)
        echo ""
        echo "=========================================="
        echo "Running Quick Ablation Studies"
        echo "=========================================="
        python ablation/ablation_experiments.py \
            --output_dir ./results/ablation_quick \
            --checkpoint_dir ./checkpoints/ablation_quick \
            --use_hf_direct \
            --quick \
            --use_wandb \
            --wandb_project textvqa-ablation-quick
        ;;
    
    evaluate)
        echo ""
        echo "=========================================="
        echo "Running Evaluation on Fine-tuned Model"
        echo "=========================================="
        MODEL_PATH=${2:-"./checkpoints/qwen_textvqa/best_model"}
        python evaluation/zero_shot_eval.py \
            --model_name "$MODEL_PATH" \
            --split validation \
            --batch_size 16 \
            --output_dir ./results/finetuned \
            --use_hf_direct
        ;;
    
    full_pipeline)
        echo ""
        echo "=========================================="
        echo "Running Full Pipeline"
        echo "=========================================="
        
        # 1. Zero-shot evaluation
        echo "Step 1: Zero-shot evaluation"
        python evaluation/zero_shot_eval.py \
            --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
            --split validation \
            --batch_size 16 \
            --output_dir ./results/zero_shot \
            --use_hf_direct
        
        # 2. Training
        echo "Step 2: Fine-tuning"
        python training/train.py \
            --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
            --output_dir ./checkpoints/qwen_textvqa \
            --learning_rate 5e-5 \
            --num_epochs 3 \
            --batch_size 4 \
            --gradient_accumulation_steps 4 \
            --lora_r 32 \
            --lora_alpha 64 \
            --warmup_steps 500 \
            --use_hf_direct \
            --use_wandb
        
        # 3. Evaluation on test set
        echo "Step 3: Test set evaluation"
        python evaluation/zero_shot_eval.py \
            --model_name "./checkpoints/qwen_textvqa/best_model" \
            --split test \
            --batch_size 16 \
            --output_dir ./results/test \
            --use_hf_direct
        
        # 4. Quick ablation studies
        echo "Step 4: Ablation studies"
        python ablation/ablation_experiments.py \
            --output_dir ./results/ablation \
            --checkpoint_dir ./checkpoints/ablation \
            --use_hf_direct \
            --quick \
            --use_wandb
        ;;
    
    *)
        echo "Unknown task: $TASK"
        echo "Available tasks: zero_shot, training, ablation, ablation_quick, evaluate, full_pipeline"
        exit 1
        ;;
esac

# Print completion information
echo ""
echo "=========================================="
echo "Job Completed Successfully"
echo "End Time: $(date)"
echo "=========================================="

# Notify completion (optional - adjust webhook URL)
# curl -X POST -H 'Content-type: application/json' \
#     --data '{"text":"Job '"$SLURM_JOB_ID"' completed on '"$SLURM_NODELIST"'"}' \
#     YOUR_SLACK_WEBHOOK_URL
