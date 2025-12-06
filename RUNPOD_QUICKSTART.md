# RunPod Quick Start Guide - TextVQA Project

## Step-by-Step Instructions

### Step 1: SSH to Your RunPod Cluster

```bash
# Connect to your RunPod instance (get SSH command from RunPod dashboard)
ssh root@<your-runpod-instance>.runpod.io -p <port>
```

---

### Step 2: Setup Environment

```bash
# Update system
apt-get update

# Install git if needed
apt-get install -y git

# Create working directory
mkdir -p /workspace
cd /workspace

# Upload your project (choose one method):

# Method 1: Clone from GitHub (if you pushed it)
git clone https://github.com/your-username/textvqa_project.git
cd textvqa_project

# Method 2: Upload via scp from your local machine
# On your local machine:
# scp -P <port> -r /Users/liyiwen/Desktop/textvqa_project root@<your-runpod-instance>.runpod.io:/workspace/

# Method 3: Use rsync (recommended for large files)
# On your local machine:
# rsync -avz -e "ssh -p <port>" /Users/liyiwen/Desktop/textvqa_project/ root@<your-runpod-instance>.runpod.io:/workspace/textvqa_project/
```

---

### Step 3: Install Dependencies

```bash
cd /workspace/textvqa_project

# Install Python packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Install OCR tools for ablations later
pip install paddleocr paddlepaddle-gpu
pip install easyocr
pip install fuzzywuzzy python-Levenshtein
```

**Expected output:**
```
PyTorch: 2.1.0+cu121
CUDA available: True
GPU count: 1
```

---

### Step 4: Download TextVQA Dataset

```bash
# Option 1: Use the download script (RECOMMENDED - already fixed for RunPod)
python data/download_data.py

# Option 2: If you still get hf_transfer error, install it first
pip install hf_transfer

# Option 3: Or disable fast transfer manually
export HF_HUB_ENABLE_HF_TRANSFER=0
python data/download_data.py

# This will download to textvqa_data/ directory
# Dataset size: ~10-15 GB
# Time: ~10-20 minutes depending on connection
```

**Note:** The script automatically handles the `hf_transfer` issue by disabling fast transfer if the package isn't available.

**What gets downloaded:**
- Training set: 34.6k samples
- Validation set: 5k samples
- Test set: 5.7k samples
- Images and annotations

---

### Step 5: Run Zero-Shot Evaluation (2-3 hours)

```bash
# Run zero-shot evaluation on validation set
python evaluation/zero_shot_eval.py \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset_name lmms-lab/textvqa \
  --split validation \
  --max_samples 1000 \
  --output_dir results/zero_shot \
  --batch_size 4

# Or use the shell script
bash scripts/run_zero_shot.sh
```

**Expected output:**
```
Loading model: Qwen/Qwen2.5-VL-3B-Instruct...
✓ Model loaded successfully
Loading dataset: lmms-lab/textvqa (validation)...
✓ Dataset loaded: 1000 samples

Running zero-shot evaluation...
[████████████████████] 100% | 1000/1000 | 4.2 samples/s

================================================================================
                         Zero-Shot Evaluation Results
================================================================================

PRIMARY METRICS (TextVQA Official)                  Score      Percentage
--------------------------------------------------------------------------------
VQA-style Accuracy (Official)                      0.6827          68.27%
Exact Match (for comparison)                       0.6523          65.23%

SEMANTIC METRICS                                    Score
--------------------------------------------------------------------------------
BLEU                                               0.3242
METEOR                                             0.4123
...
```

**Results saved to:** `results/zero_shot/metrics.json`

---

### Step 6: Train the Model (12-15 hours)

```bash
# Train with optimal configuration for maximum accuracy
python training/train.py \
  --config configs/train_max_accuracy.yaml \
  --output_dir checkpoints/qwen_max_accuracy \
  --logging_dir logs/qwen_max_accuracy

# Or use the shell script
bash scripts/run_training.sh
```

**Training configuration:**
- Epochs: 10
- Batch size: 2 (gradient accumulation: 8)
- Learning rate: 1e-5
- LoRA rank: 64
- All 34.6k training samples

**Monitor training:**
```bash
# In another terminal/tmux session:
tensorboard --logdir logs/qwen_max_accuracy --port 6006

# Then access via port forwarding or RunPod's proxy
```

**Expected training time:**
- A100 40GB: ~12-15 hours
- H100: ~8-10 hours
- A10: ~20-25 hours

**Checkpoints saved to:** `checkpoints/qwen_max_accuracy/`

---

### Step 7: Run Inference Ablations (5-8 hours)

After training completes, run ablation studies:

```bash
# Run all inference-time ablations (no retraining!)
bash scripts/run_inference_ablations.sh

# Follow the menu:
# Select option 1 for all ablations (~7-8 hours)
# Or option 3 for just OCR (~2 hours) - most important
```

**What runs:**
- OCR integration tests (PaddleOCR, EasyOCR)
- Prompt engineering variants
- Generation parameter tuning
- Post-processing strategies
- Ensemble methods

**Results saved to:** `results/inference_ablations/`

---

## Complete Command Sequence (Copy-Paste Ready)

```bash
# === STEP 1: Connect ===
ssh root@<your-runpod-instance>.runpod.io -p <port>

# === STEP 2-3: Setup ===
cd /workspace
# Upload your project here
cd textvqa_project
pip install -r requirements.txt
pip install paddleocr paddlepaddle-gpu easyocr fuzzywuzzy python-Levenshtein

# === STEP 4: Download Data ===
python data/download_data.py

# === STEP 5: Zero-Shot (2-3 hours) ===
python evaluation/zero_shot_eval.py \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset_name lmms-lab/textvqa \
  --split validation \
  --max_samples 1000 \
  --output_dir results/zero_shot \
  --batch_size 4

# === STEP 6: Training (12-15 hours) ===
python training/train.py \
  --config configs/train_max_accuracy.yaml

# === STEP 7: Ablations (5-8 hours) ===
bash scripts/run_inference_ablations.sh
```

---

## Using tmux (Recommended)

To avoid disconnection issues, use tmux:

```bash
# Install tmux if not available
apt-get install -y tmux

# Start a new tmux session
tmux new -s textvqa

# Run your commands in tmux
python data/download_data.py
python evaluation/zero_shot_eval.py ...
python training/train.py ...

# Detach from tmux: Ctrl+B then D
# Reattach later: tmux attach -t textvqa

# List sessions: tmux ls
# Kill session: tmux kill-session -t textvqa
```

---

## Monitoring Progress

### Check GPU Usage:
```bash
watch -n 1 nvidia-smi
```

### Check Training Progress:
```bash
# View latest logs
tail -f logs/qwen_max_accuracy/trainer_log.txt

# Or use tensorboard
tensorboard --logdir logs/qwen_max_accuracy --port 6006 --bind_all
```

### Check Disk Space:
```bash
df -h
du -sh /workspace/textvqa_project/*
```

---

## Downloading Results Back to Local Machine

After everything completes, download results:

```bash
# On your local machine:

# Download results
scp -P <port> -r root@<your-runpod-instance>.runpod.io:/workspace/textvqa_project/results ./

# Download checkpoints
scp -P <port> -r root@<your-runpod-instance>.runpod.io:/workspace/textvqa_project/checkpoints ./

# Download logs
scp -P <port> -r root@<your-runpod-instance>.runpod.io:/workspace/textvqa_project/logs ./

# Or use rsync (faster for large files)
rsync -avz -e "ssh -p <port>" \
  root@<your-runpod-instance>.runpod.io:/workspace/textvqa_project/results \
  root@<your-runpod-instance>.runpod.io:/workspace/textvqa_project/checkpoints \
  root@<your-runpod-instance>.runpod.io:/workspace/textvqa_project/logs \
  ./
```

---

## Expected Timeline

| Task | Duration | Cost (A100) |
|------|----------|-------------|
| Setup + Data Download | 30 min | $0.30 |
| Zero-Shot Evaluation | 2-3 hours | $2-3 |
| Training (10 epochs) | 12-15 hours | $12-15 |
| Inference Ablations | 5-8 hours | $5-8 |
| **TOTAL** | **~20-25 hours** | **~$20-26** |

*Prices based on ~$1/hour for A100 40GB on RunPod*

---

## Troubleshooting

### OOM (Out of Memory):
```bash
# Reduce batch size in configs/train_max_accuracy.yaml
# Change: per_device_train_batch_size: 2 -> 1
# Change: gradient_accumulation_steps: 8 -> 16
```

### Slow Download:
```bash
# Use dataset streaming
# Edit data/download_data.py to use streaming mode
```

### Connection Lost:
```bash
# Always use tmux!
tmux new -s textvqa
# Your commands here
# Detach: Ctrl+B then D
```

### Check if Process is Running:
```bash
ps aux | grep python
nvidia-smi  # Check if GPU is being used
```

---

## Quick Verification Checklist

Before starting long runs:

```bash
# ✓ GPU available
python -c "import torch; print(torch.cuda.is_available())"

# ✓ Dataset downloaded
ls -lh data_cache/

# ✓ Config file correct
cat configs/train_max_accuracy.yaml

# ✓ Enough disk space (need ~100GB)
df -h

# ✓ In tmux session
echo $TMUX  # Should output something if in tmux
```

---

## After Everything Completes

1. Generate visualizations locally (requires display):
```bash
# On local machine with downloaded results:
cd textvqa_project
python visualization/create_report_figures.py --output_dir results/figures
```

2. View all results:
```bash
# Check all metrics
cat results/zero_shot/metrics.json
cat results/inference_ablations/*_ablations.json

# Best checkpoint
ls -lh checkpoints/qwen_max_accuracy/checkpoint-best/
```

3. Write your report using the figures and metrics!

---

## Pro Tips

1. **Start small**: Test with `--max_samples 100` first
2. **Use tmux**: Never lose progress due to disconnection
3. **Monitor costs**: RunPod charges by the hour
4. **Save checkpoints**: Set `save_steps: 500` in config
5. **Download periodically**: Don't wait until the end
6. **Check logs**: `tail -f logs/*.log` to monitor progress

---

## Need Help?

- RunPod Documentation: https://docs.runpod.io/
- Project Issues: Check COMPLETE_WORKFLOW.md
- Training Issues: Check ACHIEVING_90_PERCENT.md
- Ablation Questions: Check INFERENCE_ABLATION_GUIDE.md

Good luck with your training! 🚀
