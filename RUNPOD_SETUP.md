# Running on RunPod.io

## Quick Setup Guide for RunPod

### 1. Create a RunPod Instance

1. Go to [runpod.io](https://runpod.io)
2. Select "GPU Instances" → "Deploy"
3. Choose GPU: **A100 80GB** (recommended) or **A100 40GB**
4. Select template: **PyTorch 2.1** or **RunPod PyTorch**
5. Configure:
   - Container Disk: 50GB minimum
   - Volume Disk: 100GB+ recommended (for datasets and checkpoints)
6. Click "Deploy On-Demand" or "Deploy Spot"

### 2. Connect to Your Instance

**Option A: JupyterLab (Recommended for beginners)**
```
Click "Connect" → "Start JupyterLab"
Open a terminal in JupyterLab
```

**Option B: SSH**
```bash
ssh root@<your-pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

**Option C: VS Code Remote SSH**
1. Install "Remote - SSH" extension
2. Add SSH config with RunPod connection details
3. Connect to remote host

### 3. Initial Setup

```bash
# Update system
apt-get update

# Clone your project (or upload)
cd /workspace
# If you have it on GitHub:
# git clone <your-repo>
# Or upload using JupyterLab interface

# Navigate to project
cd textvqa_project

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Set Environment Variables

```bash
# Create .env file
cat > .env << EOF
HF_HOME=/workspace/cache/huggingface
TRANSFORMERS_CACHE=/workspace/cache/huggingface
TORCH_HOME=/workspace/cache/torch
WANDB_API_KEY=your_wandb_key_here
EOF

# Load environment
export $(cat .env | xargs)
```

### 5. Run Your Experiments

**Zero-Shot Evaluation:**
```bash
python evaluation/zero_shot_eval.py \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --split validation \
    --batch_size 16 \
    --output_dir /workspace/results/zero_shot
```

**Training:**
```bash
python training/train.py \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --output_dir /workspace/checkpoints/qwen_textvqa \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --use_wandb \
    --wandb_project textvqa-runpod
```

**Ablation Studies:**
```bash
python ablation/ablation_experiments.py \
    --output_dir /workspace/results/ablation \
    --checkpoint_dir /workspace/checkpoints/ablation \
    --use_wandb
```

### 6. Monitor Training

**Using tmux (recommended for long runs):**
```bash
# Start tmux session
tmux new -s training

# Run your training script
python training/train.py ...

# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t training
# List sessions: tmux ls
```

**Using nohup:**
```bash
nohup python training/train.py ... > training.log 2>&1 &

# Monitor logs
tail -f training.log
```

**Using Weights & Biases:**
- Your training will automatically log to wandb.ai
- Monitor in real-time from any device

### 7. Save Results

**Important:** RunPod instances can be terminated, so save your work!

```bash
# Sync to persistent volume
mkdir -p /workspace/persistent
cp -r checkpoints results /workspace/persistent/

# Download to local machine (from your local terminal)
scp -P <port> -r root@<pod-ip>:/workspace/results ./
scp -P <port> -r root@<pod-ip>:/workspace/checkpoints ./
```

**Or use cloud storage:**
```bash
# Install rclone
apt-get install rclone

# Configure for Google Drive, S3, etc.
rclone config

# Sync results
rclone copy /workspace/results remote:textvqa-results
rclone copy /workspace/checkpoints remote:textvqa-checkpoints
```

### 8. Optimal RunPod Configuration for This Project

**Recommended Setup:**
- **GPU**: A100 80GB (best) or A100 40GB SXM
- **Container Disk**: 50GB
- **Volume Disk**: 150GB
- **Template**: PyTorch 2.1 or RunPod PyTorch
- **Region**: US or EU (lower latency)

**Cost Optimization:**
- Use **Spot instances** (~70% cheaper) if you can handle interruptions
- Use **tmux** to resume after reconnection
- Use **checkpointing** to save progress regularly
- **Stop** (don't terminate) instance when not using

### 9. RunPod-Specific Scripts

I've created a RunPod-specific script for you:

```bash
# See scripts/run_on_runpod.sh
bash scripts/run_on_runpod.sh
```

### 10. Troubleshooting

**Out of Memory:**
```bash
# Reduce batch size
python training/train.py --batch_size 4 --gradient_accumulation_steps 8
```

**Connection Lost:**
```bash
# Always use tmux for long-running jobs
tmux new -s training
# Your job will continue even if disconnected
```

**Slow Data Loading:**
```bash
# Use volume storage for datasets
export HF_HOME=/workspace/cache/huggingface
```

**GPU Not Detected:**
```python
import torch
assert torch.cuda.is_available(), "CUDA not available!"
print(torch.cuda.get_device_name(0))
```

### 11. Best Practices

1. **Always use tmux** for training sessions
2. **Save checkpoints** to /workspace (persistent)
3. **Use wandb** for monitoring (access from anywhere)
4. **Test with small dataset first** (--max_samples 100)
5. **Monitor GPU usage**: `nvidia-smi -l 1`
6. **Set up auto-sync** to cloud storage
7. **Stop instance** when idle to save costs

### 12. Complete RunPod Workflow

```bash
# 1. Start tmux
tmux new -s training

# 2. Set environment
export HF_HOME=/workspace/cache/huggingface
export WANDB_API_KEY=your_key

# 3. Test with small dataset
python evaluation/zero_shot_eval.py \
    --split validation \
    --batch_size 16 \
    --max_samples 100

# 4. Run full training
python training/train.py \
    --output_dir /workspace/checkpoints/qwen_textvqa \
    --use_wandb

# 5. Detach tmux: Ctrl+B, then D

# 6. Check progress anytime
tmux attach -t training
# Or check wandb.ai

# 7. When done, save results
cp -r /workspace/checkpoints /workspace/persistent/
cp -r /workspace/results /workspace/persistent/
```

### 13. Cost Estimates

**A100 80GB:**
- On-Demand: ~$2.50/hour
- Spot: ~$0.75/hour
- Full training (10 hours): $7.50-$25

**A100 40GB:**
- On-Demand: ~$1.50/hour
- Spot: ~$0.45/hour
- Full training (12 hours): $5.40-$18

**Tips to reduce costs:**
- Use spot instances
- Stop when not training
- Use smaller model for initial experiments
- Run ablations in parallel on multiple pods

---

Ready to start? Follow steps 1-5, then run your first experiment!
