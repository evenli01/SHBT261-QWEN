# TextVQA Visual Understanding Project

This project implements fine-tuning and evaluation of the Qwen2.5-VL-3B model on the TextVQA dataset for visual question answering tasks involving text embedded in images.

## Project Structure

```
textvqa_project/
├── data/                       # Dataset directory
│   ├── download_data.py       # Script to download TextVQA dataset
│   └── preprocess.py          # Data preprocessing utilities
├── models/                     # Model-related code
│   ├── model_config.py        # Model configuration
│   └── qwen_model.py          # Qwen2.5-VL model wrapper
├── training/                   # Training scripts
│   ├── train.py               # Main training script with LoRA
│   └── trainer_utils.py       # Training utilities
├── evaluation/                 # Evaluation scripts
│   ├── evaluate.py            # Main evaluation script
│   ├── metrics.py             # All evaluation metrics
│   └── zero_shot_eval.py      # Zero-shot evaluation
├── ablation/                   # Ablation study scripts
│   ├── ablation_experiments.py # Ablation experiment runner
│   └── ablation_configs.py    # Different ablation configurations
├── inference/                  # Inference and visualization
│   ├── inference.py           # Inference pipeline
│   └── visualize.py           # Result visualization
├── configs/                    # Configuration files
│   ├── base_config.yaml       # Base configuration
│   ├── train_config.yaml      # Training configuration
│   └── ablation_config.yaml   # Ablation study configuration
├── scripts/                    # Utility scripts
│   ├── run_zero_shot.sh       # Run zero-shot evaluation
│   ├── run_training.sh        # Run training
│   ├── run_ablation.sh        # Run ablation studies
│   └── submit_slurm.sh        # SLURM job submission script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Features

### 1. Zero-shot Evaluation
- Evaluate pretrained Qwen2.5-VL-3B on TextVQA validation/test sets
- Multiple evaluation metrics (Accuracy, BLEU, METEOR, ROUGE, F1, etc.)

### 2. Fine-tuning with LoRA
- Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
- Optimized for H200 GPU training
- Gradient accumulation and mixed precision training

### 3. Comprehensive Evaluation Metrics
- **Primary**: Exact match accuracy
- **Secondary**: BLEU, METEOR, ROUGE-L, LLM-as-a-Judge
- **Additional**: F1 score, Precision, Recall, Per-category performance

### 4. Ablation Studies
- LoRA rank variations (r=8, 16, 32, 64)
- Learning rate experiments (1e-5, 5e-5, 1e-4)
- Training data size impact (25%, 50%, 75%, 100%)
- Vision encoder freezing vs. unfreezing
- Different attention mechanisms

## Requirements

See `requirements.txt` for full list. Key dependencies:
- transformers >= 4.37.0
- torch >= 2.1.0
- datasets
- peft (for LoRA)
- accelerate
- evaluate
- nltk
- rouge_score

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download TextVQA dataset:
```bash
python data/download_data.py
```

## Usage

### Zero-shot Evaluation
```bash
bash scripts/run_zero_shot.sh
```

### Training with LoRA
```bash
bash scripts/run_training.sh
```

### Ablation Studies
```bash
bash scripts/run_ablation.sh
```

### Running on Remote Cluster (SLURM)
```bash
sbatch scripts/submit_slurm.sh
```

## Results

Results will be saved in:
- `results/zero_shot/` - Zero-shot evaluation results
- `results/finetuned/` - Fine-tuned model results
- `results/ablation/` - Ablation study results
- `checkpoints/` - Model checkpoints

## Report

The final report should include:
1. Introduction and motivation
2. Methodology (model architecture, LoRA configuration, training details)
3. Experimental design (ablation studies, evaluation setup)
4. Results and analysis (tables, figures, failure case analysis)
5. Conclusions and future work

## Citation

```bibtex
@inproceedings{singh2019textvqa,
  title={Towards VQA Models That Can Read},
  author={Singh, Amanpreet and Natarajan, Vivek and Meet, Shah and Jiang, Yu and Chen, Xinlei and Batra, Dhruv and Parikh, Devi and Rohrbach, Marcus},
  booktitle={CVPR},
  year={2019}
}
```
