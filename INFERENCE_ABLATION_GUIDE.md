# Quick Start Guide: Inference-Time Ablation Studies

This guide shows you how to run ablation studies **without retraining** - just using inference with different strategies!

## Why Inference-Time Ablations?

✅ **No retraining needed** - Use your already-trained checkpoint  
✅ **Fast** - Each ablation takes ~30 min to 2 hours  
✅ **Cost-effective** - Save 10+ hours of training time  
✅ **More experiments** - Test many strategies quickly  
✅ **Easy to interpret** - Direct A/B comparison

## Time Budget

| Ablation Type | # Experiments | Time per Exp | Total Time |
|---------------|---------------|--------------|------------|
| Generation Parameters | 5 | ~20 min | ~1.5 hours |
| Prompt Engineering | 4 | ~20 min | ~1.5 hours |
| OCR Integration | 4 | ~30 min | ~2 hours |
| Post-processing | 3 | ~15 min | ~45 min |
| Ensemble Methods | 3 | ~30 min | ~1.5 hours |
| **TOTAL** | **19** | - | **~7-8 hours** |

## Prerequisites

### 1. Install OCR Dependencies

```bash
# Install PaddleOCR (recommended - fast and accurate)
pip install paddleocr paddlepaddle-gpu

# Or install EasyOCR (alternative - very accurate)
pip install easyocr

# Or install Tesseract (traditional OCR)
# macOS:
brew install tesseract
pip install pytesseract

# Ubuntu/Linux:
sudo apt-get install tesseract-ocr
pip install pytesseract

# Also install fuzzy string matching
pip install fuzzywuzzy python-Levenshtein
```

### 2. Have Your Trained Model Ready

You need a trained checkpoint from your main training run:
```bash
checkpoints/qwen_max_accuracy/checkpoint-best/
```

## Running Ablation Studies

### Quick Start - Run All Ablations

```bash
# Run ALL inference ablations (~7-8 hours)
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --dataset lmms-lab/textvqa \
  --split validation \
  --ablation_type all \
  --output_dir results/inference_ablations
```

### Run Specific Ablation Types

#### 1. Generation Parameter Ablations (~1.5 hours)

Test different decoding strategies:
- Greedy decoding vs beam search
- Different temperatures
- Sampling strategies

```bash
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --ablation_type generation \
  --output_dir results/ablations/generation
```

**What this tests:**
- `baseline_greedy`: Greedy decoding (fastest)
- `beam_search_3`: Beam search with 3 beams
- `beam_search_5`: Beam search with 5 beams (slower, potentially better)
- `sampling_temp_03`: Sampling with temperature 0.3
- `sampling_temp_05`: Sampling with temperature 0.5

#### 2. Prompt Engineering Ablations (~1.5 hours)

Test different prompt templates:
- Standard vs detailed instructions
- Chain-of-thought reasoning
- Few-shot examples

```bash
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --ablation_type prompting \
  --output_dir results/ablations/prompting
```

**What this tests:**
- `prompt_standard`: Simple "Question: ... Answer:" format
- `prompt_detailed`: Detailed instructions about reading text
- `prompt_cot`: Chain-of-thought reasoning
- `prompt_few_shot`: With example demonstrations

#### 3. OCR Integration Ablations (~2 hours) ⭐ **IMPORTANT**

Test OCR impact - this is where you'll see **5-10% accuracy boost**!

```bash
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --ablation_type ocr \
  --output_dir results/ablations/ocr
```

**What this tests:**
- `no_ocr`: Baseline without OCR
- `paddleocr`: With PaddleOCR (fast, good)
- `easyocr`: With EasyOCR (slower, very accurate)
- `tesseract`: With Tesseract OCR (traditional)

**Expected results:**
- No OCR: ~75-78% accuracy
- With PaddleOCR: ~80-83% accuracy ⬆️
- With EasyOCR: ~80-84% accuracy ⬆️

#### 4. Post-processing Ablations (~45 min)

Test answer cleaning and normalization:

```bash
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --ablation_type postprocessing \
  --output_dir results/ablations/postprocessing
```

**What this tests:**
- `no_postprocess`: Raw model output
- `basic_postprocess`: Remove "Answer:", lowercase, etc.
- `fuzzy_postprocess`: Match against OCR text with fuzzy matching

#### 5. Ensemble Methods (~1.5 hours)

Test combining multiple predictions:

```bash
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --ablation_type ensemble \
  --output_dir results/ablations/ensemble
```

**What this tests:**
- `single_model`: Baseline single prediction
- `multi_prompt_ensemble`: Vote across multiple prompts
- `multi_prompt_with_ocr`: Ensemble with OCR integration

**Expected boost:** +2-3% accuracy from voting

## Fast Testing Mode

Want to test quickly? Use a subset of data:

```bash
# Run on only 500 samples for quick testing (~5 min per experiment)
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --ablation_type ocr \
  --max_samples 500 \
  --output_dir results/ablations/ocr_test
```

## Results Analysis

Results are saved as JSON files:

```
results/inference_ablations/
├── generation_ablations.json
├── prompting_ablations.json
├── ocr_ablations.json
├── postprocessing_ablations.json
└── ensemble_ablations.json
```

Each file contains:
- Configuration used
- Metrics (accuracy, BLEU, METEOR, etc.)
- Time taken
- Sample predictions

### View Results

```python
import json

# Load results
with open('results/inference_ablations/ocr_ablations.json', 'r') as f:
    results = json.load(f)

# Print summary
for name, result in results.items():
    metrics = result['metrics']
    print(f"{name}: {metrics['accuracy']:.2%} accuracy")
```

### Compare Results

The script automatically prints a summary table:

```
==================================================================================
ABLATION STUDY SUMMARY
==================================================================================
Experiment                     Accuracy     BLEU         Time (s)     Samples/s   
----------------------------------------------------------------------------------
no_ocr                        75.32%       0.3421       1200.5       4.17        
paddleocr                     81.45%       0.3892       1850.2       2.70        
easyocr                       82.13%       0.3945       2120.8       2.36        
tesseract                     79.87%       0.3654       1650.3       3.03        
==================================================================================
```

## Recommended Workflow

### Step 1: Train Your Model First (12-15 hours)

```bash
python training/train.py \
  --config configs/train_max_accuracy.yaml
```

**Wait for training to complete!** This is the only time-consuming step.

### Step 2: Run Priority Ablations (~3-4 hours)

```bash
# 1. OCR ablation (most important!)
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --ablation_type ocr

# 2. Prompt ablation
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --ablation_type prompting

# 3. Ensemble ablation
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --ablation_type ensemble
```

### Step 3: Generate Visualizations

```python
from inference.visualize import create_ablation_plots

create_ablation_plots(
    results_dir='results/inference_ablations',
    output_dir='results/figures'
)
```

## Expected Accuracy Improvements

Starting from fine-tuned model baseline (~75-78%):

| Strategy | Accuracy Boost | Cumulative |
|----------|----------------|------------|
| Baseline (fine-tuned) | - | 75-78% |
| + Best prompt | +1-2% | 76-79% |
| + OCR integration | +5-8% | 81-85% |
| + Beam search | +0.5-1% | 81-86% |
| + Ensemble voting | +1-2% | 82-87% |
| + Post-processing | +0.5-1% | **82-88%** |

## Tips for Best Results

### 1. OCR Quality Matters
- Use PaddleOCR for speed/accuracy balance
- Use EasyOCR if accuracy is critical
- OCR works best on clear, well-lit text

### 2. Prompt Engineering
- Detailed instructions help for complex scenes
- Few-shot examples work well for consistent format
- Chain-of-thought helps with reasoning

### 3. Ensemble Strategies
- Odd number of models (3, 5) for voting
- Combine different prompt styles
- Include OCR predictions in ensemble

### 4. Post-processing
- Always lowercase for matching
- Remove common prefixes ("Answer:", "The answer is")
- Use fuzzy matching against OCR text

## Troubleshooting

### OOM (Out of Memory) Errors

```bash
# Reduce batch size
python ablation/inference_ablations.py \
  --model_path checkpoints/qwen_max_accuracy/checkpoint-best \
  --ablation_type ocr \
  --per_device_eval_batch_size 1  # Add this parameter to script if needed
```

### OCR Installation Issues

```bash
# PaddleOCR issues - use CPU version
pip install paddleocr paddlepaddle

# EasyOCR issues - reinstall with specific version
pip install easyocr==1.7.0
```

### Slow Inference

```bash
# Use smaller sample size for testing
--max_samples 500

# Or skip slow experiments (beam search, ensemble)
--ablation_type prompting  # Fast experiments only
```

## For Your Report

### What to Include

1. **Ablation Study Design**
   - Describe each ablation category
   - Explain why inference-time ablations are efficient

2. **Results Table**
   - Show accuracy for each configuration
   - Include time measurements
   - Highlight best configurations

3. **Analysis**
   - OCR impact: "Adding PaddleOCR improved accuracy by 6.2% (from 75.3% to 81.5%)"
   - Prompt impact: "Detailed instruction prompts improved accuracy by 1.8%"
   - Ensemble impact: "Multi-prompt voting added 2.1% accuracy boost"

4. **Error Analysis**
   - Show examples where OCR helped
   - Show examples where it didn't
   - Discuss limitations

### Example Results Section

```
We conducted five categories of inference-time ablation studies using our 
fine-tuned Qwen2.5-VL-3B checkpoint (baseline: 75.3% accuracy):

1. **OCR Integration** (+6.2%): Adding PaddleOCR significantly improved 
   performance on text-reading tasks. EasyOCR achieved slightly higher 
   accuracy (+6.8%) at the cost of 1.3x slower inference.

2. **Prompt Engineering** (+1.8%): Detailed instruction prompts outperformed
   simple prompts. Few-shot examples were particularly effective for 
   questions requiring specific answer formats.

3. **Generation Parameters** (+0.8%): Beam search with 5 beams provided 
   marginal improvements over greedy decoding but increased inference time 
   by 2.5x.

4. **Ensemble Methods** (+2.1%): Voting across three different prompts 
   combined with OCR achieved our best single-model accuracy of 82.8%.

5. **Post-processing** (+0.5%): Fuzzy matching against OCR text corrected 
   minor spelling variations in model outputs.

**Final Best Configuration**: 82.8% accuracy using fine-tuned model + 
PaddleOCR + detailed prompts + multi-prompt ensemble + fuzzy post-processing.
```

## Summary

**Total Time Budget:**
- Training: 12-15 hours (do once)
- All ablations: 7-8 hours (inference only!)
- Analysis: 1-2 hours
- **Total: ~20 hours**

**Expected Final Accuracy:** 82-87% (realistic with all optimizations)

**Key Insight:** By focusing on inference-time ablations, you save massive amounts of 
compute time while still conducting comprehensive experiments!
