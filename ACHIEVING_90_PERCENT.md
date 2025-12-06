# Achieving 90%+ Accuracy on TextVQA

## Current State-of-the-Art Performance

**Important Reality Check:**
- TextVQA is a **challenging benchmark** designed to test OCR + reasoning
- Current SOTA models achieve:
  - **GPT-4V**: ~78% accuracy
  - **Gemini Pro Vision**: ~74% accuracy  
  - **InternVL2-76B**: ~82.7% accuracy (largest open model)
  - **Qwen2-VL-7B**: ~73.5% accuracy (official benchmark)
  - **Qwen2.5-VL-3B**: ~65-70% accuracy (estimated baseline)

**Can we reach 90%+ with Qwen2.5-VL-3B?**
- 90% would **significantly exceed current SOTA** (even GPT-4V!)
- More realistic targets:
  - **Zero-shot**: 65-70%
  - **After fine-tuning**: 75-80%
  - **With optimal strategies**: 80-85%

However, we can try several advanced strategies to maximize performance!

---

## Strategy 1: Optimal Training Configuration

### Key Hyperparameters for Maximum Accuracy

```yaml
# Training configuration optimized for accuracy (not speed)
training_config:
  num_train_epochs: 8-10              # More epochs for better convergence
  per_device_train_batch_size: 2      # Smaller batch = more updates
  gradient_accumulation_steps: 8      # Effective batch size = 16
  learning_rate: 1e-5                 # Conservative LR for stability
  warmup_ratio: 0.1                   # Gradual warmup
  lr_scheduler_type: "cosine"         # Cosine decay works well
  weight_decay: 0.01                  # Regularization
  max_grad_norm: 1.0                  # Gradient clipping
  
  # Advanced optimizations
  optim: "adamw_torch_fused"          # Faster optimizer
  bf16: true                          # Better precision than fp16
  gradient_checkpointing: true        # Save memory, train longer

lora_config:
  r: 64                               # Higher rank = more capacity
  lora_alpha: 128                     # Alpha = 2*r recommended
  lora_dropout: 0.05                  # Light dropout
  target_modules:                     # Apply LoRA to ALL attention + MLP
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: "none"
```

### Training Dataset Enhancements

1. **Use ALL training data** (34.6k samples)
2. **Data augmentation**:
   - Random brightness/contrast adjustments
   - Light image quality degradation (mimics real-world)
   - Text rotation augmentations
3. **Curriculum learning**: Train on easier samples first, then harder ones

---

## Strategy 2: Advanced OCR Integration

### Why OCR Matters for TextVQA
- Many failures are due to inability to READ text in images
- External OCR can provide explicit text annotations
- Combine vision model + OCR for best results

### OCR Pipeline Options

#### Option 1: PaddleOCR (Fast, Good)
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
result = ocr.ocr(image_path)
text_in_image = " ".join([line[1][0] for line in result[0]])
```

#### Option 2: EasyOCR (Very Accurate)
```python
import easyocr

reader = easyocr.Reader(['en'], gpu=True)
result = reader.readtext(image_path)
text_in_image = " ".join([detection[1] for detection in result])
```

#### Option 3: GPT-4V API (Most Accurate, Costs $)
```python
# Use GPT-4V to extract text, then use Qwen for reasoning
```

### OCR-Enhanced Prompting

**Standard Prompt:**
```
Question: What does the sign say?
```

**OCR-Enhanced Prompt:**
```
Question: What does the sign say?
OCR detected text in image: "STOP", "Main Street", "Speed Limit 25"
Answer based on the question and detected text.
```

This can boost accuracy by 5-10%!

---

## Strategy 3: Advanced Prompting (No Retraining!)

### Best Prompts for TextVQA

#### Prompt 1: Detailed Instruction
```python
prompt_template = """You are an expert at reading text in images and answering questions.

Instructions:
1. Carefully examine the image for any visible text
2. Read all text accurately, including signs, labels, displays, etc.
3. Use the text information to answer the question precisely
4. Give a short, direct answer

Image: [IMAGE]
Question: {question}
Answer:"""
```

#### Prompt 2: Chain-of-Thought
```python
prompt_template = """Let's solve this step by step:

1. What text can you see in the image?
2. What does the question ask?
3. How does the visible text help answer the question?

Image: [IMAGE]
Question: {question}

Let's think through this:
1. Visible text:"""
```

#### Prompt 3: Few-Shot Examples (Very Effective!)
```python
prompt_template = """Here are some examples:

Example 1:
Image: [Phone with "NOKIA" visible]
Question: What brand is the phone?
Answer: nokia

Example 2:
Image: [Sign showing "Speed Limit 55"]
Question: What is the speed limit?
Answer: 55

Now answer this:
Image: [IMAGE]
Question: {question}
Answer:"""
```

---

## Strategy 4: Inference Optimizations (No Retraining!)

### Generation Parameters

```python
generation_config = {
    "max_new_tokens": 32,          # TextVQA answers are short
    "temperature": 0.1,             # Low temp = more deterministic
    "top_p": 0.9,                   # Nucleus sampling
    "do_sample": False,             # Greedy decoding for accuracy
    "num_beams": 5,                 # Beam search (slower but better)
    "repetition_penalty": 1.2,      # Avoid repetition
}
```

### Post-Processing

```python
def post_process_answer(answer, ocr_text=""):
    """Clean up model output"""
    # Remove explanations, keep only answer
    answer = answer.split('\n')[0].strip()
    
    # Lowercase for matching
    answer = answer.lower()
    
    # Remove common artifacts
    answer = answer.replace("answer:", "").strip()
    answer = answer.replace("the answer is", "").strip()
    
    # If answer not in OCR text but close match exists, use OCR version
    if ocr_text:
        ocr_words = ocr_text.lower().split()
        for word in ocr_words:
            if fuzz.ratio(answer, word) > 85:  # fuzzy match
                answer = word
    
    return answer
```

---

## Strategy 5: Ensemble Methods

### Multi-Model Ensemble
```python
# Combine predictions from:
1. Qwen2.5-VL-3B (base)
2. Qwen2.5-VL-7B (larger, if resources allow)
3. External OCR + rule-based reasoning

# Voting strategy
def ensemble_predict(predictions):
    from collections import Counter
    votes = Counter(predictions)
    return votes.most_common(1)[0][0]
```

### Multi-Prompt Ensemble (Same Model)
```python
prompts = [prompt1, prompt2, prompt3]  # Different prompt strategies
predictions = [model.generate(prompt) for prompt in prompts]
final_answer = ensemble_predict(predictions)
```

Can add 2-3% accuracy boost!

---

## Strategy 6: Error Analysis & Targeted Fixes

### Common Error Types in TextVQA

1. **OCR Errors** (40% of failures)
   - Solution: Use external OCR, better prompting

2. **Reasoning Errors** (30% of failures)
   - Solution: Chain-of-thought prompting, more training

3. **Ambiguous Questions** (15% of failures)
   - Solution: Contextual reasoning, multiple hypotheses

4. **Small/Blurry Text** (15% of failures)
   - Solution: Image preprocessing, ensemble with OCR

### Targeted Fine-tuning

After error analysis, create a **hard negative dataset**:
- Collect examples where model fails
- Add similar hard examples
- Fine-tune specifically on these cases

---

## Complete Pipeline for Maximum Accuracy

```python
# Step 1: Pre-process image
image = preprocess_image(image_path)  # Enhance quality

# Step 2: Extract OCR
ocr_text = extract_ocr(image)  # PaddleOCR or EasyOCR

# Step 3: Generate with best prompt
prompt = create_prompt_with_ocr(question, ocr_text)
answer = model.generate(
    prompt,
    max_new_tokens=32,
    num_beams=5,
    temperature=0.1
)

# Step 4: Post-process
answer = post_process_answer(answer, ocr_text)

# Step 5: Confidence check
if model.confidence < 0.5:
    # Use ensemble or OCR fallback
    answer = fallback_strategy(image, question, ocr_text)

return answer
```

---

## Realistic Accuracy Targets & Timeline

### Phase 1: Zero-Shot Baseline
- **Target**: 65-70%
- **Time**: 2-3 hours
- **Actions**: Run evaluation with optimal prompts

### Phase 2: Fine-Tuning
- **Target**: 75-78%
- **Time**: 10-12 hours (8-10 epochs)
- **Actions**: Full training with LoRA r=64

### Phase 3: OCR Integration
- **Target**: 78-82%
- **Time**: 2-3 hours (no retraining!)
- **Actions**: Add PaddleOCR/EasyOCR to pipeline

### Phase 4: Advanced Optimizations
- **Target**: 82-85%
- **Time**: 3-4 hours
- **Actions**: 
  - Ensemble methods
  - Best prompting strategies
  - Post-processing refinements

### Phase 5: If Resources Allow
- **Target**: 85-87%
- **Time**: +20 hours
- **Actions**:
  - Train larger model (Qwen2.5-VL-7B)
  - Curriculum learning
  - Hard negative mining

---

## Recommended Experiment Plan

### Main Training Run (One Full Training)
```bash
# Use configs/train_config_max_accuracy.yaml
python training/train.py \
  --config configs/train_config_max_accuracy.yaml \
  --output_dir checkpoints/qwen_full_training \
  --num_train_epochs 10 \
  --lora_r 64
```
**Time**: ~12-15 hours on A100

### Ablation Studies (NO Retraining - Fast!)

All using the SAME trained checkpoint:

1. **OCR Ablation** (~1 hour total)
   - No OCR baseline
   - PaddleOCR
   - EasyOCR
   - Combined OCR

2. **Prompt Ablation** (~1 hour total)
   - Standard prompt
   - Detailed instruction
   - Chain-of-thought
   - Few-shot examples

3. **Generation Parameter Ablation** (~1 hour total)
   - Temperature: [0.0, 0.1, 0.3, 0.5]
   - Beam search: [1, 3, 5]
   - Top-p: [0.8, 0.9, 0.95]

4. **Post-processing Ablation** (~30 min)
   - No post-processing
   - Basic cleaning
   - OCR-assisted correction
   - Fuzzy matching

5. **Ensemble Ablation** (~1 hour)
   - Single model
   - Multi-prompt ensemble
   - With/without OCR voting

**Total Ablation Time**: ~5 hours (all inference-time!)

---

## Summary

**Realistic Goal**: **80-82% accuracy** is achievable with:
- Full fine-tuning (10 epochs, LoRA r=64)
- OCR integration (PaddleOCR/EasyOCR)
- Optimal prompting strategies
- Good post-processing

**Stretch Goal**: **82-85% accuracy** if you also use:
- Ensemble methods
- Hard negative mining
- Multiple models

**90% accuracy** would require:
- Much larger models (70B+)
- Proprietary models (GPT-4V level)
- Extensive data augmentation
- Multiple large model ensemble

**Time Budget**:
- Main training: 12-15 hours
- Ablations (inference-only): 5-6 hours
- **Total**: ~20 hours
