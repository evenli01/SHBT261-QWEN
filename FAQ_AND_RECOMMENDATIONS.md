# FAQ and Recommendations

## Questions Answered

### 1. Are we loading the full dataset?

**YES!** By default, all scripts load the FULL dataset:

- **Training set**: 34,602 samples (used for fine-tuning)
- **Validation set**: 5,000 samples (used for evaluation)
- **Test set**: 5,733 samples (optional, for final testing)

The `--limit` parameter is **OPTIONAL** and only for:
- Testing that scripts work (use `--limit 100`)
- Quick experiments during development
- When you have limited time

**For your final project, DON'T use `--limit`** - evaluate on full validation set!

---

### 2. Why limit 1000?

**Short answer**: You DON'T have to! I was being conservative with time estimates.

**Recommendations**:

#### **For Testing (to verify everything works)**:
```bash
# Use small limit for quick test
python evaluation/eval_zero_shot.py --limit 100 --prompt_template no_ocr
# Time: ~3-5 minutes
```

#### **For Your Final Project (RECOMMENDED)**:
```bash
# NO LIMIT = Full validation set (5000 samples)
python evaluation/eval_zero_shot.py --prompt_template no_ocr

# Or use the runner:
python experiments/run_all_ablations.py  # No --limit flag!
```

**Time estimates for FULL validation (5000 samples)**:
- Zero-shot evaluation: ~2-3 hours per ablation
- All 3 ablations: ~6-9 hours total
- Fine-tuned evaluation: ~2-3 hours per ablation

**My initial recommendation of 1000 samples**:
- Was to stay under 1 hour per ablation
- Gives ~20% of validation set
- Still statistically valid for comparison
- Good compromise if you have limited time

**For best results**: Use full 5000 samples if you have time!

---

### 3. Should I use epochs=3 for training?

**Looking at your classmate's code**:
```python
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
```
They default to **1 epoch**, but allow flexibility.

**Recommendations**:

#### **Option 1: 1 Epoch (What Your Classmate Used)**
```bash
python training/train_qwen_lora.py --epochs 1
```
- **Time**: ~12-15 hours on A100
- **Expected accuracy**: ~50-55%
- **Pros**: Faster, proven to work
- **Cons**: May not reach full potential

#### **Option 2: 3 Epochs (Better Results)**
```bash
python training/train_qwen_lora.py --epochs 3
```
- **Time**: ~36-45 hours on A100 (3x longer)
- **Expected accuracy**: ~55-60%
- **Pros**: Better performance, more thorough training
- **Cons**: Takes much longer

#### **My Recommendation**:
- **If you have time**: Use **3 epochs** for best results
- **If time is limited**: Use **1 epoch** (still gets good results)
- **Middle ground**: Use **2 epochs** (~24-30 hours)

**Important**: The script saves checkpoints after EACH epoch:
```
checkpoints/qwen_lora_full/
├── epoch_1/    # Can evaluate this if training interrupted
├── epoch_2/    # Can compare across epochs
└── epoch_3/
```

So you can:
1. Train for 1 epoch first
2. Evaluate those results
3. Continue training for more epochs if needed
4. Compare epoch_1 vs epoch_3 performance

---

## Recommended Workflow

### **For Time-Constrained (Project Due Soon)**

```bash
# 1. Quick test (5 minutes)
python data/dataset_simple.py
python evaluation/eval_zero_shot.py --limit 100 --prompt_template no_ocr

# 2. Zero-shot eval (partial, ~6-9 hours)
python experiments/run_all_ablations.py --limit 1000

# 3. Train 1 epoch (~12-15 hours)
python training/train_qwen_lora.py --epochs 1

# 4. Fine-tuned eval (partial, ~6-9 hours)
python experiments/run_all_ablations.py --limit 1000 --use_finetuned

# 5. Visualize
python experiments/visualize_results.py

# Total time: ~24-33 hours
```

### **For Best Results (Have Time)**

```bash
# 1. Quick test (5 minutes)
python data/dataset_simple.py
python evaluation/eval_zero_shot.py --limit 100 --prompt_template no_ocr

# 2. Zero-shot eval (FULL, ~6-9 hours)
python experiments/run_all_ablations.py  # No limit!

# 3. Train 3 epochs (~36-45 hours)
python training/train_qwen_lora.py --epochs 3

# 4. Fine-tuned eval (FULL, ~6-9 hours)
python experiments/run_all_ablations.py --use_finetuned  # No limit!

# 5. Visualize
python experiments/visualize_results.py

# Total time: ~48-63 hours (2-2.5 days)
```

### **Smart Middle Ground**

```bash
# Zero-shot: Full validation (best accuracy measurement)
python experiments/run_all_ablations.py

# Train: 2 epochs (good balance)
python training/train_qwen_lora.py --epochs 2

# Fine-tuned: Full validation
python experiments/run_all_ablations.py --use_finetuned

# Visualize
python experiments/visualize_results.py

# Total time: ~36-45 hours
```

---

## Statistical Validity

### **Sample Sizes**:
- **100 samples**: Good for testing code
- **1000 samples (20%)**: Statistically valid for comparison
- **5000 samples (100%)**: Most accurate, publishable results

### **For Your Report**:
Both 1000 and 5000 samples are acceptable:
- **1000 samples**: Mention "evaluated on 1000 samples (~20% of validation set)"
- **5000 samples**: "evaluated on full validation set"

Research papers often use subsets for ablations if full evaluation is costly.

---

## What Your Classmate Actually Did

Looking at their code:
1. ✅ Default to **1 epoch** training
2. ✅ Used `--limit` for testing
3. ✅ No `--limit` for final results
4. ✅ Evaluated on **validation set**

Their approach:
- Train: 1 epoch on full training set (34,602 samples)
- Eval: Full validation set (5,000 samples)
- Time: ~18-24 hours total

---

## Final Recommendations

### **Must Have (Minimum for Project)**:
- ✅ Zero-shot: 3 ablations on validation (can use 1000 samples)
- ✅ Training: 1 epoch on full training set
- ✅ Fine-tuned: 3 ablations on validation (can use 1000 samples)
- ✅ Visualizations and analysis

### **Should Have (Better Results)**:
- ✅ Zero-shot: 3 ablations on FULL validation (5000)
- ✅ Training: 1-2 epochs on full training set
- ✅ Fine-tuned: 3 ablations on FULL validation (5000)
- ✅ Per-question-type analysis

### **Nice to Have (Excellent Project)**:
- ✅ Zero-shot: 3 ablations on FULL validation
- ✅ Training: 3 epochs with checkpoint comparison
- ✅ Fine-tuned: 3 ablations on FULL validation
- ✅ Additional prompt engineering experiments
- ✅ Error analysis and failure case discussion

---

## Command Summary

### **Quick Test Everything**:
```bash
python data/dataset_simple.py
python evaluation/eval_zero_shot.py --limit 10
python training/train_qwen_lora.py --limit 10 --epochs 1
```

### **Minimum for Project** (1000 samples, 1 epoch):
```bash
python experiments/run_all_ablations.py --limit 1000
python training/train_qwen_lora.py --epochs 1
python experiments/run_all_ablations.py --limit 1000 --use_finetuned
python experiments/visualize_results.py
```

### **Recommended** (full validation, 1-2 epochs):
```bash
python experiments/run_all_ablations.py
python training/train_qwen_lora.py --epochs 2
python experiments/run_all_ablations.py --use_finetuned
python experiments/visualize_results.py
```

### **Best Results** (full validation, 3 epochs):
```bash
python experiments/run_all_ablations.py
python training/train_qwen_lora.py --epochs 3
python experiments/run_all_ablations.py --use_finetuned
python experiments/visualize_results.py
```

---

## Time Budget Planning

| Task | 1000 samples / 1 epoch | Full / 2 epochs | Full / 3 epochs |
|------|------------------------|-----------------|-----------------|
| Zero-shot eval | ~2 hours | ~6-9 hours | ~6-9 hours |
| Training | ~12-15 hours | ~24-30 hours | ~36-45 hours |
| Fine-tuned eval | ~2 hours | ~6-9 hours | ~6-9 hours |
| **Total** | **~16-19 hours** | **~36-48 hours** | **~48-63 hours** |

**Choose based on your deadline!**
