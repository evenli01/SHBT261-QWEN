"""
Run all OCR ablation experiments automatically.
Compares No-OCR vs Basic OCR vs Structured OCR.
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_ablation(prompt_template, split, limit, output_dir, lora_path=None):
    """Run a single ablation experiment."""
    
    print(f"\n{'='*70}")
    print(f"Running ablation: {prompt_template}")
    print(f"{'='*70}\n")
    
    cmd = [
        "python", "evaluation/eval_zero_shot.py",
        "--prompt_template", prompt_template,
        "--split", split,
        "--output_dir", output_dir
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    if lora_path:
        # Use finetuned evaluation script instead
        cmd = [
            "python", "evaluation/eval_finetuned.py",
            "--prompt_template", prompt_template,
            "--lora_path", lora_path,
            "--split", split,
            "--output_dir", output_dir
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])
    
    # Run command
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"⚠️  Warning: {prompt_template} failed")
        return None
    
    print(f"✓ {prompt_template} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all OCR ablation experiments")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--lora_path", default=None, help="Path to fine-tuned LoRA model")
    parser.add_argument("--use_finetuned", action="store_true", help="Evaluate fine-tuned model")
    
    args = parser.parse_args()
    
    # Determine output subdirectory
    if args.use_finetuned or args.lora_path:
        output_subdir = os.path.join(args.output_dir, "finetuned")
        lora_path = args.lora_path or "checkpoints/qwen_lora_full"
    else:
        output_subdir = os.path.join(args.output_dir, "zero_shot")
        lora_path = None
    
    print("="*70)
    print("Running All OCR Ablation Experiments")
    print("="*70)
    print(f"Split: {args.split}")
    print(f"Limit: {args.limit or 'None (all samples)'}")
    print(f"Output: {output_subdir}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print("="*70)
    
    # Define ablations to run
    ablations = ["no_ocr", "basic_ocr", "structured_ocr"]
    
    results_summary = {}
    
    # Run each ablation
    for prompt_template in ablations:
        success = run_ablation(
            prompt_template=prompt_template,
            split=args.split,
            limit=args.limit,
            output_dir=output_subdir,
            lora_path=lora_path
        )
        
        if success:
            # Load results
            result_file = os.path.join(
                output_subdir,
                f"qwen_{'finetuned' if lora_path else 'zero_shot'}_{prompt_template}_{args.split}.json"
            )
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results_summary[prompt_template] = data['metrics']
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'VQA Acc':<12} {'BLEU':<12} {'METEOR':<12}")
    print("-"*70)
    
    for method, metrics in results_summary.items():
        acc = metrics.get('accuracy', 0)
        bleu = metrics.get('bleu', 0)
        meteor = metrics.get('meteor', 0)
        print(f"{method:<20} {acc:<12.4f} {bleu:<12.4f} {meteor:<12.4f}")
    
    print("="*70)
    print(f"\n✓ All ablations completed!")
    print(f"✓ Results saved to: {output_subdir}")
    print(f"✓ Run visualization: python experiments/visualize_results.py")
    print("="*70)


if __name__ == "__main__":
    main()
