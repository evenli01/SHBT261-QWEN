"""
Zero-shot evaluation script following classmate's successful approach.
Supports OCR ablations and prompt engineering experiments.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_simple import TextVQADataset
from models.qwen_official import QwenVLModel
from evaluation.metrics_vqa import calculate_metrics, print_metrics
from evaluation.ocr_utils import format_prompt, classify_question, PROMPT_TEMPLATES


def evaluate_zero_shot(
    model,
    dataset,
    prompt_template="no_ocr",
    limit=None,
    save_predictions=True
):
    """
    Run zero-shot evaluation with specified prompt template.
    
    Args:
        model: QwenVLModel instance
        dataset: TextVQADataset instance
        prompt_template: Name of prompt template to use
        limit: Limit number of samples (for testing)
        save_predictions: Whether to save individual predictions
        
    Returns:
        Dictionary with metrics and results
    """
    results = []
    
    # Determine how many samples to process
    num_samples = min(len(dataset), limit) if limit else len(dataset)
    
    print(f"\n{'='*70}")
    print(f"Running zero-shot evaluation")
    print(f"Prompt template: {prompt_template}")
    print(f"Samples: {num_samples}")
    print(f"{'='*70}\n")
    
    for i in tqdm(range(num_samples), desc="Evaluating"):
        try:
            sample = dataset[i]
            image = sample['image']
            question = sample['question']
            ground_truth_answers = sample['answers']
            image_id = sample['image_id']
            ocr_tokens = sample.get('ocr_tokens', [])
            
            # Classify question for structured OCR
            q_type = classify_question(question)
            
            # Format prompt based on template
            formatted_question = format_prompt(
                question,
                template_name=prompt_template,
                ocr_tokens=ocr_tokens,
                q_type=q_type
            )
            
            # Generate answer
            try:
                predicted_answer = model.generate_answer(image, formatted_question)
            except (RuntimeError, torch.cuda.CudaError) as e:
                error_str = str(e)
                if "CUDA" in error_str or "device-side assert" in error_str or "nan" in error_str.lower():
                    print(f"\nCUDA error for image {image_id}")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    predicted_answer = ""
                else:
                    print(f"Runtime error for image {image_id}: {e}")
                    predicted_answer = ""
            except Exception as e:
                print(f"Error generating answer for image {image_id}: {e}")
                predicted_answer = ""
            
            # Store result
            result_item = {
                "image_id": image_id,
                "question": question,
                "formatted_question": formatted_question,
                "predicted_answer": predicted_answer,
                "ground_truth_answers": ground_truth_answers,
                "q_type": q_type
            }
            
            results.append(result_item)
            
            # Periodically clear CUDA cache
            if (i + 1) % 50 == 0:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                    
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)
    
    # Also calculate per-question-type metrics
    q_type_metrics = {}
    for q_type in ["brand", "number", "date", "time", "text", "general"]:
        q_type_results = [r for r in results if r.get("q_type") == q_type]
        if q_type_results:
            q_type_metrics[q_type] = calculate_metrics(q_type_results)
    
    output_data = {
        "config": {
            "prompt_template": prompt_template,
            "num_samples": len(results)
        },
        "metrics": metrics,
        "q_type_metrics": q_type_metrics,
    }
    
    if save_predictions:
        output_data["results"] = results
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation with OCR ablations and prompt engineering"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="no_ocr",
        choices=list(PROMPT_TEMPLATES.keys()),
        help=f"Prompt template to use. Options: {', '.join(PROMPT_TEMPLATES.keys())}"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for dataset"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Zero-Shot Evaluation - Qwen2.5-VL on TextVQA")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print(f"Prompt template: {args.prompt_template}")
    if args.limit:
        print(f"Sample limit: {args.limit}")
    print("="*70)
    
    # Load model
    print("\n[1/3] Loading model...")
    model = QwenVLModel(model_path=args.model_path)
    
    # Load dataset
    print("\n[2/3] Loading dataset...")
    dataset = TextVQADataset(split=args.split, cache_dir=args.cache_dir)
    
    # Run evaluation
    print("\n[3/3] Running evaluation...")
    output_data = evaluate_zero_shot(
        model=model,
        dataset=dataset,
        prompt_template=args.prompt_template,
        limit=args.limit,
        save_predictions=True
    )
    
    # Print results
    print_metrics(
        output_data["metrics"],
        f"Zero-Shot Results - {args.prompt_template}"
    )
    
    # Print per-question-type metrics if available
    if output_data.get("q_type_metrics"):
        print("\nPer-Question-Type Accuracy:")
        print("-" * 50)
        for q_type, metrics in sorted(output_data["q_type_metrics"].items()):
            num_samples = sum(1 for r in output_data["results"] if r.get("q_type") == q_type)
            print(f"  {q_type:10s}: {metrics['accuracy']:.4f} ({num_samples} samples)")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"qwen_zero_shot_{args.prompt_template}_{args.split}.json"
    )
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
