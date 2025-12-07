"""
Evaluate fine-tuned Qwen model with LoRA adapter.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_simple import TextVQADataset
from evaluation.metrics_vqa import calculate_metrics, print_metrics
from evaluation.ocr_utils import format_prompt, classify_question


class QwenFinetunedModel:
    """Qwen model with LoRA adapter."""
    
    def __init__(self, base_model_path, lora_path):
        print(f"Loading base model: {base_model_path}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Loading LoRA adapter: {lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        
        self.processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        self.device = self.model.device
        
        print("✓ Fine-tuned model loaded successfully")
    
    def generate_answer(self, image, question, max_new_tokens=15):
        """Generate answer for image and question."""
        messages = [
            {
                "role": "system",
                "content": "You are answering visual questions. Respond with a short phrase, not a full sentence."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Clean answer
        from models.qwen_official import QwenVLModel
        return QwenVLModel._cleanup_answer(output_text[0])


def evaluate_finetuned(args):
    """Evaluate fine-tuned model."""
    
    print("="*70)
    print("Evaluating Fine-Tuned Qwen Model")
    print("="*70)
    print(f"Base model: {args.model_path}")
    print(f"LoRA adapter: {args.lora_path}")
    print(f"Prompt template: {args.prompt_template}")
    print("="*70)
    
    # Load model
    print("\n[1/3] Loading fine-tuned model...")
    model = QwenFinetunedModel(args.model_path, args.lora_path)
    
    # Load dataset
    print("\n[2/3] Loading dataset...")
    dataset = TextVQADataset(split=args.split, cache_dir=args.cache_dir)
    
    # Run evaluation
    print("\n[3/3] Running evaluation...")
    results = []
    
    num_samples = min(len(dataset), args.limit) if args.limit else len(dataset)
    
    for i in tqdm(range(num_samples), desc="Evaluating"):
        try:
            sample = dataset[i]
            image = sample['image']
            question = sample['question']
            ground_truth_answers = sample['answers']
            image_id = sample['image_id']
            ocr_tokens = sample.get('ocr_tokens', [])
            
            q_type = classify_question(question)
            
            formatted_question = format_prompt(
                question,
                template_name=args.prompt_template,
                ocr_tokens=ocr_tokens,
                q_type=q_type
            )
            
            try:
                predicted_answer = model.generate_answer(image, formatted_question)
            except Exception as e:
                print(f"\nError for image {image_id}: {e}")
                predicted_answer = ""
            
            results.append({
                "image_id": image_id,
                "question": question,
                "formatted_question": formatted_question,
                "predicted_answer": predicted_answer,
                "ground_truth_answers": ground_truth_answers,
                "q_type": q_type
            })
            
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)
    
    # Per-question-type metrics
    q_type_metrics = {}
    for q_type in ["brand", "number", "date", "time", "text", "general"]:
        q_type_results = [r for r in results if r.get("q_type") == q_type]
        if q_type_results:
            q_type_metrics[q_type] = calculate_metrics(q_type_results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"qwen_finetuned_{args.prompt_template}_{args.split}.json"
    )
    
    output_data = {
        "config": {
            "base_model": args.model_path,
            "lora_path": args.lora_path,
            "prompt_template": args.prompt_template,
            "num_samples": len(results)
        },
        "metrics": metrics,
        "q_type_metrics": q_type_metrics,
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print results
    print_metrics(metrics, f"Fine-Tuned Results - {args.prompt_template}")
    
    if q_type_metrics:
        print("\nPer-Question-Type Accuracy:")
        print("-" * 50)
        for q_type, metrics in sorted(q_type_metrics.items()):
            num_samples = sum(1 for r in results if r.get("q_type") == q_type)
            print(f"  {q_type:10s}: {metrics['accuracy']:.4f} ({num_samples} samples)")
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen model")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--prompt_template", default="no_ocr")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default="./results/finetuned")
    parser.add_argument("--cache_dir", default=None)
    
    args = parser.parse_args()
    evaluate_finetuned(args)


if __name__ == "__main__":
    main()
