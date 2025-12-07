"""
Fine-tune Qwen2.5-VL with LoRA following classmate's successful approach.
Uses 4-bit quantization and proper answer token masking.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_simple import TextVQADataset
from models.qwen_official import QwenVLModel


def collate_fn_qwen(batch, processor):
    """
    Collate function for Qwen with answer token masking.
    
    Args:
        batch: List of dataset items
        processor: Qwen processor
        
    Returns:
        Dictionary with batched tensors
    """
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answers'][0] for item in batch]  # Take first answer
    
    # Prepare messages for each sample
    all_messages = []
    for q, a, img in zip(questions, answers, images):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"Question: {q} Answer: {a}"},
                ],
            }
        ]
        all_messages.append(messages)
    
    # Process with chat template
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
        for msg in all_messages
    ]
    
    # Extract images for processing
    image_inputs_list = [[msg[0]["content"][0]["image"]] for msg in all_messages]
    
    # Process batch
    inputs = processor(
        text=texts,
        images=image_inputs_list,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    
    # Create labels from input_ids with answer token masking
    labels = inputs['input_ids'].clone()
    
    # Mask prompt tokens, only train on answer
    for i, (q, a) in enumerate(zip(questions, answers)):
        # Tokenize just the answer to find its length
        answer_tokens = processor.tokenizer(a, add_special_tokens=False)['input_ids']
        
        # Find where answer starts (approximately)
        seq_len = (labels[i] != processor.tokenizer.pad_token_id).sum()
        answer_start = max(0, seq_len - len(answer_tokens) - 5)  # -5 for safety
        labels[i, :answer_start] = -100
    
    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    inputs['labels'] = labels
    
    return inputs


def train_qwen_lora(args):
    """
    Train Qwen2.5-VL with LoRA.
    
    Args:
        args: Command line arguments
    """
    print(f"{'='*70}")
    print("Fine-Tuning Qwen2.5-VL with LoRA")
    print(f"{'='*70}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*70}\n")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    
    # Load model with quantization
    print("[1/5] Loading model with 4-bit quantization...")
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    processor = Qwen2VLProcessor.from_pretrained(args.model_path)
    
    # Prepare for LoRA
    print("[2/5] Preparing model for LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("[3/5] Loading dataset...")
    dataset = TextVQADataset(split="train", cache_dir=args.cache_dir)
    
    if args.limit:
        print(f"Limiting to {args.limit} samples for testing")
        dataset.dataset = dataset.dataset.select(range(args.limit))
    
    # Create dataloader
    def collate_wrapper(batch):
        return collate_fn_qwen(batch, processor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_wrapper
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"[4/5] Training for {args.epochs} epochs...")
    model.train()
    
    total_steps = len(dataloader) * args.epochs
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataloader):
            try:
                # Move batch to device
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"\nWarning: NaN loss at step {step}, skipping")
                    optimizer.zero_grad()
                    continue
                
                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum_steps
                
                # Backward
                loss.backward()
                
                # Update weights every grad_accum_steps
                if (step + 1) % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Track metrics
                epoch_loss += loss.item() * args.grad_accum_steps
                num_batches += 1
                
                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * args.grad_accum_steps:.4f}',
                    'epoch': f'{epoch+1}/{args.epochs}'
                })
                
            except torch.cuda.OutOfMemoryError:
                print(f"\nCUDA OOM at step {step}, clearing cache...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as e:
                print(f"\nError in batch: {e}")
                optimizer.zero_grad()
                continue
        
        # Final optimizer step if needed
        if (step + 1) % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"\nEpoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        epoch_dir = f"{args.output_dir}/epoch_{epoch+1}"
        model.save_pretrained(epoch_dir)
        print(f"✓ Checkpoint saved to {epoch_dir}")
    
    progress_bar.close()
    
    # Save final model
    print("[5/5] Saving final model...")
    model.save_pretrained(args.output_dir)
    print(f"✓ Model saved to {args.output_dir}")
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with LoRA")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", default="./checkpoints/qwen_lora")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Limit training samples for testing")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_qwen_lora(args)
