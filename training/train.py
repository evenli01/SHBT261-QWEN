"""
Training script for fine-tuning Qwen2.5-VL-3B on TextVQA with LoRA.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import os
from pathlib import Path
import argparse
from datetime import datetime
import wandb

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.qwen_model import QwenVLModel
from models.model_config import ModelConfig, get_training_arguments
from data.preprocess import create_dataloaders
from evaluation.metrics import TextVQAMetrics, print_metrics
from evaluation.zero_shot_eval import evaluate_zero_shot


class TextVQATrainer:
    """
    Trainer for fine-tuning Qwen2.5-VL on TextVQA with LoRA.
    """
    
    def __init__(
        self,
        model_wrapper: QwenVLModel,
        config: ModelConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        use_wandb: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model_wrapper: QwenVLModel instance
            config: Model configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.get_model()
        self.processor = model_wrapper.get_processor()
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup metrics
        self.metrics_calc = TextVQAMetrics(use_llm_judge=False)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Device
        self.device = next(self.model.parameters()).device
        
    def _setup_optimizer(self):
        """Setup optimizer."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        num_training_steps = len(self.train_loader) * self.config.num_train_epochs
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def train_epoch(self, epoch: int):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_train_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch.get("labels")
            
            if labels is not None:
                labels = labels.to(self.device)
            
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            
            # Forward pass
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Log to wandb
                if self.use_wandb and self.global_step % self.config.logging_steps == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/global_step': self.global_step
                    })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def evaluate(self):
        """
        Evaluate on validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_ground_truths = []
        all_questions = []
        
        print("\nEvaluating on validation set...")
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                pixel_values = batch.get("pixel_values")
                
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)
                
                # Generate predictions
                try:
                    outputs = self.model_wrapper.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Decode predictions
                    for i in range(len(outputs)):
                        generated_ids = outputs[i][len(input_ids[i]):]
                        prediction = self.processor.tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=True
                        ).strip()
                        
                        all_predictions.append(prediction)
                        all_ground_truths.append(batch["answers"][i])
                        all_questions.append(batch["questions"][i])
                
                except Exception as e:
                    print(f"\nError during evaluation: {e}")
                    # Add empty predictions for failed batches
                    for i in range(len(batch["questions"])):
                        all_predictions.append("")
                        all_ground_truths.append(batch["answers"][i])
                        all_questions.append(batch["questions"][i])
                    continue
        
        # Compute metrics
        metrics = self.metrics_calc.compute_all_metrics(
            predictions=all_predictions,
            ground_truths_list=all_ground_truths
        )
        
        return metrics
    
    def train(self):
        """
        Main training loop.
        """
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Total epochs: {self.config.num_train_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Total training steps: {len(self.train_loader) * self.config.num_train_epochs}")
        print("=" * 70)
        
        for epoch in range(self.config.num_train_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} - Average training loss: {train_loss:.4f}")
            
            # Evaluate
            if (epoch + 1) % 1 == 0:  # Evaluate every epoch
                metrics = self.evaluate()
                print_metrics(metrics, f"Validation Metrics - Epoch {epoch+1}")
                
                # Log to wandb
                if self.use_wandb:
                    wandb_metrics = {f'val/{k}': v for k, v in metrics.items()}
                    wandb_metrics['val/epoch'] = epoch
                    wandb.log(wandb_metrics)
                
                # Save best model
                current_metric = metrics.get(self.config.metric_for_best_model, 0.0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint('best_model')
                    print(f"\n✓ New best model saved! {self.config.metric_for_best_model}: {current_metric:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}')
        
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best {self.config.metric_for_best_model}: {self.best_metric:.4f}")
        print("=" * 70)
    
    def save_checkpoint(self, name: str):
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model_wrapper.save_model(str(checkpoint_dir))
        
        # Save training state
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        
        torch.save(state, checkpoint_dir / 'training_state.pt')
        
        print(f"Checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL on TextVQA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/qwen_textvqa")
    parser.add_argument("--data_dir", type=str, default="./textvqa_data")
    parser.add_argument("--use_hf_direct", action="store_true", default=True)
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="textvqa-qwen")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Other
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"qwen_lora_r{args.lora_r}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    print("=" * 70)
    print("Fine-tuning Qwen2.5-VL-3B on TextVQA")
    print("=" * 70)
    
    # Setup configuration
    config = ModelConfig()
    config.model_name = args.model_name
    config.learning_rate = args.learning_rate
    config.num_train_epochs = args.num_epochs
    config.per_device_train_batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.warmup_steps = args.warmup_steps
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    
    # Initialize model
    print("\n[1/3] Loading model...")
    model_wrapper = QwenVLModel(config)
    
    # Load data
    print("\n[2/3] Loading data...")
    train_loader, val_loader, _ = create_dataloaders(
        processor=model_wrapper.get_processor(),
        data_dir=args.data_dir,
        train_batch_size=args.batch_size,
        use_hf_direct=args.use_hf_direct,
        subset_size=args.subset_size
    )
    
    # Initialize trainer
    print("\n[3/3] Initializing trainer...")
    trainer = TextVQATrainer(
        model_wrapper=model_wrapper,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )
    
    # Train
    trainer.train()
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
