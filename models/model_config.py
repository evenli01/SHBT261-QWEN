"""
Model configuration for Qwen2.5-VL-3B with LoRA fine-tuning.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for model initialization and training."""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_max_length: int = 512
    trust_remote_code: bool = True
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 32  # Rank of LoRA matrices
    lora_alpha: int = 64  # Scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ])
    
    # Training settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer_type: str = "adamw_torch"
    scheduler_type: str = "cosine"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Mixed precision training
    fp16: bool = False
    bf16: bool = True  # Use bf16 for better stability with H200
    
    # Batch sizes
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    
    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    
    # Logging
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: ["tensorboard", "wandb"])
    
    # Device settings
    device_map: str = "auto"
    
    # Generation settings (for inference)
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Freeze vision encoder (for ablation)
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    
    # Experiment type
    experiment_name: str = "baseline"
    
    # LoRA rank ablation
    lora_ranks: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    
    # Learning rate ablation
    learning_rates: List[float] = field(default_factory=lambda: [1e-5, 5e-5, 1e-4, 5e-4])
    
    # Data size ablation (fraction of training data)
    data_fractions: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    
    # Freezing strategies
    freeze_strategies: List[dict] = field(default_factory=lambda: [
        {"vision": False, "language": False},  # Train all
        {"vision": True, "language": False},    # Freeze vision
        {"vision": False, "language": True},    # Freeze language
    ])
    
    # Batch size ablation
    batch_sizes: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    
    # Number of epochs for quick ablation
    quick_epochs: int = 1
    full_epochs: int = 3


def get_lora_config(config: ModelConfig):
    """
    Get LoRA configuration for PEFT.
    
    Args:
        config: ModelConfig object
        
    Returns:
        LoraConfig object
    """
    from peft import LoraConfig, TaskType
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        inference_mode=False
    )


def get_training_arguments(config: ModelConfig, output_dir: str):
    """
    Get training arguments for Hugging Face Trainer.
    
    Args:
        config: ModelConfig object
        output_dir: Output directory for checkpoints
        
    Returns:
        TrainingArguments object
    """
    from transformers import TrainingArguments
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        report_to=config.report_to,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        optim=config.optimizer_type,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        lr_scheduler_type=config.scheduler_type,
        remove_unused_columns=False,
    )


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.
    
    Args:
        model: The model to analyze
    """
    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"\nTrainable Parameters:")
    print(f"  Total parameters: {all_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Percentage trainable: {100 * trainable_params / all_params:.2f}%")
    
    return trainable_params, all_params


if __name__ == "__main__":
    # Test configuration
    config = ModelConfig()
    
    print("Model Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Use LoRA: {config.use_lora}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Mixed precision: bf16={config.bf16}")
    
    print("\nAblation Configuration:")
    ablation_config = AblationConfig()
    print(f"  LoRA ranks to test: {ablation_config.lora_ranks}")
    print(f"  Learning rates to test: {ablation_config.learning_rates}")
    print(f"  Data fractions to test: {ablation_config.data_fractions}")
    
    print("\n✓ Configuration test completed!")
