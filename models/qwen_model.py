"""
Qwen2.5-VL model wrapper with LoRA support.
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict, Any
import logging

from .model_config import ModelConfig, get_lora_config, print_trainable_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenVLModel:
    """
    Wrapper class for Qwen2.5-VL model with LoRA fine-tuning support.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize Qwen2.5-VL model with optional LoRA.
        
        Args:
            config: Model configuration
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        self.config = config
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.processor = None
        
        self._initialize_model()
        self._initialize_processor()
        
    def _initialize_model(self):
        """Initialize the base Qwen2.5-VL model."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Prepare quantization config if needed
        quantization_config = None
        if self.load_in_8bit or self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load model using Auto classes (supports both Qwen2VL and Qwen2.5VL)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            quantization_config=quantization_config
        )
        
        # Apply freezing strategies if specified
        if self.config.freeze_vision_encoder:
            logger.info("Freezing vision encoder")
            self._freeze_vision_encoder()
        
        if self.config.freeze_language_model:
            logger.info("Freezing language model")
            self._freeze_language_model()
        
        # Apply LoRA if specified
        if self.config.use_lora:
            logger.info("Applying LoRA to model")
            self._apply_lora()
        
        # Print model statistics
        print_trainable_parameters(self.model)
        
    def _initialize_processor(self):
        """Initialize the processor for Qwen2.5-VL."""
        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Set padding token if not set
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
    
    def _apply_lora(self):
        """Apply LoRA to the model."""
        # Prepare model for training if using quantization
        if self.load_in_8bit or self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Get LoRA config
        lora_config = get_lora_config(self.config)
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info(f"LoRA applied with rank={self.config.lora_r}")
    
    def _freeze_vision_encoder(self):
        """Freeze the vision encoder parameters."""
        for name, param in self.model.named_parameters():
            if "visual" in name.lower() or "vision" in name.lower():
                param.requires_grad = False
    
    def _freeze_language_model(self):
        """Freeze the language model parameters."""
        for name, param in self.model.named_parameters():
            if "language" in name.lower() or "lm" in name.lower():
                param.requires_grad = False
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text given image and text inputs.
        
        Args:
            pixel_values: Image tensor
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
                top_p=self.config.top_p if self.config.do_sample else None,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **kwargs
            )
        
        return outputs
    
    def save_model(self, output_dir: str):
        """
        Save the model (or LoRA adapter if using LoRA).
        
        Args:
            output_dir: Directory to save the model
        """
        logger.info(f"Saving model to {output_dir}")
        
        if self.config.use_lora:
            # Save only the LoRA adapter
            self.model.save_pretrained(output_dir)
        else:
            # Save the full model
            self.model.save_pretrained(output_dir)
        
        # Save processor
        self.processor.save_pretrained(output_dir)
        
        logger.info("Model saved successfully")
    
    def load_adapter(self, adapter_path: str):
        """
        Load a LoRA adapter.
        
        Args:
            adapter_path: Path to the saved adapter
        """
        from peft import PeftModel
        
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        logger.info("Adapter loaded successfully")
    
    def merge_and_unload(self):
        """
        Merge LoRA weights with base model and unload adapter.
        Only works if using LoRA.
        """
        if self.config.use_lora:
            logger.info("Merging LoRA weights with base model")
            self.model = self.model.merge_and_unload()
            logger.info("Merge completed")
        else:
            logger.warning("Model is not using LoRA, nothing to merge")
    
    def get_model(self):
        """Return the underlying model."""
        return self.model
    
    def get_processor(self):
        """Return the processor."""
        return self.processor


def load_model_for_inference(
    model_path: str,
    config: Optional[ModelConfig] = None,
    use_lora: bool = False,
    device: str = "cuda"
) -> QwenVLModel:
    """
    Load a model for inference.
    
    Args:
        model_path: Path to model or adapter
        config: Model configuration (will use defaults if None)
        use_lora: Whether the model uses LoRA
        device: Device to load model on
        
    Returns:
        QwenVLModel instance
    """
    if config is None:
        config = ModelConfig()
        config.use_lora = use_lora
        config.device_map = device
    
    # Initialize model
    model_wrapper = QwenVLModel(config)
    
    # Load adapter if using LoRA
    if use_lora:
        model_wrapper.load_adapter(model_path)
    
    return model_wrapper


if __name__ == "__main__":
    # Test model initialization
    print("Testing Qwen2.5-VL model initialization...")
    
    config = ModelConfig()
    config.use_lora = True
    config.lora_r = 8  # Use small rank for testing
    
    print("\n1. Testing with LoRA:")
    try:
        model_wrapper = QwenVLModel(config)
        print("✓ Model with LoRA initialized successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n2. Testing without LoRA:")
    config.use_lora = False
    try:
        model_wrapper = QwenVLModel(config)
        print("✓ Model without LoRA initialized successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n✓ Model initialization test completed!")
