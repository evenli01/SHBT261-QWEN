"""
Official Qwen2.5-VL model wrapper following successful approach.
Uses qwen_vl_utils for proper vision processing.
"""

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import re


class QwenVLModel:
    """
    Qwen2.5-VL model wrapper with official utilities.
    """
    
    def __init__(
        self, 
        model_path="Qwen/Qwen2.5-VL-3B-Instruct", 
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16
    ):
        """
        Initialize Qwen2.5-VL model.
        
        Args:
            model_path: HuggingFace model path
            device: Device to load model on
            torch_dtype: Data type for model weights
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        
        self.model = None
        self.processor = None
        
        self.load_model()
    
    def load_model(self):
        """Load the Qwen2.5-VL model and processor."""
        print(f"Loading Qwen model from {self.model_path}...")
        
        # Load model - use AutoModel to handle version correctly
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load processor - use AutoProcessor for compatibility
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        print(f"✓ Model loaded successfully on {self.device}")
    
    @staticmethod
    def _cleanup_answer(text: str, max_words: int = 6) -> str:
        """
        Heuristic cleanup to keep answers short and comparable to VQA annotations.
        
        Args:
            text: Generated answer text
            max_words: Maximum number of words to keep
            
        Returns:
            Cleaned answer string
        """
        # Remove markdown code fences and newlines
        text = text.replace("```", " ").replace("\n", " ").strip()
        
        # Remove common prefixes
        lower = text.lower()
        prefixes = [
            "the answer is", "answer is", "answer:", 
            "the text in the image reads", "the text reads", 
            "it reads", "it says", "text:",
            "the image shows", "this is", "this shows",
        ]
        for p in prefixes:
            if lower.startswith(p):
                text = text[len(p):].strip(" :,-")
                break
        
        # Take the first clause (before punctuation)
        text = re.split(r"[.;!?]", text)[0].strip()
        
        # Limit to max_words
        words = text.split()
        if max_words and len(words) > max_words:
            text = " ".join(words[:max_words])
        
        return text.strip()
    
    def generate_answer(
        self, 
        image: Image.Image, 
        question: str, 
        max_new_tokens: int = 15,
        system_prompt: str = None
    ) -> str:
        """
        Generate an answer for the given image and question.
        
        Args:
            image: PIL Image
            question: Question string
            max_new_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            Generated answer string (cleaned)
        """
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt is None:
            system_prompt = "You are answering visual questions. Respond with a short phrase, not a full sentence."
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add user message with image and question
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        })
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate with greedy decoding (most stable)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for stability
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Clean up answer
        raw_answer = output_text[0]
        return self._cleanup_answer(raw_answer)
    
    def get_model(self):
        """Return the underlying model for training."""
        return self.model
    
    def get_processor(self):
        """Return the processor."""
        return self.processor


if __name__ == "__main__":
    # Test model initialization
    print("Testing QwenVLModel...")
    
    model = QwenVLModel()
    
    # Create a test image
    img = Image.new('RGB', (224, 224), color='red')
    
    # Test generation
    answer = model.generate_answer(img, "What color is this image?")
    print(f"\nTest question: What color is this image?")
    print(f"Answer: {answer}")
    
    print("\n✓ Model test passed!")
