"""
Inference script for trained models on TextVQA.
"""

import torch
from PIL import Image
import argparse
from pathlib import Path
from typing import List, Optional, Union
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.qwen_model import QwenVLModel, load_model_for_inference
from models.model_config import ModelConfig


class TextVQAInference:
    """
    Inference class for TextVQA models.
    """
    
    def __init__(
        self,
        model_path: str,
        use_lora: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize inference.
        
        Args:
            model_path: Path to model or LoRA adapter
            use_lora: Whether model uses LoRA
            device: Device to run inference on
        """
        self.device = device
        self.use_lora = use_lora
        
        # Load model
        print(f"Loading model from {model_path}...")
        config = ModelConfig()
        config.use_lora = use_lora
        config.device_map = device
        
        self.model_wrapper = load_model_for_inference(
            model_path=model_path,
            config=config,
            use_lora=use_lora,
            device=device
        )
        
        self.processor = self.model_wrapper.get_processor()
        print("Model loaded successfully!")
    
    def predict(
        self,
        image: Union[str, Image.Image],
        question: str,
        max_new_tokens: int = 128
    ) -> str:
        """
        Make a prediction for a single image-question pair.
        
        Args:
            image: Path to image or PIL Image
            question: Question text
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Predicted answer
        """
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = None
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"].to(self.device)
        elif "image_grid_thw" in inputs:
            pixel_values = inputs["image_grid_thw"].to(self.device)
        
        # Generate
        outputs = self.model_wrapper.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )
        
        # Decode
        generated_ids = outputs[0][len(input_ids[0]):]
        prediction = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()
        
        return prediction
    
    def batch_predict(
        self,
        images: List[Union[str, Image.Image]],
        questions: List[str],
        max_new_tokens: int = 128
    ) -> List[str]:
        """
        Make predictions for multiple image-question pairs.
        
        Args:
            images: List of image paths or PIL Images
            questions: List of questions
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of predicted answers
        """
        predictions = []
        
        for image, question in zip(images, questions):
            try:
                pred = self.predict(image, question, max_new_tokens)
                predictions.append(pred)
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                predictions.append("")
        
        return predictions


def interactive_demo(model_path: str, use_lora: bool = True):
    """
    Run interactive demo.
    
    Args:
        model_path: Path to model or adapter
        use_lora: Whether model uses LoRA
    """
    print("\n" + "=" * 70)
    print("TextVQA Interactive Demo")
    print("=" * 70)
    
    # Initialize inference
    inferencer = TextVQAInference(
        model_path=model_path,
        use_lora=use_lora,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\nDemo ready! Enter 'quit' to exit.")
    print("=" * 70)
    
    while True:
        # Get image path
        image_path = input("\nEnter image path: ").strip()
        if image_path.lower() == 'quit':
            break
        
        if not Path(image_path).exists():
            print(f"Error: Image not found at {image_path}")
            continue
        
        # Get question
        question = input("Enter question: ").strip()
        if question.lower() == 'quit':
            break
        
        # Make prediction
        print("\nGenerating answer...")
        try:
            answer = inferencer.predict(image_path, question)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


def batch_inference_from_file(
    model_path: str,
    input_file: str,
    output_file: str,
    use_lora: bool = True
):
    """
    Run batch inference from a JSON file.
    
    Args:
        model_path: Path to model or adapter
        input_file: Path to input JSON file with format:
                   [{"image": "path/to/image.jpg", "question": "..."}]
        output_file: Path to save predictions
        use_lora: Whether model uses LoRA
    """
    print("\n" + "=" * 70)
    print("Batch Inference")
    print("=" * 70)
    
    # Load input data
    print(f"\nLoading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # Initialize inference
    inferencer = TextVQAInference(
        model_path=model_path,
        use_lora=use_lora,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Make predictions
    print("\nMaking predictions...")
    results = []
    for i, item in enumerate(data, 1):
        print(f"Processing {i}/{len(data)}", end='\r')
        
        try:
            prediction = inferencer.predict(
                image=item['image'],
                question=item['question']
            )
            
            results.append({
                'image': item['image'],
                'question': item['question'],
                'prediction': prediction,
                'ground_truth': item.get('answer', None)
            })
        except Exception as e:
            print(f"\nError processing item {i}: {e}")
            results.append({
                'image': item['image'],
                'question': item['question'],
                'prediction': "",
                'error': str(e)
            })
    
    # Save results
    print(f"\n\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="TextVQA Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model or LoRA adapter"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "interactive", "batch"],
        default="interactive",
        help="Inference mode"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Whether model uses LoRA"
    )
    
    # Single prediction mode
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--question", type=str, help="Question text")
    
    # Batch mode
    parser.add_argument("--input_file", type=str, help="Input JSON file for batch mode")
    parser.add_argument("--output_file", type=str, help="Output file for batch mode")
    
    # Generation settings
    parser.add_argument("--max_new_tokens", type=int, default=128)
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.image or not args.question:
            print("Error: --image and --question required for single mode")
            return
        
        inferencer = TextVQAInference(
            model_path=args.model_path,
            use_lora=args.use_lora
        )
        
        answer = inferencer.predict(
            image=args.image,
            question=args.question,
            max_new_tokens=args.max_new_tokens
        )
        
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {answer}")
    
    elif args.mode == "interactive":
        interactive_demo(
            model_path=args.model_path,
            use_lora=args.use_lora
        )
    
    elif args.mode == "batch":
        if not args.input_file or not args.output_file:
            print("Error: --input_file and --output_file required for batch mode")
            return
        
        batch_inference_from_file(
            model_path=args.model_path,
            input_file=args.input_file,
            output_file=args.output_file,
            use_lora=args.use_lora
        )


if __name__ == "__main__":
    main()
