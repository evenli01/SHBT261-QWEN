"""
Inference-Time Ablation Studies
NO RETRAINING NEEDED - Just test different inference strategies!

Time estimate: ~5-6 hours total for all ablations
Uses pre-trained checkpoint from main training run
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import Counter

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.metrics import compute_all_metrics


@dataclass
class InferenceConfig:
    """Configuration for inference ablation"""
    name: str
    description: str
    
    # Generation parameters
    max_new_tokens: int = 32
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False
    num_beams: int = 1
    repetition_penalty: float = 1.0
    
    # Prompting strategy
    prompt_template: str = "standard"
    use_ocr: bool = False
    ocr_method: Optional[str] = None
    
    # Post-processing
    post_process: bool = True
    fuzzy_matching: bool = False
    
    # Other
    use_ensemble: bool = False
    ensemble_prompts: List[str] = field(default_factory=list)


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PROMPT_TEMPLATES = {
    "standard": """Question: {question}
Answer:""",
    
    "detailed": """You are an expert at reading text in images and answering questions.

Instructions:
1. Carefully examine the image for any visible text
2. Read all text accurately, including signs, labels, displays, etc.
3. Use the text information to answer the question precisely
4. Give a short, direct answer

Question: {question}
Answer:""",
    
    "chain_of_thought": """Let's solve this step by step:

1. What text can you see in the image?
2. What does the question ask?
3. How does the visible text help answer the question?

Question: {question}

Let's think:
1. Visible text: """,
    
    "few_shot": """Here are some examples:

Example 1:
Question: What brand is the phone?
Answer: nokia

Example 2:
Question: What is the speed limit?
Answer: 55

Now answer this:
Question: {question}
Answer:""",
    
    "with_ocr": """Question: {question}

OCR detected text in image: {ocr_text}

Based on the image and detected text, provide a short answer:
Answer:""",
}


# ============================================================================
# OCR INTEGRATION
# ============================================================================

class OCRExtractor:
    """Extract text from images using various OCR methods"""
    
    def __init__(self, method="paddleocr"):
        self.method = method
        
        if method == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=torch.cuda.is_available(),
                    show_log=False
                )
            except ImportError:
                print("Warning: PaddleOCR not installed. Install with: pip install paddleocr")
                self.ocr = None
                
        elif method == "easyocr":
            try:
                import easyocr
                self.ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            except ImportError:
                print("Warning: EasyOCR not installed. Install with: pip install easyocr")
                self.ocr = None
                
        elif method == "tesseract":
            try:
                import pytesseract
                self.ocr = pytesseract
            except ImportError:
                print("Warning: Pytesseract not installed. Install with: pip install pytesseract")
                self.ocr = None
        else:
            self.ocr = None
    
    def extract(self, image_path: str) -> str:
        """Extract text from image"""
        if self.ocr is None:
            return ""
        
        try:
            if self.method == "paddleocr":
                result = self.ocr.ocr(image_path, cls=True)
                if result and result[0]:
                    texts = [line[1][0] for line in result[0]]
                    return " ".join(texts)
                return ""
                
            elif self.method == "easyocr":
                result = self.ocr.readtext(image_path)
                texts = [detection[1] for detection in result]
                return " ".join(texts)
                
            elif self.method == "tesseract":
                from PIL import Image
                image = Image.open(image_path)
                text = self.ocr.image_to_string(image)
                return text.strip()
                
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""
        
        return ""


# ============================================================================
# POST-PROCESSING
# ============================================================================

def basic_post_process(answer: str) -> str:
    """Basic post-processing of model output"""
    # Take first line only
    answer = answer.split('\n')[0].strip()
    
    # Remove common prefixes
    for prefix in ["answer:", "the answer is", "answer is", "a:", "answer -"]:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove quotes
    answer = answer.strip('"\'')
    
    # Lowercase for matching
    answer = answer.lower()
    
    return answer


def fuzzy_post_process(answer: str, ocr_text: str = "") -> str:
    """Post-process with fuzzy matching against OCR text"""
    from fuzzywuzzy import fuzz
    
    answer = basic_post_process(answer)
    
    if not ocr_text:
        return answer
    
    # Check if answer closely matches any OCR word
    ocr_words = ocr_text.lower().split()
    best_match = answer
    best_score = 0
    
    for word in ocr_words:
        score = fuzz.ratio(answer, word)
        if score > best_score and score > 80:  # 80% similarity threshold
            best_score = score
            best_match = word
    
    return best_match


# ============================================================================
# INFERENCE ABLATION EXPERIMENTS
# ============================================================================

# 1. GENERATION PARAMETER ABLATIONS (~1 hour)
GENERATION_ABLATIONS = [
    InferenceConfig(
        name="baseline_greedy",
        description="Baseline: Greedy decoding, low temperature",
        temperature=0.1,
        do_sample=False,
        num_beams=1,
    ),
    InferenceConfig(
        name="beam_search_3",
        description="Beam search with 3 beams",
        temperature=0.1,
        do_sample=False,
        num_beams=3,
    ),
    InferenceConfig(
        name="beam_search_5",
        description="Beam search with 5 beams",
        temperature=0.1,
        do_sample=False,
        num_beams=5,
    ),
    InferenceConfig(
        name="sampling_temp_03",
        description="Sampling with temperature 0.3",
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
        num_beams=1,
    ),
    InferenceConfig(
        name="sampling_temp_05",
        description="Sampling with temperature 0.5",
        temperature=0.5,
        do_sample=True,
        top_p=0.95,
        num_beams=1,
    ),
]

# 2. PROMPT ABLATIONS (~1 hour)
PROMPT_ABLATIONS = [
    InferenceConfig(
        name="prompt_standard",
        description="Standard simple prompt",
        prompt_template="standard",
    ),
    InferenceConfig(
        name="prompt_detailed",
        description="Detailed instruction prompt",
        prompt_template="detailed",
    ),
    InferenceConfig(
        name="prompt_cot",
        description="Chain-of-thought prompt",
        prompt_template="chain_of_thought",
    ),
    InferenceConfig(
        name="prompt_few_shot",
        description="Few-shot examples prompt",
        prompt_template="few_shot",
    ),
]

# 3. OCR ABLATIONS (~1-2 hours)
OCR_ABLATIONS = [
    InferenceConfig(
        name="no_ocr",
        description="No OCR baseline",
        use_ocr=False,
    ),
    InferenceConfig(
        name="paddleocr",
        description="With PaddleOCR",
        use_ocr=True,
        ocr_method="paddleocr",
        prompt_template="with_ocr",
    ),
    InferenceConfig(
        name="easyocr",
        description="With EasyOCR",
        use_ocr=True,
        ocr_method="easyocr",
        prompt_template="with_ocr",
    ),
    InferenceConfig(
        name="tesseract",
        description="With Tesseract OCR",
        use_ocr=True,
        ocr_method="tesseract",
        prompt_template="with_ocr",
    ),
]

# 4. POST-PROCESSING ABLATIONS (~30 min)
POSTPROCESS_ABLATIONS = [
    InferenceConfig(
        name="no_postprocess",
        description="No post-processing",
        post_process=False,
    ),
    InferenceConfig(
        name="basic_postprocess",
        description="Basic post-processing",
        post_process=True,
        fuzzy_matching=False,
    ),
    InferenceConfig(
        name="fuzzy_postprocess",
        description="Fuzzy matching post-processing",
        post_process=True,
        fuzzy_matching=True,
        use_ocr=True,
        ocr_method="paddleocr",
    ),
]

# 5. ENSEMBLE ABLATIONS (~1 hour)
ENSEMBLE_ABLATIONS = [
    InferenceConfig(
        name="single_model",
        description="Single model, single prompt",
        use_ensemble=False,
    ),
    InferenceConfig(
        name="multi_prompt_ensemble",
        description="Ensemble of multiple prompts",
        use_ensemble=True,
        ensemble_prompts=["standard", "detailed", "few_shot"],
    ),
    InferenceConfig(
        name="multi_prompt_with_ocr",
        description="Ensemble with OCR integration",
        use_ensemble=True,
        use_ocr=True,
        ocr_method="paddleocr",
        ensemble_prompts=["standard", "detailed", "with_ocr"],
    ),
]

# ALL ABLATIONS
ALL_ABLATIONS = {
    "generation": GENERATION_ABLATIONS,
    "prompting": PROMPT_ABLATIONS,
    "ocr": OCR_ABLATIONS,
    "postprocessing": POSTPROCESS_ABLATIONS,
    "ensemble": ENSEMBLE_ABLATIONS,
}


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """Run inference with specific configuration"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        print(f"Loading model from {model_path}...")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        
        self.ocr_cache = {}  # Cache OCR results
    
    def generate(
        self,
        image_path: str,
        question: str,
        config: InferenceConfig
    ) -> str:
        """Generate answer with given configuration"""
        
        # 1. Extract OCR if needed
        ocr_text = ""
        if config.use_ocr and config.ocr_method:
            cache_key = f"{image_path}_{config.ocr_method}"
            if cache_key in self.ocr_cache:
                ocr_text = self.ocr_cache[cache_key]
            else:
                ocr_extractor = OCRExtractor(config.ocr_method)
                ocr_text = ocr_extractor.extract(image_path)
                self.ocr_cache[cache_key] = ocr_text
        
        # 2. Create prompt
        prompt_template = PROMPT_TEMPLATES.get(config.prompt_template, PROMPT_TEMPLATES["standard"])
        prompt = prompt_template.format(question=question, ocr_text=ocr_text)
        
        # 3. Process image and text
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # 4. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                num_beams=config.num_beams,
                repetition_penalty=config.repetition_penalty,
            )
        
        # 5. Decode
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        if prompt in answer:
            answer = answer.replace(prompt, "").strip()
        
        # 6. Post-process
        if config.post_process:
            if config.fuzzy_matching and ocr_text:
                answer = fuzzy_post_process(answer, ocr_text)
            else:
                answer = basic_post_process(answer)
        
        return answer
    
    def ensemble_generate(
        self,
        image_path: str,
        question: str,
        config: InferenceConfig
    ) -> str:
        """Generate with ensemble of prompts"""
        
        predictions = []
        
        for prompt_name in config.ensemble_prompts:
            temp_config = InferenceConfig(
                name=f"ensemble_{prompt_name}",
                description="",
                prompt_template=prompt_name,
                use_ocr=config.use_ocr,
                ocr_method=config.ocr_method,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                num_beams=config.num_beams,
            )
            
            answer = self.generate(image_path, question, temp_config)
            predictions.append(answer)
        
        # Voting
        vote_counts = Counter(predictions)
        final_answer = vote_counts.most_common(1)[0][0]
        
        return final_answer


# ============================================================================
# RUN ABLATION STUDY
# ============================================================================

def run_ablation_study(
    model_path: str,
    dataset_name: str = "lmms-lab/textvqa",
    split: str = "validation",
    max_samples: int = None,
    output_dir: str = "results/inference_ablations",
    ablation_type: str = "all"
):
    """Run inference ablation studies"""
    
    print("=" * 80)
    print("INFERENCE-TIME ABLATION STUDIES")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name} ({split})")
    print(f"Ablation type: {ablation_type}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize inference engine
    engine = InferenceEngine(model_path)
    
    # Select ablations to run
    if ablation_type == "all":
        ablations_to_run = []
        for ablations in ALL_ABLATIONS.values():
            ablations_to_run.extend(ablations)
    else:
        ablations_to_run = ALL_ABLATIONS.get(ablation_type, [])
    
    print(f"Running {len(ablations_to_run)} ablation experiments\n")
    
    # Run each ablation
    all_results = {}
    
    for ablation_config in ablations_to_run:
        print(f"\n{'='*80}")
        print(f"Running: {ablation_config.name}")
        print(f"Description: {ablation_config.description}")
        print(f"{'='*80}")
        
        start_time = time.time()
        predictions = []
        references = []
        
        for sample in tqdm(dataset, desc=ablation_config.name):
            image_path = sample['image_path']  # Adjust based on actual dataset structure
            question = sample['question']
            answers = sample['answers']  # List of acceptable answers
            
            # Generate answer
            if ablation_config.use_ensemble:
                pred = engine.ensemble_generate(image_path, question, ablation_config)
            else:
                pred = engine.generate(image_path, question, ablation_config)
            
            predictions.append(pred)
            references.append(answers)
        
        elapsed_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_all_metrics(predictions, references)
        metrics['time_seconds'] = elapsed_time
        metrics['samples_per_second'] = len(dataset) / elapsed_time
        
        # Save results
        result = {
            "config": ablation_config.__dict__,
            "metrics": metrics,
            "predictions": predictions[:100],  # Save first 100 for analysis
        }
        
        all_results[ablation_config.name] = result
        
        # Print summary
        print(f"\nResults for {ablation_config.name}:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Time: {elapsed_time:.1f}s ({metrics['samples_per_second']:.1f} samples/s)")
        print()
    
    # Save all results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{ablation_type}_ablations.json")
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {output_path}")
    
    # Print summary table
    print_summary_table(all_results)
    
    return all_results


def print_summary_table(results: Dict[str, Any]):
    """Print summary table of all ablation results"""
    print("\n" + "=" * 100)
    print("ABLATION STUDY SUMMARY")
    print("=" * 100)
    print(f"{'Experiment':<30} {'Accuracy':<12} {'BLEU':<12} {'Time (s)':<12} {'Samples/s':<12}")
    print("-" * 100)
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"{name:<30} {metrics['accuracy']:<12.2%} {metrics.get('bleu', 0):<12.4f} "
              f"{metrics['time_seconds']:<12.1f} {metrics['samples_per_second']:<12.2f}")
    
    print("=" * 100)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference-time ablation studies")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="lmms-lab/textvqa", help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/inference_ablations", help="Output directory")
    parser.add_argument(
        "--ablation_type",
        type=str,
        default="all",
        choices=["all", "generation", "prompting", "ocr", "postprocessing", "ensemble"],
        help="Type of ablation to run"
    )
    
    args = parser.parse_args()
    
    run_ablation_study(
        model_path=args.model_path,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        ablation_type=args.ablation_type
    )
