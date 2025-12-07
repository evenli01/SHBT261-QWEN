"""
Metrics calculation following classmate's successful approach.
Primary metric: VQA-style accuracy (min(1, matches/3))
"""

import torch
import evaluate
from sentence_transformers import SentenceTransformer, util
import numpy as np


# Load metrics globally to avoid reloading
try:
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    rouge_metric = evaluate.load("rouge")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
except Exception as e:
    print(f"Warning: Failed to load some metrics: {e}")
    bleu_metric = None
    meteor_metric = None
    rouge_metric = None
    semantic_model = None


def preprocess_answer(answer):
    """
    Standard VQA text preprocessing.
    Lowercase, remove extra whitespace.
    """
    answer = str(answer).lower()
    answer = answer.replace('\n', ' ')
    answer = answer.replace('\t', ' ')
    answer = answer.strip()
    return answer


def compute_vqa_accuracy(ground_truth_list, predicted_answer):
    """
    Compute VQA-style consensus accuracy (official metric).
    
    Formula: acc = min(1, (number of matching human answers) / 3)
    
    Args:
        ground_truth_list: List of ground truth answers
        predicted_answer: Predicted answer string
        
    Returns:
        Accuracy score between 0 and 1
    """
    if not ground_truth_list:
        return 0.0

    predicted_answer = preprocess_answer(predicted_answer)
    ground_truth_list = [preprocess_answer(ans) for ans in ground_truth_list]

    # Count exact matches
    match_count = sum(1 for gt in ground_truth_list if gt == predicted_answer)
    
    # VQA formula: min(1, matches/3)
    return min(1.0, match_count / 3.0)


def compute_semantic_similarity(ground_truth_list, predicted_answer):
    """
    Compute semantic similarity between predicted answer and ground truth answers.
    Returns the maximum similarity score among all ground truth answers.
    """
    if semantic_model is None or not ground_truth_list:
        return 0.0
    
    try:
        # Encode
        pred_emb = semantic_model.encode(predicted_answer, convert_to_tensor=True)
        gt_embs = semantic_model.encode(ground_truth_list, convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = util.cos_sim(pred_emb, gt_embs)
        
        # Return max similarity
        return float(torch.max(cosine_scores).item()) if cosine_scores.numel() > 0 else 0.0
    except Exception as e:
        print(f"Error computing semantic similarity: {e}")
        return 0.0


def calculate_metrics(results):
    """
    Calculate average metrics over all results.
    
    Args:
        results: List of dicts with 'predicted_answer' and 'ground_truth_answers'
        
    Returns:
        Dict of metric name -> score
    """
    if not results:
        return {}

    total_acc = 0
    predictions = []
    references = []
    semantic_scores = []
    
    for item in results:
        pred = item['predicted_answer']
        gts = item['ground_truth_answers']
        
        # VQA Accuracy
        acc = compute_vqa_accuracy(gts, pred)
        total_acc += acc
        
        # For BLEU/METEOR/ROUGE, we need lists
        predictions.append(pred)
        references.append(gts)
        
        # Semantic Similarity
        if semantic_model:
            sim = compute_semantic_similarity(gts, pred)
            semantic_scores.append(sim)

    metrics = {
        "accuracy": total_acc / len(results)
    }
    
    # BLEU - use bigrams and smoothing for short VQA answers
    if bleu_metric is not None:
        try:
            bleu_score = bleu_metric.compute(
                predictions=predictions, 
                references=references,
                max_order=2,  # Bigrams (VQA answers are short)
                smooth=True   # Enable smoothing for short texts
            )
            metrics["bleu"] = bleu_score['bleu']
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            metrics["bleu"] = 0.0

    # METEOR
    if meteor_metric is not None:
        try:
            meteor_score = meteor_metric.compute(
                predictions=predictions, 
                references=references
            )
            metrics["meteor"] = meteor_score['meteor']
        except Exception as e:
            print(f"Error computing METEOR: {e}")
            metrics["meteor"] = 0.0

    # ROUGE
    if rouge_metric is not None:
        try:
            rouge_score = rouge_metric.compute(
                predictions=predictions, 
                references=references
            )
            metrics["rouge1"] = rouge_score['rouge1']
            metrics["rouge2"] = rouge_score['rouge2']
            metrics["rougeL"] = rouge_score['rougeL']
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            metrics["rouge1"] = 0.0
            metrics["rouge2"] = 0.0
            metrics["rougeL"] = 0.0
    
    # Semantic Similarity
    if semantic_scores:
        metrics["semantic_similarity"] = sum(semantic_scores) / len(semantic_scores)
    else:
        metrics["semantic_similarity"] = 0.0
    
    return metrics


def print_metrics(metrics, title="Evaluation Results"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dict of metric name -> score
        title: Title for the results
    """
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    
    # Print VQA accuracy first (primary metric)
    if "accuracy" in metrics:
        print(f"VQA Accuracy (Primary):    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    # Print other metrics
    print("\nSecondary Metrics:")
    for key, value in sorted(metrics.items()):
        if key != "accuracy":
            print(f"  {key:25s}: {value:.4f}")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing metrics calculation...")
    
    # Example results
    test_results = [
        {
            "predicted_answer": "red",
            "ground_truth_answers": ["red", "red", "crimson"]
        },
        {
            "predicted_answer": "blue",
            "ground_truth_answers": ["blue", "azure", "sky blue"]
        },
        {
            "predicted_answer": "green",
            "ground_truth_answers": ["red", "red", "crimson"]
        },
    ]
    
    metrics = calculate_metrics(test_results)
    print_metrics(metrics, "Test Results")
    
    print("✓ Metrics test passed!")
