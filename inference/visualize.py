"""
Visualization utilities for TextVQA results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def visualize_predictions(
    image_paths: List[str],
    questions: List[str],
    predictions: List[str],
    ground_truths: List[List[str]],
    correct: List[bool],
    num_samples: int = 10,
    save_path: str = None
):
    """
    Visualize model predictions with images and questions.
    
    Args:
        image_paths: List of image paths
        questions: List of questions
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        correct: List of booleans indicating correctness
        num_samples: Number of samples to visualize
        save_path: Path to save the figure
    """
    num_samples = min(num_samples, len(image_paths))
    
    fig, axes = plt.subplots(num_samples//2, 2, figsize=(15, 5*num_samples//2))
    axes = axes.flatten()
    
    for i in range(num_samples):
        try:
            # Load and display image
            img = Image.open(image_paths[i])
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Create title with question and answers
            color = 'green' if correct[i] else 'red'
            title = f"Q: {questions[i]}\n"
            title += f"Pred: {predictions[i]}\n"
            title += f"GT: {', '.join(ground_truths[i])}"
            
            axes[i].set_title(title, fontsize=10, color=color, pad=10)
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading image:\n{e}",
                        ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str = None
):
    """
    Plot comparison of metrics across different models/experiments.
    
    Args:
        results: Dictionary mapping model names to metric dictionaries
        save_path: Path to save the figure
    """
    # Prepare data
    models = list(results.keys())
    metrics = ['accuracy', 'f1', 'bleu', 'meteor', 'rougeL']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]
        
        bars = axes[idx].bar(range(len(models)), values, color=sns.color_palette("husl", len(models)))
        axes[idx].set_xlabel('Model')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].set_title(f'{metric.capitalize()} Comparison')
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels(models, rotation=45, ha='right')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=9)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    log_file: str,
    save_path: str = None
):
    """
    Plot training curves from training logs.
    
    Args:
        log_file: Path to training log JSON file
        save_path: Path to save the figure
    """
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    if 'train_loss' in logs:
        axes[0, 0].plot(logs['train_loss'], label='Train Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Validation metrics
    if 'val_accuracy' in logs:
        axes[0, 1].plot(logs['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in logs:
        axes[1, 0].plot(logs['learning_rate'], label='Learning Rate')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Multiple metrics
    if 'val_f1' in logs and 'val_bleu' in logs:
        axes[1, 1].plot(logs.get('val_f1', []), label='F1')
        axes[1, 1].plot(logs.get('val_bleu', []), label='BLEU')
        axes[1, 1].plot(logs.get('val_meteor', []), label='METEOR')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_error_analysis_report(
    predictions: List[str],
    ground_truths: List[List[str]],
    questions: List[str],
    save_path: str = None
):
    """
    Create error analysis report.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        questions: List of questions
        save_path: Path to save the report
    """
    from evaluation.metrics import TextVQAMetrics
    
    metrics_calc = TextVQAMetrics()
    
    # Categorize errors
    correct = []
    incorrect = []
    
    for i, (pred, gts) in enumerate(zip(predictions, ground_truths)):
        is_correct = metrics_calc.compute_exact_match(pred, gts) > 0.5
        
        if is_correct:
            correct.append({
                'question': questions[i],
                'prediction': pred,
                'ground_truth': gts
            })
        else:
            incorrect.append({
                'question': questions[i],
                'prediction': pred,
                'ground_truth': gts
            })
    
    # Create report
    report = []
    report.append("=" * 80)
    report.append("ERROR ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal Examples: {len(predictions)}")
    report.append(f"Correct: {len(correct)} ({100*len(correct)/len(predictions):.2f}%)")
    report.append(f"Incorrect: {len(incorrect)} ({100*len(incorrect)/len(predictions):.2f}%)")
    report.append("\n" + "=" * 80)
    report.append("SAMPLE ERRORS")
    report.append("=" * 80)
    
    # Show first 20 errors
    for i, error in enumerate(incorrect[:20], 1):
        report.append(f"\nError {i}:")
        report.append(f"  Question: {error['question']}")
        report.append(f"  Prediction: {error['prediction']}")
        report.append(f"  Ground Truth: {', '.join(error['ground_truth'])}")
        report.append("-" * 80)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Error analysis saved to {save_path}")
    else:
        print(report_text)


def visualize_ablation_results(results_dir: str, save_dir: str = None):
    """
    Visualize ablation study results.
    
    Args:
        results_dir: Directory containing ablation results
        save_dir: Directory to save visualizations
    """
    results_path = Path(results_dir)
    
    # Load comparison CSV
    comparison_file = results_path / "comparison.csv"
    if not comparison_file.exists():
        print(f"Comparison file not found: {comparison_file}")
        return
    
    df = pd.read_csv(comparison_file)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot accuracy by experiment type
    plt.figure(figsize=(14, 6))
    
    # Sort by accuracy
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    colors = sns.color_palette("coolwarm", len(df_sorted))
    bars = plt.bar(range(len(df_sorted)), df_sorted['accuracy'], color=colors)
    
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Ablation Study: Accuracy by Configuration', fontsize=14, fontweight='bold')
    plt.xticks(range(len(df_sorted)), df_sorted['experiment'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / "ablation_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    
    print(f"✓ Ablation visualizations complete")


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities for TextVQA")
    print("Import and use functions in your scripts")
