"""
Visualize experiment results for report.
Creates publication-quality figures comparing ablations.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_results(results_dir):
    """Load all result JSON files."""
    results = {
        'zero_shot': {},
        'finetuned': {}
    }
    
    # Load zero-shot results
    zero_shot_dir = os.path.join(results_dir, 'zero_shot')
    if os.path.exists(zero_shot_dir):
        for file in os.listdir(zero_shot_dir):
            if file.endswith('.json'):
                with open(os.path.join(zero_shot_dir, file), 'r') as f:
                    data = json.load(f)
                    # Extract prompt template from filename
                    for template in ['no_ocr', 'basic_ocr', 'structured_ocr']:
                        if template in file:
                            results['zero_shot'][template] = data
                            break
    
    # Load fine-tuned results
    finetuned_dir = os.path.join(results_dir, 'finetuned')
    if os.path.exists(finetuned_dir):
        for file in os.listdir(finetuned_dir):
            if file.endswith('.json'):
                with open(os.path.join(finetuned_dir, file), 'r') as f:
                    data = json.load(f)
                    for template in ['no_ocr', 'basic_ocr', 'structured_ocr']:
                        if template in file:
                            results['finetuned'][template] = data
                            break
    
    return results


def plot_accuracy_comparison(results, output_dir):
    """Plot zero-shot vs fine-tuned accuracy comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['no_ocr', 'basic_ocr', 'structured_ocr']
    method_names = ['No OCR', 'Basic OCR', 'Structured OCR']
    
    zero_shot_acc = []
    finetuned_acc = []
    
    for method in methods:
        zs = results['zero_shot'].get(method, {}).get('metrics', {}).get('accuracy', 0) * 100
        ft = results['finetuned'].get(method, {}).get('metrics', {}).get('accuracy', 0) * 100
        zero_shot_acc.append(zs)
        finetuned_acc.append(ft)
    
    x = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, zero_shot_acc, width, label='Zero-Shot', color='#3498db')
    bars2 = ax.bar(x + width/2, finetuned_acc, width, label='Fine-Tuned', color='#e74c3c')
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('VQA Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Zero-Shot vs Fine-Tuned Performance', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: accuracy_comparison.png")
    plt.close()


def plot_ocr_ablation(results, output_dir):
    """Plot OCR ablation study results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = ['no_ocr', 'basic_ocr', 'structured_ocr']
    method_names = ['No OCR', 'Basic OCR', 'Structured OCR']
    
    # Zero-shot metrics
    zs_metrics = {
        'accuracy': [],
        'bleu': [],
        'meteor': []
    }
    
    for method in methods:
        data = results['zero_shot'].get(method, {}).get('metrics', {})
        zs_metrics['accuracy'].append(data.get('accuracy', 0) * 100)
        zs_metrics['bleu'].append(data.get('bleu', 0) * 100)
        zs_metrics['meteor'].append(data.get('meteor', 0) * 100)
    
    # Fine-tuned metrics
    ft_metrics = {
        'accuracy': [],
        'bleu': [],
        'meteor': []
    }
    
    for method in methods:
        data = results['finetuned'].get(method, {}).get('metrics', {})
        ft_metrics['accuracy'].append(data.get('accuracy', 0) * 100)
        ft_metrics['bleu'].append(data.get('bleu', 0) * 100)
        ft_metrics['meteor'].append(data.get('meteor', 0) * 100)
    
    # Plot zero-shot
    x = np.arange(len(method_names))
    width = 0.25
    
    ax1.bar(x - width, zs_metrics['accuracy'], width, label='Accuracy', color='#3498db')
    ax1.bar(x, zs_metrics['bleu'], width, label='BLEU', color='#2ecc71')
    ax1.bar(x + width, zs_metrics['meteor'], width, label='METEOR', color='#f39c12')
    
    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Zero-Shot Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot fine-tuned
    ax2.bar(x - width, ft_metrics['accuracy'], width, label='Accuracy', color='#3498db')
    ax2.bar(x, ft_metrics['bleu'], width, label='BLEU', color='#2ecc71')
    ax2.bar(x + width, ft_metrics['meteor'], width, label='METEOR', color='#f39c12')
    
    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Fine-Tuned Performance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ocr_ablation_comparison.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: ocr_ablation_comparison.png")
    plt.close()


def plot_question_type_breakdown(results, output_dir):
    """Plot per-question-type accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    q_types = ['brand', 'number', 'date', 'time', 'text', 'general']
    q_type_names = ['Brand', 'Number', 'Date', 'Time', 'Text', 'General']
    
    # Get best method (structured_ocr)
    method = 'structured_ocr'
    
    zero_shot_acc = []
    finetuned_acc = []
    
    for q_type in q_types:
        zs_data = results['zero_shot'].get(method, {}).get('q_type_metrics', {}).get(q_type, {})
        ft_data = results['finetuned'].get(method, {}).get('q_type_metrics', {}).get(q_type, {})
        
        zero_shot_acc.append(zs_data.get('accuracy', 0) * 100)
        finetuned_acc.append(ft_data.get('accuracy', 0) * 100)
    
    x = np.arange(len(q_type_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, zero_shot_acc, width, label='Zero-Shot', color='#3498db')
    bars2 = ax.bar(x + width/2, finetuned_acc, width, label='Fine-Tuned', color='#e74c3c')
    
    ax.set_xlabel('Question Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('VQA Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance by Question Type (Structured OCR)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(q_type_names)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_question_type_accuracy.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: per_question_type_accuracy.png")
    plt.close()


def create_summary_table(results, output_dir):
    """Create a summary table in text format."""
    summary = []
    summary.append("="*80)
    summary.append("EXPERIMENTAL RESULTS SUMMARY")
    summary.append("="*80)
    summary.append("")
    
    # Zero-shot results
    summary.append("Zero-Shot Performance:")
    summary.append("-"*80)
    summary.append(f"{'Method':<20} {'Accuracy':<12} {'BLEU':<12} {'METEOR':<12} {'ROUGE-L':<12}")
    summary.append("-"*80)
    
    for method in ['no_ocr', 'basic_ocr', 'structured_ocr']:
        metrics = results['zero_shot'].get(method, {}).get('metrics', {})
        acc = metrics.get('accuracy', 0) * 100
        bleu = metrics.get('bleu', 0)
        meteor = metrics.get('meteor', 0)
        rouge = metrics.get('rougeL', 0)
        summary.append(f"{method:<20} {acc:<12.2f} {bleu:<12.4f} {meteor:<12.4f} {rouge:<12.4f}")
    
    summary.append("")
    
    # Fine-tuned results
    summary.append("Fine-Tuned Performance:")
    summary.append("-"*80)
    summary.append(f"{'Method':<20} {'Accuracy':<12} {'BLEU':<12} {'METEOR':<12} {'ROUGE-L':<12}")
    summary.append("-"*80)
    
    for method in ['no_ocr', 'basic_ocr', 'structured_ocr']:
        metrics = results['finetuned'].get(method, {}).get('metrics', {})
        acc = metrics.get('accuracy', 0) * 100
        bleu = metrics.get('bleu', 0)
        meteor = metrics.get('meteor', 0)
        rouge = metrics.get('rougeL', 0)
        summary.append(f"{method:<20} {acc:<12.2f} {bleu:<12.4f} {meteor:<12.4f} {rouge:<12.4f}")
    
    summary.append("="*80)
    
    # Save to file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
    
    print("✓ Saved: summary.txt")
    
    # Also print to console
    print('\n'.join(summary))


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--results_dir", default="results", help="Directory containing result JSON files")
    parser.add_argument("--output_dir", default="figures", help="Output directory for figures")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Generating Visualizations")
    print("="*70)
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    results = load_results(args.results_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_comparison(results, args.output_dir)
    plot_ocr_ablation(results, args.output_dir)
    plot_question_type_breakdown(results, args.output_dir)
    
    # Create summary
    print("\nCreating summary...")
    create_summary_table(results, args.output_dir)
    
    print("\n" + "="*70)
    print("✓ All visualizations generated!")
    print(f"✓ Check {args.output_dir}/ for figures")
    print("="*70)


if __name__ == "__main__":
    main()
