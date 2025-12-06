"""
Script to download and prepare the TextVQA dataset from Hugging Face.
"""

import os
from datasets import load_dataset
from pathlib import Path
import json
import argparse


def download_textvqa_dataset(output_dir="./textvqa_data", cache_dir=None):
    """
    Download TextVQA dataset from Hugging Face.
    
    Args:
        output_dir: Directory to save the dataset
        cache_dir: Cache directory for Hugging Face datasets
    """
    print("=" * 60)
    print("Downloading TextVQA Dataset from Hugging Face")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    print("\n[1/4] Loading dataset from Hugging Face...")
    
    # Disable hf_transfer if it causes issues
    if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1':
        try:
            import hf_transfer
        except ImportError:
            print("  ⚠ Warning: HF_HUB_ENABLE_HF_TRANSFER is set but hf_transfer is not installed.")
            print("  → Disabling fast transfer mode...")
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    
    # Load dataset (trust_remote_code removed as it's deprecated)
    dataset = load_dataset(
        "lmms-lab/textvqa",
        cache_dir=cache_dir
    )
    
    print(f"\n[2/4] Dataset loaded successfully!")
    print(f"  - Train samples: {len(dataset['train'])}")
    print(f"  - Validation samples: {len(dataset['validation'])}")
    print(f"  - Test samples: {len(dataset['test'])}")
    
    # Save dataset splits
    print("\n[3/4] Saving dataset splits...")
    for split in ['train', 'validation', 'test']:
        split_path = output_path / split
        split_path.mkdir(exist_ok=True)
        
        # Save the dataset in arrow format (efficient)
        dataset[split].save_to_disk(str(split_path / "data"))
        
        # Also save a JSON summary for quick inspection
        summary = []
        for idx, example in enumerate(dataset[split]):
            if idx < 10:  # Save first 10 examples as preview
                summary.append({
                    'question_id': example.get('question_id', idx),
                    'question': example.get('question', ''),
                    'answers': example.get('answers', []),
                    'image_id': example.get('image_id', '')
                })
        
        with open(split_path / "preview.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ {split} split saved to {split_path}")
    
    # Print dataset statistics
    print("\n[4/4] Dataset Statistics:")
    print("-" * 60)
    
    # Analyze a sample from training set
    train_sample = dataset['train'][0]
    print("\nSample data structure:")
    for key in train_sample.keys():
        value = train_sample[key]
        if isinstance(value, list):
            print(f"  - {key}: list with {len(value)} items")
        elif hasattr(value, 'size'):  # Image
            print(f"  - {key}: Image")
        else:
            print(f"  - {key}: {type(value).__name__}")
    
    print("\n" + "=" * 60)
    print("Dataset download completed successfully!")
    print(f"Data saved to: {output_path.absolute()}")
    print("=" * 60)
    
    return dataset


def inspect_dataset(data_dir="./textvqa_data"):
    """
    Inspect the downloaded dataset and print sample information.
    
    Args:
        data_dir: Directory containing the dataset
    """
    from datasets import load_from_disk
    
    print("\n" + "=" * 60)
    print("Dataset Inspection")
    print("=" * 60)
    
    data_path = Path(data_dir)
    
    for split in ['train', 'validation', 'test']:
        split_path = data_path / split / "data"
        if split_path.exists():
            dataset = load_from_disk(str(split_path))
            print(f"\n{split.upper()} Split:")
            print(f"  - Total samples: {len(dataset)}")
            
            # Show a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"\n  Sample from {split}:")
                print(f"    Question: {sample.get('question', 'N/A')}")
                print(f"    Answers: {sample.get('answers', 'N/A')}")
                if 'image' in sample:
                    print(f"    Image: {type(sample['image'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TextVQA dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./textvqa_data",
        help="Directory to save the dataset"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for Hugging Face datasets"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect the dataset after downloading"
    )
    
    args = parser.parse_args()
    
    # Download dataset
    dataset = download_textvqa_dataset(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    # Optionally inspect
    if args.inspect:
        inspect_dataset(args.output_dir)
    
    print("\n✓ All done! You can now use the dataset for training and evaluation.")
