"""
Simplified TextVQA dataset loader following classmate's successful approach.
Direct loading from HuggingFace with OCR token support.
"""

from datasets import load_dataset
from torch.utils.data import Dataset


class TextVQADataset(Dataset):
    """
    Simple TextVQA dataset wrapper.
    Loads directly from HuggingFace and provides access to all fields including OCR tokens.
    """
    
    def __init__(self, split="train", cache_dir=None):
        """
        Args:
            split (str): One of "train", "validation", "test".
            cache_dir (str, optional): Directory to cache the dataset.
        """
        self.split = split
        print(f"Loading TextVQA dataset split: {split}...")
        self.dataset = load_dataset(
            "lmms-lab/textvqa", 
            split=split, 
            cache_dir=cache_dir
        )
        print(f"Loaded {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            dict with keys:
                - image: PIL Image
                - question: str
                - answers: list of str (ground truth answers)
                - image_id: str
                - ocr_tokens: list of str (OCR text detected in image)
        """
        item = self.dataset[idx]
        
        # Extract fields
        image = item['image']
        question = item['question']
        image_id = item['image_id']
        
        # 'answers' might not be present in test split or might be in a specific format
        answers = item.get('answers', [])
        ocr_tokens = item.get('ocr_tokens', [])
        
        return {
            "image": image,
            "question": question,
            "answers": answers,
            "image_id": image_id,
            "ocr_tokens": ocr_tokens
        }


if __name__ == "__main__":
    # Simple test
    print("Testing TextVQADataset...")
    
    ds = TextVQADataset(split="validation")
    sample = ds[0]
    
    print("\nSample 0:")
    print(f"  Image size: {sample['image'].size}")
    print(f"  Question: {sample['question']}")
    print(f"  Answers: {sample['answers']}")
    print(f"  Image ID: {sample['image_id']}")
    print(f"  OCR tokens ({len(sample['ocr_tokens'])}): {sample['ocr_tokens'][:5]}...")
    
    print("\n✓ Dataset test passed!")
