"""
Data preprocessing utilities for TextVQA + Qwen2.5-VL.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import AutoProcessor
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
#                   Load Local TextVQA Dataset
# ============================================================

def load_textvqa_data(
    data_dir: str,
    split: str = "train",
    use_hf_direct: bool = False,
):
    """
    Load TextVQA dataset from local disk (HuggingFace arrow).

    Folder layout (already in your pod):
        data_dir/
            train/data/...
            validation/data/...
            test/data/...
    """
    if use_hf_direct:
        raise ValueError(
            "Direct HF loading is disabled. Use local dataset at data_dir instead."
        )

    split_path = Path(data_dir) / split / "data"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {split_path}.\n"
            f"Expected structure: {data_dir}/{split}/data/"
        )

    print(f"Loading local dataset: {split_path}")
    dataset = load_from_disk(str(split_path))
    print(f"Loaded {len(dataset)} samples from split '{split}'")
    return dataset


# ============================================================
#                       Dataset Class
# ============================================================

class TextVQADataset(Dataset):
    """
    TextVQA dataset adapted for Qwen2.5-VL.
    """

    def __init__(
        self,
        dataset,
        processor,
        max_length: int = 512,
        split: str = "train"
    ):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Image: either already PIL or file path
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        question = example["question"]
        answers = example.get("answers", [])
        answer = answers[0] if isinstance(answers, list) and len(answers) > 0 else ""

        # Qwen chat-style prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process text + image jointly
        inputs = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Qwen2.5-VL expects BOTH pixel_values and image_grid_thw
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise ValueError(f"Processor missing required keys. Got: {inputs.keys()}")

        pixel_values = inputs["pixel_values"].squeeze(0)      # [num_tokens, dim]
        image_grid_thw = inputs["image_grid_thw"].squeeze(0)  # [3]
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Labels for training only
        if self.split == "train":
            label_ids = self.processor.tokenizer(
                answer,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
        else:
            label_ids = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": label_ids,
            "question": question,
            "answers": answers,
            "question_id": example.get("question_id", idx),
            "image_id": example.get("image_id", ""),
        }


# ============================================================
#                       Collate Function
# ============================================================

def collate_fn(batch: List[Dict]):
    """
    Collate function for Qwen2.5-VL:
    - Pads input_ids + attention_mask to max length in batch
    - Pads pixel_values to max visual token length in batch
    - Stacks image_grid_thw (shape [3] per image)
    """

    # ---- Text padding ----
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list = []
    attention_mask_list = []

    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        pad_len = max_len - ids.shape[0]

        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])

        input_ids_list.append(ids)
        attention_mask_list.append(mask)

    input_ids = torch.stack(input_ids_list)          # [B, L]
    attention_mask = torch.stack(attention_mask_list)

    # ---- Visual tokens padding ----
    pv_list = [item["pixel_values"] for item in batch]        # each [T_i, D]
    max_tokens = max(pv.shape[0] for pv in pv_list)
    hidden_dim = pv_list[0].shape[1]

    padded_pv = []
    for pv in pv_list:
        num_tokens = pv.shape[0]
        if num_tokens < max_tokens:
            pad_tokens = max_tokens - num_tokens
            pad = torch.zeros(pad_tokens, hidden_dim, dtype=pv.dtype)
            pv = torch.cat([pv, pad], dim=0)
        padded_pv.append(pv)

    pixel_values = torch.stack(padded_pv)  # [B, max_tokens, D]

    # ---- image_grid_thw: [3] per image → [B, 3] ----
    grid_list = [item["image_grid_thw"] for item in batch]  # each [3]
    image_grid_thw = torch.stack(grid_list)

    # ---- labels (optional) ----
    labels = (
        torch.stack([item["labels"] for item in batch])
        if batch[0]["labels"] is not None
        else None
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "labels": labels,
        "questions": [b["question"] for b in batch],
        "answers": [b["answers"] for b in batch],
        "question_ids": [b["question_id"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
    }


# ============================================================
#               Helper for full train/val/test loaders
# ============================================================

def create_dataloaders(
    processor,
    data_dir: str,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    num_workers: int = 4,
    subset_size: Optional[int] = None,
):
    train_ds = load_textvqa_data(data_dir, "train")
    val_ds = load_textvqa_data(data_dir, "validation")
    test_ds = load_textvqa_data(data_dir, "test")

    if subset_size and subset_size < len(train_ds):
        idx = np.random.choice(len(train_ds), subset_size, replace=False)
        train_ds = train_ds.select(idx)

    train_data = TextVQADataset(train_ds, processor, split="train")
    val_data = TextVQADataset(val_ds, processor, split="validation")
    test_data = TextVQADataset(test_ds, processor, split="test")

    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader
