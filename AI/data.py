import os

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def load_data(name="imdb", tokenizer_name="deepseek-ai/DeepSeek-V4.1-Flash", limit=None, cache=True):
    """Tokenize a text dataset into two flat streams of ids for causal LM training.

    Returns (tokenizer, train_ids, val_ids) where the ids are 1-D int64 tensors
    holding every document concatenated together, separated by the eos token.
    """
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    tag = f"{name}-{tokenizer_name}-{limit}".replace("/", "_")
    cache_path = os.path.join(CACHE_DIR, f"{tag}.pt")
    if cache and os.path.exists(cache_path):
        blob = torch.load(cache_path)
        return tokenizer, blob["train"], blob["val"]

    dataset = load_dataset(name)

    def encode_split(split):
        texts = dataset[split]["text"]
        if limit is not None:
            texts = texts[:limit]
        ids = []
        # encode_batch is parallel under the hood, so chunk instead of looping per doc
        for start in range(0, len(texts), 1000):
            for encoding in tokenizer.encode_batch(texts[start : start + 1000]):
                ids.extend(encoding.ids)
                if eos_id is not None:
                    ids.append(eos_id)
        return torch.tensor(ids, dtype=torch.long)

    train_ids = encode_split("train")
    val_ids = encode_split("test" if "test" in dataset else "validation")

    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save({"train": train_ids, "val": val_ids}, cache_path)

    return tokenizer, train_ids, val_ids


def batches(ids, block_size=512, batch_size=128, device=None, shuffle=True, generator=None):
    """Yield (x, y) batches of shape (batch_size, block_size) for next-token prediction.

    The stream is cut into non-overlapping windows, shuffled once, then served in
    batches — one pass over the data per call, dropping any short final batch.
    """
    ids = ids.view(-1)
    n_blocks = (ids.numel() - 1) // block_size
    if n_blocks < batch_size:
        raise ValueError(
            f"need at least {batch_size * block_size + 1} tokens for "
            f"block_size={block_size}, batch_size={batch_size}, got {ids.numel()}"
        )

    starts = torch.arange(n_blocks) * block_size
    if shuffle:
        starts = starts[torch.randperm(n_blocks, generator=generator)]

    for i in range(0, n_blocks - batch_size + 1, batch_size):
        offsets = starts[i : i + batch_size].unsqueeze(1) + torch.arange(block_size)
        x = ids[offsets]
        y = ids[offsets + 1]
        if device is not None:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        yield x, y
