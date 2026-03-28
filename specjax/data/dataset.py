"""
Data pipeline for EAGLE3 JAX training.

Pure-numpy dataset and bucketed batch sampler for EAGLE3 training in JAX.

Supported input formats:
  ShareGPT: {"conversations": [{"from": "human"/"gpt", "value": "..."}]}
  OpenAI:   {"messages": [{"role": "user"/"assistant", "content": "..."}]}

Sequences are truncated to max_length and padded to the nearest SEQ_BUCKET.
BucketBatchSampler ensures all samples in a batch share the same bucket length,
giving JAX static shapes and avoiding XLA recompilation per batch.
"""

import json
import logging
import random
from collections import defaultdict
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

SEQ_BUCKETS = [128, 256, 512, 1024, 2048]


def _pad_to_bucket(seq_len: int) -> int:
    for b in SEQ_BUCKETS:
        if seq_len <= b:
            return b
    return SEQ_BUCKETS[-1]


class Eagle3Dataset:
    """
    Load a JSONL file and pre-tokenize all samples.

    After construction:
      dataset[i] returns (input_ids, attention_mask) as numpy int32 arrays
      of shape [bucket_length].
      dataset.bucket_lengths[i] gives the padded length of sample i.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        logger.warning(f"Loading dataset from {jsonl_path} ...")
        raw_samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    msgs = self._parse_messages(obj)
                    if msgs:
                        raw_samples.append(msgs)
                except json.JSONDecodeError:
                    continue
        logger.warning(f"Dataset: {len(raw_samples)} samples loaded.")

        # Pre-tokenize once
        logger.warning(f"Pre-tokenising {len(raw_samples)} samples ...")
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []
        self.bucket_lengths: list[int] = []
        for msgs in raw_samples:
            ids, mask = self._tokenize(msgs)
            self._cache.append((ids, mask))
            self.bucket_lengths.append(ids.shape[0])
        logger.warning("Pre-tokenisation complete.")

    def _parse_messages(self, obj: dict) -> list[dict] | None:
        if "messages" in obj:
            return obj["messages"]
        if "conversations" in obj:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            msgs = []
            for turn in obj["conversations"]:
                role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
                content = turn.get("value", "")
                if content:
                    msgs.append({"role": role, "content": content})
            return msgs if msgs else None
        return None

    def _tokenize(self, messages: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in messages
            )

        enc = self.tokenizer(
            text,
            return_tensors=None,      # return plain lists, no torch tensors
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = np.array(enc["input_ids"], dtype=np.int32)

        T = input_ids.shape[0]
        bucket = _pad_to_bucket(T)

        if bucket > T:
            pad = np.full((bucket - T,), self.pad_token_id, dtype=np.int32)
            input_ids = np.concatenate([input_ids, pad])
            mask = np.concatenate([np.ones(T, dtype=np.int32), np.zeros(bucket - T, dtype=np.int32)])
        else:
            mask = np.ones(T, dtype=np.int32)

        return input_ids, mask   # both [bucket]

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self._cache[idx]


class BucketBatchSampler:
    """
    Groups dataset indices by bucket length and yields shuffled batches
    where every sample in a batch has the same bucket length.

    Required for JAX static-shape compilation: different bucket sizes trigger
    separate XLA programs, but within a bucket the shape is always constant.
    """

    def __init__(self, bucket_lengths: list[int], batch_size: int, drop_last: bool = True):
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, blen in enumerate(bucket_lengths):
            buckets[blen].append(idx)

        self._batches: list[list[int]] = []
        for blen, indices in buckets.items():
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start: start + batch_size]
                if drop_last and len(batch) < batch_size:
                    continue
                self._batches.append(batch)

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches[:]
        random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)


def collate(batch: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    """
    Stack a list of (input_ids, mask) pairs into a batch dict.
    All items must share the same sequence length (guaranteed by BucketBatchSampler).
    """
    input_ids = np.stack([b[0] for b in batch])   # [B, T]
    masks     = np.stack([b[1] for b in batch])   # [B, T]
    return {"input_ids": input_ids, "attention_mask": masks}


class DataLoader:
    """
    Minimal data loader that iterates over BucketBatchSampler and collates batches.
    Returns dicts of numpy arrays — no torch DataLoader dependency.
    """

    def __init__(self, dataset: Eagle3Dataset, batch_sampler: BucketBatchSampler):
        self.dataset = dataset
        self.batch_sampler = batch_sampler

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self):
        for indices in self.batch_sampler:
            batch = [self.dataset[i] for i in indices]
            yield collate(batch)
