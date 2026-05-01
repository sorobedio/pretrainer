import logging
from typing import Iterator, Dict

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class FinemathDataset(IterableDataset):
    """Streams finemath text and packs tokens into fixed-length sequences for pretraining."""

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        world_rank: int = 0,
        world_size: int = 1,
        dataset_name: str = "HuggingFaceTB/finemath",
        subset: str = "finemath-4plus",
        split: str = "train",
        num_proc: int = 8,
        streaming: bool = True,
        buffer_size: int = 10_000,
        seed: int = 42,
        max_samples: int = 0,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.world_rank = world_rank
        self.world_size = world_size
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.num_proc = num_proc
        self.streaming = streaming
        self.buffer_size = buffer_size
        self.seed = seed
        self.max_samples = max_samples

    def _get_dataset(self):
        load_kwargs = dict(split=self.split)
        args = [self.dataset_name]
        if self.subset:
            args.append(self.subset)

        if self.streaming:
            ds = load_dataset(*args, streaming=True, **load_kwargs)
            if self.world_size > 1:
                ds = ds.shard(num_shards=self.world_size, index=self.world_rank)
            return ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)
        else:
            ds = load_dataset(*args, num_proc=self.num_proc, **load_kwargs)
            if self.world_size > 1:
                ds = ds.shard(num_shards=self.world_size, index=self.world_rank)
            return ds.shuffle(seed=self.seed)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ds = self._get_dataset()
        eos_id = self.tokenizer.eos_token_id
        buffer = []
        n_yielded = 0

        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if eos_id is not None:
                ids.append(eos_id)
            buffer.extend(ids)

            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": t, "labels": t.clone()}
                n_yielded += 1
                if self.max_samples > 0 and n_yielded >= self.max_samples:
                    return
