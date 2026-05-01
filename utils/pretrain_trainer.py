# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_ratio: float = 0.1,
    theta: float = 1,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step <= num_training_steps:
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        lr = min_ratio + 0.5 * (1 - min_ratio) * (
            math.cos(math.pi * progress**theta / num_cycles) + 1
        )
    else:
        lr = min_ratio
    return lr


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class FlopsCallback(TrainerCallback):
    """Logs total tokens processed and estimated compute (FLOPs) to wandb / tensorboard.

    Uses the Chinchilla approximation: FLOPs ≈ 6 × N × D
    where N = total model parameters and D = total tokens seen.
    """

    def __init__(self, num_params: int, seq_len: int, world_size: int):
        self.num_params = num_params
        self.seq_len = seq_len
        self.world_size = world_size

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if logs is None or state.global_step == 0:
            return
        tokens_per_step = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.seq_len
            * self.world_size
        )
        total_tokens = state.global_step * tokens_per_step
        total_flops = 6 * self.num_params * total_tokens
        logs["tokens_seen"] = total_tokens
        logs["total_flops"] = float(total_flops)
        logs["total_flops_e21"] = total_flops / 1e21  # ZettaFLOPs, convenient scale


class PretrainMixin:
    def __init__(
        self,
        manifold_ckpt_dir: Optional[str] = None,
        max_parallel_files: int = 5,
        resume: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.manifold_ckpt_dir = manifold_ckpt_dir
        self.max_parallel_files = max_parallel_files
        self.resume = resume

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> LambdaLR:
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler


class PretrainTrainer(PretrainMixin, Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if isinstance(self.train_dataset, IterableDataset):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=True,
            )

        # Legacy path: dataset yields pre-batched tensors (e.g. MultiJSONLIterator)
        return self.train_dataset
