# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Dict, Optional

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
        if logs is None or state.global_step == 0 or not state.is_world_process_zero:
            return
        tokens_per_step = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.seq_len
            * self.world_size
        )
        total_tokens = state.global_step * tokens_per_step
        total_flops = 6 * self.num_params * total_tokens
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "tokens_seen": total_tokens,
                        "total_flops": float(total_flops),
                        "total_flops_e21": total_flops / 1e21,
                    },
                    step=state.global_step,
                )
        except Exception:
            pass


def _tokens_per_step(args: TrainingArguments) -> int:
    ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * getattr(args, "model_max_length", 2048)
        * ws
    )


class PerplexityCallback(TrainerCallback):
    """Triggers test-set evaluation at a fixed token interval (or on every save as fallback)."""

    def __init__(self, eval_tokens_interval: int = 0):
        self.eval_tokens_interval = eval_tokens_interval
        self._last_eval_tokens = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        control.should_save = True
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        # Fallback: trigger on save when no token interval is configured
        if self.eval_tokens_interval <= 0:
            control.should_evaluate = True
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if self.eval_tokens_interval <= 0:
            return control
        total_tokens = state.global_step * _tokens_per_step(args)
        boundary = (total_tokens // self.eval_tokens_interval) * self.eval_tokens_interval
        if boundary > self._last_eval_tokens and boundary > 0:
            self._last_eval_tokens = boundary
            control.should_evaluate = True
        return control

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ) -> None:
        if not metrics or not state.is_world_process_zero:
            return
        loss = metrics.get("eval_loss")
        if loss is None:
            return
        ppl = math.exp(min(loss, 20.0))
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"eval/perplexity": ppl, "eval/loss": loss}, step=state.global_step)
        except Exception:
            pass


class VariableCheckpointCallback(TrainerCallback):
    """Saves checkpoints at token-based intervals that tighten early in training:
      - every 100M tokens until 1B
      - every 500M tokens from 1B to 5B
      - every 1B tokens after 5B
    """

    # (exclusive upper bound, interval)
    _SCHEDULE = [
        (1_000_000_000,   100_000_000),
        (5_000_000_000,   500_000_000),
        (float("inf"),  1_000_000_000),
    ]

    def __init__(self):
        self._last_save_tokens = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        control.should_save = True
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        total_tokens = state.global_step * _tokens_per_step(args)
        for upper, interval in self._SCHEDULE:
            if total_tokens < upper:
                break
        boundary = (total_tokens // interval) * interval
        if boundary > self._last_save_tokens and boundary > 0:
            self._last_save_tokens = boundary
            control.should_save = True
        return control


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
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

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
