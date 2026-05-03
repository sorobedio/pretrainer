# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import os
from logging import Logger
from typing import Optional

import torch
import transformers
from torch import distributed as dist
from transformers import AutoConfig, default_data_collator

from utils.finemath_dataset import FinemathDataset
from utils.pretrain_trainer import FlopsCallback, PerplexityCallback, PretrainTrainer, VariableCheckpointCallback
from utils.process_args import process_args


def get_logger(logger_name: Optional[str]) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


log: Logger = get_logger("mobileLLM")


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    logging.warning(
        "LOCAL_RANK not in os.environ, falling back to torch.distributed"
    )
    return torch.distributed.get_rank()


def get_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(environ_rank)
    return 0


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    environ_ws = os.environ.get("WORLD_SIZE", "")
    if environ_ws.isdecimal():
        return int(environ_ws)
    return 1


def train() -> None:
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=8)
    )
    model_args, data_args, training_args = process_args()

    global_rank = get_global_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()

    log.info(f"Global rank {global_rank} / world size {world_size}")

    if model_args.init_from_pretrained:
        log.info(f"Loading pretrained weights from {model_args.input_model_filename} …")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.input_model_filename,
            torch_dtype=torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32),
        )
    else:
        log.info("Initialising model from scratch (random weights) …")
        config = AutoConfig.from_pretrained(model_args.input_model_filename)
        model = transformers.AutoModelForCausalLM.from_config(config=config)

    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {num_params / 1e6:.1f}M  ({num_params:,})")

    log.info("Loading tokenizer …")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    log.info("Tokenizer ready.")

    # ------------------------------------------------------------------ #
    # Derive save_steps and max_steps from token counts                   #
    # ------------------------------------------------------------------ #
    tokens_per_step = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.model_max_length
        * world_size
    )
    if data_args.variable_checkpoint_schedule:
        log.info(
            f"tokens/step={tokens_per_step:,}  |  "
            "variable checkpoint schedule: every 100M until 1B, 500M until 5B, 1B after"
        )
        training_args.save_strategy = "no"
    else:
        save_steps = max(1, data_args.tokens_per_checkpoint // tokens_per_step)
        log.info(
            f"tokens/step={tokens_per_step:,}  |  "
            f"checkpoint every {save_steps} steps  "
            f"({data_args.tokens_per_checkpoint / 1e9:.2f}B tokens)"
        )
        training_args.save_steps = save_steps
        training_args.save_strategy = "steps"

    if training_args.max_steps <= 0 and data_args.total_tokens > 0:
        training_args.max_steps = max(1, data_args.total_tokens // tokens_per_step)
        log.info(
            f"Derived max_steps={training_args.max_steps:,} from "
            f"total_tokens={data_args.total_tokens / 1e9:.2f}B"
        )

    # ------------------------------------------------------------------ #
    # Datasets                                                             #
    # ------------------------------------------------------------------ #
    train_data = FinemathDataset(
        tokenizer=tokenizer,
        seq_len=training_args.model_max_length,
        world_rank=global_rank,
        world_size=world_size,
        dataset_name=data_args.dataset_name,
        subset=data_args.dataset_subset,
        split="train",
        num_proc=data_args.num_proc,
        streaming=data_args.streaming,
        buffer_size=data_args.buffer_size,
        seed=training_args.seed,
    )

    eval_dataset_name = data_args.eval_dataset_name or data_args.dataset_name
    eval_dataset_subset = data_args.eval_dataset_subset if data_args.eval_dataset_name else data_args.dataset_subset

    test_data = FinemathDataset(
        tokenizer=tokenizer,
        seq_len=training_args.model_max_length,
        world_rank=0,
        world_size=1,
        dataset_name=eval_dataset_name,
        subset=eval_dataset_subset,
        split=data_args.eval_split,
        num_proc=data_args.num_proc,
        streaming=data_args.streaming,
        buffer_size=data_args.buffer_size,
        seed=training_args.seed,
        max_samples=data_args.eval_max_samples,
    )

    # ------------------------------------------------------------------ #
    # Trainer                                                              #
    # ------------------------------------------------------------------ #
    flops_cb = FlopsCallback(
        num_params=num_params,
        seq_len=training_args.model_max_length,
        world_size=world_size,
    )
    ppl_cb = PerplexityCallback(eval_tokens_interval=data_args.eval_tokens_interval)

    callbacks = [flops_cb, ppl_cb]
    if data_args.variable_checkpoint_schedule:
        callbacks.append(VariableCheckpointCallback())

    trainer = PretrainTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=test_data,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    torch.distributed.barrier(device_ids=[local_rank])

    if training_args.do_train:
        _ = trainer.train()
        trainer.save_state()

    torch.distributed.barrier(device_ids=[local_rank])


if __name__ == "__main__":
    train()
