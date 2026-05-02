# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    local_dir: str = field(
        default=None, metadata={"help": "Local path for inputs and outputs"}
    )
    input_model_filename: Optional[str] = field(
        default="facebook/MobileLLM-R1-950M",
        metadata={"help": "HuggingFace model ID or local path (used for config + tokenizer)"},
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative path"}
    )
    init_from_pretrained: bool = field(
        default=False,
        metadata={"help": "Load pretrained weights via from_pretrained instead of random init from config"},
    )


@dataclass
class DataArguments:
    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Legacy: local JSONL data root path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Legacy: local eval data path"}
    )
    dataset_name: str = field(
        default="HuggingFaceTB/finemath",
        metadata={"help": "HuggingFace dataset name"},
    )
    dataset_subset: str = field(
        default="finemath-4plus",
        metadata={"help": "Dataset configuration / subset name"},
    )
    num_proc: int = field(
        default=8,
        metadata={"help": "Workers for non-streaming dataset download/cache"},
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Stream the dataset instead of downloading it fully"},
    )
    buffer_size: int = field(
        default=10_000,
        metadata={"help": "Shuffle buffer size (streaming mode only)"},
    )
    tokens_per_checkpoint: int = field(
        default=1_000_000_000,
        metadata={"help": "Save a checkpoint every this many tokens (default 1B)"},
    )
    total_tokens: int = field(
        default=0,
        metadata={"help": "Total dataset tokens; used to derive max_steps and cosine schedule when max_steps is not set"},
    )
    eval_max_samples: int = field(
        default=500,
        metadata={"help": "Number of packed sequences to evaluate perplexity on (0 = full test split)"},
    )
    eval_split: str = field(
        default="test",
        metadata={"help": "Dataset split to use for perplexity evaluation (e.g. 'test', 'validation')"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "Sequence length / context window"},
    )


def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args
