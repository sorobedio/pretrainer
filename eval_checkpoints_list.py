"""
eval_checkpoints_list.py
========================
Evaluate a list of pretraining checkpoints on the SlimPajama-6B test split.
Computes perplexity and loss, logs to wandb aligned to the training token axis,
and saves a JSON summary to disk.

Usage:
  python eval_checkpoints_list.py \
      --checkpoint_dir /c2/soro/checkpoints-tuner/mobilellm-360m-slimpajama-scratch \
      --output_path    /c2/soro/checkpoints-tuner/mobilellm-360m-slimpajama-scratch/eval_results.json \
      --wandb_project  slimpajama-scratch \
      --wandb_run      mobilellm-360m-eval \
      --per_device_batch_size 8 \
      --max_eval_samples 500 \
      --seq_len 2048 \
      --tokens_per_step 32768
"""

import argparse
import json
import math
import os
import re

import torch
import transformers
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, default_data_collator


# ── Dataset ──────────────────────────────────────────────────────────────────

class SlimPajamaTestDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len: int, max_samples: int = 500, seed: int = 42):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.seed = seed

    def __iter__(self):
        ds = load_dataset(
            "DKYoon/SlimPajama-6B",
            split="test",
            streaming=True,
        ).shuffle(seed=self.seed, buffer_size=2000)

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


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    tokenizer,
    seq_len: int,
    max_eval_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    print(f"\n  Loading model from {checkpoint_path} ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    dataset = SlimPajamaTestDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_samples=max_eval_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        n_batches += 1

    del model
    torch.cuda.empty_cache()

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    return {"loss": avg_loss, "perplexity": perplexity, "n_batches": n_batches}


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def discover_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
    """Return (step, path) pairs sorted by step."""
    pattern = re.compile(r"^checkpoint-(\d+)$")
    entries = []
    for name in os.listdir(checkpoint_dir):
        m = pattern.match(name)
        if m:
            step = int(m.group(1))
            entries.append((step, os.path.join(checkpoint_dir, name)))
    return sorted(entries, key=lambda x: x[0])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing checkpoint-XXXXX folders")
    parser.add_argument("--output_path", required=True,
                        help="Path to save JSON results")
    parser.add_argument("--wandb_project", default="slimpajama-eval")
    parser.add_argument("--wandb_run", default=None,
                        help="W&B run name (default: auto-generated)")
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--max_eval_samples", type=int, default=500,
                        help="Number of packed sequences to eval on")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--tokens_per_step", type=int, default=32768,
                        help="Tokens consumed per optimizer step (bs × grad_acc × seq_len × gpus)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Discover checkpoints ──
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint-* folders found in {args.checkpoint_dir}")
    print(f"Found {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")

    # ── Load tokenizer from first checkpoint ──
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0][1])

    # ── Init wandb ──
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            "checkpoint_dir": args.checkpoint_dir,
            "max_eval_samples": args.max_eval_samples,
            "seq_len": args.seq_len,
            "tokens_per_step": args.tokens_per_step,
        },
    )

    # ── Evaluate ──
    all_results = {}
    for step, ckpt_path in checkpoints:
        total_tokens = step * args.tokens_per_step
        print(f"\n{'─' * 60}")
        print(f"  checkpoint-{step}  |  {total_tokens / 1e9:.3f}B tokens")

        metrics = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            max_eval_samples=args.max_eval_samples,
            batch_size=args.per_device_batch_size,
            device=device,
        )
        metrics["step"] = step
        metrics["tokens"] = total_tokens
        all_results[f"checkpoint-{step}"] = metrics

        print(f"  loss={metrics['loss']:.4f}  perplexity={metrics['perplexity']:.2f}")

        wandb.log(
            {
                "eval/loss":       metrics["loss"],
                "eval/perplexity": metrics["perplexity"],
                "tokens":          total_tokens,
            },
            step=step,
        )

    # ── Save to disk ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")

    # ── Final summary ──
    print(f"\n{'═' * 60}")
    print(f"  {'Checkpoint':<20}  {'Tokens':>12}  {'Loss':>8}  {'PPL':>8}")
    print(f"  {'─'*20}  {'─'*12}  {'─'*8}  {'─'*8}")
    for name, m in sorted(all_results.items(), key=lambda x: x[1]["step"]):
        print(f"  {name:<20}  {m['tokens']/1e9:>10.2f}B  {m['loss']:>8.4f}  {m['perplexity']:>8.2f}")

    wandb.finish()
    run.finish()


if __name__ == "__main__":
    main()
