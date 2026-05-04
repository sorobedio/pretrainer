"""
eval_checkpoints_list.py
========================
Evaluate all pretraining checkpoints on:
  - perplexity  : one or more HuggingFace text datasets (default: wikitext, lambada, text8)
  - benchmarks  : winogrande, mmlu, arc_challenge, arc_easy, hellaswag

Results are logged to wandb (aligned to the training token axis) and saved
to disk as JSON.

Usage:
  # All three perplexity datasets (default)
  python eval_checkpoints_list.py \
      --checkpoint_dir ./checkpoints/my-model \
      --output_path    ./checkpoints/my-model/eval_results.json \
      --wandb_project  my-project \
      --wandb_run      my-eval \
      --mode perplexity

  # Custom perplexity datasets  (format: name:subset:split or name:subset:split:text_key)
  python eval_checkpoints_list.py ... \
      --ppl_datasets "Salesforce/wikitext:wikitext-2-raw-v1:test" "cimec/lambada::test"

  # Benchmarks only
  python eval_checkpoints_list.py ... --mode benchmarks

  # Both perplexity and benchmarks
  python eval_checkpoints_list.py ... --mode all
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass

import torch
import transformers
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, default_data_collator

# ── Benchmark config ──────────────────────────────────────────────────────────

TASKS = {
    "winogrande":    5,
    "mmlu":          5,
    "arc_challenge": 25,
    "arc_easy":      25,
    "hellaswag":     10,
}

SCORE_KEY = {
    "winogrande":    "acc,none",
    "mmlu":          "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy":      "acc_norm,none",
    "hellaswag":     "acc_norm,none",
}

DEFAULT_PPL_DATASETS = [
    "Salesforce/wikitext:wikitext-2-raw-v1:test",
    "cimec/lambada::test",
    "afmck/text8::test",
]

# ── Perplexity datasets ───────────────────────────────────────────────────────

@dataclass
class PplDatasetSpec:
    name: str
    subset: str
    split: str
    text_key: str
    tag: str  # short name used for wandb keys


def parse_ppl_dataset(spec: str) -> PplDatasetSpec:
    """Parse 'name:subset:split[:text_key]' into a PplDatasetSpec."""
    parts = spec.split(":")
    name     = parts[0]
    subset   = parts[1] if len(parts) > 1 else ""
    split    = parts[2] if len(parts) > 2 else "test"
    text_key = parts[3] if len(parts) > 3 else "text"
    tag      = name.split("/")[-1]          # e.g. "wikitext", "lambada", "text8"
    return PplDatasetSpec(name=name, subset=subset, split=split,
                          text_key=text_key, tag=tag)


class TextDataset(IterableDataset):
    """Streams any HuggingFace text dataset and packs tokens into fixed-length sequences."""

    def __init__(self, tokenizer, seq_len: int, spec: PplDatasetSpec):
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.spec      = spec

    def __iter__(self):
        load_args = [self.spec.name]
        if self.spec.subset:
            load_args.append(self.spec.subset)
        ds = load_dataset(*load_args, split=self.spec.split, streaming=True)

        eos_id = self.tokenizer.eos_token_id
        buffer = []
        for example in ds:
            text = example.get(self.spec.text_key, "")
            if isinstance(text, str):
                text = text.strip()
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if eos_id is not None:
                ids.append(eos_id)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                chunk  = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": t, "labels": t.clone()}


@torch.no_grad()
def eval_perplexity_all(ckpt_path: str, tokenizer, seq_len: int,
                        batch_size: int, device: torch.device,
                        specs: list[PplDatasetSpec]) -> dict:
    """Load the model once and evaluate perplexity on every dataset in specs."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        ckpt_path, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    metrics = {}
    for spec in specs:
        print(f"  [{spec.tag}] evaluating …", flush=True)
        dataset = TextDataset(tokenizer=tokenizer, seq_len=seq_len, spec=spec)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             collate_fn=default_data_collator)

        total_loss, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total_loss += model(**batch).loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        ppl      = math.exp(min(avg_loss, 20.0))
        metrics[f"{spec.tag}_loss"]        = avg_loss
        metrics[f"{spec.tag}_perplexity"]  = ppl
        print(f"  [{spec.tag}] loss={avg_loss:.4f}  perplexity={ppl:.2f}", flush=True)

    del model
    torch.cuda.empty_cache()
    return metrics


# ── Benchmarks ────────────────────────────────────────────────────────────────

def eval_benchmarks(ckpt_path: str, batch_size: int,
                    device: str, limit: int | None) -> dict:
    import lm_eval
    all_results = {}
    for task, num_fewshot in TASKS.items():
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={ckpt_path},dtype=bfloat16",
            tasks=[task],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
            log_samples=False,
        )
        all_results[task] = results["results"][task]
    return all_results


# ── Utilities ─────────────────────────────────────────────────────────────────

def discover_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
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
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_path",    required=True)
    parser.add_argument("--wandb_project",  default="slimpajama-eval")
    parser.add_argument("--wandb_run",      default=None)
    parser.add_argument("--mode", choices=["perplexity", "benchmarks", "all"],
                        default="perplexity",
                        help="perplexity | benchmarks | all")
    parser.add_argument("--ppl_datasets", nargs="+", default=DEFAULT_PPL_DATASETS,
                        help="Perplexity datasets as 'name:subset:split[:text_key]'. "
                             "Subset and text_key may be omitted (default text_key='text').")
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--seq_len",        type=int,   default=2048)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--tokens_per_step", type=int,  default=32768,
                        help="Tokens per optimizer step (bs × grad_acc × seq_len × gpus)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per benchmark task (debug)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ppl_specs = [parse_ppl_dataset(s) for s in args.ppl_datasets]
    print("Perplexity datasets:", [s.tag for s in ppl_specs])

    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint-* folders found in {args.checkpoint_dir}")
    print(f"Found {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0][1])

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            "checkpoint_dir":  args.checkpoint_dir,
            "mode":            args.mode,
            "tokens_per_step": args.tokens_per_step,
            "ppl_datasets":    args.ppl_datasets,
        },
    )

    all_results = {}

    for step, ckpt_path in checkpoints:
        total_tokens = step * args.tokens_per_step
        print(f"\n{'═' * 70}")
        print(f"checkpoint-{step}  |  {total_tokens / 1e9:.3f}B tokens")
        print(f"{'═' * 70}")

        result = {"step": step, "tokens": total_tokens}
        log    = {"tokens": total_tokens}

        if args.mode in ("perplexity", "all"):
            ppl_metrics = eval_perplexity_all(
                ckpt_path=ckpt_path,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                device=device,
                specs=ppl_specs,
            )
            result.update(ppl_metrics)
            for key, val in ppl_metrics.items():
                log[f"eval/{key}"] = val

        if args.mode in ("benchmarks", "all"):
            task_results = eval_benchmarks(
                ckpt_path=ckpt_path,
                batch_size=args.batch_size,
                device=str(device),
                limit=args.limit,
            )
            scores = {t: task_results[t].get(SCORE_KEY[t], None) for t in TASKS}
            result["benchmark_scores"] = scores
            result["benchmark_full"]   = task_results
            for t, s in scores.items():
                if isinstance(s, float):
                    log[f"eval/{t}"] = s
            print(f"  {'Task':<20}  {'Score':>8}")
            for t, s in scores.items():
                print(f"  {t:<20}  {s:.4f}" if isinstance(s, float) else f"  {t:<20}  N/A")

        all_results[f"checkpoint-{step}"] = result
        wandb.log(log, step=step)

    # ── Save ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")

    # ── Summary ──
    print(f"\n{'═' * 60}")
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["step"]):
        parts = [f"{r['tokens']/1e9:.2f}B tokens"]
        for spec in ppl_specs:
            ppl = r.get(f"{spec.tag}_perplexity")
            if ppl:
                parts.append(f"{spec.tag}_ppl={ppl:.2f}")
        print(f"  {name:<25}  {'  '.join(parts)}")

    wandb.finish()


if __name__ == "__main__":
    main()
