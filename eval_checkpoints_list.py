"""
eval_checkpoints_list.py
========================
Evaluate all pretraining checkpoints on:
  - perplexity  : Salesforce/wikitext (wikitext-2-raw-v1) test set
  - benchmarks  : winogrande, mmlu, arc_challenge, arc_easy, hellaswag

Results are logged to wandb (aligned to the training token axis) and saved
to disk as JSON.

Usage:
  # Wikitext perplexity only (fast)
  python eval_checkpoints_list.py \
      --checkpoint_dir /c2/soro/checkpoints-tuner/mobilellm-360m-slimpajama-scratch \
      --output_path    /c2/soro/checkpoints-tuner/mobilellm-360m-slimpajama-scratch/eval_results.json \
      --wandb_project  slimpajama-scratch \
      --wandb_run      mobilellm-360m-eval \
      --mode perplexity

  # Benchmarks only
  python eval_checkpoints_list.py ... --mode benchmarks

  # Both
  python eval_checkpoints_list.py ... --mode all
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


# ── Wikitext perplexity ───────────────────────────────────────────────────────

class WikitextDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="test",
            streaming=True,
        )
        eos_id = self.tokenizer.eos_token_id
        buffer = []
        for example in ds:
            text = example.get("text", "").strip()
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


@torch.no_grad()
def eval_perplexity(ckpt_path: str, tokenizer, seq_len: int,
                    batch_size: int, device: torch.device) -> dict:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        ckpt_path, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    dataset = WikitextDataset(tokenizer=tokenizer, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=default_data_collator)

    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total_loss += model(**batch).loss.item()
        n += 1

    del model
    torch.cuda.empty_cache()

    avg_loss = total_loss / max(n, 1)
    return {"wikitext_loss": avg_loss, "wikitext_perplexity": math.exp(min(avg_loss, 20.0))}


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
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--wandb_project", default="slimpajama-eval")
    parser.add_argument("--wandb_run", default=None)
    parser.add_argument("--mode", choices=["perplexity", "benchmarks", "all"],
                        default="perplexity",
                        help="perplexity=wikitext only | benchmarks=lm-harness | all=both")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens_per_step", type=int, default=32768,
                        help="Tokens per optimizer step (bs × grad_acc × seq_len × gpus)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per benchmark task (debug)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint-* folders found in {args.checkpoint_dir}")
    print(f"Found {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0][1])

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            "checkpoint_dir": args.checkpoint_dir,
            "mode": args.mode,
            "tokens_per_step": args.tokens_per_step,
        },
    )

    all_results = {}

    for step, ckpt_path in checkpoints:
        total_tokens = step * args.tokens_per_step
        print(f"\n{'═' * 70}")
        print(f"checkpoint-{step}  |  {total_tokens / 1e9:.3f}B tokens")
        print(f"{'═' * 70}")

        result = {"step": step, "tokens": total_tokens}
        log = {"tokens": total_tokens}

        if args.mode in ("perplexity", "all"):
            ppl_metrics = eval_perplexity(
                ckpt_path=ckpt_path,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                device=device,
            )
            result.update(ppl_metrics)
            log["eval/wikitext_loss"] = ppl_metrics["wikitext_loss"]
            log["eval/wikitext_perplexity"] = ppl_metrics["wikitext_perplexity"]
            print(f"  wikitext loss={ppl_metrics['wikitext_loss']:.4f}  "
                  f"perplexity={ppl_metrics['wikitext_perplexity']:.2f}")

        if args.mode in ("benchmarks", "all"):
            task_results = eval_benchmarks(
                ckpt_path=ckpt_path,
                batch_size=args.batch_size,
                device=str(device),
                limit=args.limit,
            )
            scores = {
                t: task_results[t].get(SCORE_KEY[t], None) for t in TASKS
            }
            result["benchmark_scores"] = scores
            result["benchmark_full"] = task_results
            for t, s in scores.items():
                if isinstance(s, float):
                    log[f"eval/{t}"] = s
            print(f"  {'Task':<20}  {'Score':>8}")
            for t, s in scores.items():
                print(f"  {t:<20}  {s:.4f}" if isinstance(s, float) else f"  {t:<20}  N/A")

        all_results[f"checkpoint-{step}"] = result
        wandb.log(log, step=step)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")

    # Summary
    print(f"\n{'═' * 60}")
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["step"]):
        ppl = r.get("wikitext_perplexity")
        ppl_str = f"ppl={ppl:.2f}" if ppl else ""
        print(f"  {name:<25}  {r['tokens']/1e9:.2f}B tokens  {ppl_str}")

    wandb.finish()


if __name__ == "__main__":
    main()
