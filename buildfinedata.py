"""
Download and combine:
  - HuggingFaceTB/smollm-corpus  fineweb-edu-dedup  (sample-10BT split)
  - HuggingFaceTB/finemath        finemath-4plus      (half of train split)
into a single JSON-lines file: finedata.json
"""

import json
from datasets import load_dataset

OUTPUT = "finedata.json"

# ── 1. FineWeb-edu-dedup  (10BT sample) ─────────────────────────────────
print("⏳ Loading FineWeb-edu-dedup (sample-10BT) ...")
fineweb = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "fineweb-edu-dedup",
    split="train",
    num_proc=16,
)
print(f"   ✓ FineWeb loaded: {len(fineweb):,} rows")

# ── 2. FineMath 4+ (full train, then take half) ─────────────────────────
print("⏳ Loading FineMath (finemath-4plus) ...")
finemath_full = load_dataset(
    "HuggingFaceTB/finemath",
    "finemath-4plus",
    split="train",
    num_proc=8,
)
half = len(finemath_full) // 2
finemath = finemath_full.select(range(half))
print(f"   ✓ FineMath loaded: {len(finemath_full):,} rows → using half: {half:,}")

# ── 3. Write combined JSONL ──────────────────────────────────────────────
print(f"⏳ Writing {OUTPUT} ...")
count = 0
with open(OUTPUT, "w", encoding="utf-8") as f:
    # FineWeb rows
    for row in fineweb:
        record = {
            "text": row["text"],
            "source": "fineweb-edu-dedup",
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
        if count % 1_000_000 == 0:
            print(f"   ... {count:,} rows written")

    # FineMath rows
    for row in finemath:
        record = {
            "text": row["text"],
            "source": "finemath-4plus",
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
        if count % 1_000_000 == 0:
            print(f"   ... {count:,} rows written")

print(f"✅ Done — {count:,} total rows written to {OUTPUT}")