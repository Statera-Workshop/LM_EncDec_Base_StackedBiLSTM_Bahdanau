"""
Unified settings for the Language Model stage ([Unified] protocol).

Every model is trained and evaluated with the values below. Anything a model
needs that differs from its upstream repository is written in its registry entry
(models/registry.py) as an "[Unified]" note and copied into the run results.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("LM_DATA", ROOT / "data"))   # set LM_DATA=D:\LM_data to move data
RUNS_DIR = Path(os.environ.get("LM_RUNS", ROOT / "runs"))   # set LM_RUNS=D:\LM_runs to move results

VOCAB = 256          # byte level for every task, so every model sees the same tokens
MAX_LEN = 2048       # longest sequence any model is built for (LM-05 tests go up to here)

MODEL = dict(        # nominal size; hierarchical models split it over stages (see registry)
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
)

OPTIM = dict(        # [Unified] one recipe for all models (lucidrains train.py majority: Adam 2e-4, clip 0.5)
    lr=2e-4,
    warmup_steps=200,  # [Unified] short linear warmup added for stability across very different models
    betas=(0.9, 0.99),
    weight_decay=0.0,
    clip=0.5,
)

# LM-02 Next Token Prediction on enwik8 (bytes)
LM02 = dict(
    seq_len=512,
    batch=16,                 # one optimizer step = 16 x 512 = 8,192 tokens
    tokens=200_000_000,       # token budget per model (~24,400 steps)
    max_hours=4.0,            # wall-clock cap; a capped run is marked "capped", not hidden
    eval_every=1000,          # steps
    eval_batches=100,         # 100 x 16 x 512 = 819,200 validation bytes per eval
    final_eval_batches=600,   # ~4.9 M bytes, essentially the whole validation split
)

ENWIK8 = dict(
    file="enwik8.gz",
    total_bytes=100_000_000,
    train_bytes=90_000_000,   # standard split: first 90M train, next 5M valid, last 5M test
    valid_bytes=5_000_000,
    test_bytes=5_000_000,
    url="http://mattmahoney.net/dc/enwik8.zip",
)

# LM-05 Long-Horizon Recall (synthetic): key/value pairs, a long gap of distractors, then queries
LM05 = dict(
    pairs=16,
    train_len=512,
    test_lens=(512, 1024, 2048),
    batch=32,
    steps=6000,
    eval_every=500,
    eval_samples=512,
    max_hours=2.0,            # wall-clock cap for slow sequential models; marked "capped"
)

OVERFIT = dict(       # cheap implementation check before any long run
    samples=4,
    seq_len=256,
    steps=400,
    pass_loss=0.5,      # nats/token on the memorised batch; the run continues only below this
)

PROBE = dict(          # memory / speed growth with sequence length (untrained weights, batch 1)
    lens=(512, 1024, 2048, 4096, 8192),
)

SEED = 1234

# LM-05 v2 (2026-10-21): queries with replacement (no elimination shortcut), two difficulty levels
LM05V2 = dict(
    pairs=(4, 16),
    queries=16,
    train_len=512,
    test_lens=(512, 2048),
    batch=32,
    steps=6000,
    eval_every=500,
    eval_samples=512,
    max_hours=1.0,            # per difficulty level
)

# Round 3 (2026-10-22): long runs for the 4-day block
LM02_LONG = dict(tokens=1_000_000_000, max_hours=12.0, eval_every=2000)          # 5x the round-1 budget
LM05V2_LONG = dict(steps=30_000, max_hours=4.0, eval_every=2500,
                   test_lens=(512, 2048, 8192))                                    # 5x steps, test up to 8192
SEEDS = {"lm05v2s2": 2, "lm05v2s3": 3}                                             # repeat LM-05 v2 with new seeds

# LM-05 v3 (2026-10-26): v2 let models copy answers of repeated queries; v3 asks DIFFERENT keys, few per sample.
# Levels (pairs, queries): easy (4, 1) value-set baseline 0.25; hard (16, 4) baseline about 0.07.
LM05V3 = dict(LM05V2, levels=((4, 1), (16, 4)), distinct=True)
LM05V3_LONG = dict(steps=30_000, max_hours=4.0, eval_every=2500, test_lens=(512, 2048, 8192))

# LM-06 memory update (2026-10-29): 8 pairs, 4 keys get a new value inside the gap, 4 distinct queries
# (2 on updated keys, 2 on unchanged keys). Correct answer = latest value; "stale" = answering the old value.
LM06 = dict(pairs=8, updates=4, queries=4, train_len=512, test_lens=(512, 2048), batch=32, steps=6000,
            eval_every=500, eval_samples=512, max_hours=1.5)
LM06_LONG = dict(steps=30_000, max_hours=5.0, eval_every=2500, test_lens=(512, 2048, 8192))
SEEDS.update({"lm05v3s2": 2, "lm05v3s3": 3, "lm05v3longs2": 2})
SEEDS.update({"lm06s2": 2, "lm06s3": 3, "lm06longs2": 2})
SEEDS.update({"lm05v3s4": 4, "lm05v3s5": 5, "lm05v3s6": 6, "lm06longs3": 3})
SEEDS.update({"lm05v3longs3": 3})
SEEDS.update({"lm07s2": 2, "lm07s3": 3})
SEEDS.update({"lm07longs2": 2})
