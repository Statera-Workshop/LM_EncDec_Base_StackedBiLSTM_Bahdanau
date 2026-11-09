"""
Data for the Language Model stage.

LM-02  enwik8 (raw bytes of English Wikipedia XML), standard 90M / 5M / 5M split.
LM-05  Long-Horizon Recall: synthetic key/value recall across a long distractor gap.

Lesson carried over from the detection stage: count what the loader really reads
before training and compare it with the expected numbers. `check_enwik8()` and
`check_recall()` do that and raise if anything differs.
"""
import gzip
import hashlib
import zipfile
import urllib.request

import numpy as np
import torch

from common.config import DATA_DIR, ENWIK8, LM05, VOCAB

ENWIK8_SHA256 = "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"


# --------------------------------------------------------------------------- enwik8

def _raw_enwik8():
    gz = DATA_DIR / ENWIK8["file"]
    zp = DATA_DIR / "enwik8.zip"
    if gz.exists():
        with gzip.open(gz) as f:
            return f.read()
    if not zp.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {ENWIK8['url']} ...", flush=True)
        urllib.request.urlretrieve(ENWIK8["url"], zp)
    with zipfile.ZipFile(zp) as z:
        return z.read("enwik8")


def load_enwik8():
    """Returns dict(train, valid, test) as uint8 torch tensors. Cached as .npy."""
    npy = DATA_DIR / "enwik8.npy"
    if npy.exists():
        data = np.load(npy, mmap_mode="r")
    else:
        raw = _raw_enwik8()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != ENWIK8_SHA256:
            raise RuntimeError(f"enwik8 checksum mismatch: {digest[:16]}... (expected {ENWIK8_SHA256[:16]}...)")
        data = np.frombuffer(raw, dtype=np.uint8)
        np.save(npy, data)
        data = np.load(npy, mmap_mode="r")
    a = ENWIK8["train_bytes"]
    b = a + ENWIK8["valid_bytes"]
    c = b + ENWIK8["test_bytes"]
    return dict(
        train=torch.from_numpy(np.array(data[:a])),
        valid=torch.from_numpy(np.array(data[a:b])),
        test=torch.from_numpy(np.array(data[b:c])),
    )


def check_enwik8(splits):
    """Count bytes per split and compare with the expected split sizes."""
    expected = dict(train=ENWIK8["train_bytes"], valid=ENWIK8["valid_bytes"], test=ENWIK8["test_bytes"])
    lines = []
    for k, n in expected.items():
        got = splits[k].numel()
        status = "OK" if got == n else "MISMATCH"
        lines.append(f"enwik8 {k:5s} bytes {got:>11,d} expected {n:>11,d} {status}")
        if got != n:
            raise RuntimeError("\n".join(lines))
    distinct = int(torch.unique(splits["train"][:10_000_000]).numel())
    lines.append(f"enwik8 distinct byte values in first 10M train bytes: {distinct} (vocab {VOCAB})")
    return lines


class RandomWindows:
    """Random contiguous windows of length seq_len+1 (input + next-token target)."""

    def __init__(self, data, seq_len, batch, generator):
        self.data, self.seq_len, self.batch, self.g = data, seq_len, batch, generator

    def __iter__(self):
        return self

    def __next__(self):
        hi = self.data.numel() - self.seq_len - 1
        idx = torch.randint(0, hi, (self.batch,), generator=self.g)
        return torch.stack([self.data[i:i + self.seq_len + 1] for i in idx.tolist()]).long()


def eval_windows(data, seq_len, batch, n_batches):
    """Deterministic, non-overlapping validation windows from the start of the split."""
    step = seq_len + 1
    need = n_batches * batch * step
    n_batches = min(n_batches, data.numel() // (batch * step))
    flat = data[: n_batches * batch * step].long().view(n_batches, batch, step)
    return [flat[i] for i in range(n_batches)], n_batches * batch * seq_len, need


# --------------------------------------------------------------------------- LM-05 recall

# token layout inside the shared 256-byte vocabulary
KEY_LO, KEY_HI = 2, 66          # 64 key symbols
VAL_LO, VAL_HI = 66, 130        # 64 value symbols
DIS_LO, DIS_HI = 130, 250       # 120 distractor symbols (never keys or values)


def recall_batch(batch, length, pairs, generator, max_gap=True):
    """
    Layout: k1 v1 ... kP vP | gap of distractors | q1 a1 ... qP aP | distractor tail
    The model must output a_i at the position of q_i. Loss and accuracy use only those P positions.
    max_gap=True puts the queries at the very end (largest recall distance) - used for tests.
    max_gap=False draws the gap uniformly - used for training.
    Returns x (B, L) and target positions mask over predictions (B, L-1).
    """
    P = pairs
    free = length - 4 * P
    assert free >= 0, "sequence too short for the number of pairs"
    keys = torch.stack([torch.randperm(KEY_HI - KEY_LO, generator=generator)[:P] for _ in range(batch)]) + KEY_LO
    vals = torch.randint(VAL_LO, VAL_HI, (batch, P), generator=generator)
    x = torch.randint(DIS_LO, DIS_HI, (batch, length), generator=generator)
    x[:, 0:2 * P:2] = keys
    x[:, 1:2 * P:2] = vals
    if max_gap:
        gaps = torch.full((batch,), free)
    else:
        gaps = torch.randint(0, free + 1, (batch,), generator=generator)
    order = torch.stack([torch.randperm(P, generator=generator) for _ in range(batch)])
    mask = torch.zeros(batch, length - 1, dtype=torch.bool)
    for b in range(batch):
        start = 2 * P + int(gaps[b])
        q = keys[b, order[b]]
        a = vals[b, order[b]]
        x[b, start:start + 2 * P:2] = q
        x[b, start + 1:start + 2 * P:2] = a
        mask[b, start:start + 2 * P:2] = True   # prediction at the query position targets the answer
    return x.long(), mask


def check_recall(length=LM05["train_len"], pairs=LM05["pairs"]):
    """Count answer positions and verify every target is the value paired with its query key."""
    g = torch.Generator().manual_seed(0)
    lines = []
    for max_gap in (False, True):
        x, mask = recall_batch(8, length, pairs, g, max_gap=max_gap)
        per_sample = mask.sum(1)
        if not bool((per_sample == pairs).all()):
            raise RuntimeError(f"recall: answer positions per sample {per_sample.tolist()} != {pairs}")
        for b in range(8):
            table = {int(k): int(v) for k, v in zip(x[b, 0:2 * pairs:2], x[b, 1:2 * pairs:2])}
            pos = mask[b].nonzero().flatten()
            for p in pos.tolist():
                if table[int(x[b, p])] != int(x[b, p + 1]):
                    raise RuntimeError("recall: an answer does not match its key")
            # distractors never collide with keys or values
            outside = torch.ones(length, dtype=torch.bool)
            outside[0:2 * pairs] = False
            for p in pos.tolist():
                outside[p] = outside[p + 1] = False
            d = x[b][outside]
            if d.numel() and (d.min() < DIS_LO or d.max() >= DIS_HI):
                raise RuntimeError("recall: distractor token outside its range")
        gap_kind = "max gap" if max_gap else "random gap"
        lines.append(f"recall L={length} {gap_kind}: {pairs} answers/sample, keys/values/distractors disjoint OK")
    return lines


def recall_batch_v2(batch, length, pairs, queries, generator, max_gap=True, distinct=False):
    """LM-05 v2: like recall_batch, but the queries are drawn WITH replacement from the pairs, so answering by
    elimination (picking a value not yet used) gives no advantage. A model that only knows which values are in
    the context scores about 1/pairs; the key-removed control measures that baseline directly."""
    P, Q = pairs, queries
    free = length - 2 * P - 2 * Q
    assert free >= 0, "sequence too short"
    keys = torch.stack([torch.randperm(KEY_HI - KEY_LO, generator=generator)[:P] for _ in range(batch)]) + KEY_LO
    vals = torch.stack([torch.randperm(VAL_HI - VAL_LO, generator=generator)[:P] for _ in range(batch)]) + VAL_LO
    x = torch.randint(DIS_LO, DIS_HI, (batch, length), generator=generator)
    x[:, 0:2 * P:2] = keys
    x[:, 1:2 * P:2] = vals
    gaps = torch.full((batch,), free) if max_gap else torch.randint(0, free + 1, (batch,), generator=generator)
    if distinct:   # LM-05 v3: queries ask different keys and Q << P, so neither copying nor elimination helps much
        pick = torch.stack([torch.randperm(P, generator=generator)[:Q] for _ in range(batch)])
    else:
        pick = torch.randint(0, P, (batch, Q), generator=generator)
    mask = torch.zeros(batch, length - 1, dtype=torch.bool)
    for b in range(batch):
        start = 2 * P + int(gaps[b])
        x[b, start:start + 2 * Q:2] = keys[b, pick[b]]
        x[b, start + 1:start + 2 * Q:2] = vals[b, pick[b]]
        mask[b, start:start + 2 * Q:2] = True
    return x.long(), mask


def check_recall_v2(length=512, pairs=(4, 16), queries=16):
    g = torch.Generator().manual_seed(1)
    lines = []
    for P in pairs:
        x, mask = recall_batch_v2(8, length, P, queries, g, max_gap=False)
        assert bool((mask.sum(1) == queries).all())
        for b in range(8):
            table = {int(k): int(v) for k, v in zip(x[b, 0:2 * P:2], x[b, 1:2 * P:2])}
            assert len(table) == P and len(set(table.values())) == P
            for pos in mask[b].nonzero().flatten().tolist():
                assert table[int(x[b, pos])] == int(x[b, pos + 1])
        lines.append(f"recall v2 L={length} pairs={P}: {queries} answers/sample drawn with replacement, keys and values distinct OK")
    return lines


def check_recall_v3(length=512, levels=((4, 1), (16, 4))):
    g = torch.Generator().manual_seed(2)
    lines = []
    for P, Q in levels:
        x, mask = recall_batch_v2(8, length, P, Q, g, max_gap=False, distinct=True)
        assert bool((mask.sum(1) == Q).all())
        for b in range(8):
            table = {int(k): int(v) for k, v in zip(x[b, 0:2 * P:2], x[b, 1:2 * P:2])}
            qs = [int(x[b, pos]) for pos in mask[b].nonzero().flatten().tolist()]
            assert len(set(qs)) == Q, "v3 queries must ask different keys"
            for pos in mask[b].nonzero().flatten().tolist():
                assert table[int(x[b, pos])] == int(x[b, pos + 1])
        lines.append(f"recall v3 L={length} pairs={P} queries={Q}: distinct keys, answers match OK")
    return lines


def update_batch(batch, length, pairs, updates, queries, generator, max_gap=True, return_src=False):
    """LM-06 memory update: k1 v1 ... kP vP | gap with U updates "k v_new" placed at random positions |
    Q queries (distinct keys, half of them updated keys). The answer is always the LATEST value.
    Returns x, mask (answer positions), upd (answer positions whose key was updated), old (stale value there)."""
    P, U, Q = pairs, updates, queries
    assert U <= P and Q <= P
    head, tail = 2 * P, 2 * Q
    free = length - head - tail
    assert free >= 2 * U + 2
    keys = torch.stack([torch.randperm(KEY_HI - KEY_LO, generator=generator)[:P] for _ in range(batch)]) + KEY_LO
    vals = torch.stack([torch.randperm(VAL_HI - VAL_LO, generator=generator)[:P] for _ in range(batch)]) + VAL_LO
    x = torch.randint(DIS_LO, DIS_HI, (batch, length), generator=generator)
    x[:, 0:head:2] = keys
    x[:, 1:head:2] = vals
    mask = torch.zeros(batch, length - 1, dtype=torch.bool)
    upd = torch.zeros(batch, length - 1, dtype=torch.bool)
    old = torch.zeros(batch, length - 1, dtype=torch.long)
    src = torch.zeros(batch, length - 1, dtype=torch.long)   # position of the value token the answer must copy
    for b in range(batch):
        src_of = {i: 2 * i + 1 for i in range(P)}
        gap = free if max_gap else int(torch.randint(2 * U + 2, free + 1, (1,), generator=generator))
        q_start = head + gap
        which = torch.randperm(P, generator=generator)[:U]                      # keys that get a new value
        slots = torch.randperm((gap - 2) // 2, generator=generator)[:U] * 2 + head  # even offsets inside the gap
        latest = vals[b].clone()
        for i, pos in zip(which.tolist(), sorted(slots.tolist())):
            new_v = int(torch.randint(VAL_LO, VAL_HI, (1,), generator=generator))
            while new_v == int(latest[i]):
                new_v = int(torch.randint(VAL_LO, VAL_HI, (1,), generator=generator))
            x[b, pos] = keys[b, i]
            x[b, pos + 1] = new_v
            latest[i] = new_v
            src_of[i] = pos + 1
        n_upd = min(Q // 2, U)
        not_upd = [i for i in range(P) if i not in set(which.tolist())]
        pick = which[:n_upd].tolist() + [not_upd[j] for j in torch.randperm(len(not_upd), generator=generator)[:Q - n_upd].tolist()]
        order = torch.randperm(Q, generator=generator).tolist()
        for j, o in enumerate(order):
            i = pick[o]
            pos = q_start + 2 * j
            x[b, pos] = keys[b, i]
            x[b, pos + 1] = latest[i]
            mask[b, pos] = True
            src[b, pos] = src_of[i]
            if i in set(which.tolist()):
                upd[b, pos] = True
                old[b, pos] = vals[b, i]
    if return_src:
        return x.long(), mask, upd, old, src
    return x.long(), mask, upd, old


def check_update(length=512, pairs=8, updates=4, queries=4):
    g = torch.Generator().manual_seed(3)
    x, mask, upd, old = update_batch(8, length, pairs, updates, queries, g, max_gap=False)
    for b in range(8):
        table = {}
        for pos in range(0, length - 1, 1):
            pass
        # replay: scan left to right, a key token followed by a value token sets the table
        latest = {}
        qpos = set(mask[b].nonzero().flatten().tolist())
        for pos in range(length - 1):
            if pos in qpos:
                assert latest[int(x[b, pos])] == int(x[b, pos + 1]), "answer is not the latest value"
                continue
            k, v = int(x[b, pos]), int(x[b, pos + 1])
            if KEY_LO <= k < KEY_HI and VAL_LO <= v < VAL_HI and (pos - 1) not in qpos:
                latest[k] = v
        assert int(mask[b].sum()) == queries and int(upd[b].sum()) == min(queries // 2, updates)
        for pos in upd[b].nonzero().flatten().tolist():
            assert int(old[b, pos]) != int(x[b, pos + 1]), "stale value equals latest"
    return [f"update task L={length} pairs={pairs} updates={updates} queries={queries}: answers are latest values, "
            f"{min(queries // 2, updates)} updated + {queries - min(queries // 2, updates)} unchanged per sample OK"]


# --------------------------------------------------------------------------- LM-07 state tracking (PASG)
# K objects on a ring of 16 positions. Stream of events: "SET o p" (absolute observation) and "MOVE o d"
# (relative motion, d in -3..+3 without 0), mixed with sensor-noise tokens. At the end "QUERY o" -> current position.
# Unlike LM-05/06 the answer is not copied from the context: it must be integrated from all moves after the
# last observation (state tracking, as in odometry between GPS fixes).
ST_OBJ, ST_POS, ST_MOVE = 2, 20, 40          # object ids 2..17, positions 20..35, moves 40..45
ST_SET, ST_MV, ST_Q = 50, 51, 52             # event markers
ST_DELTAS = (-3, -2, -1, 1, 2, 3)


def state_batch(batch, length, objects, queries, generator, event_rate=0.3, set_prob=0.2):
    """Returns x, mask (answer positions), last_set (what a 'copy the last SET' shortcut would answer)."""
    K, Q = objects, queries
    x = torch.randint(DIS_LO, DIS_HI, (batch, length), generator=generator)
    mask = torch.zeros(batch, length - 1, dtype=torch.bool)
    shortcut = torch.zeros(batch, length - 1, dtype=torch.long)
    body_end = length - 3 * Q
    for b in range(batch):
        pos = torch.randint(0, 16, (K,), generator=generator).tolist()
        last_set = list(pos)
        t = 0
        for o in range(K):                       # initial observation of every object
            x[b, t:t + 3] = torch.tensor([ST_SET, ST_OBJ + o, ST_POS + pos[o]])
            t += 3
        while t + 3 <= body_end:
            if float(torch.rand(1, generator=generator)) < event_rate:
                o = int(torch.randint(0, K, (1,), generator=generator))
                if float(torch.rand(1, generator=generator)) < set_prob:
                    x[b, t:t + 3] = torch.tensor([ST_SET, ST_OBJ + o, ST_POS + pos[o]])
                    last_set[o] = pos[o]
                else:
                    k = int(torch.randint(0, len(ST_DELTAS), (1,), generator=generator))
                    pos[o] = (pos[o] + ST_DELTAS[k]) % 16
                    x[b, t:t + 3] = torch.tensor([ST_MV, ST_OBJ + o, ST_MOVE + k])
                t += 3
            else:
                t += 1                           # one noise token
        order = torch.randperm(K, generator=generator)[:Q].tolist()
        t = body_end
        for o in order:
            x[b, t:t + 3] = torch.tensor([ST_Q, ST_OBJ + o, ST_POS + pos[o]])
            mask[b, t + 1] = True                # prediction at the object token targets the position
            shortcut[b, t + 1] = ST_POS + last_set[o]
            t += 3
    return x.long(), mask, shortcut


def check_state(length=512, objects=4, queries=4):
    """Replay every sample and verify each answer equals the integrated position."""
    g = torch.Generator().manual_seed(4)
    x, mask, short = state_batch(8, length, objects, queries, g)
    for b in range(8):
        pos = {}
        seq = x[b].tolist()
        t = 0
        while t < length - 2:
            m = seq[t]
            if m == ST_SET and ST_OBJ <= seq[t + 1] < ST_OBJ + objects and ST_POS <= seq[t + 2] < ST_POS + 16:
                pos[seq[t + 1]] = seq[t + 2] - ST_POS
                t += 3
            elif m == ST_MV and ST_OBJ <= seq[t + 1] < ST_OBJ + objects and ST_MOVE <= seq[t + 2] < ST_MOVE + 6:
                pos[seq[t + 1]] = (pos[seq[t + 1]] + ST_DELTAS[seq[t + 2] - ST_MOVE]) % 16
                t += 3
            elif m == ST_Q:
                assert bool(mask[b, t + 1]), "query without answer mask"
                assert pos[seq[t + 1]] == seq[t + 2] - ST_POS, "answer is not the integrated position"
                t += 3
            else:
                t += 1
        assert int(mask[b].sum()) == queries
    moves = int(((x == ST_MV).sum()).item()) / 8
    return [f"state task L={length} objects={objects}: answers replayed and verified, about {moves:.0f} moves per sample"]


# --------------------------------------------------------------------------- LM-08 sensor forecasting (PASG)
# A quantised multi-component sensor signal (bytes 0..255): slow periodic components (some with periods longer
# than the 512-token training window), a step change (regime), and noise. Next-sample prediction; the test
# compares the loss on the last 256 samples with the FULL history vs only the last 512 samples, i.e. how much
# the model gains from history beyond its training length.

def sensor_batch(batch, length, generator):
    t = torch.arange(length, dtype=torch.float32)
    out = torch.empty(batch, length, dtype=torch.long)
    for b in range(batch):
        r = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1, generator=generator))
        p1, p2, p3 = r(24, 96), r(256, 1024), r(1500, 6000)          # fast, medium, slow (beyond 512)
        a1, a2, a3 = r(10, 25), r(15, 35), r(20, 45)
        ph = [r(0, 6.283) for _ in range(3)]
        y = 128 + a1 * torch.sin(6.283 * t / p1 + ph[0]) + a2 * torch.sin(6.283 * t / p2 + ph[1]) \
            + a3 * torch.sin(6.283 * t / p3 + ph[2])
        step_at = int(r(0, length))
        y[step_at:] += r(-30, 30)
        y += torch.randn(length, generator=generator) * 2.0
        out[b] = y.round().clamp(0, 255).long()
    return out


def check_sensor(length=2048):
    g = torch.Generator().manual_seed(5)
    x = sensor_batch(4, length, g)
    assert int(x.min()) >= 0 and int(x.max()) <= 255
    spread = float(x.float().std())
    return [f"sensor task L={length}: values in 0..255, std {spread:.1f} (signal varies)"]
