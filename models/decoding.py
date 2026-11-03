"""
Decoding for the encoder-decoder models: greedy and beam search (replaces the TF Addons BasicDecoder /
BeamSearchDecoder used in C9; TF Addons is no longer maintained), and corpus BLEU-4 without extra packages.
"""
import math
from collections import Counter

import torch

SOS, EOS = 1, 2


@torch.no_grad()
def greedy(model, src, max_len=24):
    keys, mask, state, feed = model.start(src)
    y = torch.full((src.shape[0],), SOS, dtype=torch.long, device=src.device)
    out, done = [], torch.zeros_like(y, dtype=torch.bool)
    for _ in range(max_len):
        logits, state, feed = model.step(y, state, keys, mask, feed)
        y = logits.argmax(-1)
        out.append(y)
        done |= y == EOS
        if bool(done.all()):
            break
    return torch.stack(out, 1).tolist()


def _select(state, idx, cell):
    if cell == "lstm":
        return [(h[idx], c[idx]) for h, c in state]
    return [h[idx] for h in state]


@torch.no_grad()
def beam(model, src, width=4, max_len=24, alpha=0.6):
    """Beam search per sentence; score = log-prob / ((5 + len) / 6) ** alpha (GNMT length penalty)."""
    results = []
    for b in range(src.shape[0]):
        s = src[b:b + 1]
        keys, mask, state, feed = model.start(s)
        beams = [([], 0.0, state, feed, False)]
        for _ in range(max_len):
            live = [bm for bm in beams if not bm[4]]
            if not live:
                break
            ys = torch.tensor([bm[0][-1] if bm[0] else SOS for bm in live], device=src.device)
            n = len(live)
            st = _cat_states([bm[2] for bm in live], model.cell)
            fd = torch.cat([bm[3] for bm in live]) if live[0][3] is not None else None
            logits, st, fd = model.step(ys, st, keys.expand(n, -1, -1), mask.expand(n, -1), fd)
            logp = logits.log_softmax(-1)
            top_v, top_i = logp.topk(width, -1)
            cand = [bm for bm in beams if bm[4]]
            for j, bm in enumerate(live):
                for v, i in zip(top_v[j].tolist(), top_i[j].tolist()):
                    cand.append((bm[0] + [i], bm[1] + v, _select(st, slice(j, j + 1), model.cell),
                                 fd[j:j + 1] if fd is not None else None, i == EOS))
            cand.sort(key=lambda c: c[1] / (((5 + len(c[0])) / 6) ** alpha), reverse=True)
            beams = cand[:width]
        best = max(beams, key=lambda c: c[1] / (((5 + len(c[0])) / 6) ** alpha))
        results.append(best[0])
    return results


def _cat_states(states, cell):
    layers = len(states[0])
    if cell == "lstm":
        return [(torch.cat([s[l][0] for s in states]), torch.cat([s[l][1] for s in states])) for l in range(layers)]
    return [torch.cat([s[l] for s in states]) for l in range(layers)]


def corpus_bleu(hyps, refs, n=4):
    """hyps, refs: lists of token lists (one reference each). Returns BLEU-4 in [0, 100]."""
    match, total = [0] * n, [0] * n
    hl = rl = 0
    for h, r in zip(hyps, refs):
        hl += len(h)
        rl += len(r)
        for k in range(1, n + 1):
            hc = Counter(tuple(h[i:i + k]) for i in range(len(h) - k + 1))
            rc = Counter(tuple(r[i:i + k]) for i in range(len(r) - k + 1))
            match[k - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            total[k - 1] += max(len(h) - k + 1, 0)
    if min(match) == 0 or hl == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / n
    bp = 1.0 if hl > rl else math.exp(1 - rl / hl)
    return 100 * bp * math.exp(log_p)
