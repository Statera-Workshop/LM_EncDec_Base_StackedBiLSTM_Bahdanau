"""
LM-03 translation (English -> French) for the legacy encoder-decoder ports.
Legacy settings kept: embedding 128, hidden 1024, batch 256, 20 epochs, Adam 1e-3, 12,800 pairs, 80/20 split.
Reported: test loss / perplexity (teacher forcing, the decoder's next-token prediction), next-token accuracy,
BLEU-4 with greedy decoding (whole test split) and with beam search (width 4, first 500 test pairs).
"""
import math
import time

import torch
import torch.nn.functional as F

from common.s2s_data import PAD, batches, check_pairs, load_pairs
from models.decoding import beam, corpus_bleu, greedy
from models.seq2seq import S2S, build_s2s

LM03 = dict(epochs=20, batch=256, lr=1e-3, clip=1.0, max_hours=2.0, beam_width=4, beam_pairs=500)


def _loss(model, xs, ys):
    logits = model(xs, ys[:, :-1])
    tgt = ys[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=PAD)
    acc = ((logits.argmax(-1) == tgt) & (tgt != PAD)).sum() / (tgt != PAD).sum()
    return loss, acc


@torch.no_grad()
def evaluate(model, pairs, sv, tv, device):
    model.eval()
    tot = n = acc_sum = 0.0
    for xs, ys, _ in batches(pairs, sv, tv, 256, False):
        xs, ys = xs.to(device), ys.to(device)
        loss, acc = _loss(model, xs, ys)
        k = (ys[:, 1:] != PAD).sum().item()
        tot += loss.item() * k
        acc_sum += acc.item() * k
        n += k
    model.train()
    return tot / n, acc_sum / n


VARIANTS = {   # lm03: legacy setting; lm03big: about 8x more pairs, 10 epochs; lm03s2: legacy setting, init seed 2
    "lm03": dict(),
    "lm03big": dict(n_pairs=100_000, epochs=10, max_hours=3.0),
    "lm03s2": dict(seed=2),
}


def run_lm03(name, device, run_dir, log, smoke=False, variant="lm03"):
    cfg = dict(LM03, n_pairs=12_800, seed=1234)
    cfg.update(VARIANTS[variant])
    if smoke:
        cfg.update(epochs=2, batch=32, beam_pairs=8)
    train, test, sv, tv = load_pairs(n=cfg["n_pairs"], smoke=smoke)
    for line in check_pairs(train, test, sv, tv):
        log(line)
    torch.manual_seed(cfg["seed"])
    model = build_s2s(name, len(sv), len(tv))
    if smoke:   # small model for the CPU code-path test
        from models.seq2seq import Seq2Seq
        model = Seq2Seq(len(sv), len(tv), **{**S2S[name]["cfg"], "emb": 16, "hid": 32})
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    log(f"LM-03 {name}: {params / 1e6:.1f}M params, {len(train)} train / {len(test)} test pairs")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(cfg["seed"])
    t0, hist, capped = time.time(), [], False
    for ep in range(cfg["epochs"]):
        tl = steps = 0
        for xs, ys, _ in batches(train, sv, tv, cfg["batch"], True, g):
            xs, ys = xs.to(device), ys.to(device)
            loss, _ = _loss(model, xs, ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            tl += loss.item()
            steps += 1
        vl, va = evaluate(model, test, sv, tv, device)
        hist.append(dict(epoch=ep + 1, train_loss=tl / steps, test_loss=vl, test_acc=va))
        log(f"epoch {ep + 1:2d} train {tl / steps:.3f} test loss {vl:.3f} ppl {math.exp(vl):.1f} acc {va:.3f} "
            f"({(time.time() - t0) / 60:.1f} min)")
        if (time.time() - t0) / 3600 > cfg["max_hours"]:
            capped = True
            break
    if not smoke:
        torch.save(model.state_dict(), run_dir / f"{variant}_model.pt")
    model.eval()
    refs, hyps_g = [], []
    for xs, ys, chunk in batches(test, sv, tv, 256, False):
        for ids in greedy(model, xs.to(device)):
            hyps_g.append(tv.decode(ids))
        refs += [t.split() for _, t in chunk]
    bleu_g = corpus_bleu(hyps_g, refs)
    sub = test[:cfg["beam_pairs"]]
    hyps_b = []
    for xs, _, _ in batches(sub, sv, tv, 32, False):
        hyps_b += [tv.decode(ids) for ids in beam(model, xs.to(device), width=cfg["beam_width"])]
    bleu_b = corpus_bleu(hyps_b, refs[:len(sub)])
    bleu_g_sub = corpus_bleu(hyps_g[:len(sub)], refs[:len(sub)])
    examples = [dict(src=s, ref=t, greedy=" ".join(h)) for (s, t), h in zip(test[:5], hyps_g[:5])]
    for e in examples:
        log(f"  {e['src']}  ->  {e['greedy']}   (ref: {e['ref']})")
    vl, va = evaluate(model, test, sv, tv, device)
    return dict(params=params, epochs=len(hist), capped=capped, minutes=round((time.time() - t0) / 60, 1),
                test_loss=vl, test_ppl=math.exp(vl), test_acc=va, bleu_greedy=bleu_g,
                bleu_greedy_beam_subset=bleu_g_sub, bleu_beam=bleu_b, beam_width=cfg["beam_width"],
                beam_pairs=len(sub), n_train=len(train), n_test=len(test), variant=variant, history=hist, examples=examples, legacy_source=S2S[name]["src"],
                config=S2S[name]["cfg"])
