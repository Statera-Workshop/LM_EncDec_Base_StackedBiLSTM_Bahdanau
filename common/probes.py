"""
Cheap checks run before any long training.

causality  changing x[t] must not change logits before t (no look-ahead), and must change something at >= t
overfit    the model can memorise a tiny batch (implementation check, as in the detection stage)
scaling    peak GPU memory and time vs sequence length, for training (fwd+bwd) and inference (fwd)
"""
import time

import torch
import torch.nn.functional as F

from common.config import OVERFIT, PROBE, VOCAB


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def causality(model, device, length=256, positions=(17, 63, 100, 130, 200), seed=0):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (2, length), generator=g).to(device)
    base = model(x).float()
    worst_leak, min_effect = 0.0, float("inf")
    for t in positions:
        x2 = x.clone()
        x2[:, t] = (x2[:, t] + 1 + torch.randint(0, VOCAB - 1, (1,), generator=g).item()) % VOCAB
        out = model(x2).float()
        diff = (out - base).abs()
        leak = diff[:, :t].max().item() if t > 0 else 0.0
        effect = diff[:, t].max().item()   # 2026-10-21: logits[t] itself must use x[t] (catches outputs shifted back)
        worst_leak = max(worst_leak, leak)
        min_effect = min(min_effect, effect)
    scale = base.abs().max().item() + 1e-6
    causal = worst_leak <= 1e-4 * scale + 1e-5
    uses_input = min_effect > 1e-4 * scale
    model.train()
    return dict(causal=bool(causal), uses_input=bool(uses_input), worst_leak=worst_leak,
                min_effect=min_effect, logit_scale=scale, length=length, positions=list(positions))


def overfit(model, train_bytes, device, use_amp, log=print, steps=None):
    cfg = dict(OVERFIT)
    if steps:
        cfg["steps"] = steps
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, train_bytes.numel() - cfg["seq_len"] - 1, (cfg["samples"],), generator=g)
    x = torch.stack([train_bytes[i:i + cfg["seq_len"] + 1] for i in idx.tolist()]).long().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    first = last = None
    t0 = time.time()
    for step in range(cfg["steps"]):
        opt.zero_grad(set_to_none=True)
        mb = getattr(model, "micro_batch", None) or x.shape[0]
        loss = ce = 0.0
        for s in range(0, x.shape[0], mb):
            with autocast(device, use_amp):
                l, c = model.train_loss(x[s:s + mb])
            w = x[s:s + mb].shape[0] / x.shape[0]
            (l * w).backward()
            loss, ce = loss + l.detach() * w, ce + c.detach() * w
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        v = ce.item()
        if first is None:
            first = v
        last = v
        if step % 100 == 0 or step == cfg["steps"] - 1:
            log(f"overfit step {step:4d} ce {v:.4f}")
        if not torch.isfinite(ce):
            break
    secs = time.time() - t0
    return dict(first_ce=first, last_ce=last, passed=bool(last is not None and last < cfg["pass_loss"]),
                pass_loss=cfg["pass_loss"], steps=cfg["steps"], seconds=round(secs, 1),
                lr=1e-3, note="overfit uses Adam 1e-3 on 4 x 256 bytes; only an implementation check")


def _measure(build_fn, device, L, mode, use_amp):
    model = build_fn(max(L, 512)).to(device)
    x = torch.randint(0, VOCAB, (1, L), device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    try:
        times = []
        for _ in range(3):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            if mode == "train":
                model.train()
                with autocast(device, use_amp):
                    logits = model(x)
                F.cross_entropy(logits.float().reshape(-1, VOCAB)[:-1], x.reshape(-1)[1:]).backward()
                model.zero_grad(set_to_none=True)
            else:
                model.eval()
                with torch.no_grad(), autocast(device, use_amp):
                    model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if device.type == "cuda" else float("nan")
        return dict(peak_mb=round(peak, 1), ms=round(1000 * min(times[1:]), 1), dtype="bf16" if use_amp else "fp32")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def scaling(build_fn, device, max_len, log=print):
    """Each length gets a freshly built model so allocator state does not carry over.
    bf16 autocast first; a model whose ops do not support bf16 (Mega FFT, GateLoop complex numbers)
    is measured in fp32 instead and marked dtype=fp32 (2026-10-19 fix)."""
    rows = []
    stop = {"train": False, "infer": False}
    amp = {"train": True, "infer": True}
    for L in PROBE["lens"]:
        if L > max_len:
            rows.append(dict(len=L, note=f"not supported (model built for <= {max_len})"))
            continue
        row = dict(len=L)
        for mode in ("train", "infer"):
            if stop[mode]:
                row[mode] = "skipped after OOM"
                continue
            for attempt in range(2):
                try:
                    row[mode] = _measure(build_fn, device, L, mode, amp[mode] and device.type == "cuda")
                    break
                except torch.cuda.OutOfMemoryError:
                    row[mode] = "OOM"
                    stop[mode] = True
                    break
                except Exception as e:
                    if amp[mode] and attempt == 0 and device.type == "cuda":
                        log(f"scaling L={L} {mode}: bf16 failed ({type(e).__name__}: {str(e)[:80]}) -> fp32")
                        amp[mode] = False
                        continue
                    row[mode] = f"error: {type(e).__name__}: {str(e)[:120]}"
                    break
        log(f"scaling L={L}: {row}")
        rows.append(row)
    return rows
