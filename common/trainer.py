"""
Training loops shared by every model.

- bf16 autocast on CUDA; if the first step is not finite or raises, the run falls back
  to fp32 and records it (an [Unified] deviation written to the result file).
- Checkpoint every eval so an interrupted run resumes where it stopped.
- LM-02 stops at the token budget or the wall-clock cap, whichever comes first.
"""
import json
import math
import time

import torch
import torch.nn.functional as F

from common.config import LM02, LM05, LM05V2, LM06, OPTIM, SEED
from common.data import RandomWindows, eval_windows, recall_batch, recall_batch_v2, update_batch, VAL_LO, VAL_HI, DIS_LO, DIS_HI
from common.probes import autocast

LN2 = math.log(2)


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=OPTIM["lr"], betas=OPTIM["betas"],
                            weight_decay=OPTIM["weight_decay"])


def lr_at(step, base=None):
    w = OPTIM["warmup_steps"]
    return (base or OPTIM["lr"]) * min(1.0, (step + 1) / w)


def step_once(model, opt, x, step, use_amp, device, loss_fn):
    """One optimizer step. The batch is split into micro-batches of model.micro_batch samples
    (gradient accumulation, mathematically the same update). On CUDA out-of-memory the micro-batch
    is halved and the step is retried; the final value is recorded in the results."""
    for g in opt.param_groups:
        g["lr"] = lr_at(step, getattr(model, "lr", None))
    B = x.shape[0]
    while True:
        mb = min(getattr(model, "micro_batch", None) or B, B)
        try:
            opt.zero_grad(set_to_none=True)
            tot_loss = tot_metric = 0.0
            for s in range(0, B, mb):
                sl = slice(s, min(s + mb, B))
                with autocast(device, use_amp):
                    loss, metric = loss_fn(model, x[sl], sl)
                w = (sl.stop - sl.start) / B
                (loss * w).backward()
                tot_loss += loss.detach() * w
                tot_metric += metric.detach() * w
            break
        except torch.cuda.OutOfMemoryError:
            if mb == 1:
                raise
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            model.micro_batch = max(1, mb // 2)
            print(f"CUDA OOM at micro-batch {mb}; retrying with {model.micro_batch}", flush=True)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), OPTIM["clip"])
    opt.step()
    return tot_loss, tot_metric, gnorm


def first_step_with_fallback(model, opt, x, device, loss_fn, log):
    """Try bf16; fall back to fp32 if it errors or is not finite. Returns use_amp."""
    if device.type != "cuda":
        return False
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        loss, _, gnorm = step_once(model, opt, x, 0, True, device, loss_fn)
        if torch.isfinite(loss) and torch.isfinite(gnorm):
            return True
        log("bf16 first step not finite -> fp32")
    except Exception as e:
        log(f"bf16 first step failed ({type(e).__name__}: {str(e)[:150]}) -> fp32")
    model.load_state_dict(state)
    opt.state.clear()
    return False


def save_ckpt(path, model, opt, extra):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), **extra), path)


@torch.no_grad()
def forward_chunked(model, x, device, use_amp):
    """Inference in micro-batches (model.micro_batch), halving on CUDA OOM."""
    while True:
        mb = min(getattr(model, "micro_batch", None) or x.shape[0], x.shape[0])
        try:
            outs = []
            for s in range(0, x.shape[0], mb):
                with autocast(device, use_amp):
                    outs.append(model(x[s:s + mb]).float())
            return torch.cat(outs)
        except torch.cuda.OutOfMemoryError:
            if mb == 1:
                raise
            torch.cuda.empty_cache()
            model.micro_batch = max(1, mb // 2)


# --------------------------------------------------------------------------- LM-02

def lm02_loss(model, x, sl=None):
    return model.train_loss(x)


@torch.no_grad()
def evaluate_bpc(model, data, device, use_amp, n_batches):
    model.eval()
    batches, n_tokens, _ = eval_windows(data, LM02["seq_len"], LM02["batch"], n_batches)
    total = 0.0
    for i, b in enumerate(batches):
        if i and i % 50 == 0:
            print(f"  eval batch {i}/{len(batches)}", flush=True)
        b = b.to(device)
        logits = forward_chunked(model, b[:, :-1], device, use_amp)
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1),
                                 reduction="sum").item()
    model.train()
    return total / n_tokens / LN2, n_tokens


@torch.no_grad()
def sample_text(model, prime, device, use_amp, length=256, top_k=1):
    model.eval()
    seq = prime.clone().to(device)[None]
    for _ in range(length):
        with autocast(device, use_amp):
            logits = model(seq)[:, -1].float()
        if top_k == 1:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            v, i = logits.topk(top_k)
            nxt = i.gather(-1, torch.multinomial(v.softmax(-1), 1))
        seq = torch.cat([seq, nxt], dim=1)
    model.train()
    out = seq[0, prime.numel():].tolist()
    return "".join(chr(max(32, c)) if c < 128 else "?" for c in out)


def train_lm02(model, splits, device, run_dir, log, smoke=False, overrides=None, tag="lm02"):
    cfg = dict(LM02)
    cfg.update(overrides or {})
    if smoke:
        cfg.update(tokens=cfg["batch"] * cfg["seq_len"] * 4, eval_every=2, eval_batches=1, final_eval_batches=1, batch=4)
    tokens_per_step = cfg["batch"] * cfg["seq_len"]
    total_steps = math.ceil(cfg["tokens"] / tokens_per_step)
    opt = make_optimizer(model)
    ckpt = run_dir / f"{tag}_ckpt.pt"
    g = torch.Generator().manual_seed(SEED)
    hist, step, elapsed, use_amp = [], 0, 0.0, None
    if ckpt.exists():
        s = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step, elapsed, hist, use_amp = s["step"], s["elapsed"], s["hist"], s["use_amp"]
        g.set_state(s["rng"].cpu())   # 2026-10-22 fix: map_location moved the CPU RNG state to the GPU
        log(f"resumed LM-02 at step {step} ({elapsed / 3600:.2f} h)")
    loader = RandomWindows(splits["train"], cfg["seq_len"], cfg["batch"], g)
    if use_amp is None:
        use_amp = first_step_with_fallback(model, opt, next(loader).to(device), device, lm02_loss, log)
        opt = make_optimizer(model) if not use_amp else opt
    log(f"LM-02 {total_steps} steps x {tokens_per_step} tokens, bf16={use_amp}")
    t_start = time.time() - elapsed
    capped = False
    run_loss, run_n = 0.0, 0
    while step < total_steps:
        x = next(loader).to(device)
        loss, ce, _ = step_once(model, opt, x, step, use_amp, device, lm02_loss)
        run_loss += ce.item()
        run_n += 1
        step += 1
        if not math.isfinite(loss.item()):
            log(f"STOP non-finite loss at step {step}")
            break
        over_time = (time.time() - t_start) / 3600 > cfg["max_hours"]
        if step % 100 == 0 and step % cfg["eval_every"] != 0:
            log(f"step {step:6d} train_bpc {ce.item() / LN2:.3f} ({(time.time() - t_start) / 3600:.2f} h)")
        if step % cfg["eval_every"] == 0 or step == total_steps or over_time:
            bpc, _ = evaluate_bpc(model, splits["valid"], device, use_amp, cfg["eval_batches"])
            elapsed = time.time() - t_start
            row = dict(step=step, tokens=step * tokens_per_step, train_bpc=run_loss / max(run_n, 1) / LN2,
                       valid_bpc=bpc, hours=elapsed / 3600)
            hist.append(row)
            run_loss, run_n = 0.0, 0
            log(f"step {step:6d} tokens {row['tokens'] / 1e6:7.1f}M train_bpc {row['train_bpc']:.3f} "
                f"valid_bpc {bpc:.3f} {row['hours']:.2f} h")
            save_ckpt(ckpt, model, opt, dict(step=step, elapsed=elapsed, hist=hist, use_amp=use_amp, rng=g.get_state()))
            if elapsed / 3600 > cfg["max_hours"]:
                capped = True
                log(f"time cap {cfg['max_hours']} h reached at {row['tokens'] / 1e6:.1f}M tokens")
                break
    valid_bpc, n_valid = evaluate_bpc(model, splits["valid"], device, use_amp, cfg["final_eval_batches"])
    test_bpc, n_test = evaluate_bpc(model, splits["test"], device, use_amp, cfg["final_eval_batches"])
    sample = sample_text(model, splits["valid"][:128].long(), device, use_amp, length=64 if smoke else 256)
    if not smoke:
        torch.save(model.state_dict(), run_dir / f"{tag}_model.pt")
    return dict(micro_batch=getattr(model, "micro_batch", None), steps=step, tokens=step * tokens_per_step, budget_tokens=cfg["tokens"], capped=capped,
                bf16=use_amp, hours=round((time.time() - t_start) / 3600, 3), valid_bpc=valid_bpc,
                test_bpc=test_bpc, valid_tokens=n_valid, test_tokens=n_test, history=hist,
                prime=splits["valid"][:128].tolist(), sample=sample)


# --------------------------------------------------------------------------- LM-05

def recall_loss_fn(mask):
    def fn(model, x, sl=slice(None)):
        logits = model(x[:, :-1]).float()
        m = mask[sl].to(x.device)
        sel = logits[m]
        tgt = x[:, 1:][m]
        loss = F.cross_entropy(sel, tgt)
        return loss, loss
    return fn


@torch.no_grad()
def recall_accuracy(model, device, use_amp, length, n_samples, batch=16, seed=99, control=None):
    """control='values': the values written in the key/value block are replaced by fresh random values
    after the targets are fixed, so the answer is no longer in the context -> accuracy must fall to chance.
    control='keys': the keys in the block are replaced, so the queries cannot be matched -> chance."""
    batch = min(batch, n_samples)
    model.eval()
    g = torch.Generator().manual_seed(seed + length)
    correct = total = 0
    for _ in range(math.ceil(n_samples / batch)):
        x, mask = recall_batch(batch, length, LM05["pairs"], g, max_gap=True)
        P = LM05["pairs"]
        if control == "values":
            x[:, 1:2 * P:2] = torch.randint(VAL_LO, VAL_HI, (x.shape[0], P), generator=g)
        elif control == "keys":
            x[:, 0:2 * P:2] = torch.randint(DIS_LO, DIS_HI, (x.shape[0], P), generator=g)
        x = x.to(device)
        logits = forward_chunked(model, x[:, :-1], device, use_amp)
        pred = logits.argmax(-1)
        m = mask.to(device)
        correct += (pred[m] == x[:, 1:][m]).sum().item()
        total += m.sum().item()
    model.train()
    return correct / total, total


def train_lm05(model, device, run_dir, log, smoke=False):
    cfg = dict(LM05)
    if smoke:
        cfg.update(steps=10, eval_every=10, eval_samples=4, batch=4, test_lens=(512, 1024))
    opt = make_optimizer(model)
    g = torch.Generator().manual_seed(SEED + 5)
    x, mask = recall_batch(cfg["batch"], cfg["train_len"], cfg["pairs"], g, max_gap=False)
    use_amp = first_step_with_fallback(model, opt, x.to(device), device, recall_loss_fn(mask), log)
    if not use_amp:
        opt = make_optimizer(model)
    log(f"LM-05 {cfg['steps']} steps batch {cfg['batch']} len {cfg['train_len']}, bf16={use_amp}")
    hist, t0, capped, done = [], time.time(), False, 0
    for step in range(cfg["steps"]):
        x, mask = recall_batch(cfg["batch"], cfg["train_len"], cfg["pairs"], g, max_gap=False)
        loss, _, _ = step_once(model, opt, x.to(device), step, use_amp, device, recall_loss_fn(mask))
        if not math.isfinite(loss.item()):
            log(f"STOP non-finite loss at step {step}")
            break
        done = step + 1
        if (time.time() - t0) / 3600 > cfg["max_hours"]:
            capped = True
            log(f"time cap {cfg['max_hours']} h reached at step {step + 1}")
            break
        if (step + 1) % 50 == 0 and (step + 1) % cfg["eval_every"] != 0:
            log(f"step {step + 1:5d} loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} min)")
        if (step + 1) % cfg["eval_every"] == 0:
            acc, _ = recall_accuracy(model, device, use_amp, cfg["train_len"], cfg["eval_samples"] // 4)
            hist.append(dict(step=step + 1, loss=loss.item(), acc_train_len=acc))
            log(f"step {step + 1:5d} loss {loss.item():.4f} recall@{cfg['train_len']} {acc:.3f}")
            if (time.time() - t0) / 3600 > cfg["max_hours"]:
                capped = True
                log(f"time cap {cfg['max_hours']} h reached at step {step + 1}")
                break
    results = {}
    for L in cfg["test_lens"]:
        try:
            acc, n = recall_accuracy(model, device, use_amp, L, cfg["eval_samples"])
            results[str(L)] = dict(acc=acc, answers=n)
        except torch.cuda.OutOfMemoryError:
            results[str(L)] = "OOM"
            torch.cuda.empty_cache()
        except Exception as e:
            results[str(L)] = f"error: {type(e).__name__}: {str(e)[:120]}"
        log(f"recall test L={L} (gap {L - 4 * cfg['pairs']}): {results[str(L)]}")
    if not smoke:
        torch.save(model.state_dict(), run_dir / "lm05_model.pt")
    return dict(micro_batch=getattr(model, "micro_batch", None), steps=done, budget_steps=cfg["steps"], capped=capped, batch=cfg["batch"], train_len=cfg["train_len"], pairs=cfg["pairs"],
                bf16=use_amp, minutes=round((time.time() - t0) / 60, 1), history=hist, test=results,
                chance=1 / 64)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=1, default=str))


# --------------------------------------------------------------------------- LM-05 v2

@torch.no_grad()
def recall_accuracy_v2(model, device, use_amp, length, pairs, n_samples, control=None, seed=99, batch=16,
                       queries=None, distinct=False):
    """control='keys': keys in the pair block replaced by distractors -> only the value set is known (~1/pairs).
    control='values': values replaced by fresh values -> answer absent (chance 1/64)."""
    model.eval()
    g = torch.Generator().manual_seed(seed + 31 * length + pairs)
    correct = total = 0
    batch = min(batch, n_samples)
    for i in range(math.ceil(n_samples / batch)):
        if i and i % 8 == 0:
            print(f"  recall eval L={length} batch {i}", flush=True)
        x, mask = recall_batch_v2(batch, length, pairs, queries or LM05V2["queries"], g, max_gap=True,
                                  distinct=distinct)
        if control == "values":
            x[:, 1:2 * pairs:2] = torch.randint(VAL_LO, VAL_HI, (batch, pairs), generator=g)
        elif control == "keys":
            x[:, 0:2 * pairs:2] = torch.randint(DIS_LO, DIS_HI, (batch, pairs), generator=g)
        x = x.to(device)
        logits = forward_chunked(model, x[:, :-1], device, use_amp)
        m = mask.to(device)
        correct += (logits.argmax(-1)[m] == x[:, 1:][m]).sum().item()
        total += m.sum().item()
    model.train()
    return correct / total


def train_lm05v2(build_fn, device, run_dir, log, smoke=False, overrides=None, seed_offset=0, tag="lm05v2"):
    """Trains one fresh model per difficulty level (pairs 4 and 16) and evaluates normal / no-keys / no-values."""
    cfg = dict(LM05V2)
    cfg.update(overrides or {})
    if smoke:
        cfg.update(steps=6, eval_every=3, eval_samples=4, batch=4, test_lens=(512,))
    out = dict(levels={})
    levels = cfg.get("levels") or [(P, cfg["queries"]) for P in cfg["pairs"]]
    distinct = cfg.get("distinct", False)
    for P, Q in levels:
        torch.manual_seed(SEED + P + 1000 * seed_offset)
        model = build_fn().to(device)
        opt = make_optimizer(model)
        g = torch.Generator().manual_seed(SEED + 50 + P + 1000 * seed_offset)
        x, mask = recall_batch_v2(cfg["batch"], cfg["train_len"], P, Q, g, max_gap=False, distinct=distinct)
        use_amp = first_step_with_fallback(model, opt, x.to(device), device, recall_loss_fn(mask), log)
        if not use_amp:
            opt = make_optimizer(model)
        log(f"LM-05 recall pairs={P} queries={Q} distinct={distinct}: {cfg['steps']} steps batch {cfg['batch']} len {cfg['train_len']}, bf16={use_amp}")
        t0, done, capped, hist, last, diverged = time.time(), 0, False, [], float("nan"), False
        for step in range(cfg["steps"]):
            x, mask = recall_batch_v2(cfg["batch"], cfg["train_len"], P, Q, g, max_gap=False, distinct=distinct)
            loss, _, _ = step_once(model, opt, x.to(device), step, use_amp, device, recall_loss_fn(mask))
            last = loss.item()
            if not math.isfinite(last):
                log(f"STOP non-finite loss at step {step} (recorded as diverged)")
                diverged = True
                break
            done = step + 1
            if (step + 1) % 100 == 0:
                log(f"pairs={P} step {step + 1:5d} loss {last:.4f} ({(time.time() - t0) / 60:.1f} min)")
            if (step + 1) % cfg["eval_every"] == 0:
                acc = recall_accuracy_v2(model, device, use_amp, cfg["train_len"], P, 64, queries=Q, distinct=distinct)
                hist.append(dict(step=step + 1, loss=last, acc=acc))
                log(f"pairs={P} step {step + 1:5d} recall@{cfg['train_len']} {acc:.3f}")
            if (time.time() - t0) / 3600 > cfg["max_hours"]:
                capped = True
                log(f"pairs={P} time cap {cfg['max_hours']} h at step {step + 1}")
                break
        if not smoke:   # 2026-10-26: keep the trained weights before testing, so a failed test never loses training
            torch.save(model.state_dict(), run_dir / f"lm05v2_{tag}_P{P}.pt")
        res = {}
        for L in cfg["test_lens"]:
            row = {}
            limit = getattr(model, "max_positions", None)
            if limit and L > limit:
                res[str(L)] = f"n/a (model built for <= {limit} positions)"
                log(f"pairs={P} L={L}: skipped, model built for <= {limit} positions")
                continue
            for ctl in (None, "keys", "values"):
                try:
                    row[ctl or "normal"] = recall_accuracy_v2(model, device, use_amp, L, P, cfg["eval_samples"],
                                                              control=ctl, queries=Q, distinct=distinct)
                except torch.cuda.OutOfMemoryError:
                    row[ctl or "normal"] = "OOM"
                    torch.cuda.empty_cache()
                except Exception as e:   # e.g. a model built for <= 2048 positions tested at 8192: record, do not crash
                    row[ctl or "normal"] = f"n/a ({type(e).__name__})"
                    log(f"pairs={P} L={L} {ctl or 'normal'}: {type(e).__name__}: {str(e)[:100]}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            if all(isinstance(v, float) for v in row.values()):
                row["true_recall"] = row["normal"] - row["keys"]
            res[str(L)] = row
            log(f"pairs={P} L={L}: {row}")
        out["levels"][str(P)] = dict(queries=Q, distinct=distinct, steps=done, capped=capped, diverged=diverged, bf16=use_amp, final_loss=last,
                                     minutes=round((time.time() - t0) / 60, 1), history=hist, test=res,
                                     value_set_baseline=1 / P, chance=1 / 64,
                                     micro_batch=getattr(model, "micro_batch", None))
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- LM-06 memory update

DIST_BINS = ((0, 256), (256, 1024), (1024, 4096), (4096, 10**9))


@torch.no_grad()
def update_accuracy(model, device, use_amp, length, cfg, n_samples, seed=77, batch=16, control=None):
    """control='nohead': keys of the first block replaced by distractors, so unchanged keys become unanswerable
    (accuracy there must fall to chance if the model really reads the first block). Also reports accuracy by the
    distance between the answer and the value token it has to copy."""
    model.eval()
    g = torch.Generator().manual_seed(seed + length)
    c_all = c_upd = c_keep = stale = n_all = n_upd = n_keep = 0
    bins = [[0, 0] for _ in DIST_BINS]
    batch = min(batch, n_samples)
    P = cfg["pairs"]
    for i in range(math.ceil(n_samples / batch)):
        if i and i % 8 == 0:
            print(f"  update eval L={length} batch {i}", flush=True)
        x, mask, upd, old, src = update_batch(batch, length, P, cfg["updates"], cfg["queries"], g, max_gap=True,
                                              return_src=True)
        if control == "nohead":
            x[:, 0:2 * P:2] = torch.randint(DIS_LO, DIS_HI, (x.shape[0], P), generator=g)
        x = x.to(device)
        pred = forward_chunked(model, x[:, :-1], device, use_amp).argmax(-1).cpu()
        tgt = x[:, 1:].cpu()
        ok = pred == tgt
        keep = mask & ~upd
        c_all += ok[mask].sum().item(); n_all += mask.sum().item()
        c_upd += ok[upd].sum().item(); n_upd += upd.sum().item()
        c_keep += ok[keep].sum().item(); n_keep += keep.sum().item()
        stale += (pred[upd] == old[upd]).sum().item()
        pos = torch.arange(length - 1).expand_as(mask)
        dist = (pos + 1 - src)[mask]
        okm = ok[mask]
        for j, (lo, hi) in enumerate(DIST_BINS):
            sel = (dist >= lo) & (dist < hi)
            bins[j][0] += okm[sel].sum().item(); bins[j][1] += sel.sum().item()
    model.train()
    return dict(acc=c_all / n_all, acc_updated=c_upd / n_upd, acc_unchanged=c_keep / n_keep, stale_rate=stale / n_upd,
                by_distance={f"{lo}-{hi if hi < 10**9 else 'inf'}": (round(a / n, 3) if n else None, n)
                             for (lo, hi), (a, n) in zip(DIST_BINS, bins)})


def train_lm06(build_fn, device, run_dir, log, smoke=False, overrides=None, tag="lm06", seed_offset=0):
    cfg = dict(LM06)
    cfg.update(overrides or {})
    if smoke:
        cfg.update(steps=4, eval_every=2, eval_samples=4, batch=4, test_lens=(512,))
    torch.manual_seed(SEED + 6 + 1000 * seed_offset)
    model = build_fn().to(device)
    opt = make_optimizer(model)
    g = torch.Generator().manual_seed(SEED + 66 + 1000 * seed_offset)

    def batch():
        x, mask, _, _ = update_batch(cfg["batch"], cfg["train_len"], cfg["pairs"], cfg["updates"], cfg["queries"], g,
                                     max_gap=False)
        return x, mask

    x, mask = batch()
    use_amp = first_step_with_fallback(model, opt, x.to(device), device, recall_loss_fn(mask), log)
    if not use_amp:
        opt = make_optimizer(model)
    log(f"LM-06 update: {cfg['steps']} steps batch {cfg['batch']} len {cfg['train_len']}, bf16={use_amp}")
    t0, done, capped, diverged, hist, last = time.time(), 0, False, False, [], float("nan")
    for step in range(cfg["steps"]):
        x, mask = batch()
        loss, _, _ = step_once(model, opt, x.to(device), step, use_amp, device, recall_loss_fn(mask))
        last = loss.item()
        if not math.isfinite(last):
            log(f"STOP non-finite loss at step {step} (recorded as diverged)")
            diverged = True
            break
        done = step + 1
        if (step + 1) % 100 == 0:
            log(f"update step {step + 1:5d} loss {last:.4f} ({(time.time() - t0) / 60:.1f} min)")
        if (step + 1) % cfg["eval_every"] == 0:
            r = update_accuracy(model, device, use_amp, cfg["train_len"], cfg, 64)
            hist.append(dict(step=step + 1, loss=last, **r))
            log(f"update step {step + 1:5d} {r}")
        if (time.time() - t0) / 3600 > cfg["max_hours"]:
            capped = True
            log(f"update time cap {cfg['max_hours']} h at step {step + 1}")
            break
    if not smoke:
        torch.save(model.state_dict(), run_dir / f"{tag}_model.pt")
    res = {}
    limit = getattr(model, "max_positions", None)
    for L in cfg["test_lens"]:
        if limit and L > limit:
            res[str(L)] = f"n/a (model built for <= {limit} positions)"
            continue
        try:
            res[str(L)] = update_accuracy(model, device, use_amp, L, cfg, cfg["eval_samples"])
        except torch.cuda.OutOfMemoryError:
            res[str(L)] = "OOM"
            torch.cuda.empty_cache()
        except Exception as e:
            res[str(L)] = f"n/a ({type(e).__name__})"
        log(f"update L={L}: {res[str(L)]}")
    return dict(steps=done, capped=capped, diverged=diverged, bf16=use_amp, final_loss=last,
                minutes=round((time.time() - t0) / 60, 1), history=hist, test=res, config=cfg,
                micro_batch=getattr(model, "micro_batch", None))
