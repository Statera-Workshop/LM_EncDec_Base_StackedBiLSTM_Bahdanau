"""
Run one stage for one model.

    python run.py <stage> <model> [--smoke] [--cpu]

stages
  check     data counts (enwik8, recall), causality, parameter count          (~1 min)
  scaling   peak memory / time vs sequence length 512..8192                   (~1-5 min)
  overfit   memorise 4 x 256 bytes; must reach < 0.5 nats                      (~1-10 min)
  lm05      Long-Horizon Recall: train at 512, test recall at 512/1024/2048     (~0.3-3 h)
  lm02      enwik8 next-byte prediction, 200M token budget, 4 h cap            (~0.5-4 h)
  all       check -> scaling -> overfit -> (only if causal and overfit passed) lm05 -> lm02

Results: runs/<model>/<stage>.json, one summary line per stage in runs/summary.log
"""
import argparse
import json
import datetime
import sys
import time
import traceback

import torch

from common.config import RUNS_DIR, SEED, MAX_LEN
from common.data import load_enwik8, check_enwik8, check_recall
from common import probes, trainer
from models.registry import REGISTRY, build, n_params, repo_name

STAGES = ("check", "scaling", "overfit", "lm05", "lm02")
EXTRA = ("verify5", "lm05v2", "lm05v2long", "lm05v2s2", "lm05v2s3", "lm02long", "lm05v3", "lm05v3long", "lm05v3s2", "lm05v3s3", "lm05v3longs2", "lm06", "lm06long", "lm06s2", "lm06s3", "lm06longs2", "lm06v", "lm05v3s4", "lm05v3s5", "lm05v3s6", "lm06longs3", "lm05v3longs3", "lm07", "lm07long", "lm07s2", "lm07s3", "lm08")   # run on demand, not part of "all"


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def summary(line):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "summary.log", "a", encoding="utf-8") as f:
        f.write(f"{now()}  {line}\n")
    print(f"SUMMARY {line}", flush=True)


def meta(name):
    e = REGISTRY[name]
    return dict(model=name, repo=e["repo"], target_repo=repo_name(name), category=e["category"],
                upstream=e["upstream"], unified=e["unified"], torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


def gate(name, smoke):
    """Long stages run only for a causal model that passed the overfit check."""
    sfx = "_smoke" if smoke else ""
    chk, ov = RUNS_DIR / name / f"check{sfx}.json", RUNS_DIR / name / f"overfit{sfx}.json"
    if not chk.exists() or not ov.exists():
        return "run check and overfit first"
    c = json.loads(chk.read_text())["causality"]
    if not (c["causal"] and c["uses_input"]):
        return "not a valid causal LM (check)"
    if not json.loads(ov.read_text())["passed"] and not smoke:  # smoke overfit is too short to pass
        return "overfit FAIL"
    return None


def run_stage(stage, name, device, smoke, log):
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    out = dict(meta(name), stage=stage, started=now(), smoke=smoke)
    global build
    if smoke:  # CPU code-path test in a small container: one sample at a time
        _b = build
        def build(n, max_len=MAX_LEN, _b=_b):
            m = _b(n, max_len)
            m.micro_batch = 1
            return m
    t0 = time.time()

    if stage == "check":
        lines = check_recall()
        if not smoke:
            lines = check_enwik8(load_enwik8()) + lines
        for line in lines:
            log(line)
        model = build(name).to(device)
        out["params"] = n_params(model)
        out["causality"] = probes.causality(model, device)
        out["data_checks"] = lines
        c = out["causality"]
        status = "OK" if c["causal"] and c["uses_input"] else ("NON-CAUSAL" if not c["causal"] else "MISALIGNED (logits[t] ignores x[t])")
        msg = f"{name:12s} check    params {out['params'] / 1e6:6.2f}M causality {status} (leak {c['worst_leak']:.2e})"

    elif stage == "scaling":
        lim = min(REGISTRY[name]["probe_max"], 512 if smoke else 10**9)
        out["rows"] = probes.scaling(lambda L: build(name, max_len=max(L, 512)), device, lim, log)
        last = [r for r in out["rows"] if isinstance(r.get("train"), dict)]
        top = last[-1] if last else None
        msg = (f"{name:12s} scaling  longest train L={top['len']} peak {top['train']['peak_mb']} MB "
               f"({top['train']['dtype']})"
               if top else f"{name:12s} scaling  no length completed")

    elif stage == "overfit":
        splits = load_enwik8() if not smoke else dict(train=torch.randint(0, 256, (100_000,), dtype=torch.uint8))
        model = build(name).to(device)
        use_amp = device.type == "cuda"
        try:
            out.update(probes.overfit(model, splits["train"], device, use_amp, log, steps=6 if smoke else None))
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as e:  # ops without bf16 support (Mega FFT, GateLoop complex): redo in fp32
            if not use_amp:
                raise
            log(f"overfit bf16 failed ({str(e)[:80]}) -> fp32")
            torch.manual_seed(SEED)
            model = build(name).to(device)
            out.update(probes.overfit(model, splits["train"], device, False, log, steps=6 if smoke else None))
        out["bf16"] = "fp32" not in " ".join(str(v) for v in out.values()) and use_amp
        msg = (f"{name:12s} overfit  ce {out['first_ce']:.3f} -> {out['last_ce']:.3f} "
               f"{'PASS' if out['passed'] else 'FAIL'} ({out['seconds']} s)")

    elif stage in ("lm05", "lm02") + EXTRA[1:] and gate(name, smoke):
        reason = gate(name, smoke)
        summary(f"{name:12s} {stage:8s} SKIPPED: {reason}")
        return dict(skipped=reason)

    elif stage in ("lm07", "lm07long", "lm07s2", "lm07s3"):
        from common.data import check_state
        from common.config import SEEDS
        for line in check_state() + check_state(2048):
            log(line)
        out.update(trainer.train_lm07(lambda: build(name), device, run_dir, log, smoke=smoke, tag=stage,
                                      overrides=trainer.LM07_LONG if stage == "lm07long" else None,
                                      seed_offset=SEEDS.get(stage, 0)))
        parts = [f"{L}: {r['acc']:.2f} (copy-last-SET {r['shortcut_copy_last_set']:.2f})" if isinstance(r, dict)
                 else f"{L}: n/a" for L, r in out["test"].items()]
        msg = f"{name:15s} {stage:8s} state acc " + " | ".join(parts) + (" CAPPED" if out["capped"] else "")

    elif stage == "lm08":
        from common.data import check_sensor
        for line in check_sensor():
            log(line)
        out.update(trainer.train_lm08(lambda: build(name), device, run_dir, log, smoke=smoke))
        parts = []
        for L, r in out["test"].items():
            if isinstance(r, dict):
                parts.append(f"{L}: {r['bits_full']:.3f}" + (f" (gain {r['gain_from_history']:+.3f})"
                                                             if "gain_from_history" in r else ""))
            else:
                parts.append(f"{L}: n/a")
        msg = f"{name:15s} lm08     sensor bits " + " | ".join(parts)

    elif stage == "lm06v":
        # verification of trained LM-06 models: first-block control and accuracy by copy distance
        from common.config import LM06
        wpath = run_dir / "lm06long_model.pt"
        if not wpath.exists():
            wpath = run_dir / "lm06_model.pt"
        model = build(name).to(device)
        model.load_state_dict(torch.load(wpath, map_location=device))
        amp = device.type == "cuda" and bool(json.loads((run_dir / (wpath.stem.replace("_model", "") + ".json")).read_text()).get("bf16"))
        out["weights"] = wpath.name
        out["rows"] = {}
        for L in ((512,) if smoke else (512, 2048, 8192)):
            if model.max_positions and L > model.max_positions:
                out["rows"][str(L)] = "n/a"
                continue
            row = {}
            for ctl in (None, "nohead"):
                row[ctl or "normal"] = trainer.update_accuracy(model, device, amp, L, LM06, 4 if smoke else 256, control=ctl)
            out["rows"][str(L)] = row
            log(f"lm06v L={L}: {row}")
        parts = []
        for L, r in out["rows"].items():
            if isinstance(r, dict):
                parts.append(f"{L}: keep {r['normal']['acc_unchanged']:.2f}->nohead {r['nohead']['acc_unchanged']:.2f}, "
                             f"upd {r['normal']['acc_updated']:.2f}")
        msg = f"{name:15s} lm06v      " + " | ".join(parts) + f" ({wpath.name})"

    elif stage in ("lm06", "lm06long", "lm06s2", "lm06s3", "lm06longs2", "lm06longs3"):
        from common.data import check_update
        from common.config import LM06_LONG, SEEDS
        for line in check_update():
            log(line)
        out.update(trainer.train_lm06(lambda: build(name), device, run_dir, log, smoke=smoke,
                                      overrides=LM06_LONG if stage.startswith("lm06long") else None, tag=stage,
                                      seed_offset=SEEDS.get(stage, 0)))
        parts = []
        for L, r in out["test"].items():
            parts.append(f"{L}: upd {r['acc_updated']:.2f} keep {r['acc_unchanged']:.2f} stale {r['stale_rate']:.2f}"
                         if isinstance(r, dict) else f"{L}: n/a")
        msg = f"{name:15s} {stage:10s} " + " | ".join(parts) + (" CAPPED" if out["capped"] else "")

    elif stage in ("lm05v2", "lm05v2long", "lm05v2s2", "lm05v2s3", "lm05v3", "lm05v3long",
                   "lm05v3s2", "lm05v3s3", "lm05v3longs2", "lm05v3s4", "lm05v3s5", "lm05v3s6", "lm05v3longs3"):
        from common.data import check_recall_v2, check_recall_v3
        from common.config import LM05V2_LONG, LM05V3, LM05V3_LONG, SEEDS
        for line in (check_recall_v3() if stage.startswith("lm05v3") else check_recall_v2()):
            log(line)
        if stage in ("lm05v3", "lm05v3s2", "lm05v3s3", "lm05v3s4", "lm05v3s5", "lm05v3s6"):
            ov = LM05V3
        elif stage in ("lm05v3long", "lm05v3longs2", "lm05v3longs3"):
            ov = dict(LM05V3, **LM05V3_LONG)
        else:
            ov = LM05V2_LONG if stage == "lm05v2long" else None
        out.update(trainer.train_lm05v2(lambda: build(name), device, run_dir, log, smoke=smoke, overrides=ov,
                                        seed_offset=SEEDS.get(stage, 0), tag=stage))
        parts = []
        for P, lv in out["levels"].items():
            t = lv["test"]
            parts.append(f"P{P}: " + " ".join(
                f"{L}:{r['normal']:.2f}/{r['keys']:.2f}" if isinstance(r, dict) and isinstance(r.get("normal"), float) else f"{L}:n/a"
                for L, r in t.items()) + (" CAPPED" if lv["capped"] else ""))
        msg = f"{name:15s} {stage:10s} normal/no-keys  " + "  |  ".join(parts)

    elif stage == "verify5":
        # control test for LM-05: the trained model must drop to chance when the answer is removed
        wpath = run_dir / "lm05_model.pt"
        if not wpath.exists():
            raise FileNotFoundError(f"{wpath} (run lm05 first)")
        model = build(name).to(device)
        model.load_state_dict(torch.load(wpath, map_location=device))
        r5 = json.loads((run_dir / "lm05.json").read_text())
        amp = bool(r5.get("bf16")) and device.type == "cuda"
        out["rows"] = {}
        for L in (512, 2048):
            row = {}
            for ctl in (None, "values", "keys"):
                acc, n = trainer.recall_accuracy(model, device, amp, L, 256, control=ctl, seed=7)
                row[ctl or "normal"] = acc
            out["rows"][str(L)] = row
            log(f"verify L={L}: {row}")
        r = out["rows"]
        msg = (f"{name:12s} verify5  normal {r['512']['normal']:.3f}/{r['2048']['normal']:.3f}  "
               f"no-values {r['512']['values']:.3f}/{r['2048']['values']:.3f}  "
               f"no-keys {r['512']['keys']:.3f}/{r['2048']['keys']:.3f}  (L=512/2048, chance 0.016)")

    elif stage == "lm05":
        model = build(name).to(device)
        out["params"] = n_params(model)
        out.update(trainer.train_lm05(model, device, run_dir, log, smoke=smoke))
        accs = " ".join(f"{k}:{v['acc']:.3f}" if isinstance(v, dict) else f"{k}:{v}" for k, v in out["test"].items())
        msg = (f"{name:12s} lm05     recall {accs} ({out['minutes']} min, bf16={out['bf16']}"
               f"{', CAPPED at step ' + str(out['steps']) if out['capped'] else ''})")

    elif stage in ("lm02", "lm02long"):
        if smoke:
            data = torch.randint(0, 256, (200_000,), dtype=torch.uint8)
            splits = dict(train=data, valid=data[:50_000], test=data[50_000:100_000])
        else:
            splits = load_enwik8()
            for line in check_enwik8(splits):
                log(line)
        model = build(name, max_len=MAX_LEN).to(device)
        out["params"] = n_params(model)
        from common.config import LM02_LONG
        out.update(trainer.train_lm02(model, splits, device, run_dir, log, smoke=smoke,
                                      overrides=LM02_LONG if stage == "lm02long" else None, tag=stage))
        msg = (f"{name:12s} {stage:8s} valid {out['valid_bpc']:.3f} test {out['test_bpc']:.3f} bpc, "
               f"{out['tokens'] / 1e6:.0f}M tokens{' CAPPED' if out['capped'] else ''} ({out['hours']} h)")
    else:
        raise ValueError(stage)

    out["seconds"] = round(time.time() - t0, 1)
    trainer.write_json(run_dir / f"{stage}{'_smoke' if smoke else ''}.json", out)
    summary(msg + (" [smoke]" if smoke else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES + EXTRA + ("all", "lm03", "lm03big", "lm03s2"))
    from models.seq2seq import S2S
    ap.add_argument("model", choices=sorted(REGISTRY) + sorted(S2S))
    ap.add_argument("--smoke", action="store_true", help="tiny budgets, random data: code path test only")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")

    def log(msg):
        print(f"{now()}  {msg}", flush=True)

    log(f"{a.model} {a.stage} on {device} (torch {torch.__version__})")
    if a.model in S2S:   # legacy encoder-decoder ports: only the LM-03 translation stage applies
        if a.stage not in ("lm03", "lm03big", "lm03s2"):
            summary(f"{a.model:15s} {a.stage:8s} SKIPPED: encoder-decoder models run only lm03")
            sys.exit(0)
        from common.lm03 import run_lm03
        run_dir = RUNS_DIR / a.model
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            out = run_lm03(a.model, device, run_dir, log, smoke=a.smoke, variant=a.stage)
        except Exception as e:
            traceback.print_exc()
            summary(f"{a.model:15s} {a.stage:8s} FAILED {type(e).__name__}: {str(e)[:150]}")
            sys.exit(1)
        out.update(model=a.model, stage=a.stage, smoke=a.smoke, torch=torch.__version__)
        trainer.write_json(run_dir / f"{a.stage}{'_smoke' if a.smoke else ''}.json", out)
        summary(f"{a.model:15s} {a.stage:8s} BLEU greedy {out['bleu_greedy']:.1f} beam{out['beam_width']} "
                f"{out['bleu_beam']:.1f} (greedy same subset {out['bleu_greedy_beam_subset']:.1f}), test ppl "
                f"{out['test_ppl']:.1f}, {out['params'] / 1e6:.0f}M params, {out['minutes']} min"
                + (" CAPPED" if out["capped"] else "") + (" [smoke]" if a.smoke else ""))
        sys.exit(0)
    stages = STAGES if a.stage == "all" else (a.stage,)
    code = 0
    for st in stages:
        try:
            out = run_stage(st, a.model, device, a.smoke, log)
        except Exception as e:
            traceback.print_exc()
            summary(f"{a.model:12s} {st:8s} FAILED {type(e).__name__}: {str(e)[:150]}")
            code = 1
            break
        if out.get("skipped"):
            break
        if a.stage == "all":
            if st == "check" and not (out["causality"]["causal"] and out["causality"]["uses_input"]):
                summary(f"{a.model:12s} all      stopped after check: not a valid causal LM, long runs skipped")
                break
            if st == "overfit" and not out["passed"]:
                summary(f"{a.model:12s} all      stopped after overfit FAIL, long runs skipped")
                break
        if device.type == "cuda":
            torch.cuda.empty_cache()
    sys.exit(code)


if __name__ == "__main__":
    main()
