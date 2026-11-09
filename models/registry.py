"""
Model registry for the Language Model stage (memory-first set, 13 models).

Every wrapper exposes the same interface:
    logits = model(x)            x: (B, T) bytes, logits: (B, T, 256)
    logits[:, t] predicts x[:, t + 1] and may depend only on x[:, :t + 1]
The causality probe (common/probes.py) checks the last sentence for every model.

Model code is vendored unchanged from the RichardMinsooGo repositories under
models/vendor/ (each keeps its upstream LICENSE). What the wrappers change is
listed in `unified` and copied into every result file.

Repository name rule: LM_<Enc|Dec|EncDec>_<Category>_<Model>
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from common.config import MODEL, VOCAB, MAX_LEN  # noqa: E402


class LM(nn.Module):
    """Common interface. Subclasses implement forward(x) -> logits."""

    def train_loss(self, x):
        """x: (B, T+1). Returns (loss to backpropagate, next-token cross-entropy in nats)."""
        logits = self(x[:, :-1])
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        return ce, ce


class Wrap(LM):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


# --------------------------------------------------------------------------- builders

def build_llama2(max_len):
    from llama2_rs.llama2 import Llama2Model
    cfg = dict(vocab_size=VOCAB, context_length=max_len, emb_dim=MODEL["dim"], n_heads=MODEL["heads"],
               n_layers=MODEL["depth"], hidden_dim=1408, dtype=torch.float32)
    return Wrap(Llama2Model(cfg))


def build_megabyte(max_len):
    from MEGABYTE_pytorch import MEGABYTE
    return Wrap(MEGABYTE(num_tokens=VOCAB, dim=(MODEL["dim"], 256), depth=(MODEL["depth"], 2),
                         max_seq_len=(max_len // 8, 8), dim_head=MODEL["dim_head"], heads=MODEL["heads"],
                         flash_attn=True))


def build_rq(max_len):
    from rq_transformer import RQTransformer
    return Wrap(RQTransformer(num_tokens=VOCAB, dim=MODEL["dim"], max_spatial_seq_len=max_len // 8,
                              depth_seq_len=8, spatial_layers=MODEL["depth"], depth_layers=2,
                              dim_head=MODEL["dim_head"], heads=MODEL["heads"]))


def build_hourglass(max_len):
    from hourglass_transformer_pytorch import HourglassTransformerLM
    return Wrap(HourglassTransformerLM(num_tokens=VOCAB, dim=MODEL["dim"], max_seq_len=max_len,
                                       depth=(2, 2, 2), shorten_factor=2, heads=MODEL["heads"],
                                       dim_head=MODEL["dim_head"]))


class SimpleHier(LM):
    """Uses the model's own training objective (CE + recon + hierarchical AR losses) for LM-02."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)

    def train_loss(self, x):
        total, (ce, *_rest) = self.inner(x, return_loss=True)
        return total, ce


def build_simple_hier(max_len):
    from simple_hierarchical_transformer import HierarchicalTransformer
    return SimpleHier(HierarchicalTransformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"],
                                              seq_len=max_len, hierarchies=(1, 2), window_sizes=(32, 64),
                                              dim_head=MODEL["dim_head"], heads=MODEL["heads"],
                                              ignore_index=-100, use_flash_attn=True))


def build_feedback(max_len):
    from feedback_transformer_pytorch import FeedbackTransformer
    return Wrap(FeedbackTransformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], seq_len=2,
                                    mem_len=256, heads=MODEL["heads"], dim_head=MODEL["dim_head"]))


class Memorizing(LM):
    """Processes the sequence in segments; kNN and XL memories carry information across segments."""

    def __init__(self, inner, segment):
        super().__init__()
        self.inner, self.segment = inner, segment

    def forward(self, x):
        outs, xl = [], None
        with self.inner.knn_memories_context(batch_size=x.shape[0]) as knn:
            for seg in x.split(self.segment, dim=1):
                logits, xl = self.inner(seg, knn_memories=knn, xl_memories=xl)
                outs.append(logits)
        return torch.cat(outs, dim=1)


def build_memorizing(max_len):
    from memorizing_transformers_pytorch import MemorizingTransformer
    inner = MemorizingTransformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"],
                                  dim_head=MODEL["dim_head"], heads=MODEL["heads"], memorizing_layers=4,
                                  max_knn_memories=max(max_len, 512) * 4, num_retrieved_memories=32,
                                  xl_memory_layers=(5, 6), xl_max_memories=128)
    return Memorizing(inner, segment=128)


def build_gmlp(max_len):
    from g_mlp_pytorch import gMLP
    return Wrap(gMLP(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], seq_len=max_len, causal=True))


def build_metaformer(max_len):
    from metaformer_gpt import MetaformerGPT
    return Wrap(MetaformerGPT(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], heads=MODEL["heads"],
                              dim_head=MODEL["dim_head"], max_seq_len=max_len))


class MambaLM(LM):
    """Embedding -> Mamba blocks (vendored) -> RMSNorm -> head. Replaces the Lightning MambaLLM wrapper."""

    def __init__(self):
        super().__init__()
        from mamba_pscan.mamba import Mamba, RMSNorm
        d = MODEL["dim"]
        self.emb = nn.Embedding(VOCAB, d)
        self.body = Mamba(num_layers=MODEL["depth"], d_input=d, d_model=2 * d, d_state=16, parallel=True)
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, x):
        h, _ = self.body(self.emb(x))
        return self.head(self.norm(h))


def build_mamba(max_len):
    return MambaLM()


def build_mega(max_len):
    from mega_pytorch import Mega
    return Wrap(Mega(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"]))


def build_gateloop(max_len):
    from gateloop_transformer import Transformer
    return Wrap(Transformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], dim_head=MODEL["dim_head"],
                            heads=MODEL["heads"], use_gate_looped_attn=True, gate_loop_heads=MODEL["dim"],
                            data_dependent_rel_pos=False, attn_softmax_normalize=True, post_ln_norm=True))


def build_perceiver(max_len):
    from perceiver_pytorch import PerceiverLM
    return Wrap(PerceiverLM(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], max_seq_len=max_len,
                            num_latents=256, latent_dim=MODEL["dim"], cross_heads=1, latent_heads=MODEL["heads"],
                            cross_dim_head=MODEL["dim_head"], latent_dim_head=MODEL["dim_head"]))


# --------------------------------------------------------------------------- phase 2 (pip packages, lucidrains)

class RMT(LM):
    """Recurrent Memory Transformer: segments of 128 tokens, memory tokens and XL memories carried over.
    detach=True cuts the gradient between segments (truncated backprop through the memory)."""

    def __init__(self, inner, segment, detach=False):
        super().__init__()
        self.inner, self.segment, self.detach = inner, segment, detach

    def forward(self, x):
        outs, mem, xl = [], None, None
        for seg in x.split(self.segment, dim=1):
            if self.detach and mem is not None:
                mem = mem.detach()
                xl = [t.detach() for t in xl] if xl is not None else None
            logits, mem, xl = self.inner(seg, mem, xl_memories=xl)
            outs.append(logits)
        return torch.cat(outs, dim=1)


def build_rmt(max_len, detach=False):
    from recurrent_memory_transformer_pytorch import RecurrentMemoryTransformer
    return RMT(RecurrentMemoryTransformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"],
                                          dim_head=MODEL["dim_head"], heads=MODEL["heads"], num_memory_tokens=16,
                                          seq_len=128), segment=128, detach=detach)


def build_rmt_detach(max_len):
    return build_rmt(max_len, detach=True)


class BRT(LM):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)[0]


def build_block_recurrent(max_len):
    from block_recurrent_transformer_pytorch import BlockRecurrentTransformer
    return BRT(BlockRecurrentTransformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"],
                                         dim_head=MODEL["dim_head"], heads=MODEL["heads"], max_seq_len=max(max_len, 8192),
                                         block_width=128, recurrent_layers=(4,)))


def build_mingru(max_len):
    from minGRU_pytorch.minLM import minLM
    return Wrap(minLM(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"]))


def build_minlstm(max_len):
    from minGRU_pytorch.minLM import minLM
    return Wrap(minLM(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], use_lstm=True))


# --------------------------------------------------------------------------- phase 4 (2026-10-26)
# classic recurrent baselines (PyTorch / cuDNN) and lucidrains LMs relevant to PASG memory

class RNNLM(LM):
    """Embedding -> stacked nn.LSTM / nn.GRU (cuDNN) -> LayerNorm -> head. Constant inference memory per token."""

    def __init__(self, cell, hidden=768, layers=4):
        super().__init__()
        d = MODEL["dim"]
        self.emb = nn.Embedding(VOCAB, d)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(d, hidden, num_layers=layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, VOCAB, bias=False)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(self.norm(h))


def build_lstm(max_len):
    return RNNLM("lstm")


def build_gru(max_len):
    return RNNLM("gru", hidden=896)


class Infini(LM):
    """Infini-attention: segments of 128 with a compressive memory carried across segments
    (memories detached every 2 segments, as the upstream wrapper does by default)."""

    def __init__(self, inner, segment=128):
        super().__init__()
        self.inner, self.segment = inner, segment

    def forward(self, x):
        from infini_transformer_pytorch import detach_memories_
        outs, mem = [], None
        for i, seg in enumerate(x.split(self.segment, dim=1)):
            r = self.inner(seg, past_memories=mem, return_new_memories=True)
            outs.append(r[0])
            mem = r[2]
            if i % 2 == 1:
                detach_memories_(mem)
        return torch.cat(outs, dim=1)


def build_infini(max_len):
    from infini_transformer_pytorch import InfiniTransformer
    return Infini(InfiniTransformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"],
                                    dim_head=MODEL["dim_head"], heads=MODEL["heads"]))


class NSA(LM):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x, disable_flex=True, disable_triton_kernel=True)


def build_nsa(max_len):
    from native_sparse_attention_pytorch.transformer import Transformer
    return NSA(Transformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], dim_head=MODEL["dim_head"],
                           heads=MODEL["heads"], kv_heads=MODEL["heads"] // 2, use_sparse_attn=True, causal=True,
                           sparse_attn_kwargs=dict(sliding_window_size=64, compress_block_size=16,
                                                   compress_block_sliding_stride=16, selection_block_size=16,
                                                   num_selected_blocks=4)))


class TaylorBlockLM(LM):
    """Based-style LM: pre-norm blocks of causal Taylor-series linear attention + feedforward (written here;
    the package provides the attention layer only). Linear attention has a constant-size recurrent state."""

    def __init__(self):
        super().__init__()
        from taylor_series_linear_attention import TaylorSeriesLinearAttn
        d = MODEL["dim"]
        self.emb = nn.Embedding(VOCAB, d)
        self.layers = nn.ModuleList([])
        for _ in range(MODEL["depth"]):
            self.layers.append(nn.ModuleList([
                TaylorSeriesLinearAttn(dim=d, dim_head=16, heads=MODEL["heads"], causal=True, rotary_emb=True,
                                       prenorm=True),
                nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
            ]))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for attn, ff in self.layers:
            h = attn(h) + h
            h = ff(h) + h
        return self.head(self.norm(h))


def build_based(max_len):
    return TaylorBlockLM()


def build_xt_alibi(max_len):
    from x_transformers import TransformerWrapper, Decoder
    return Wrap(TransformerWrapper(num_tokens=VOCAB, max_seq_len=max(max_len, 8192), use_abs_pos_emb=False,
                                   attn_layers=Decoder(dim=MODEL["dim"], depth=MODEL["depth"], heads=MODEL["heads"],
                                                       attn_dim_head=MODEL["dim_head"], alibi_pos_bias=True,
                                                       attn_flash=True)))


class XL(LM):
    """Transformer-XL recurrence via x-transformers: segments of 128, each attends to the cached
    (gradient-stopped) hidden states of the previous segment(s)."""

    def __init__(self, inner, segment=128):
        super().__init__()
        self.inner, self.segment = inner, segment

    def forward(self, x):
        outs, mems = [], None
        for seg in x.split(self.segment, dim=1):
            logits, mems = self.inner(seg, mems=mems, return_mems=True)
            outs.append(logits)
        return torch.cat(outs, dim=1)


def build_xt_xl(max_len):
    from x_transformers import TransformerWrapper, Decoder
    return XL(TransformerWrapper(num_tokens=VOCAB, max_seq_len=128, max_mem_len=256, use_abs_pos_emb=False,
                                 attn_layers=Decoder(dim=MODEL["dim"], depth=MODEL["depth"], heads=MODEL["heads"],
                                                     attn_dim_head=MODEL["dim_head"], rel_pos_bias=True)))


# --------------------------------------------------------------------------- phase 5 (2026-10-29)
# position-encoding ablation: the same x-transformers decoder as xt_alibi, only the position scheme changes.
# Question from round 4: is length generalisation of recall decided by the position representation?

def build_xt_pos(kind):
    def build(max_len):
        from x_transformers import TransformerWrapper, Decoder
        kw = dict(dim=MODEL["dim"], depth=MODEL["depth"], heads=MODEL["heads"], attn_dim_head=MODEL["dim_head"],
                  attn_flash=True)
        abs_pos = False
        if kind == "rope":
            kw["rotary_pos_emb"] = True
        elif kind == "t5":
            kw.update(rel_pos_bias=True, attn_flash=False)       # flash attention cannot add a learned bias
        elif kind == "dynpos":
            kw.update(dynamic_pos_bias=True, attn_flash=False)
        elif kind == "abs":
            abs_pos = True
        # kind == "nope": no position information at all (causal mask only)
        return Wrap(TransformerWrapper(num_tokens=VOCAB, max_seq_len=max_len if abs_pos else max(max_len, 8192),
                                       use_abs_pos_emb=abs_pos, attn_layers=Decoder(**kw)))
    return build


# --------------------------------------------------------------------------- phase 3 (2026-11-02): remaining 12 repos

class PadToLen(LM):
    """For models built for one fixed length (Linformer): right-pad to that length, then cut the logits."""

    def __init__(self, inner, length):
        super().__init__()
        self.inner, self.length = inner, length

    def forward(self, x):
        n = x.shape[1]
        if n < self.length:
            x = F.pad(x, (0, self.length - n))
        return self.inner(x)[:, :n]


def build_linformer(max_len):
    from linformer import LinformerLM
    return PadToLen(LinformerLM(num_tokens=VOCAB, dim=MODEL["dim"], seq_len=512, depth=MODEL["depth"], k=128,
                                heads=MODEL["heads"]), 512)


class BlocksLM(LM):
    """Embedding + learned positions + a sequence module from the repo + norm + head (for repos that ship only
    attention / encoder blocks: Nystromformer, Attention-on-Attention)."""

    def __init__(self, body, max_len):
        super().__init__()
        d = MODEL["dim"]
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(max_len, d)
        self.body = body
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        return self.head(self.norm(self.body(h)))


def build_nystrom(max_len):
    from nystrom_attention import Nystromformer
    return BlocksLM(Nystromformer(dim=MODEL["dim"], depth=MODEL["depth"], dim_head=MODEL["dim_head"],
                                  heads=MODEL["heads"], num_landmarks=64), max_len)


class AoABody(nn.Module):
    def __init__(self):
        super().__init__()
        from aoa_pytorch import AttentionOnAttention
        d = MODEL["dim"]
        self.layers = nn.ModuleList([nn.ModuleList([
            nn.LayerNorm(d), AttentionOnAttention(dim=d, dim_head=MODEL["dim_head"], heads=MODEL["heads"]),
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))])
            for _ in range(MODEL["depth"])])

    def forward(self, h):
        for n1, attn, ff in self.layers:
            h = attn(n1(h)) + h
            h = ff(h) + h
        return h


def build_aoa(max_len):
    return BlocksLM(AoABody(), max_len)


class FastLM(LM):
    """FastTransformer.forward fails when mask is None (einops on None); pass an all-True mask."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x, mask=torch.ones_like(x, dtype=torch.bool))


def build_fast(max_len):
    from fast_transformer_pytorch import FastTransformer
    return FastLM(FastTransformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], max_seq_len=max_len,
                                heads=MODEL["heads"], dim_head=MODEL["dim_head"], absolute_pos_emb=True))


def build_rela(max_len):
    from rela_transformer import ReLATransformer
    return Wrap(ReLATransformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], max_seq_len=max_len,
                                causal=True, heads=MODEL["heads"], dim_head=MODEL["dim_head"]))


def build_multistream(max_len):
    from multistream_transformers import MultistreamTransformer
    return Wrap(MultistreamTransformer(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], max_seq_len=max_len,
                                       causal=True, heads=MODEL["heads"], dim_head=MODEL["dim_head"], num_streams=2))


def build_palm(max_len):
    from palm_pytorch import PaLM
    return Wrap(PaLM(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], dim_head=MODEL["dim_head"],
                     heads=MODEL["heads"]))


def build_lamda(max_len):
    from lamda_rmg.model import LaMDA
    return Wrap(LaMDA(num_tokens=VOCAB, dim=MODEL["dim"], depth=MODEL["depth"], dim_head=MODEL["dim_head"],
                      heads=MODEL["heads"]))


def build_llama3(max_len):
    from llama3_rmg.model import Llama3Model
    cfg = dict(vocab_size=VOCAB, context_length=max_len, emb_dim=MODEL["dim"], n_heads=MODEL["heads"],
               n_layers=MODEL["depth"], hidden_dim=1408, n_kv_groups=2, rope_base=500_000.0, rope_freq=None,
               dtype=torch.float32)
    return Wrap(Llama3Model(cfg))


def build_mgqa(module, **attn_kw):
    def build(max_len):
        m = __import__(module, fromlist=["MGQATransformer", "Decoder"])
        return Wrap(m.MGQATransformer(num_tokens=VOCAB, max_seq_len=max_len,
                                      attn_layers=m.Decoder(dim=MODEL["dim"], depth=MODEL["depth"],
                                                            heads=MODEL["heads"], **attn_kw)))
    return build


def build_xt_rmg(max_len):
    from x_transformers_rmg import TransformerWrapper, Decoder
    return Wrap(TransformerWrapper(num_tokens=VOCAB, max_seq_len=max_len,
                                   attn_layers=Decoder(dim=MODEL["dim"], depth=MODEL["depth"], heads=MODEL["heads"])))


# --------------------------------------------------------------------------- phase 6 (2026-11-03): hybrid
# Round-5/6 finding: GRU tracks recent updates reliably at any distance; attention (ALiBi / NoPE) can retrieve
# distant items but learns it only in some runs. Hybrid: a GRU layer in front of the attention decoder.

class HybridGRUAttn(LM):
    def __init__(self, pos, cell="gru", window=None, gru=True):
        super().__init__()
        from x_transformers import Decoder
        d = MODEL["dim"]
        self.emb = nn.Embedding(VOCAB, d)
        self.gru = (nn.LSTM if cell == "lstm" else nn.GRU)(d, d, num_layers=1, batch_first=True) if gru else None
        kw = dict(dim=d, depth=MODEL["depth"], heads=MODEL["heads"], attn_dim_head=MODEL["dim_head"], attn_flash=True)
        if pos == "alibi":
            kw["alibi_pos_bias"] = True
        elif pos == "rope":
            kw["rotary_pos_emb"] = True
        if window:   # 2026-11-09: sliding-window attention (bounded KV cache); the recurrent layer carries the rest
            kw["attn_max_attend_past"] = window
        self.attn = Decoder(**kw)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, x):
        h = self.emb(x)
        if self.gru is not None:
            r, _ = self.gru(h)
            h = h + r
        h = self.attn(h)
        return self.head(self.norm(h))


def build_hyb(pos, cell="gru", window=None, gru=True):
    return lambda max_len: HybridGRUAttn(pos, cell, window, gru)


# --------------------------------------------------------------------------- registry

LUCID = "lucidrains/{} (MIT)"

REGISTRY = {
    "llama2": dict(
        max_positions=MAX_LEN,
        build=build_llama2, repo="LM_2023_llama2-pytorch", category="Base", model="LLaMA2",
        upstream="rasbt/LLMs-from-scratch Llama 2 conversion code (Apache-2.0); license file missing in repo - add",
        probe_max=MAX_LEN, phase=1, role="baseline",
        unified=["hidden_dim 1408 (about 8/3 x dim); dtype float32 weights, bf16 autocast"]),
    "megabyte": dict(
        max_positions=MAX_LEN,
        build=build_megabyte, repo="LM_2023_Megabyte-pytorch", category="Long", model="MEGABYTE",
        upstream=LUCID.format("MEGABYTE-pytorch"), probe_max=8192, phase=1, role="memory",
        unified=["2 stages dim (512, 256) depth (6, 2), patch 8 (upstream train.py: 3 stages, 768/512/256)"]),
    "rq": dict(
        max_positions=MAX_LEN,
        build=build_rq, repo="LM_2022_RQ-Transformer", category="Long", model="RQTransformer",
        upstream=LUCID.format("RQ-Transformer"), probe_max=8192, phase=1, role="memory",
        unified=["bytes reshaped to spatial x depth with depth 8; spatial 6 layers, depth 2 layers",
                 "BUG FIX in vendored forward (2026-10-21): outputs were shifted by one position per 8-byte group "
                 "(logits[t] did not use x[t]); results before the fix (bpc 4.65) are invalid and kept as *_offset_bug.json"]),
    "hourglass": dict(
        max_positions=MAX_LEN,
        build=build_hourglass, repo="LM_2021_hourglass-transformer-pytorch", category="Long", model="Hourglass",
        upstream=LUCID.format("hourglass-transformer-pytorch"), probe_max=8192, phase=1, role="memory",
        unified=["depth (2, 2, 2) instead of (4, 2, 4) to match the 6-layer budget"]),
    "simple_hier": dict(
        max_positions=MAX_LEN,
        build=build_simple_hier, repo="LM_2023_Simple-hierarchical-transformer", category="Long",
        model="SimpleHierarchical", upstream=LUCID.format("simple-hierarchical-transformer"), probe_max=8192,
        phase=1, role="memory",
        unified=["dim 512 depth 6 (upstream 1024/8); LM-02 uses its own loss (CE + recon + hierarchical AR), "
                 "LM-05 uses CE on answers only; ignore_index -100 so byte 0 is not ignored"]),
    "feedback": dict(
        build=build_feedback, repo="LM_2020_feedback-transformer", category="Long", model="Feedback",
        upstream=LUCID.format("feedback-transformer-pytorch"), probe_max=8192, phase=1, role="memory",
        micro_batch=2,
        unified=["seq_len (step) 2, mem_len 256 (upstream defaults)",
                 "trained in micro-batches of 2 with gradient accumulation (same update; the token-by-token "
                 "recurrence keeps a long graph and does not fit batch 16 x 512 in 8 GB)"]),
    "memorizing": dict(
        build=build_memorizing, repo="LM_2022_Memorizing-transformers-pytorch", category="Long",
        model="Memorizing", upstream=LUCID.format("memorizing-transformers-pytorch"), probe_max=8192,
        phase=1, role="memory",
        unified=["segments of 128 inside each 512 window (upstream: 5 segments of 512); kNN layer 4, "
                 "XL layers 5-6 with 128 memories; faiss-cpu instead of faiss-gpu"]),
    "gmlp": dict(
        max_positions=MAX_LEN,
        build=build_gmlp, repo="LM_2021_G-mlp-pytorch", category="AttnFree", model="gMLP",
        upstream=LUCID.format("g-mlp-pytorch"), probe_max=MAX_LEN, phase=1, role="memory",
        unified=["seq_len built at 2048 so LM-05 can test up to 2048; trained on 512"]),
    "metaformer": dict(
        max_positions=MAX_LEN,
        build=build_metaformer, repo="LM_2021_Metaformer-pytorch", category="AttnFree", model="MetaformerGPT",
        upstream=LUCID.format("metaformer-gpt"), probe_max=MAX_LEN, phase=1, role="memory",
        unified=["heads 8 dim_head 64 (upstream 16/32)"]),
    "mamba": dict(
        build=build_mamba, repo="LM_2023_Mamba_transformer-pytorch", category="AttnFree", model="Mamba",
        upstream="pscan from alxndrTL/mamba.py (MIT); Mamba blocks: source not stated in repo - confirm",
        probe_max=8192, phase=1, role="memory",
        micro_batch=8,
        unified=["LM head (embedding, final RMSNorm, linear) written here; Lightning MambaLLM not used; "
                 "d_model 1024, d_state 16, parallel scan",
                 "micro-batch 8 with gradient accumulation (same update): the pure-PyTorch scan keeps about "
                 "0.9 MB per token, so batch 32 x 512 exceeds 8 GB; on Windows this spilled into system RAM "
                 "and the first LM-05 run reached only 500 steps in 2 h (kept as lm05_capped_1019.json)",
                 "BUG FIX in vendored pscan (2026-10-19): backward returned gradients in the wrong layout and "
                 "used the already-scanned A, so gradients were wrong; also padded to power-of-two lengths. "
                 "Verified equal to the sequential scan (forward and gradients, error < 1e-14)"]),
    "mega": dict(
        build=build_mega, repo="LM_2022_Mega-pytorch", category="AttnFree", model="Mega",
        upstream=LUCID.format("Mega-pytorch"), probe_max=8192, phase=1, role="memory", micro_batch=4,
        unified=["depth 6 (upstream 8)",
                 "fp32 (its FFT does not support bf16) and micro-batch 4 with gradient accumulation (same update): "
                 "about 0.75 GB per 512-token sample in fp32; the first LM-05 run spilled past 8 GB and stalled"]),
    "gateloop": dict(
        build=build_gateloop, repo="LM_2023_GateLoop-transformer-pytorch", category="AttnFree", model="GateLoop",
        upstream=LUCID.format("gateloop-transformer"), probe_max=8192, phase=1, role="memory",
        unified=["upstream train.py hparams (gate_loop_heads 512, post-LN)"]),
    "perceiver": dict(
        max_positions=MAX_LEN,
        build=build_perceiver, repo="LM_2021_Perceiver-pytorch", category="Long", model="Perceiver",
        upstream=LUCID.format("perceiver-pytorch"), probe_max=MAX_LEN, phase=1, role="memory",
        unified=["PerceiverLM from perceiver_io.py (as used in 2021_Perceiver_dec_train.py)"]),
    "rmt": dict(
        build=build_rmt, repo="(new) lucidrains/recurrent-memory-transformer-pytorch 0.7.0", category="Long",
        model="RecurrentMemory", upstream=LUCID.format("recurrent-memory-transformer-pytorch") + ", pip 0.7.0",
        probe_max=8192, phase=2, role="memory",
        unified=["segments of 128 with 16 memory tokens and XL memories; gradients flow across segments "
                 "(upstream wrapper can also do memory-replay backprop, not used)"]),
    "block_recurrent": dict(
        build=build_block_recurrent, repo="(new) lucidrains/block-recurrent-transformer-pytorch 0.4.4",
        category="Long", model="BlockRecurrent",
        upstream=LUCID.format("block-recurrent-transformer-pytorch") + ", pip 0.4.4", probe_max=8192, phase=2,
        role="memory", unified=["block_width 128, recurrent layer 4"]),
    "mingru": dict(
        build=build_mingru, repo="(new) lucidrains/minGRU-pytorch 0.2.1", category="AttnFree", model="minGRU",
        upstream=LUCID.format("minGRU-pytorch") + ", pip 0.2.1", probe_max=8192, phase=2, role="memory",
        unified=["minLM defaults (expansion 1.5, ff_mult 4, no conv)"]),
    "minlstm": dict(
        build=build_minlstm, repo="(new) lucidrains/minGRU-pytorch 0.2.1", category="AttnFree", model="minLSTM",
        upstream=LUCID.format("minGRU-pytorch") + ", pip 0.2.1", probe_max=8192, phase=2, role="memory",
        unified=["minLM with use_lstm=True"]),
    "rmt_detach": dict(
        build=build_rmt_detach, repo="(new) lucidrains/recurrent-memory-transformer-pytorch 0.7.0", category="Long",
        model="RecurrentMemory", upstream=LUCID.format("recurrent-memory-transformer-pytorch") + ", pip 0.7.0",
        probe_max=8192, phase=3, role="stability test",
        unified=["stability variant (a): memory and XL memories detached between segments (truncated backprop); "
                 "the first rmt LM-02 run diverged after 16M tokens (valid bpc 2.94 -> 4.2)"]),
    "rmt_lr1e4": dict(
        build=build_rmt, repo="(new) lucidrains/recurrent-memory-transformer-pytorch 0.7.0", category="Long",
        model="RecurrentMemory", upstream=LUCID.format("recurrent-memory-transformer-pytorch") + ", pip 0.7.0",
        probe_max=8192, phase=3, role="stability test", lr=1e-4,
        unified=["stability variant (b): learning rate 1e-4 instead of 2e-4, gradients through memory kept"]),
    "lstm": dict(
        build=build_lstm, repo="(new) Legacy baseline, PyTorch nn.LSTM", category="Base", model="LSTM",
        upstream="torch.nn.LSTM (cuDNN); classic baseline requested 2026-10-22", probe_max=8192, phase=4,
        role="classic memory baseline", unified=["4 layers, hidden 768, input dim 512, LayerNorm before head"]),
    "gru": dict(
        build=build_gru, repo="(new) Legacy baseline, PyTorch nn.GRU", category="Base", model="GRU",
        upstream="torch.nn.GRU (cuDNN); classic baseline requested 2026-10-22", probe_max=8192, phase=4,
        role="classic memory baseline", unified=["4 layers, hidden 896 (to match LSTM size), input dim 512"]),
    "infini": dict(
        build=build_infini, repo="(new) lucidrains/infini-transformer-pytorch 0.2.1", category="Long",
        model="InfiniAttention", upstream=LUCID.format("infini-transformer-pytorch") + ", pip 0.2.1", probe_max=8192, phase=4,
        role="compressive memory", unified=["segments of 128, compressive memory detached every 2 segments"]),
    "nsa": dict(
        build=build_nsa, repo="(new) lucidrains/native-sparse-attention-pytorch 0.2.3", category="Attention",
        model="NativeSparseAttention", upstream=LUCID.format("native-sparse-attention-pytorch") + ", pip 0.2.3", probe_max=8192,
        phase=4, role="long-context sparse attention",
        unified=["sliding window 64, compress block 16, select 4 blocks of 16, kv heads 4; pure PyTorch path "
                 "(flex attention and Triton kernels are not available on Windows)"]),
    "xt_alibi": dict(
        build=build_xt_alibi, repo="(new) lucidrains/x-transformers 2.31.7", category="Base", model="ALiBi",
        upstream=LUCID.format("x-transformers") + ", pip 2.31.7", probe_max=8192, phase=4, role="length extrapolation",
        unified=["decoder with ALiBi position bias, no absolute positions (tests length generalisation)"]),
    "xt_xl": dict(
        build=build_xt_xl, repo="(new) lucidrains/x-transformers 2.31.7", category="Long", model="TransformerXL",
        upstream=LUCID.format("x-transformers") + ", pip 2.31.7", probe_max=8192, phase=4, role="segment recurrence",
        unified=["segments of 128 with 256 cached memories, relative position bias"]),
    "xt_rope": dict(
        build=build_xt_pos("rope"), repo="(new) lucidrains/x-transformers 2.31.7", category="Base", model="RoPE",
        upstream=LUCID.format("x-transformers") + ", pip 2.31.7", probe_max=8192, phase=5, role="position ablation",
        unified=["same decoder as xt_alibi; rotary position embedding (as in LLaMA)"]),
    "xt_nope": dict(
        build=build_xt_pos("nope"), repo="(new) lucidrains/x-transformers 2.31.7", category="Base", model="NoPE",
        upstream=LUCID.format("x-transformers") + ", pip 2.31.7", probe_max=8192, phase=5, role="position ablation",
        unified=["same decoder as xt_alibi; no position information (causal mask only)"]),
    "xt_abs": dict(
        build=build_xt_pos("abs"), repo="(new) lucidrains/x-transformers 2.31.7", category="Base", model="AbsPos",
        upstream=LUCID.format("x-transformers") + ", pip 2.31.7", probe_max=8192, phase=5, role="position ablation",
        max_positions=MAX_LEN,
        unified=["same decoder as xt_alibi; learned absolute positions built for 2048"]),
    "xt_t5": dict(
        build=build_xt_pos("t5"), repo="(new) lucidrains/x-transformers 2.31.7", category="Base", model="T5RelBias",
        upstream=LUCID.format("x-transformers") + ", pip 2.31.7", probe_max=8192, phase=5, role="position ablation",
        unified=["same decoder as xt_alibi; T5 bucketed relative position bias; non-flash attention (bias not supported by flash)"]),
    "xt_dynpos": dict(
        build=build_xt_pos("dynpos"), repo="(new) lucidrains/x-transformers 2.31.7", category="Base", model="DynamicPosBias",
        upstream=LUCID.format("x-transformers") + ", pip 2.31.7", probe_max=8192, phase=5, role="position ablation",
        unified=["same decoder as xt_alibi; dynamic position bias (small MLP over distance); non-flash attention"]),
    "linformer": dict(
        build=build_linformer, repo="LM_2020_Linformer", category="Attention", model="Linformer",
        upstream="vendored from RichardMinsooGo/LM_2020_Linformer (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        max_positions=MAX_LEN,
        unified=["fixed length 512 (inputs right-padded), k 128"]),
    "nystrom": dict(
        build=build_nystrom, repo="LM_2021_Nystrom-transformer", category="Attention", model="Nystromformer",
        upstream="vendored from RichardMinsooGo/LM_2021_Nystrom-transformer (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        max_positions=MAX_LEN,
        unified=["Nystromformer blocks + embedding/positions/head written here, 64 landmarks"]),
    "fast": dict(
        build=build_fast, repo="LM_2021_fast-transformer", category="Attention", model="Fastformer",
        upstream="vendored from RichardMinsooGo/LM_2021_fast-transformer (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        max_positions=MAX_LEN,
        unified=["absolute positions; all-True mask passed (forward fails with mask=None)"]),
    "rela": dict(
        build=build_rela, repo="LM_2021_ReLA-transformer-pytorch", category="Attention", model="ReLA",
        upstream="vendored from RichardMinsooGo/LM_2021_ReLA-transformer-pytorch (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        max_positions=MAX_LEN,
        unified=["causal=True"]),
    "aoa": dict(
        build=build_aoa, repo="LM_2021_Attention_on_Attention-pytorch", category="Attention", model="AttentionOnAttention",
        upstream="vendored from RichardMinsooGo/LM_2021_Attention_on_Attention-pytorch (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        max_positions=MAX_LEN,
        unified=["AoA attention blocks + FF/embedding/head written here (module has no causal mask)"]),
    "gqa": dict(
        build=build_mgqa('gqa_rmg.model', attn_kv_heads=2), repo="LM_2023_Grouped_Query_Attention-pytorch", category="Attention", model="GQA",
        upstream="vendored from RichardMinsooGo/LM_2023_Grouped_Query_Attention-pytorch (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        max_positions=MAX_LEN,
        unified=["2 key/value heads for 8 query heads"]),
    "mqa": dict(
        build=build_mgqa('mqa_rmg.model', attn_kv_heads=1), repo="LM_2023_Mutli_Query_Attention_transformer", category="Attention", model="MQA",
        upstream="vendored from RichardMinsooGo/LM_2023_Mutli_Query_Attention_transformer (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        max_positions=MAX_LEN,
        unified=["1 key/value head"]),
    "multistream": dict(
        build=build_multistream, repo="LM_2021_multistream-transformers", category="Attention", model="Multistream",
        upstream="vendored from RichardMinsooGo/LM_2021_multistream-transformers (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        max_positions=MAX_LEN,
        unified=["2 streams, causal=True"]),
    "palm": dict(
        build=build_palm, repo="LM_2022_PaLM", category="Base", model="PaLM",
        upstream="vendored from RichardMinsooGo/LM_2022_PaLM (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        unified=["parallel blocks, rotary, SwiGLU (upstream defaults)"]),
    "lamda": dict(
        build=build_lamda, repo="LM_2022_LaMDA-pytorch", category="Base", model="LaMDA",
        upstream="vendored from RichardMinsooGo/LM_2022_LaMDA-pytorch (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        unified=["T5 relative position bias (upstream)"]),
    "llama3": dict(
        build=build_llama3, repo="LM_2024_llama3-pytorch", category="Base", model="LLaMA3",
        upstream="vendored from RichardMinsooGo/LM_2024_llama3-pytorch (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        max_positions=MAX_LEN,
        unified=["2 kv groups, rope base 500k, hidden 1408; hard-coded bfloat16 casts removed from the vendored file"]),
    "xt_repo": dict(
        build=build_xt_rmg, repo="LM_2023_x-transformers-pytorch", category="Base", model="XTransformers",
        upstream="vendored from RichardMinsooGo/LM_2023_x-transformers-pytorch (see LICENSE in vendor folder if present)", probe_max=MAX_LEN,
        phase=3, role="standardisation",
        max_positions=MAX_LEN,
        unified=["the repository's own x-transformers copy, library defaults (absolute positions)"]),
    "hyb_gru_alibi": dict(
        build=build_hyb("alibi"), repo="(new) hybrid: GRU layer + x-transformers decoder", category="Long",
        model="HybridGRU_ALiBi", upstream=LUCID.format("x-transformers") + ", pip 2.31.7 + torch.nn.GRU", probe_max=8192, phase=6, role="hybrid memory",
        unified=["1 GRU layer (dim 512) added residually before a 6-layer alibi decoder (same decoder as xt_alibi)"]),
    "hyb_gru_nope": dict(
        build=build_hyb("nope"), repo="(new) hybrid: GRU layer + x-transformers decoder", category="Long",
        model="HybridGRU_NoPE", upstream=LUCID.format("x-transformers") + ", pip 2.31.7 + torch.nn.GRU", probe_max=8192, phase=6, role="hybrid memory",
        unified=["1 GRU layer (dim 512) added residually before a 6-layer nope decoder (same decoder as xt_nope)"]),
    "hyb_gru_rope": dict(
        build=build_hyb("rope"), repo="(new) hybrid: GRU layer + x-transformers decoder", category="Long",
        model="HybridGRU_RoPE", upstream=LUCID.format("x-transformers") + ", pip 2.31.7 + torch.nn", probe_max=8192, phase=7, role="hybrid ablation",
        unified=["ablation of hyb_gru_nope: same GRU layer, decoder with rotary positions (does GRU fix RoPE length generalisation?)"]),
    "hyb_lstm_nope": dict(
        build=build_hyb("nope", "lstm"), repo="(new) hybrid: LSTM layer + x-transformers decoder", category="Long",
        model="HybridLSTM_NoPE", upstream=LUCID.format("x-transformers") + ", pip 2.31.7 + torch.nn", probe_max=8192, phase=7, role="hybrid ablation",
        unified=["ablation of hyb_gru_nope: LSTM instead of GRU in front of the NoPE decoder"]),
    "hyb_gru_nope_w256": dict(
        build=build_hyb("nope", window=256), repo="(new) hybrid with sliding-window attention", category="Long",
        model="HybridGRU_NoPE_W256", upstream=LUCID.format("x-transformers") + ", pip 2.31.7 + torch.nn", probe_max=8192, phase=8, role="bounded KV cache",
        unified=["hyb_gru_nope with attention limited to the last 256 tokens"]),
    "hyb_gru_nope_w1024": dict(
        build=build_hyb("nope", window=1024), repo="(new) hybrid with sliding-window attention", category="Long",
        model="HybridGRU_NoPE_W1024", upstream=LUCID.format("x-transformers") + ", pip 2.31.7 + torch.nn", probe_max=8192, phase=8, role="bounded KV cache",
        unified=["hyb_gru_nope with attention limited to the last 1024 tokens"]),
    "hyb_gru_rope_w256": dict(
        build=build_hyb("rope", window=256), repo="(new) hybrid with sliding-window attention", category="Long",
        model="HybridGRU_RoPE_W256", upstream=LUCID.format("x-transformers") + ", pip 2.31.7 + torch.nn", probe_max=8192, phase=8, role="bounded KV cache",
        unified=["hyb_gru_rope with attention limited to the last 256 tokens"]),
    "xt_rope_w256": dict(
        build=build_hyb("rope", window=256, gru=False), repo="(new) hybrid with sliding-window attention", category="Long",
        model="RoPE_W256", upstream=LUCID.format("x-transformers") + ", pip 2.31.7 + torch.nn", probe_max=8192, phase=8, role="bounded KV cache",
        unified=["control: RoPE decoder limited to the last 256 tokens, no recurrent layer"]),
}


def build(name, max_len=MAX_LEN):
    entry = REGISTRY[name]
    model = entry["build"](max_len)
    model.name = name
    model.micro_batch = entry.get("micro_batch")   # None = whole batch at once
    model.lr = entry.get("lr")                     # None = unified learning rate
    # 2026-10-26: longest sequence the built model can take. Models with learned/fixed position tables built at
    # MAX_LEN must not be run longer: on CUDA an out-of-range position index is a device-side assert that kills
    # the whole process (it cannot be caught like a normal error).
    model.max_positions = entry.get("max_positions")
    return model


def repo_name(name):
    e = REGISTRY[name]
    return f"LM_Dec_{e['category']}_{e['model']}"


def n_params(model):
    return sum(p.numel() for p in model.parameters())
