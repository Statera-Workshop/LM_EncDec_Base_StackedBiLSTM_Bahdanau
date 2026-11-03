"""
PyTorch ports of the legacy TF2 encoder-decoder models (RichardMinsooGo, C0-C8), one configurable class.

  C0/C1  basic encoder-decoder (GRU / LSTM), the encoder's final state initialises the decoder
  C2/C4  + Bahdanau (additive) attention: context from the previous decoder state, fed with the input
  C3/C5  + Luong attention (dot / general / concat): context from the current decoder output,
         attentional hidden h~ = tanh(Wc [c; h]) predicts the next word
  C7     stacked bidirectional RNN encoder + Bahdanau
  C8     Google NMT: first encoder layer bidirectional, residual connections from layer 3, attention
         computed from the bottom decoder layer and fed to every decoder layer

Shapes: src (B, S), trg_in (B, T) -> logits (B, T, V_trg). Beam search lives in models/decoding.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0


class Attention(nn.Module):
    def __init__(self, kind, q_dim, k_dim, units):
        super().__init__()
        self.kind = kind
        if kind == "bahdanau" or kind == "concat":
            self.Wq = nn.Linear(q_dim, units, bias=False)
            self.Wk = nn.Linear(k_dim, units, bias=False)
            self.v = nn.Linear(units, 1, bias=False)
        elif kind == "general":
            self.W = nn.Linear(k_dim, q_dim, bias=False)
        elif kind == "dot":
            assert q_dim == k_dim
        self.keys_cache = None

    def forward(self, q, keys, mask):
        """q (B, Dq), keys (B, S, Dk), mask (B, S) True = real token -> context (B, Dk), weights (B, S)."""
        if self.kind in ("bahdanau", "concat"):
            score = self.v(torch.tanh(self.Wq(q)[:, None] + self.Wk(keys))).squeeze(-1)
        elif self.kind == "general":
            score = torch.einsum("bd,bsd->bs", q, self.W(keys))
        else:
            score = torch.einsum("bd,bsd->bs", q, keys)
        score = score.masked_fill(~mask, float("-inf"))
        w = score.softmax(-1)
        return torch.einsum("bs,bsd->bd", w, keys), w


class Encoder(nn.Module):
    def __init__(self, vocab, emb, hid, cell, layers=1, bi_layers=0, residual_from=None):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.cell, self.layers, self.bi_layers, self.residual_from = cell, layers, bi_layers, residual_from
        self.rnns = nn.ModuleList()
        d = emb
        for i in range(layers):
            bi = i < bi_layers
            self.rnns.append(rnn(d, hid, batch_first=True, bidirectional=bi))
            d = 2 * hid if bi else hid
        self.out_dim = d
        # a bidirectional last state is projected back to hid for the decoder (as in C7/C8: Dense on concat(fw,bw))
        self.bridge_h = nn.Linear(2 * hid, hid) if bi_layers else None
        self.bridge_c = nn.Linear(2 * hid, hid) if bi_layers and cell == "lstm" else None

    def forward(self, src):
        x = self.emb(src)
        states = []
        for i, rnn in enumerate(self.rnns):
            y, st = rnn(x)
            if self.residual_from is not None and i >= self.residual_from and y.shape[-1] == x.shape[-1]:
                y = y + x
            x = y
            states.append(st)
        return x, states

    def init_decoder(self, states, dec_layers):
        """Final encoder states -> initial decoder state for dec_layers layers (last layers aligned)."""
        hs, cs = [], []
        for st, rnn in zip(states, self.rnns):
            h, c = (st if self.cell == "lstm" else (st, None))
            if rnn.bidirectional:
                h = self.bridge_h(torch.cat([h[0], h[1]], -1))[None]
                if c is not None:
                    c = self.bridge_c(torch.cat([c[0], c[1]], -1))[None]
            hs.append(h)
            cs.append(c)
        hs = (hs * dec_layers)[-dec_layers:] if len(hs) < dec_layers else hs[-dec_layers:]
        cs = (cs * dec_layers)[-dec_layers:] if len(cs) < dec_layers else cs[-dec_layers:]
        h = torch.cat(hs, 0)
        return (h, torch.cat(cs, 0)) if self.cell == "lstm" else h


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, trg_vocab, cell="gru", attention=None, emb=128, hid=1024, enc_layers=1,
                 dec_layers=1, bi_layers=0, residual_from=None, gnmt=False):
        super().__init__()
        self.cell, self.attn_kind, self.gnmt, self.dec_layers, self.hid = cell, attention, gnmt, dec_layers, hid
        self.encoder = Encoder(src_vocab, emb, hid, cell, enc_layers, bi_layers, residual_from)
        self.emb = nn.Embedding(trg_vocab, emb, padding_idx=PAD)
        k_dim = self.encoder.out_dim
        rnn = nn.LSTMCell if cell == "lstm" else nn.GRUCell
        luong = attention in ("dot", "general", "concat")
        self.attn = Attention(attention if attention else "none", hid, k_dim, hid) if attention else None
        if attention == "dot" and k_dim != hid:
            raise ValueError("dot attention needs encoder output size == hidden size")
        ctx = k_dim if attention else 0
        self.cells = nn.ModuleList()
        for i in range(dec_layers):
            if i == 0:
                d_in = emb + (ctx if (attention == "bahdanau" or gnmt) else 0) + (hid if luong else 0)
            else:
                d_in = hid + (ctx if gnmt else 0)
            self.cells.append(rnn(d_in, hid))
        self.residual_from = residual_from
        self.Wc = nn.Linear(hid + ctx, hid, bias=False) if luong else None   # Luong attentional hidden
        self.out = nn.Linear(hid + (ctx if attention == "bahdanau" and not gnmt else 0), trg_vocab)

    # ---- one decoding step (used by teacher forcing, greedy and beam search)
    def step(self, y_prev, state, keys, mask, feed):
        """y_prev (B,), state: list per layer, feed: previous attentional hidden (Luong input feeding) or None."""
        x = self.emb(y_prev)
        luong = self.attn_kind in ("dot", "general", "concat")
        h_top_prev = state[-1][0] if self.cell == "lstm" else state[-1]
        ctx = None
        if self.attn_kind == "bahdanau" and not self.gnmt:
            ctx, _ = self.attn(h_top_prev, keys, mask)
            x = torch.cat([x, ctx], -1)
        if luong:
            x = torch.cat([x, feed], -1)
        new_state = []
        for i, cell in enumerate(self.cells):
            if self.gnmt and i == 0:
                h0_prev = state[0][0] if self.cell == "lstm" else state[0]
                ctx, _ = self.attn(h0_prev, keys, mask)
                inp = torch.cat([x, ctx], -1)
            elif self.gnmt:
                inp = torch.cat([x, ctx], -1)
            else:
                inp = x
            st = cell(inp, state[i])
            h = st[0] if self.cell == "lstm" else st
            if self.residual_from is not None and i >= self.residual_from and h.shape == x.shape:
                h = h + x
            new_state.append(st)
            x = h
        if luong:
            ctx, _ = self.attn(x, keys, mask)
            x = torch.tanh(self.Wc(torch.cat([ctx, x], -1)))
            feed = x
            logits = self.out(x)
        elif self.attn_kind == "bahdanau" and not self.gnmt:
            logits = self.out(torch.cat([x, ctx], -1))
        else:
            logits = self.out(x)
        return logits, new_state, feed

    def start(self, src):
        keys, states = self.encoder(src)
        mask = src != PAD
        init = self.encoder.init_decoder(states, self.dec_layers)
        if self.cell == "lstm":
            state = [(init[0][i], init[1][i]) for i in range(self.dec_layers)]
        else:
            state = [init[i] for i in range(self.dec_layers)]
        feed = keys.new_zeros(src.shape[0], self.hid) if self.attn_kind in ("dot", "general", "concat") else None
        return keys, mask, state, feed

    def forward(self, src, trg_in):
        keys, mask, state, feed = self.start(src)
        outs = []
        for t in range(trg_in.shape[1]):
            logits, state, feed = self.step(trg_in[:, t], state, keys, mask, feed)
            outs.append(logits)
        return torch.stack(outs, 1)


S2S = {   # legacy file -> configuration. Repository names follow LM_EncDec_<Category>_<Model>.
    "s2s_gru":            dict(src="C0", model="Seq2SeqGRU", cfg=dict(cell="gru")),
    "s2s_lstm":           dict(src="C1", model="Seq2SeqLSTM", cfg=dict(cell="lstm")),
    "s2s_bahdanau_gru":   dict(src="C2", model="BahdanauGRU", cfg=dict(cell="gru", attention="bahdanau")),
    "s2s_luong_gru":      dict(src="C3", model="LuongGRU", cfg=dict(cell="gru", attention="general")),
    "s2s_bahdanau_lstm":  dict(src="C4", model="BahdanauLSTM", cfg=dict(cell="lstm", attention="bahdanau")),
    "s2s_luong_lstm":     dict(src="C5", model="LuongLSTM", cfg=dict(cell="lstm", attention="general")),
    "s2s_luong_lstm_dot": dict(src="C5 (dot)", model="LuongLSTM_dot", cfg=dict(cell="lstm", attention="dot")),
    "s2s_luong_lstm_concat": dict(src="C5 (concat)", model="LuongLSTM_concat",
                                  cfg=dict(cell="lstm", attention="concat")),
    "s2s_bilstm":         dict(src="C7", model="StackedBiLSTM_Bahdanau",
                               cfg=dict(cell="lstm", attention="bahdanau", enc_layers=2, dec_layers=2, bi_layers=2)),
    "s2s_gnmt":           dict(src="C8", model="GNMT",
                               cfg=dict(cell="lstm", attention="bahdanau", enc_layers=4, dec_layers=4, bi_layers=1,
                                        residual_from=2, gnmt=True)),
}


def build_s2s(name, src_vocab, trg_vocab):
    return Seq2Seq(src_vocab, trg_vocab, **S2S[name]["cfg"])
