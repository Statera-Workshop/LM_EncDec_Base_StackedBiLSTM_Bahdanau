"""
LM-03 data: English -> French sentence pairs (manythings.org / Tatoeba, CC-BY 2.0 FR), prepared as in the
legacy TF2 code (C0-C9): 5-11 words on both sides, duplicates dropped, 12,800 pairs sampled with seed 1234,
80/20 train/test split, word-level vocabulary with <pad> <start> <end> <unk>.

Two fixes against the legacy code, recorded as [Unified]:
  - the vocabulary is built from the training split only (legacy built it from all pairs, leaking test words)
  - "what's" is expanded to "what is" (legacy wrote "that is")
"""
import random
import re
import unicodedata
import urllib.request
import zipfile

import torch

from common.config import DATA_DIR

URL = "https://www.manythings.org/anki/fra-eng.zip"
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<start>", "<end>", "<unk>"]

CONTRACTIONS = [
    (r"i'm", "i am"), (r"he's", "he is"), (r"she's", "she is"), (r"it's", "it is"), (r"that's", "that is"),
    (r"what's", "what is"), (r"where's", "where is"), (r"how's", "how is"), (r"\'ll", " will"),
    (r"\'ve", " have"), (r"\'re", " are"), (r"\'d", " would"), (r"won't", "will not"), (r"can't", "cannot"),
    (r"n't", " not"), (r"n'", "ng"),
]


def _ascii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def clean(sentence, english):
    s = sentence.lower().strip()
    if english:
        for a, b in CONTRACTIONS:
            s = re.sub(a, b, s)
    else:
        s = _ascii(s)
    s = re.sub(r"([?.!,])", r" \1 ", s)
    s = re.sub(r"[^a-z?.!,'\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _raw_pairs():
    txt = DATA_DIR / "fra.txt"
    if not txt.exists():
        zp = DATA_DIR / "fra-eng.zip"
        if not zp.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r, open(zp, "wb") as f:
                    f.write(r.read())
            except Exception as e:
                raise RuntimeError(f"could not download {URL} ({e}). Download it in a browser and put "
                                   f"fra-eng.zip or fra.txt into {DATA_DIR}") from e
        with zipfile.ZipFile(zp) as z:
            z.extract("fra.txt", DATA_DIR)
    pairs = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
    return pairs


class Vocab:
    def __init__(self, sentences):
        words = sorted({w for s in sentences for w in s.split()})
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, s):
        return [SOS] + [self.stoi.get(w, UNK) for w in s.split()] + [EOS]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i not in (PAD, SOS):
                out.append(self.itos[i])
        return out

    def __len__(self):
        return len(self.itos)


def load_pairs(n=12_800, seed=1234, smoke=False, dedup_clean=False):
    if smoke:   # code-path test without the download
        rnd = random.Random(0)
        words = ["the", "cat", "dog", "is", "big", "small", "red", "a", "house", "runs"]
        fr = ["le", "chat", "chien", "est", "grand", "petit", "rouge", "un", "maison", "court"]
        pairs = []
        for _ in range(400):
            idx = [rnd.randrange(10) for _ in range(rnd.randint(5, 8))]
            pairs.append((" ".join(words[i] for i in idx), " ".join(fr[i] for i in idx)))
    else:
        raw = _raw_pairs()
        seen_s, seen_t, seen_clean, pairs = set(), set(), set(), []
        for s, t in raw:
            if s in seen_s or t in seen_t:
                continue
            seen_s.add(s)
            seen_t.add(t)
            if 4 < len(s.split()) < 12 and 4 < len(t.split()) < 12:
                cs = clean(s, True)
                # 2026-11-09: different raw sentences can become the same after cleaning (case, contractions,
                # punctuation); with 100k pairs this put identical sources in train and test. The legacy-size
                # runs (12,800 pairs) had no overlap and keep the original sampling.
                if dedup_clean:
                    if cs in seen_clean:
                        continue
                    seen_clean.add(cs)
                pairs.append((cs, clean(t, False)))
        random.Random(seed).shuffle(pairs)
        pairs = pairs[:n]
    cut = int(len(pairs) * 0.8)
    train, test = pairs[:cut], pairs[cut:]
    src_vocab = Vocab([s for s, _ in train])
    trg_vocab = Vocab([t for _, t in train])
    return train, test, src_vocab, trg_vocab


def check_pairs(train, test, sv, tv):
    lines = [f"LM-03 pairs: train {len(train)}, test {len(test)}; vocab src {len(sv)}, trg {len(tv)}"]
    tr_src = {s for s, _ in train}
    overlap = sum(1 for s, _ in test if s in tr_src)
    lines.append(f"LM-03 test sources also in train: {overlap} (duplicates were dropped, expect 0)")
    unk = sum(1 for _, t in test for w in t.split() if w not in tv.stoi)
    lines.append(f"LM-03 test target words unknown to the train vocabulary: {unk}")
    if overlap:
        raise RuntimeError("test sources overlap the training sources")
    return lines


def batches(pairs, sv, tv, batch, shuffle, generator=None):
    order = list(range(len(pairs)))
    if shuffle:
        order = torch.randperm(len(pairs), generator=generator).tolist()
    for i in range(0, len(order), batch):
        chunk = [pairs[j] for j in order[i:i + batch]]
        src = [sv.encode(s) for s, _ in chunk]
        trg = [tv.encode(t) for _, t in chunk]
        S, T = max(map(len, src)), max(map(len, trg))
        xs = torch.full((len(chunk), S), PAD, dtype=torch.long)
        ys = torch.full((len(chunk), T), PAD, dtype=torch.long)
        for k, (a, b) in enumerate(zip(src, trg)):
            xs[k, :len(a)] = torch.tensor(a)
            ys[k, :len(b)] = torch.tensor(b)
        yield xs, ys, chunk
