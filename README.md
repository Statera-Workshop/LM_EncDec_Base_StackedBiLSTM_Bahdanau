# LM_EncDec_Base_StackedBiLSTM_Bahdanau

Encoder-decoder translation model ported from legacy TF2 code C7 to PyTorch (cell=lstm, attention=bahdanau, enc_layers=2, dec_layers=2, bi_layers=2).

## Status

- Verification level: **Benchmarked (LM-03, one run)**

## Changes from the legacy code

- PyTorch port; beam search reimplemented (TF Addons is no longer maintained)
- Vocabulary built from the training split only (legacy built it from all pairs)
- "what's" expanded to "what is" (legacy wrote "that is")

## LM-03 setting

English -> French, 5-11 words, 12,800 pairs (seed 1234), 80/20 split, embedding 128, hidden 1024, batch 256, 20 epochs, Adam 1e-3. Small data: compare structures, not absolute BLEU.

## Results

- Parameters: 88.4 M
- Test perplexity 40.7, next-word accuracy 0.562
- BLEU-4 greedy (whole test split): 20.0
- BLEU-4 beam width 4 on 500 pairs: 20.7 (greedy on the same pairs 19.7)
- Examples:
  - `this winter has been mild .` -> `cette histoire a ete prise .` (reference `cet hiver a ete doux .`)
  - `they will be here at three .` -> `ils pourraient etre ici a cette vie .` (reference `ils seront ici a trois heures .`)
  - `i once lived in rome .` -> `j'ai besoin de retour en cours .` (reference `j'ai vecu un temps a rome .`)

## Run

```bash
pip install torch==2.5.1 einops numpy
# put fra-eng.zip from https://www.manythings.org/anki/ into data/
python run.py lm03 s2s_bilstm
```
