# Transformer for Machine Translation (EN → DE)

A from-scratch implementation of the Transformer architecture
([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) for English-to-German
machine translation, including a custom Word2Vec Skip-Gram model
implemented from scratch in NumPy.

> **Scope.** This is a teaching-style implementation focused on the architecture
> itself — multi-head attention, masking, positional encoding, training loop,
> greedy decoding — trained on a small subset of Europarl (25k sentence pairs)
> on a single Colab GPU. It is not intended to compete with production MT
> systems, which use orders of magnitude more data and parameters.

---

## Results

| Metric                     | Value                  |
| -------------------------- | ---------------------- |
| BLEU score (validation)    | **3.23**               |
| Final train / val loss     | 4.01 / 4.61            |
| Training data              | 22,500 sentence pairs  |
| Validation data            | 2,500 sentence pairs   |
| Source / target vocab size | 14,590 / 20,000        |
| Model parameters           | ~7.9 million           |
| Training time              | ~15 min on a Colab T4 GPU |

> Tokenization: lowercase + whitespace split. BLEU computed on a 200-sample
> subset of the validation set using greedy decoding.

The training loss keeps decreasing while the validation loss flattens out
around epoch 7 — a clear sign of mild overfitting on this small data subset.

### Example Translations

| English (input) | German (model output) | German (reference) |
| --------------- | --------------------- | ------------------ |
| Madam President, Mr Prodi, Europe needs a strong Commission. | frau präsidentin, herr prodi, muß ein europa `<unk>` | (reference missing in dataset) |
| Is that not the lesson of the European Union itself and of the single market in particular? | das ist nicht der europäischen union und der union in den `<unk>` | Belegen dies nicht auch die Europäische Union selbst und ganz besonders der Binnenmarkt? |
| The Council of Ministers, however, is still putting a spanner in the works. | die ratsverordnung ist jedoch nicht `<unk>` | Der Ministerrat blockiert jedoch noch immer. |

The model has clearly learned German morphology, vocabulary and basic
clause structure, but with this dataset size it cannot produce fully
fluent translations and falls back to `<unk>` for rare words.

---

## Architecture

**Word2Vec Skip-Gram (Part 1) — implemented from scratch**

| Hyperparameter   | Value  |
| ---------------- | ------ |
| Embedding dim    | 50     |
| Window size      | 2      |
| Max vocab size   | 5,000  |
| Min frequency    | 2      |
| Learning rate    | 0.05   |
| Epochs           | 3      |
| Training pairs   | 50,000 |

Includes manual softmax, gradient computation, parameter updates, and 2D
PCA visualization of the learned embeddings.

**Transformer Seq2Seq (Part 2) — built on `torch.nn.Transformer`**

| Hyperparameter      | Value         |
| ------------------- | ------------- |
| Embedding dim       | 128           |
| Attention heads     | 4             |
| Encoder / decoder layers | 2 / 2    |
| Feed-forward dim    | 512           |
| Dropout             | 0.1           |
| Batch size          | 32            |
| Optimizer           | Adam, lr=5e-4 |
| Gradient clipping   | 1.0           |
| Epochs              | 10            |
| Loss                | CrossEntropy (ignore PAD) |

Sinusoidal positional encoding, causal masking for the decoder, padding
masks for both encoder and decoder, and greedy decoding for inference.

---

## Possible Improvements

- Train on the full Europarl corpus (~1.9M pairs) instead of 25k
- Subword tokenization (BPE / SentencePiece) to reduce `<unk>` tokens
- Larger model: more layers, higher embedding dim, more attention heads
- Beam search instead of greedy decoding
- Learning-rate warmup + inverse-square-root schedule (as in the original paper)
- Label smoothing in the loss
- Early stopping based on validation BLEU instead of fixed epoch count

---

## Dataset

[Europarl v7](https://www.statmt.org/europarl/) German–English parallel corpus.
A subset of **25,000 sentence pairs** is used (90 % train / 10 % validation,
seed 42).

The raw files are not included in the repository.
Download them manually:

```bash
wget https://www.statmt.org/europarl/v7/de-en.tgz
tar -xzf de-en.tgz
```

You should end up with `europarl-v7.de-en.en` and `europarl-v7.de-en.de`.

---

## Setup

```bash
# clone the repo
git clone https://github.com/hkuschnarev/transformer-translation.git
cd transformer-translation

# install dependencies
pip install -r requirements.txt

# launch the notebook
jupyter notebook transformer_translation.ipynb
```

A GPU is strongly recommended for the Transformer part (Google Colab works
out of the box).

---

## Tech Stack

Python · PyTorch · NumPy · NLTK · Matplotlib · scikit-learn (PCA) ·
Jupyter Notebook

---

## Project Context

Originally developed as part of the *Natural Language Processing* course at
Technische Hochschule Ingolstadt (B.Sc. Artificial Intelligence).
