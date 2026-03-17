# Day 19 — Word Embeddings + Word2Vec
**Date:** March 17, 2026
**Notebook:** `day19_word_embeddings.ipynb`
**Dataset:** text8 (500k tokens, Wikipedia)
**Machine:** Mac MPS
**Paper:** Word2Vec — Mikolov et al. (2013)

---

## What Was Built

- Skip-gram Word2Vec from scratch in PyTorch
- Negative sampling loss function (manual implementation)
- PCA visualization of learned embeddings
- Cosine similarity and word analogy functions
- Trained on real text8 data (500k tokens, ~20k vocab)

---

## Core Concepts

### Why Embeddings Over One-Hot Vectors

One-hot vectors are binary sparse vectors — every word gets a vector of zeros with a single 1. A vocab of 10,000 words means 10,000-dimensional vectors. They have zero semantic information — `king` and `queen` are as different as `king` and `pizza`.

Embeddings are dense low-dimensional vectors (e.g. 100 dimensions) where **similar words have similar vectors**. The distance between vectors encodes meaning. This is the property one-hot vectors completely lack.

### Skip-gram Objective

Given a **center word**, predict its surrounding **context words** within a window.

```
sentence: "the cat sat on the mat"
center:   "sat"  (window=2)
predict:  "the", "cat", "on", "mat"
```

The model is not predicting the next word — it's predicting *any* word that appears nearby. Every word gets a turn as center, and context pairs become training data.

### Negative Sampling

Without negative sampling, the model cheats — it can make all scores huge and the loss goes to zero without learning anything meaningful.

Negative sampling adds **fake pairs** alongside real ones:
- Real pair `(sat, cat)` → score should be HIGH
- Fake pairs `(sat, king)`, `(sat, run)` → score should be LOW

The model now has to push real pairs up AND fake pairs down simultaneously. It turns Word2Vec into a **binary classifier**: real pair = 1, fake pair = 0.

Typical: 5–20 negatives per real pair.

### The Loss Function

```python
loss = -log(sigmoid(real_score))
       - Σ log(sigmoid(-neg_score))
```

**Why sigmoid:** converts any score to (0, 1) — a probability.

**Why log:** punishes confident wrong answers exponentially harder than uncertain ones.
```
-log(0.99) = 0.01   # confident and correct  → tiny loss
-log(0.50) = 0.69   # uncertain              → medium loss
-log(0.01) = 4.60   # confident and wrong    → huge loss
```

**The sign flip trick:** `-log(sigmoid(-neg_score))` — for fake pairs, a low score is correct. But the loss function only rewards high inputs. Flipping the sign means a low neg_score becomes a high input, so the model gets rewarded for correctly identifying fake pairs.

### nn.Embedding — Just a Lookup Table

`nn.Embedding(vocab_size, embed_size)` is a matrix where each row is a word's vector. Pass in an integer index, get back that row. It learns because **backprop adjusts the rows** — words that appear in similar contexts get nudged toward each other over thousands of updates. The meaning is not programmed in; it emerges from co-occurrence statistics.

---

## Architecture

```
Two embedding tables (both vocab_size × embed_size):
  center_embed   ← what we keep after training
  context_embed  ← discarded after training

forward(center, context):
  c_emb   = center_embed[center]      # row lookup
  ctx_emb = context_embed[context]    # row lookup
  score   = (c_emb * ctx_emb).sum(dim=1)  # dot product → scalar
  return score
```

The dot product score = how confident the model is that these two words co-occur.

---

## Key Code Patterns

### Building Skip-gram Pairs
```python
def build_skipgram_pairs(corpus, word2idx, window_size=2):
    pairs = []
    for sentence in corpus:
        tokens = tokenize(sentence)
        idxs = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
        for i, center in enumerate(idxs):
            left  = max(0, i - window_size)
            right = min(len(idxs), i + window_size + 1)
            for j in range(left, right):
                if j != i:
                    pairs.append((center, idxs[j]))
    return pairs
```

### Negative Sampling Loss
```python
def negative_sampling_loss(model, center, context, n_neg=5):
    batch_size = center.shape[0]
    pos_score  = model(center, context)
    pos_loss   = F.logsigmoid(pos_score)

    neg_words = torch.randint(0, VOCAB_SIZE, (batch_size, n_neg)).to(device)
    neg_loss  = torch.zeros(batch_size).to(device)

    for k in range(n_neg):
        neg_score  = model(center, neg_words[:, k])
        neg_loss  += F.logsigmoid(-neg_score)   # sign flip here

    return -(pos_loss + neg_loss).mean()
```

---

## Training Results

| Epoch | Loss |
|-------|------|
| 1/5   | 1.5206 |
| 2/5   | 1.3155 |
| 3/5   | 1.2307 |
| 4/5   | 1.1722 |
| 5/5   | 1.1299 |

Clean monotonic decrease. No bouncing. Loss in the 0.8–1.1 range is normal for Word2Vec on limited data — the loss number matters less than whether embeddings are geometrically meaningful.

---

## Similarity Results

```
computer → desktop(0.57), apple(0.56), macintosh(0.53)   ✅ real structure
river    → valley(0.48), mississippi(0.47), delaware(0.46) ✅ geography cluster
king     → afghana(0.52), solomon(0.44), viii(0.44)        → Wikipedia "King X" pattern
```

`computer → macintosh` is the key win — the model learned tech co-occurrence purely from text statistics.

`king` not returning `queen` is expected: text8 is Wikipedia where "king" appears mostly as a title before proper nouns (`King Solomon`, `King VIII`), not next to royal family words.

---

## PCA Visualization Results

**Technology (purple)** — tightest cluster. `desktop, apple, computer, microsoft, software` all grouped right. Makes sense: these words appear in highly specific, consistent tech contexts in Wikipedia.

**Geography (brown)** — clean cluster. `mountain, river, forest, valley, island` grouped together, `ocean` slightly separated.

**Royalty (red)** — visible cluster. `emperor, king, dynasty, prince, queen` grouped.

**Politics (gray)** — partially separated. `president, government, parliament` close, `minister` and `election` drifting.

---

## Reflection Questions

**1. Why are word embeddings better than one-hot vectors?**

One-hot vectors treat every word as completely independent — there is no notion of similarity. The dot product of any two one-hot vectors is always 0 (they share no dimensions). Embeddings encode meaning geometrically: similar words have high cosine similarity because they've been pushed together by training on co-occurrence. The property one-hot vectors completely lack is **semantic similarity** — they cannot represent that "cat" and "dog" are more related to each other than "cat" and "democracy".

**2. Skip-gram objective in your own words**

The model is trained to predict which words tend to appear near a given center word. Given "sat", it should assign high probability to words like "cat", "on", "the" — words that actually appear nearby in real text. It's not predicting the next word in sequence; it's predicting any word that co-occurs within a window. The embeddings that enable this prediction end up encoding semantic similarity as a side effect.

**3. What is negative sampling and why is it needed?**

Without it, the model cheats. If you only show it real pairs and push scores up, the model can maximize scores by making all embedding vectors very large — every dot product becomes huge, loss goes to zero, but the embeddings are meaningless. Negative sampling forces the model to also push scores down for fake pairs (randomly sampled words that didn't appear near the center). Now the model must genuinely discriminate between real and fake co-occurrences, which forces the embeddings to become meaningful.

**4. What does "king - man + woman ≈ queen" mean geometrically?**

It means meaning is encoded as **directions in vector space**. The vector difference `king - man` captures the concept of "royalty without gender". Adding `woman` then moves along the gender axis to land near `queen`. This tells us the embedding space has learned to separate different semantic dimensions (royalty vs. gender vs. profession etc.) into different geometric directions. Meaning isn't stored in any single dimension — it's distributed across the whole vector, but the arithmetic works because similar concepts occupy similar regions of the space.

**5. PCA plot — which group clustered tightest and why?**

Technology had the tightest cluster. This is because tech words (`computer`, `apple`, `microsoft`, `desktop`) appear in very specific, consistent Wikipedia contexts — they co-occur with the same vocabulary repeatedly. This gives the model a strong, consistent signal. Royalty words appear in more varied historical contexts across many centuries and cultures, so their signal is noisier and the cluster is looser.

**6. How does nn.Embedding learn anything if it's just a lookup table?**

The lookup table is differentiable — backprop can compute gradients with respect to the specific rows that were looked up in a forward pass. When the loss says the score for `(sat, cat)` should be higher, backprop flows back through the dot product to the specific rows for "sat" and "cat" in both embedding tables and nudges those numbers. After thousands of such nudges across the whole corpus, the numbers in the table stop being random and start encoding semantic relationships. The table is simple; the learning comes from the accumulated effect of millions of gradient updates.

**7. What would have happened with one-hot vectors in Seq2Seq and Attention?**

Several problems. First, the input dimensionality would be vocab_size (potentially 10,000+) instead of embed_size (100–300) — much heavier computation. Second and more critically, the model would have no way to generalize across similar words. Every word would start as completely orthogonal to every other word. The encoder hidden state would have to learn word similarity from scratch purely from sequence context, with no prior geometric structure to build on. Training would be much slower, require far more data, and the model would struggle to generalize to unseen word combinations. Learned embeddings give the model a head start — words that are semantically similar already have similar representations before training even begins.

---

## Bugs Fixed Today

- `for i in enumerate(idxs)` → `for i, center in enumerate(idxs)` (enumerate returns tuple)
- `if j != 1` → `if j != i` (was hardcoded, skipped wrong position)

---

