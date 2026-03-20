# Day 21 — Project: Text Generation with LSTM (Shakespeare)
**Date:** March 20, 2026
**Notebook:** `day21_text_generation.ipynb`
**Dataset:** Shakespeare complete works (~1.1M characters)
**Machine:** Windows RTX 4070 ⚡ CUDA
**Paper:** Karpathy — "The Unreasonable Effectiveness of Recurrent Neural Networks"

---

## What Was Built

- Character-level LSTM language model trained on full Shakespeare corpus
- Stacked 3-layer LSTM with embed_size=128, hidden_size=512
- Stateful autoregressive text generation with seed priming
- Temperature-controlled sampling
- ReduceLROnPlateau scheduler + early checkpoint saving
- Loss curve + PPL visualization

---

## Results

```
Best val loss: 1.258
Best val PPL:  3.5
Training time: ~3s per epoch on RTX 4070
```

### Perplexity comparison across all days

| Model | Val PPL |
|-------|---------|
| Bigram (Day 20) | 1766 |
| Trigram (Day 20) | 4564 |
| LSTM Mac 200k tokens (Day 20) | 721 |
| CharLSTM Windows full corpus (Day 21) | **3.5** |

200x improvement over the Day 20 Mac LSTM — entirely explained by more data, bigger model, and CUDA.

---

## Core Concepts

### Character-level vs Word-level Language Modeling

**Word-level:**
```
vocab size: 10,000–50,000 words
input:      "the" → "cat" → "sat"
strength:   semantic meaning, larger context per step
weakness:   can't handle unseen words (OOV problem)
            large embedding tables
            can't generate new words
```

**Character-level:**
```
vocab size: ~65 characters (letters + punctuation + spaces)
input:      "t" → "h" → "e" → " " → "c" → "a" → "t"
strength:   tiny vocab, no OOV problem, can invent new words
            learns spelling, morphology automatically
weakness:   sequences are much longer
            harder to learn long range meaning
            more steps to generate same amount of text
```

The tradeoff is vocabulary size vs sequence length. Character models are simpler to set up and can generalize to any text, but need more steps to learn the same semantic content.

### Architecture — Stacked LSTM

```
input (batch, seq_len=100)
→ Embedding (65 → 128)
→ Dropout(0.3)
→ LSTM layer 1  (128 → 512)   learns: letter combos, "th", "ing"
→ Dropout(0.3)
→ LSTM layer 2  (512 → 512)   learns: word patterns, common phrases
→ Dropout(0.3)
→ LSTM layer 3  (512 → 512)   learns: sentence structure, style
→ Linear (512 → 65)           predicts next character
```

Hidden state shape with num_layers=3: `(3, batch, 512)` — one h and c per layer.

### init_hidden

```python
def init_hidden(self, batch_size):
    return (
        torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
        torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
    )
```

Explicitly creates `(h_0, c_0)` on the correct device. Each layer gets its own hidden state. Previously used `h=None` which lets PyTorch default to zeros — same thing, but explicit device placement matters for CUDA.

---

## Data Pipeline

### make_batches

```python
n_seqs = (len(encoded) - 1) // seq_len   # 11,153 sequences
X_all  = zeros(n_seqs, seq_len)           # (11153, 100)
y_all  = zeros(n_seqs, seq_len)           # (11153, 100) — shifted by 1

perm   = randperm(n_seqs)                 # shuffle SEQUENCE order
split  = int(0.9 * n_seqs)               # 90/10 train/val

# result:
X_train: (10037, 100)
X_val:   (1116,  100)
```

The `-1` in `n_seqs` calculation is a safety buffer — the last target needs to read one char past the last input position.

### Two-stage shuffling

**Stage 1 (make_batches):** shuffles which sequences go into train vs val — runs once.

**Stage 2 (training loop):** `randperm` every epoch shuffles the order the model sees training sequences — runs 30 times. Without this the model would memorize the batch order, not the data.

### Manual mini-batching

```python
perm = torch.randperm(X_train.shape[0])
for i in range(0, X_train.shape[0], BATCH_SIZE):
    idx     = perm[i:i + BATCH_SIZE]      # 128 random indices
    X_batch = X_train[idx].to(device)     # slice those rows
    y_batch = y_train[idx].to(device)
```

Equivalent to DataLoader with shuffle=True. Preferred when data is already a tensor in memory — avoids Dataset wrapper overhead.

---

## Training

### ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)
scheduler.step(avg_val)   # called every epoch
```

Monitors val loss. If it doesn't improve for 3 consecutive epochs, halves the learning rate. Solves the problem of a fixed LR that's too large to find a precise minimum late in training — instead of stopping (early stopping), it slows down and keeps looking.

---

## Stateful Generation

### Priming the hidden state

```python
# run entire seed through at once — builds up h
_, h = model(seed_tensor)

# then generate one char at a time, passing h forward
for _ in range(length):
    logits, h = model(x, h)        # h carries full context
    next_char = sample(logits)
    x = tensor([[next_char]])      # feed prediction back
```

Training throws away h between batches (`_, h = model(x)` discards h). Generation passes h forward continuously — the model remembers the full seed + everything generated so far through the hidden state.

### Temperature sampling

```python
logits    = logits[0, -1, :] / temperature   # shape: (vocab,)
probs     = F.softmax(logits, dim=0)
next_char = torch.multinomial(probs, 1).item()
```

`torch.multinomial` samples weighted by probability — not argmax. `.item()` extracts the raw Python int from the tensor so it can be used as a dict key in `idx2char`.

---

## Generated Text Samples

**Seed:** `"ROMEO:\nWhat light through yonder window breaks"`

```
temp 0.5: "LUCENTIO: The market-place, I will be so."
          "ARIEL: Every word is of the book of the spirits"
          → clean sentences, real character names, play format correct

temp 0.8: "LEONTES: I am required from me To any exchange"
          → mostly grammatical, Shakespeare vocabulary intact

temp 1.0: "BAPTISTA: Sir Hach, Paulina, you do, as it were valiant"
          → meaning dissolving but words are real

temp 1.2: "cardant-lank", "avained", "uaked"
          → inventing words, structure breaking down
```

---

## What the model learned (no words, no grammar — only characters)

```
✅ play format: "CHARACTER:\nDialogue"
✅ real Shakespeare character names
✅ archaic vocabulary: hath, thou, darest, ta'en
✅ punctuation patterns: commas, semicolons, stage directions
✅ sentence-level structure
❌ long range coherence — characters say disconnected things
❌ plot awareness — no memory across sequences
```

---

## Reflection Questions

**1. Character-level vs word-level — tradeoffs**

Character-level models have a tiny vocabulary (~65 chars) so the embedding table is small, there's no out-of-vocabulary problem, and they can generate novel words. The tradeoff is that sequences are much longer — to say "the cat sat" you need 11 character steps vs 3 word steps. This makes it harder to learn long-range dependencies since the same number of LSTM steps covers much less semantic ground. Word-level models capture meaning faster per step but need large vocabularies and can't handle unseen words at all. For creative text generation, character-level is often preferred because it can invent words and handle any character set. For meaning-heavy tasks like translation, word-level (or subword) is better.

**2. Stateful generation — why pass h at inference but not training**

During training batches are independent — each batch starts fresh with h=None because you don't care about coherence across batches, you just care about the loss within each sequence. The LSTM never needs to remember across batch boundaries to compute correct gradients. During generation you're producing one continuous sequence and every character prediction depends on everything that came before it — the seed text, the characters already generated, the overall style. Passing h forward is what gives the model that persistent memory. Without it, the model would forget the seed after the first step and start generating incoherently from blank context every step.

**3. Temperature observations**

temp=0.5 felt like the most structured output — clean sentences, real character names placed correctly, proper play formatting throughout. The model was playing it very safe, always picking the statistically most common next character, which happens to be correct punctuation and common words. temp=1.0 felt like the model was trying to be creative but losing the thread — individual words were real Shakespeare vocabulary but sentences started contradicting themselves or trailing off. temp=1.2 was the most chaotic — the model started inventing words like "cardant-lank" and "avained" that don't exist but sound vaguely Elizabethan. There's a sweet spot around 0.8 where you get coherent sentences with some variation.

**4. ReduceLROnPlateau**

A fixed learning rate has a fundamental problem — the same step size that's good for making fast progress early in training is too large for fine-tuning late in training. You overshoot the minimum and oscillate around it instead of settling in. ReduceLROnPlateau watches val loss and whenever it stops improving for `patience=3` epochs, it multiplies the LR by `factor=0.5`. This is adaptive — the model gets to make big steps when there's a lot to learn and small precise steps when it's close to the optimum. The advantage over early stopping is that instead of giving up when progress stalls, it slows down and tries harder.

**5. Best val PPL — better or worse than Day 20?**

Best val PPL was 3.5 — dramatically better than everything from Day 20. The bigram got 1766, the trigram got 4564, and the Day 20 Mac LSTM got 721. The Day 21 CharLSTM is 200x better than the Day 20 LSTM on the same architecture. The difference is entirely data and compute — 1.1M characters vs 200k tokens, hidden_size 512 vs 256, 3 layers vs 2, and CUDA vs MPS. This is the scaling law in action: the same architecture with more data and more capacity predictably produces better results. PPL 3.5 means the model is effectively choosing between 3.5 characters at each step — on a vocab of 65 that means it's eliminated 95% of the uncertainty.

**6. Karpathy's blog — most surprising thing**

Karpathy's "Unreasonable Effectiveness of RNNs" demonstrated that a character-level LSTM trained on raw text could produce outputs that looked superficially like LaTeX mathematical papers — complete with \begin{} and \end{} blocks, theorem environments, and fake but plausible-looking equations. It had never been taught LaTeX syntax explicitly — it just learned the structural patterns from character co-occurrence statistics. Similarly it generated Linux kernel code with correct C syntax, function signatures, and comment styles. The most surprising thing was that the model learned hierarchical structure — matching brackets, consistent indentation, opening and closing XML tags — purely from seeing enough examples at the character level. Structure that humans think of as requiring semantic understanding emerged from pure pattern matching over characters.

**7. What "learning" means for a neural network**

This model has no concept of words. It has never been told what a noun or verb is. It doesn't know that "Romeo" is a person or that Shakespeare lived in the 16th century. It only ever saw a sequence of integers representing characters and learned to predict the next integer. Yet it produced character names, play structure, archaic vocabulary, and grammatically plausible sentences. What this tells you is that "learning" for a neural network is fundamentally about **compressing statistical regularities**. Every pattern that appears consistently in the training data — "ROMEO:" always followed by a newline and then words, "thou" more likely after "dost", uppercase letters more likely after newlines — gets encoded as a numerical relationship in the weights. The model doesn't understand Shakespeare. It has memorized the distribution of characters in Shakespeare so precisely that sampling from that distribution produces text that looks Shakespearean. Intelligence and pattern compression are much closer to the same thing than we intuitively expect.

---

## Bugs / Notes

- `make_batches` takes `batch_size` as argument but doesn't use it — batching happens manually in the training loop with `randperm` instead
- `logits[0, -1, :]` during generation: `[0]` = first batch, `[-1]` = last timestep, `[:]` = all vocab logits — collapses `(1, 1, vocab)` to `(vocab,)`
- `.item()` on `multinomial` output needed to convert tensor → plain int for `idx2char` dict lookup

---
