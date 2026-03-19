# Day 20 — Language Modeling Basics + Perplexity
**Date:** March 19, 2026
**Notebook:** `day20_language_modeling.ipynb`
**Dataset:** text8 (200k tokens, Wikipedia)
**Machine:** Mac MPS
**Paper:** Scaling Laws — Kaplan et al. (2020) intro

---

## What Was Built

- N-gram language model (bigram + trigram) with Laplace smoothing from scratch
- LSTM language model with temperature-controlled text generation
- Perplexity comparison across all three models
- Early stopping to prevent overfitting
- Loss curve + perplexity bar chart visualization

---

## Core Concepts

### What a Language Model Does

A language model assigns a probability to any sequence of words. It does this by decomposing the sequence using the **chain rule of probability** — predicting each word given everything that came before it:

```
P("the cat sat") = P("the")
                 × P("cat" | "the")
                 × P("sat" | "the", "cat")
```

At every step it produces a probability distribution over the entire vocabulary. The model is computing: *"given everything I've seen so far, how likely is each possible next word?"*

### Perplexity

Perplexity is the standard metric for evaluating language models. The intuition is:

> **Perplexity ≈ how many words the model is choosing between at each step on average.**

```
PPL = 1      → perfect model, always certain
PPL = 10     → effectively choosing between 10 options per step
PPL = 100    → choosing between 100 options — decent model
PPL = 1000   → nearly random — almost no useful signal learned
PPL = vocab  → completely random model
```

Formally:
```
Perplexity = exp(cross_entropy_loss)
           = exp(-(1/N) Σ log P(wᵢ | context))
```

### Why exp(cross_entropy)?

Cross entropy loss is the average negative log probability per word:
```
CE = -(1/N) Σ log P(wᵢ | context)
```

Log probabilities are negative numbers (since probabilities are between 0 and 1). A better model assigns higher probabilities → less negative logs → lower CE loss.

The exp reverses the log and converts back to probability space, recovering the "effective number of choices" interpretation:

```
if model assigns p = 1.0  → -log(1.0) = 0    → exp(0)    = 1     choice
if model assigns p = 0.5  → -log(0.5) = 0.69 → exp(0.69) = 2     choices
if model assigns p = 0.1  → -log(0.1) = 2.30 → exp(2.30) = 10    choices
if model assigns p = 0.01 → -log(0.01)= 4.60 → exp(4.60) = 100   choices
```

The exponential is the right function because log and exp are inverses — exp(CE) cleanly undoes the log and gives you back a number in the "how many choices" space rather than the "bits of information" space.

### The Chain Rule in Language Modeling

```
P("the cat sat on mat")
  = P("the")
  × P("cat" | "the")
  × P("sat" | "the", "cat")
  × P("on"  | "the", "cat", "sat")
  × P("mat" | "the", "cat", "sat", "on")
```

Each term is a conditional probability. The model scores the entire sentence by multiplying these together (or summing in log space for numerical stability). You can use this to compare any two sentences — a good LM assigns much higher probability to grammatical, coherent text than to word salad.

---

## N-gram Language Model

### How It Works

N-gram models approximate the chain rule by truncating context to the last n-1 words:

```
Bigram  (n=2): P(word | previous 1 word)
Trigram (n=3): P(word | previous 2 words)
```

The model counts every context → next_word pair in training data and converts to probabilities.

### Laplace Smoothing

Without smoothing, any unseen n-gram gets probability 0 — and one zero in the chain rule collapses the entire sentence probability to 0, making perplexity infinite.

Laplace smoothing adds a small value (0.1) to every count:
```
P(word | context) = (count(context, word) + 0.1) / (count(context) + 0.1 × vocab_size)
```

This ensures no probability is ever exactly zero.

### Key Code Pattern

```python
# counting
context = tuple(tokens[i:i + n - 1])   # tuple — hashable, can be dict key
word    = tokens[i + n - 1]
counts[context][word] += 1

# probability with smoothing
count = counts[context][word] + smoothing
total = sum(counts[context].values()) + smoothing * VOCAB_SIZE
prob  = count / total
```

---

## LSTM Language Model

### Architecture

```
input (batch, seq_len)
→ Embedding (vocab → embed_size)
→ Dropout
→ LSTM (embed_size → hidden_size, num_layers=2)
→ Dropout
→ Linear (hidden_size → vocab_size)
→ logits (batch, seq_len, vocab_size)
```

Key difference from sequence classification: use **all timesteps** (`out`) not just the last (`out[:, -1, :]`). Every position simultaneously predicts the next word — so you get seq_len predictions per sample per batch.

### Input/Target Construction

The entire language modeling setup is one line:

```python
X.append(all_idxs[i:i + seq_len])          # positions 0..n
y.append(all_idxs[i + 1:i + seq_len + 1])  # positions 1..n+1
```

Target = input shifted by 1. Every word's job is to predict the word immediately to its right.

### Loss Reshape

```python
loss = criterion(
    logits.reshape(-1, VOCAB_SIZE),   # (batch × seq_len, vocab)
    targets.reshape(-1)               # (batch × seq_len,)
)
```

Flatten batch and sequence dimensions together so CrossEntropyLoss sees one prediction per token.

### Temperature Sampling

```python
logits = logits[:, -1, :] / temperature
probs  = F.softmax(logits, dim=-1)
probs[0, word2idx['<UNK>']] = 0    # exclude UNK from generation
probs  = probs / probs.sum()
next_idx = torch.multinomial(probs, 1)
```

Temperature controls the sharpness of the probability distribution:
- `temp < 1.0` → sharper distribution → conservative, repetitive, structured
- `temp = 1.0` → unchanged distribution → balanced
- `temp > 1.0` → flatter distribution → creative, random, incoherent

---

## Results

### Perplexity Comparison

| Model | Train PPL | Val PPL |
|-------|-----------|---------|
| Bigram | 336.2 | 1766.2 |
| Trigram | 498.5 | 4564.5 |
| LSTM | 266.3 | 721.0 |

LSTM beats bigram by **2.4x on validation** — purely because the LSTM remembers the full sequence context through its hidden state while bigram only sees 1 word back.

### Generated Text

```
temp 0.5: "the economy of the state of the north and the main system..."
temp 1.0: "the disorder was clearly by a young travel produced but..."
temp 1.5: "the snow niger simply belonged on nathaniel entry include..."
```

temp=0.5 produces the most grammatically structured output — "the X of the Y" is a real Wikipedia noun phrase pattern the LSTM learned. temp=1.5 is mostly word salad.

### Training Behavior

- Early stopping triggered at epoch 36
- Val loss plateaued around 6.6 after epoch 10 — data ceiling with 200k tokens
- Train/val gap is expected — model size (hidden=256) vs vocab (7045) is tight

---

## Why Trigram Was Worse Than Bigram on Val

Trigrams need to have seen the exact 2-word context before. With only 200k tokens, most trigram contexts appear rarely or never in val data. The model constantly falls back to smoothing, which is weak signal. More context hurts when you don't have enough data to cover that context space.

This is the fundamental problem with n-gram models — they scale poorly to larger n. Every extra word you add to the context multiplies the number of possible contexts exponentially, but your training data stays fixed.

---

## Reflection Questions

**1. What does a language model actually do?**

A language model computes the probability of a sequence of words. At each step it takes everything seen so far as context and outputs a probability distribution over the vocabulary — essentially asking "what word is most likely to come next given this history?" The chain rule lets you decompose the probability of any sentence into a product of these conditional probabilities, one per word. This means you can both score existing sentences and generate new ones by sampling from the distribution one word at a time.

**2. Perplexity of 10 vs 100 vs 1000**

PPL=10 means the model is effectively choosing between 10 equally likely options at each step — it has strong opinions about what comes next, narrows the field dramatically. PPL=100 means 100 options — the model has learned something but still has significant uncertainty. PPL=1000 means the model is nearly random — it's barely better than guessing uniformly across the vocab. For reference: our vocab was 7045 words, so a completely random model would have PPL=7045. Getting to 721 means the model has genuinely eliminated about 90% of the uncertainty.

**3. Why perplexity = exp(cross_entropy)?**

Cross entropy is the average negative log probability per word. The negative log converts small probabilities (which the model assigns to rare words) into large positive numbers — the loss. Averaging across all words gives one number summarizing how surprised the model was by the text. Taking exp() of that average reverses the log and converts back to probability space, giving you the geometric mean of (1/probability) across all predictions. This has the natural interpretation of "how many equally likely choices would produce this same average surprise" — the effective branching factor at each step. The exp is the right function because it's the exact inverse of the log that created the CE loss.

**4. Chain rule for P("the cat sat")**

```
P("the cat sat")
  = P("the")
  × P("cat" | "the")
  × P("sat" | "the", "cat")
```

Each term conditions on the full left context. P("the") is just how often sentences start with "the". P("cat"|"the") is how often "cat" follows "the". P("sat"|"the","cat") is how often "sat" follows that exact two-word context. Multiply them together and you get the joint probability of the whole sequence. In log space you sum instead of multiply, which is numerically stable.

**5. Why did trigram perform worse than bigram on val?**

The trigram needs to have seen the exact 2-word context before making a prediction. With only 200k tokens, most 2-word contexts appear very rarely. On the val set, most trigram contexts are completely unseen, so the model falls back entirely to Laplace smoothing — which is essentially a flat prior over the vocab. The bigram only needs to have seen 1-word contexts, which are far more likely to have appeared in training. This exposes the core weakness of n-gram models: adding more context exponentially increases the context space while your training data stays fixed. More context only helps when you have enough data to have seen those contexts.

**6. What does temperature do?**

Temperature divides the logits before softmax, controlling how peaked the probability distribution is. Low temperature (0.5) amplifies the gaps between logits — the highest-probability word gets an even larger share, so the model almost always picks the safe, common word. This produced the most grammatically structured output: "the economy of the state of the north" reads like a real Wikipedia noun phrase. High temperature (1.5) shrinks the gaps — rare words get real probability mass, so the model samples more freely. This produced word salad. Temperature=1.0 is the default unmodified distribution. The UNK problem at temp=0.5 was because UNK aggregates all out-of-vocab words so it had the highest raw frequency — fixing it by zeroing out UNK before sampling gave much cleaner output.

**7. Scaling Laws — core claim and why it matters**

The Scaling Laws paper (Kaplan et al. 2020) found that language model performance follows smooth power laws with respect to three variables: model size (number of parameters), dataset size (number of tokens), and compute budget. Critically, these relationships are predictable — you can extrapolate from small experiments to large ones. The core claim is that bigger models trained on more data with more compute consistently produce better models, and the improvement follows a precise mathematical curve rather than being random or unpredictable.

For someone building their own LLM this matters enormously for two reasons. First, it tells you that you can run small experiments and predict what a large run will achieve before committing the compute. Second, it tells you the optimal allocation — if you double your compute budget, how much should go to making the model bigger vs. training on more data? The paper gives a concrete answer. The lesson for this curriculum: the ceiling we hit at PPL=721 with 200k tokens and hidden_size=256 is not a bug — it's the scaling law in action. More tokens and larger model = predictably better perplexity.

---

## Bugs Fixed Today

- Print statement outside training loop → moved inside loop
- `loss.item()` in print was last batch loss not epoch average → fixed to `epoch_loss / len(train_dl)`
- `next_idx.unsqueeze(0)` added extra dimension each iteration → removed unsqueeze
- `<UNK>` dominating temp=0.5 generation → zeroed out UNK probability before sampling

---