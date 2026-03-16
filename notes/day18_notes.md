# Day 18 — Bahdanau Attention

**Date:** March 15, 2026
**Status:** Complete ✅
**Notebook:** `day18_attention.ipynb`
**Paper:** Neural Machine Translation by Jointly Learning to Align and Translate — Bahdanau et al. (2015)

---

## What I Built
- `BahdanauAttention` module — W_s, W_h, v projection layers
- `Encoder` — same as Day 17 but now returns ALL hidden states `outputs` not just final h_n, c_n
- `AttentionDecoder` — decoder with attention, LSTM input is `embed + context` concatenated
- `Seq2SeqAttention` — full model connecting encoder → attention → decoder
- `translate_beam` — beam search inference with `beam_size=3`
- Attention heatmap visualization showing which input tokens the decoder attended to at each output step
- Trained on word reversal task: `['three','one','four','two','five']` → `['five','two','four','one','three']`
- Final result: 3/3 test cases correct

---

## The One Structural Change From Day 17

Everything is the same as Day 17 except one thing — the encoder now returns its outputs at every timestep, and the decoder uses them at every step:

```
Day 17:  encoder → h_n, c_n (final state only) → decoder
Day 18:  encoder → outputs (ALL states) + h_n, c_n → decoder
                      ↑
             attention uses this at every decoder step
```

---

## Core Concepts

### The Bottleneck Problem (recap)
In vanilla Seq2Seq the encoder compresses the entire input into one fixed-size context vector — `h_n`. For long sequences this loses information. The decoder has no way to access the parts of the input it needs most at each step.

### What Attention Does
Instead of one fixed context vector, attention gives the decoder a dynamic context vector at every step — a weighted blend of ALL encoder hidden states. The decoder learns to "attend" to the relevant parts of the input for each output token it generates.

---

## The Three Steps of Bahdanau Attention

### Step 1 — Alignment Scores
For each encoder position `i`, compute how relevant it is to the current decoder state `s`:

```
e_ti = v^T · tanh(W_s · s_{t-1} + W_h · h_i)
```

In code:
```python
s = self.W_s(decoder_hidden).unsqueeze(1)   # project decoder state: (batch, 1, attn)
h = self.W_h(encoder_outputs)               # project all encoder states: (batch, seq, attn)
energy = self.v(torch.tanh(s + h)).squeeze(2)  # one score per encoder position: (batch, seq)
```

- `W_s` projects the decoder's current state (what I want)
- `W_h` projects every encoder hidden state (what's available)
- `v` collapses the combined representation to a single scalar score
- `v` has no hand-crafted meaning — it's a learned collapse operation

### Step 2 — Attention Weights
Normalize the scores into a probability distribution with softmax:

```python
attn_weights = F.softmax(energy, dim=1)   # sums to 1 across seq positions: (batch, seq)
```

Now each encoder position has a weight between 0 and 1 representing "how much to look here."

### Step 3 — Context Vector
Weighted sum of all encoder hidden states:

```python
context = torch.bmm(
    attn_weights.unsqueeze(1),   # (batch, 1, seq)
    encoder_outputs              # (batch, seq, hidden)
).squeeze(1)                     # → (batch, hidden)
```

The context vector is a blend of all encoder states, emphasizing the most relevant positions. This replaces the fixed `h_n` from Day 17.

---

## Key Code Patterns

### Encoder — now returns outputs too
```python
def forward(self, src):
    embedded = self.dropout(self.embedding(src))
    outputs, (h_n, c_n) = self.lstm(embedded)
    return outputs, h_n, c_n   # outputs shape: (batch, seq_len, hidden)
                                # Day 17 threw outputs away — Day 18 keeps them
```

### AttentionDecoder — one step
```python
def forward(self, token, h, c, encoder_outputs):
    embedded = self.dropout(self.embedding(token.unsqueeze(1)))  # (batch, 1, embed)
    query = h[-1]                                                 # (batch, hidden) — top layer
    context, attn_weights = self.attention(query, encoder_outputs)

    lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (batch, 1, embed+hidden)
    output, (h, c) = self.lstm(lstm_input, (h, c))
    prediction = self.fc(output.squeeze(1))                           # (batch, vocab_size)
    return prediction, h, c, attn_weights
```

### Training loop — teacher forcing decay
```python
for epoch in range(EPOCHS):
    teacher_ratio = max(0.0, 0.7 - epoch * 0.02)   # 0.7 → 0.0 over 35 epochs

    for src_batch, tgt_batch in dataloader:
        output, _ = model(src_batch, tgt_batch, teacher_forcing_ratio=teacher_ratio)
        # CRITICAL: pass teacher_ratio explicitly — never hardcode 0.5
```

### Beam Search inference
```python
def translate_beam(model, src_words, beam_size=3, max_len=10):
    beams = [(0.0, [], torch.tensor([SOS_IDX]).to(device), h, c)]
    # each beam: (cumulative_score, tokens_so_far, last_token, h, c)

    for _ in range(max_len):
        new_beams = []
        for score, tokens, dec_input, h, c in beams:
            pred, h_new, c_new, _ = model.decoder(dec_input, h, c, encoder_outputs)
            log_probs = torch.log_softmax(pred[0], dim=0)       # log probs are addable
            top_scores, top_idx = log_probs.topk(beam_size)     # expand K ways

            for i in range(beam_size):
                new_score = score + top_scores[i].item()        # accumulate score
                if idx2word[top_idx[i].item()] == EOS:
                    completed.append((new_score, tokens))
                else:
                    new_beams.append((new_score, tokens + [word], ...))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        # prune to top K, NO early break — let all paths finish

    return max(completed, key=lambda x: x[0])[1]
```

---

## Shape Reference

```
# ENCODER
src                       (batch, seq_len)
embedding                 (batch, seq_len, embed)
lstm outputs              (batch, seq_len, hidden)   ← ALL timesteps, kept now
lstm h_n                  (num_layers, batch, hidden) ← final state
lstm c_n                  (num_layers, batch, hidden) ← final state

# ATTENTION (one decoder step)
decoder_hidden = h[-1]    (batch, hidden)            ← query: what do I want?
W_s(query).unsqueeze(1)   (batch, 1, hidden)         ← broadcast over seq_len
W_h(encoder_outputs)      (batch, seq_len, hidden)   ← all encoder projections
energy (after tanh + v)   (batch, seq_len)            ← one score per position
attn_weights (softmax)    (batch, seq_len)            ← sums to 1
context (bmm)             (batch, hidden)             ← weighted blend

# DECODER INPUT (concat)
embedded                  (batch, 1, embed)
context.unsqueeze(1)      (batch, 1, hidden)
cat dim=2                 (batch, 1, embed+hidden)    ← richer than Day 17
```

---

## Reflection Questions

**1. Fundamental difference between vanilla Seq2Seq context vector vs attention context vector?**

In vanilla Seq2Seq the context vector is fixed — it's `h_n`, the encoder's final hidden state, compressed once and handed to the decoder at `t=0`. Every decoder step uses the same vector regardless of what it's generating. In attention the context vector is dynamic — it's recomputed at every decoder step as a weighted blend of ALL encoder hidden states. The decoder effectively gets a fresh, relevant summary of the input tailored to what it's currently trying to generate.

**2. Walk through the three steps of Bahdanau attention.**

Step 1 — alignment scores: for each encoder position `i`, compute a scalar score measuring how compatible the current decoder state `s` is with that encoder state `h_i`. This uses a small feedforward network: `e_ti = v^T · tanh(W_s·s + W_h·h_i)`. The result is one raw score per encoder position.

Step 2 — attention weights: normalize the alignment scores into a probability distribution using softmax along the sequence dimension. Now each encoder position has a weight between 0 and 1 summing to 1 across all positions. These weights answer "how much should I look at each part of the input right now?"

Step 3 — context vector: take the weighted sum of all encoder hidden states using the attention weights as coefficients. The result is a single vector that blends all encoder states, emphasizing the most relevant ones. This context vector goes into the decoder LSTM alongside the embedded current token.

**3. Why does the context vector change every decoder step in attention but stay fixed in vanilla Seq2Seq?**

Because in attention the context is computed fresh each step using the decoder's current hidden state as the query. As the decoder generates tokens its hidden state evolves — and a different hidden state produces different alignment scores against the encoder outputs, which produces a different weighted blend. In vanilla Seq2Seq there is no query mechanism at all — `h_n` is computed once by the encoder and passed as `h_0` to the decoder. It never changes regardless of what the decoder generates.

**4. What does the attention heatmap tell you? What pattern did you see?**

The heatmap shows which encoder positions the decoder was attending to when generating each output token. Rows are output tokens, columns are input tokens. A bright cell at row `i`, column `j` means "when generating output token `i`, the decoder was heavily attending to input token `j`."

For the reversal task `['three','one','four','two','five']` → `['five','two','four','one','three']`, the heatmap showed a clear anti-diagonal pattern:
- Output "five" → attends to input "five" (0.97)
- Output "two" → attends to input "two" (0.66)
- Output "four" → attends to input "four" (0.83)
- Output "one" → attends to input "one" (0.73)
- Output "three" → attends to input "three" (0.61)

This makes perfect sense — reversing a sequence means each output token corresponds exactly to the mirrored input position. The model learned this alignment structure entirely from data with no hardcoded rules.

**5. Bahdanau (additive) vs Luong (multiplicative) attention — what's the difference?**

Bahdanau (additive): `score = v^T · tanh(W_s·s + W_h·h)` — projects both states separately then combines with tanh and a learned vector. More parameters, slightly more expressive, original formulation.

Luong (multiplicative/dot-product): `score = s^T · h` (dot) or `score = s^T · W · h` (general) — computes similarity directly via dot product. Simpler, fewer parameters, faster, and scales better. Works well when hidden dimensions are compatible.

In practice Luong is preferred when speed and scalability matter — it's the stepping stone to the scaled dot-product attention in Transformers. Bahdanau is slightly more expressive for small models but the difference usually doesn't matter much in practice.

**6. Why do we use `h[-1]` as the attention query rather than the full `h` tensor?**

`h` has shape `(num_layers, batch, hidden)` — it contains the hidden state for every layer. The attention module expects a query of shape `(batch, hidden)` — a single hidden state per sample. `h[-1]` takes only the top (last) layer's hidden state, which is the one that directly informs the output. The lower layers learn more primitive representations; the top layer is the most semantically meaningful. Using the full `h` tensor would require either flattening or summing across layers, which adds complexity without much benefit.

**7. Transformers use attention but get rid of the RNN entirely — what might that look like?**

In the attention model today, the RNN is still the backbone — it processes tokens sequentially, and attention is a plug-in that gives the decoder better access to the encoder. The bottleneck is the sequential processing: you can't compute step `t+1` until step `t` is done, which means no parallelism.

If you removed the RNN entirely, you'd need attention to do all the work — not just between encoder and decoder, but within the encoder itself (self-attention: each token attending to every other token in the same sequence) and within the decoder too. Without sequential processing you'd also need to inject position information explicitly since the model would have no other way to know token order. That's essentially what a Transformer is: stacked self-attention layers with positional encodings, no recurrence, fully parallelizable. The attention mechanism stays — the RNN disappears.

---

## Bugs Debugged Today

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `AttributeError: cannot assign module before Module.__init__()` | Missing `super().__init__()` in `Seq2SeqAttention` | Add `super().__init__()` as first line |
| Tokens doubling (e.g. `['three','three','two','one']`) | Exposure bias — `teacher_forcing_ratio=0.5` hardcoded, decay variable `teacher_ratio` computed but ignored | Wire up `teacher_forcing_ratio=teacher_ratio` in model call |
| `teacher_forcing_ratio = teacher_ratio` as default arg | `teacher_ratio` captured at class definition time before it exists | Change default to plain `0.5` |
| Beam search stopping too early | `if completed: break` fires when first short/wrong path hits EOS | Remove early break, let all beams run to `max_len` |
| Length penalty hurting beam search | Dividing by length penalized longer correct sequences, causing early stop | Remove length penalty entirely |
| `ValueError: too many values to unpack` | `translate_beam` returns a list, but code unpacked as `predicted, _` | Remove unpacking — `predicted = translate_beam(...)` |
| Random results between runs | No seed set, fresh random dataset and init each run | Add `torch.manual_seed(42)` + `random.seed(42)` + `np.random.seed(42)` |
| Model not saved after good run | Weights lost when kernel restarts | `torch.save(model.state_dict(), 'day18_model.pth')` after training |

---

## Beam Search — How It Works

Greedy decoding (argmax) picks the single highest-scoring token at every step and commits forever. If it makes one wrong choice early on, it can't recover.

Beam search keeps `K` active paths simultaneously. At every step each path expands into `K` candidates, giving `K²` total. You keep only the best `K` by cumulative score. When any path generates `<EOS>` it moves to a completed list. After `max_len` steps the best completed sequence wins.

```
Why log_softmax instead of argmax?
  log probabilities are addable across steps
  log(0.9) ≈ -0.1  (good token)
  log(0.1) ≈ -2.3  (bad token)
  sequence score = sum of log probs per step — no numerical underflow

Each beam carries its own h and c:
  when two beams diverge at different tokens
  their decoder states evolve differently from that point
  this is why beam search uses K× more memory than greedy
```

Key lesson: beam search helps with exposure bias but doesn't fix it — a properly trained model with decayed teacher forcing should work with greedy decoding. Beam search is a useful safety net, not a substitute for good training.

---

## Watch Out Fors
- `teacher_forcing_ratio = teacher_ratio` as a default arg — `teacher_ratio` doesn't exist at class definition time, only inside the training loop. Always use a plain number as the default: `= 0.5`
- Always pass `teacher_forcing_ratio=teacher_ratio` explicitly in the training loop — never let the default kick in
- Beam search: do NOT break early when first EOS is found — let all beams run to completion so longer correct paths aren't cut off
- Length penalty in beam search hurts short-sequence tasks — only use it for long sequence translation
- Always save model weights after a good training run — `torch.save(model.state_dict(), 'day18_model.pth')`
- Set random seed at the top of the training cell — results are not reproducible without it
- `nn.LSTM` dropout is silently ignored with `num_layers=1` — need at least 2 layers

---

## Key Insight from Today

The attention heatmap is not just a visualization — it's proof of what the model actually learned. The near-perfect anti-diagonal pattern on the reversal task means the model discovered, entirely from data, that to reverse a sequence you should attend to positions in reverse order. No rules, no hardcoded logic — just geometry shaped by 50 epochs of gradient descent on 10,000 examples.

That's the power of the attention mechanism: it makes the model's internal decision process interpretable. When we move to Transformers the same interpretability carries over, but applied everywhere — not just between encoder and decoder but within every layer of both.

---
