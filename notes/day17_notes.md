# Day 17 — Seq2Seq + Encoder-Decoder

**Date:** March 15, 2026
**Status:** Complete
**Notebook:** `day17_seq2seq.ipynb`
**Paper:** Sequence to Sequence Learning with Neural Networks — Sutskever et al. (2014)

---

## What I Built
- Encoder class using `nn.LSTM` — reads full input sequence, returns `h_n` and `c_n`
- Decoder class using `nn.LSTM` — generates one token at a time using context vector
- `Seq2Seq` wrapper class connecting encoder to decoder
- Full training loop with teacher forcing and scheduled sampling
- Tested on word reversal task — `['one','two','three']` → `['three','two','one']`
- Debugged `tgt` vs `tgts` typo, missing forward pass, `tgt.shape[0]` vs `shape[1]` off-by-one
- Final result: 3/3 test cases correct after 50 epochs on 10k samples

---

## Core Concepts

### Why Seq2Seq — the problem it solves
Standard LSTMs output one value per timestep — locked to the same length as input. Translation doesn't work that way. "How are you" → "Comment allez-vous" is 3 words in, 3 words out by coincidence. But "I love dogs" → "J'aime les chiens" could be different lengths. You need a way to decouple input length from output length entirely. That's Seq2Seq.

| Model | Input | Output |
|-------|-------|--------|
| Day 16 LSTM | sequence of 50 timesteps | one number |
| Day 17 Seq2Seq | sequence of any length | sequence of any (different) length |

### The Encoder-Decoder Architecture

```
encoder:  reads ALL input tokens at once → compresses to context vector (h_n, c_n)
                ↓
        context vector — the only bridge
                ↓
decoder:  receives context vector as h_0, c_0 → generates output one token at a time
```

### Context Vector
The encoder's final hidden state `h_n` and cell state `c_n` — shape `(num_layers, batch, hidden)` each. This is the only information the decoder has about the input. It carries the compressed meaning of the entire sentence as floating point numbers. No words, no rules — just geometry learned from training.

### Autoregressive Generation
The decoder generates output token by token. Each predicted token becomes the input for the next step:

```
step 0: input=<SOS>,  h=encoder_h  →  predicts token 1
step 1: input=token1, h=step0_h    →  predicts token 2
step 2: input=token2, h=step1_h    →  predicts <EOS>  →  stop
```

### The Bottleneck Problem
A fixed-size vector has limited capacity. For a 3-word sentence, compressing meaning into 256 numbers is fine. For a 40-word sentence, those same 256 numbers have to carry proportionally more information — and they can't. Information gets squashed out. This is structural — you cannot put 40 words of meaning into a fixed 256-dimensional vector without loss. The decoder then reconstructs output from an incomplete representation of the input. This is exactly what attention (Day 18) fixes.

### Teacher Forcing
During training, instead of feeding the decoder its own (potentially wrong) predictions, you feed it the correct ground truth token at every step:

```
# WITHOUT teacher forcing — errors compound
step 0: correct input → wrong prediction "le"
step 1: wrong input "le" → even more wrong prediction
step 2: completely derailed

# WITH teacher forcing — clean signal always
step 0: correct input <SOS>   → wrong prediction "le", but loss computed vs "J'aime"
step 1: correct input "J'aime" → train from correct context regardless of step 0
step 2: correct input "les"    → train from correct context regardless of step 1
```

We use it during training for stability. We don't use it during inference because there is no ground truth available — the model has to use its own outputs. This is by design.

### Exposure Bias
The gap between training and inference caused by teacher forcing. During training the decoder always sees perfect inputs. During inference it sees its own outputs which can be wrong. The model was never trained to recover from its own mistakes — it was never "exposed" to the distribution of tokens it actually generates at inference time.

Fix — scheduled sampling: start with high teacher forcing ratio and decay it over epochs so the model progressively learns to handle its own predictions.

```python
teacher_ratio = max(0.1, 0.5 - epoch * 0.01)   # 0.5 → 0.1 over 40 epochs
```

### PAD, SOS, EOS tokens

| Token | Meaning | Used where |
|-------|---------|------------|
| `<PAD>` | Filler to equalise batch lengths | `collate_fn`, ignored in loss |
| `<SOS>` | Start of sequence — kick off decoder | First decoder input every step |
| `<EOS>` | End of sequence — decoder stops | Last target token, stop condition at inference |

---

## Key Code Patterns

### Encoder forward
```python
def forward(self, src):
    embedded = self.dropout(self.embedding(src))   # (batch, seq, embed)
    outputs, (h_n, c_n) = self.lstm(embedded)
    return h_n, c_n    # (num_layers, batch, hidden) each — passed to decoder
```

### Decoder forward — one step at a time
```python
def forward(self, token, h, c):
    token = token.unsqueeze(1)                      # (batch,) → (batch, 1)
    embedded = self.dropout(self.embedding(token))  # (batch, 1, embed)
    output, (h, c) = self.lstm(embedded, (h, c))    # (batch, 1, hidden)
    prediction = self.fc(output.squeeze(1))         # (batch, vocab_size)
    return prediction, h, c
```

### Seq2Seq forward with teacher forcing
```python
def forward(self, src, tgt, teacher_forcing_ratio=0.5):
    tgt_len = tgt.shape[1]                          # ← shape[1] not shape[0]
    h, c = self.encoder(src)
    dec_input = tgt[:, 0]                           # SOS token for all samples
    outputs = []

    for t in range(1, tgt_len):
        pred, h, c = self.decoder(dec_input, h, c)
        outputs.append(pred.unsqueeze(1))
        use_teacher = random.random() < teacher_forcing_ratio
        dec_input = tgt[:, t] if use_teacher else pred.argmax(dim=1)

    return torch.cat(outputs, dim=1)                # (batch, tgt_len-1, vocab)
```

### Training loop
```python
for src_batch, tgt_batch in dataloader:
    src_batch = src_batch.to(device)
    tgt_batch = tgt_batch.to(device)

    optimizer.zero_grad()                           # ← always first

    output = model(src_batch, tgt_batch)            # ← forward pass

    output_flat = output[:, 1:, :].reshape(-1, VOCAB_SIZE)
    tgt_flat    = tgt_batch[:, 1:].reshape(-1)      # skip SOS in both

    loss = criterion(output_flat, tgt_flat)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### Loss — ignoring PAD
```python
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

Without this, the model wastes capacity learning to "predict" meaningless PAD tokens and the loss is artificially inflated by padding positions.

### Inference — no teacher forcing
```python
def translate(model, src_words, max_len=10):
    model.eval()
    with torch.no_grad():
        src = torch.tensor([[word2idx[w] for w in src_words]]).to(device)
        h, c = model.encoder(src)
        dec_input = torch.tensor([SOS_IDX]).to(device)
        result = []

        for _ in range(max_len):
            pred, h, c = model.decoder(dec_input, h, c)
            top_token = pred.argmax(dim=1)
            word = idx2word[top_token.item()]
            if word == EOS:
                break
            result.append(word)
            dec_input = top_token

    return result
```

---

## Shape Reference — Full Pipeline

```
# ENCODER
src                       (batch, seq_len)          integers — token IDs
embedding(src)            (batch, seq_len, embed)   floats — looked up vectors
dropout(embedded)         (batch, seq_len, embed)   unchanged shape
lstm(embedded):
  outputs                 (batch, seq_len, hidden)  h at every timestep — THROWN AWAY
  h_n                     (num_layers, batch, hidden) final hidden — KEPT
  c_n                     (num_layers, batch, hidden) final cell   — KEPT

# BRIDGE — h_n and c_n passed as h_0, c_0 to decoder
# shape stays (num_layers, batch, hidden) the entire time — never changes

# DECODER — one step
token                     (batch,)                  one token ID per sample
unsqueeze(1)              (batch, 1)                add fake seq_len dim
embedding                 (batch, 1, embed)         token → float vector
dropout                   (batch, 1, embed)         unchanged
lstm(emb, (h,c)):
  output                  (batch, 1, hidden)        h at this one step
  h                       (num_layers, batch, hidden) updated hidden — next step
  c                       (num_layers, batch, hidden) updated cell   — next step
squeeze(1)                (batch, hidden)           remove fake seq_len dim
fc Linear                 (batch, vocab_size)       score per vocab token
argmax(dim=1)             (batch,)                  predicted token ID
```

---

## Reflection Questions

**1. What is the role of the context vector — what does it carry and where does it come from?**

The context vector is the compressed summary of the entire input sequence. It comes from the encoder's final hidden state `h_n` and cell state `c_n` after the encoder LSTM has processed every input token. It carries the learned meaning of the sentence encoded as floating point numbers — not words, not rules, just numbers whose geometry was shaped by training on thousands of examples. The decoder receives `h_n` and `c_n` as its starting `h_0` and `c_0` and this is the sole source of knowledge about the input. Everything the decoder generates comes from those two tensors.

**2. What is the bottleneck problem? Why does a fixed-size vector cause issues for long sequences?**

A fixed-size vector has limited capacity. For a short sentence compressing meaning into 256 numbers works fine. For a 40-word sentence those same 256 numbers have to carry proportionally more information and they can't — information gets squashed out. The decoder then has to reconstruct output from an incomplete or lossy representation. This is structural: no amount of training can make a 256-dimensional vector carry arbitrarily long sentences without loss. The problem scales with sentence length, which is why translation quality degrades on long inputs.

**3. Explain teacher forcing in your own words. Why training only?**

Teacher forcing is feeding the decoder the correct ground truth token at each step during training, regardless of what the model actually predicted. Without it, one wrong prediction poisons every subsequent step — the model never gets a clean signal to learn from because it's always chasing its own mistakes. With it, every step starts from a correct input and learning is stable and fast. We don't use it at inference because there is no ground truth available — the model has to use its own predictions. By inference time the model should be good enough to generate correctly on its own.

**4. What is exposure bias and why does it happen because of teacher forcing?**

Exposure bias is the gap between training and inference behavior. During training with teacher forcing the decoder always sees perfect inputs — it was never exposed to the distribution of tokens it actually generates. At inference, if it makes one wrong prediction it has no learned mechanism to recover from it, and errors compound. The model was trained in an environment that doesn't exist at test time. The fix is scheduled sampling — gradually reducing the teacher forcing ratio so the model increasingly sees its own predictions during training and learns to handle imperfect inputs.

**5. In the Sutskever paper, why does reversing the input help gradient flow?**

Normally the first source word corresponds to the first target word but they are far apart in the unrolled sequence — the gradient has to travel through many timesteps to connect them. By reversing the source, the first source word is now right next to the start of decoding — a short gradient path. This dramatically reduces the minimum distance for the most important correspondences without changing the average distance. In the paper this single change jumped BLEU from 25.9 to 30.6 with no architecture changes. In the word reversal task this was even more direct — the last input word becomes the first output word, and reversing the target puts that correspondence immediately at the start of decoding where the gradient path is shortest.

**6. Why do we use ignore_index=PAD_IDX in CrossEntropyLoss?**

PAD tokens are meaningless filler added only to make sequences the same length within a batch. They carry no semantic content. If loss is computed on PAD positions, the model wastes capacity learning to predict a meaningless token, and the loss number is artificially inflated by padding positions rather than reflecting real translation quality. `ignore_index=PAD_IDX` tells the loss function to skip those positions — gradients do not flow from PAD positions and the loss only reflects predictions on real tokens.

**7. In one sentence — what problem is Attention (Day 18) going to solve?**

Attention solves the bottleneck problem by letting the decoder look at ALL encoder hidden states at every generation step instead of being limited to the single compressed context vector — so information from any part of the input is directly accessible regardless of sequence length.

---

## Bugs Debugged Today

| Bug | Error | Fix |
|-----|-------|-----|
| `tgt` instead of `tgts` in `collate_fn` | `NameError: name 'tgt' is not defined` | Change to `tgts` (the tuple from `zip`) |
| Missing `output = model(src, tgt)` | `NameError: output not defined` | Add forward pass before computing loss |
| Missing `optimizer.zero_grad()` | Gradients accumulate incorrectly | Add before forward pass every iteration |
| `tgt.shape[0]` instead of `tgt.shape[1]` | `IndexError: index 8 out of bounds for size 8` | `shape[0]` is batch, `shape[1]` is seq_len |
| EOS token not predicted, extra tokens generated | Model generates beyond sequence end | Train longer — 50 epochs fixed it |

---

## Paper Notes — Sutskever et al. (2014)

**Full title:** Sequence to Sequence Learning with Neural Networks
**Authors:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le — Google
**Published:** 2014

### The problem they solved
Standard DNNs require fixed-size inputs and outputs. Sequences are variable length and input/output lengths don't have to match in translation. Their solution: one LSTM encoder reads the input and compresses it to a fixed vector, a second LSTM decoder generates variable-length output from that vector. End-to-end trainable, no hand-crafted rules.

### Three key architectural choices and their parallels to your code

Two separate LSTMs — encoder and decoder are independent models. You built this as separate `Encoder` and `Decoder` classes connected by `Seq2Seq`.

Four layers deep — they found each extra layer reduced perplexity by ~10%. You used `NUM_LAYERS=2` for the same reason at smaller scale.

Reversed source sentences — their most memorable contribution.

### The reversing trick — Section 3.3
```python
# your code — same trick
tgt = [SOS_IDX] + [word2idx[w] for w in reversed(words)] + [EOS_IDX]
```

By reversing the source, the first source word is right next to the first target word in the unrolled sequence. Short gradient path for the most important correspondence. BLEU jumped from 25.9 → 30.6 with zero other changes — no new parameters, no architecture change, one data transformation.

### The historically significant result
BLEU of 34.8 vs phrase-based SMT baseline of 33.3 — first time a pure neural system beat a rule-based translation system at scale on a large MT benchmark. The architecture you built today is the direct implementation of this paper at toy scale.

### BLEU Score — Bilingual Evaluation Understudy
Measures word and n-gram overlap between predicted and reference translation. Score from 0–100:

| Range | Quality |
|-------|---------|
| 0–10 | Almost useless |
| 10–20 | Getting something right |
| 20–30 | Understandable |
| 30–40 | Good, fluent translation |
| 40–50 | High quality |
| 50+ | Near human |

Limitation: measures overlap not meaning. "The cat is happy" and "The feline is joyful" score poorly despite meaning the same thing. It is a proxy metric — useful for benchmarking but imperfect.

---

## Watch Out Fors
- `tgt.shape[1]` for sequence length — `shape[0]` is batch, `shape[1]` is seq_len
- `tgts` not `tgt` inside `collate_fn` — `tgt` is not defined in that scope
- Always `optimizer.zero_grad()` before the forward pass
- Always `output = model(src, tgt)` before computing loss
- EOS learning comes last — if words are right but model won't stop, train longer
- Teacher forcing with too few epochs → exposure bias at test time
- `nn.LSTM` dropout silently ignored with `num_layers=1` — need at least 2 layers
- Sequential train/test split on ordered data = data leakage (Day 16 lesson — still applies)

---

## Key Insight from Today
The decoder has never seen the original input sentence directly. It only ever gets `h_n` and `c_n` — two tensors of floating point numbers — and from those alone it reconstructs the output. The knowledge of how to reverse (or translate) is entirely distributed across 466,189 weight values that got nudged in the right direction over 50 epochs. No rules, no dictionary, just geometry learned from examples.

That also makes the bottleneck feel real — those same weights work fine for 3–6 word reversals. Scale it to 40-word sentences and the fixed-size context vector starts losing information. That is exactly the problem attention solves on Day 18.

---
