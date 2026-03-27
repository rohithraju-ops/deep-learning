# Day 26 — Attention Is All You Need: Transformer Encoder + Self-Attention
**Date:** March 27, 2026
**Notebook:** `day26_transformer_pt1.ipynb`
**Machine:** Mac MPS
**Paper:** Attention Is All You Need — Vaswani et al. (2017)

---

## What Was Built

- Scaled dot-product attention from scratch
- MultiHeadAttention with correct shape splitting and merging
- Sinusoidal PositionalEncoding with register_buffer
- EncoderBlock — Pre-LN style with residual connections
- TransformerEncoder — 4 stacked blocks, 919k parameters
- Attention heatmaps for all 8 heads across layers

---

## Architecture Config

```
d_model   = 128
num_heads = 8
d_k       = 16   (128 / 8 — per head)
d_ff      = 512  (4 × d_model)
num_layers = 4
params    = 919,296
```

---

## Why The Transformer Exists — Three RNN Problems

### Problem 1 — Sequential Computation
RNNs process tokens one at a time: t1 → t2 → t3. Cannot parallelize. On modern GPUs with thousands of cores, this wastes almost all compute.

Transformer fix: all tokens processed simultaneously in parallel. Every GPU core used.

### Problem 2 — Long Range Dependencies
To connect token 1 to token 500 in an RNN, information travels through 499 hidden states — it degrades. Even LSTMs struggle over very long distances.

Transformer fix: every token attends directly to every other token. Path length is always 1, regardless of distance.

### Problem 3 — Bottleneck
Seq2Seq encoder compresses entire input into one fixed-size vector h. Long sequences lose information (Day 17 problem).

Transformer fix: no bottleneck. Every layer has direct access to all positions via attention. No compression required.

---

## Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k     = Q.shape[-1]
    scores  = torch.matmul(Q, K.transpose(-2, -1))   # Q·Kᵀ
    scores  = scores / math.sqrt(d_k)                  # scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)                # normalize
    output  = torch.matmul(weights, V)                 # weighted sum
    return output, weights
```

### Step by Step

**Step 1 — Q · Kᵀ (similarity scores)**
```
Q: (batch, heads, seq, d_k)
Kᵀ:(batch, heads, d_k, seq)
→  (batch, heads, seq, seq)   ← d_k cancels
```
Every token's query dot-producted against every token's key. High score = similar, likely relevant.

**Step 2 — ÷ √d_k (scaling)**
Dot products grow with dimension size. With d_k=64, scores can be ~64 magnitude. Softmax on large values → near-zero gradients (saturation). Dividing by √d_k keeps scores at stable magnitude regardless of d_k.

**Step 3 — softmax(dim=-1)**
Converts scores to attention weights summing to 1. `dim=-1` means softmax across the key positions for each query token — each row of the (seq×seq) matrix sums to 1.

**Step 4 — weights · V (weighted sum)**
```
weights:(batch, heads, seq, seq)
V:      (batch, heads, seq, d_k)
→       (batch, heads, seq, d_k)   ← seq cancels
```
Each token gets a new representation = weighted mix of all value vectors.

### The Two Cancellations
```
Q · Kᵀ:    d_k  cancels → (seq, seq)   attention matrix
weights·V: seq  cancels → (seq, d_k)  enriched representations
```

---

## Multi-Head Attention

### Why Multiple Heads
Single attention head learns one type of relationship. Multiple heads learn different relationships simultaneously:

```
head 1: syntactic   — subject attends to verb
head 2: coreference — pronoun attends to antecedent
head 3: local       — each token attends to neighbors
head 4: semantic    — similar meaning words attend to each other
```

Each head gets its own W_Q, W_K, W_V — independent learned projections.

### Shape Journey (d_model=128, heads=8, seq=10, batch=2)

```
input x:            (2, 10, 128)
W_Q/K/V projection: (2, 10, 128)   ← Linear(128→128)
split_heads:        (2, 8, 10, 16) ← view then transpose(1,2)
Q·Kᵀ:              (2, 8, 10, 10) ← d_k=16 cancels
weights:            (2, 8, 10, 10) ← softmax per row
weights·V:          (2, 8, 10, 16) ← seq cancels
transpose+reshape:  (2, 10, 128)   ← 8×16 = 128, concat heads
W_O projection:     (2, 10, 128)   ← final linear
```

Input and output always `(batch, seq, d_model)`.

### split_heads Explained
```python
x = x.view(batch, seq, num_heads, d_k)   # (2,10,8,16) — split 128→8×16
x = x.transpose(1, 2)                     # (2,8,10,16) — heads to front
```
`view` reshapes without copying data — total elements must stay equal. After transpose use `reshape` not `view` because transpose makes memory non-contiguous.

### bias=False in W_Q, W_K, W_V
LayerNorm before attention already centers activations to zero mean — bias would undo this. Bias also corrupts the Q·Kᵀ scores with position-independent noise. FFN layers keep bias because they store factual associations, not compute content similarity.

---

## Self-Attention vs Cross-Attention

**Self-attention (today):** Q, K, V all from same sequence. Every token attends to every other token in the same sequence. Used in encoder and decoder self-attention.

```python
# Q = K = V = x  (same source)
self.attention(norm1(x), norm1(x), norm1(x), mask)
```

**Cross-attention (Day 18 Bahdanau / Day 30 decoder):** Q from decoder, K and V from encoder output. Decoder attends to encoder to figure out what source information is relevant.

```python
# Q from decoder, K and V from encoder
self.attention(norm(decoder_x), encoder_output, encoder_output, mask)
```

Same three-step structure (scores → weights → weighted sum). Different sources for Q vs K/V.

---

## Positional Encoding

Without PE: "cat sat mat" and "mat cat sat" look identical to the Transformer — no inherent notion of order.

### Sinusoidal Formula
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Fast vs Slow Dimensions
```
early dims (0-1):   fast oscillation — changes dramatically each position
                    distinguishes nearby tokens (1-10 steps apart)
late dims (126-127): very slow — barely moves between positions
                    distinguishes distant tokens (100-500 steps apart)
```

Every position gets all 128 dimensions simultaneously — a unique fingerprint combining fast and slow signals.

### Why Sinusoidal Works for Relative Position
```
sin(pos + k) = sin(pos)cos(k) + cos(pos)sin(k)
```
Moving k positions ahead is always a linear transformation of the current position's PE. The model learns "k steps ahead" once and it applies everywhere — even to positions longer than seen in training. Learned PE (GPT-2) can't extrapolate beyond max training length.

### Code Pattern
```python
pe = pe.unsqueeze(0)           # (512,128) → (1,512,128) for batch broadcast
self.register_buffer('pe', pe) # saved with model, moved to GPU, NOT trained
x = x + self.pe[:, :x.shape[1], :]  # slice to actual seq length
```

### Embedding Scaling — × √d_model
```python
x = self.embedding(x) * self.scale   # self.scale = math.sqrt(d_model)
```
Embeddings initialize small (~1). PE values are in (-1, 1). Without scaling PE dominates. Multiplying embeddings by √128 ≈ 11.3 makes them comparable magnitude to PE.

---

## Encoder Block — Full Pattern

```python
def forward(self, x, mask=None):
    # sublayer 1 — self-attention (Pre-LN)
    attn_out, attn_w = self.attention(norm1(x), norm1(x), norm1(x), mask)
    x = x + dropout(attn_out)                 # residual

    # sublayer 2 — feedforward (Pre-LN)
    x = x + dropout(self.ff(norm2(x)))        # norm → ff → residual

    return x, attn_w
```

Every sublayer follows: `x = x + dropout(sublayer(norm(x)))`

Pre-LN (modern): normalize BEFORE sublayer → stable gradients in deep networks.
Post-LN (original paper): normalize AFTER residual → less stable at scale.

---

## TransformerEncoder — nn.ModuleList

```python
self.blocks = nn.ModuleList([
    EncoderBlock(d_model, num_heads, d_ff, dropout)
    for _ in range(num_layers)
])
```

`nn.ModuleList` vs `nn.Sequential` — ModuleList lets you loop manually so you can collect `attn_weights` from each block. Sequential runs all layers automatically without letting you intercept intermediate outputs.

---

## Visualization Results

**Positional Encoding heatmap (50 positions × 64 dims):**
Top rows — rapid red/blue stripes (fast oscillation). Bottom rows — solid blue, barely any variation (slow oscillation). Exactly confirms fast vs slow dimension theory.

**Self-attention weights (Layer 1, Head 1):**
Scattered uniform pattern — expected for untrained model. After training on real task, clear structural patterns emerge.

**All 8 heads:**
Each head shows a different random pattern — 8 independent W_Q/W_K/W_V matrices produce 8 different attention distributions. After training these specialize into different linguistic relationship detectors.

---

## Reflection Questions

**1. Three RNN problems and Transformer solutions**

Sequential computation — RNNs process one token at a time, making GPU parallelization impossible. The Transformer processes all tokens simultaneously through matrix operations, using every GPU core. Long-range dependencies — information from token 1 reaching token 500 in an RNN must travel through 499 hidden states and degrades over distance. Every token in a Transformer attends directly to every other token so path length is always 1 regardless of distance. Bottleneck — the Seq2Seq encoder compresses everything into one fixed vector h which can't hold all information from long sequences. The Transformer eliminates the bottleneck entirely — every layer has direct access to all positions through attention.

**2. Scaled dot-product attention step by step**

Q·Kᵀ computes the raw similarity between every query-key pair. This produces a (seq×seq) matrix where entry [i,j] measures how relevant token j is to token i. The d_k dimensions cancel because you're doing a (seq,d_k)×(d_k,seq) matrix multiplication. Scaling by √d_k is necessary because the dot product sum grows with dimension — without scaling, large d_k produces large scores that push softmax into saturation where gradients vanish. Softmax with dim=-1 converts each row of scores to weights summing to 1 — giving each query token a probability distribution over all key tokens. The weighted sum weights·V then produces a new representation for each token that is a mixture of all value vectors weighted by relevance — the token now "knows about" what it attended to.

**3. Why √d_k scaling**

Each score in Q·Kᵀ is a sum of d_k multiplications. The variance of that sum grows linearly with d_k — so larger d_k means larger scores. With d_k=64, typical scores might have magnitude ~8 (√64). Without scaling, scores of magnitude 64 pushed into softmax cause the output to become a near-one-hot distribution — effectively argmax. The gradients of softmax in this regime approach zero. Dividing by √d_k normalizes the variance back to ~1 regardless of how large d_k is, keeping softmax in a regime where gradients flow.

**4. Self-attention vs cross-attention**

Self-attention has Q, K, and V all coming from the same sequence — every token attends to every other token in the same sequence to build richer representations of itself. It's used in the encoder (attend to all input tokens) and the decoder's first sublayer (attend to previously generated tokens, with masking). Cross-attention has Q coming from one sequence (decoder) and K, V from another (encoder output) — the decoder asks "what in the encoder is relevant to what I'm currently generating?" It's used in the decoder's second sublayer. Day 18's Bahdanau attention was cross-attention — decoder hidden state as Q, encoder hidden states as K and V. The Transformer generalizes and vectorizes this exact mechanism.

**5. Multi-head attention vs single-head**

Single-head attention learns one type of relationship across the whole d_model space. With 8 heads each operating in 16-dimensional subspaces, the model can simultaneously learn different types of relationships — one head might track syntactic subject-verb agreement, another tracks coreference (pronoun → antecedent), another attends to local context (neighboring words), another tracks semantic similarity. The heads run in parallel, then their outputs are concatenated and projected back through W_O which mixes all the learned relationships into the final representation. Single-head has to compromise between all these relationship types simultaneously in one attention pattern.

**6. Why positional encoding**

The Transformer has no recurrence and no convolution — it processes all tokens as a set with no inherent ordering. Without PE, "the cat sat on the mat" and "the mat on sat cat the" produce identical representations because the model sees the same set of tokens. PE adds a unique position-dependent signal to each token embedding before the first layer. The sinusoidal formula produces signals at multiple frequencies — fast frequencies distinguish nearby positions, slow frequencies distinguish distant positions — giving the model enough information to understand both local order and global structure.

**7. Phase 2 concepts appearing in today's architecture**

Embeddings (Day 19) — `nn.Embedding(vocab_size, d_model)` converts token indices to dense vectors, exactly as in Word2Vec. Attention mechanism (Day 18) — Bahdanau's three steps (alignment scores → softmax weights → weighted sum) are the direct ancestor of Q·Kᵀ/√d_k → softmax → ·V. Encoder-decoder structure (Day 17) — the Q from decoder, K/V from encoder pattern in cross-attention is exactly the Seq2Seq architecture with attention. Hidden state as feature vector — d_model plays the same role as hidden_size in LSTMs, carrying the representation forward. LayerNorm (Day 22) — Pre-LN applied before every sublayer, with learned γ and β. Residual connections (Day 23) — `x = x + sublayer(x)` pattern appears twice per encoder block, same gradient highway as ResNet. Dropout (Day 11) — applied after each sublayer before the residual add. Teacher forcing parallel training (Day 17) — decoder processes all positions in parallel during training using the same shifted-target pattern from language modeling days. Perplexity and language modeling (Day 20) — the Transformer encoder will be used for exactly the same next-token prediction task.

---

## Key Code Patterns

```python
# scaled dot product attention
scores  = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
weights = F.softmax(scores, dim=-1)
output  = weights @ V

# split heads
x = x.view(batch, seq, heads, d_k).transpose(1, 2)   # (B,H,S,dk)

# merge heads
x = x.transpose(1, 2).reshape(batch, seq, d_model)   # view fails after transpose

# encoder block — Pre-LN
x = x + dropout(attention(norm1(x), norm1(x), norm1(x)))
x = x + dropout(ff(norm2(x)))

# positional encoding
x = embedding(x) * math.sqrt(d_model)   # scale before adding PE
x = x + pe[:, :seq_len, :]              # slice PE to actual length
```

---
