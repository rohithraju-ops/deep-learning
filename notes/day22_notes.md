# Day 22 — Normalization Deep Dive: BatchNorm vs LayerNorm
**Date:** March 20, 2026
**Notebook:** `day22_normalization.ipynb`
**Dataset:** Synthetic binary classification (2000 × 20)
**Machine:** Mac MPS
**Paper:** Batch Normalization — Ioffe & Szegedy (2015)

---

## What Was Built

- Manual BatchNorm and LayerNorm from scratch — matched PyTorch exactly (max diff = 0.000000)
- 3-way MLP experiment: No Norm vs BatchNorm vs LayerNorm on tabular data
- PostLNBlock and PreLNBlock — Transformer feedforward blocks
- Training curves + accuracy plots showing normalization impact

---

## Results

```
High LR + 20 layers experiment:
  No Norm:   val acc = 48.0%  ← coin flip, training collapsed
  BatchNorm: val acc = 85.5%  ← only one that trained
  LayerNorm: val acc = 48.0%  ← also collapsed on tabular data
```

Key finding — normalization choice depends on data structure:
```
tabular data:  BatchNorm ✅  LayerNorm ❌
sequence data: LayerNorm ✅  BatchNorm ❌
```

---

## Core Concepts

### Internal Covariate Shift

As data flows through a deep network, each layer's output distribution keeps shifting as earlier layers update their weights. Layer 2 adjusts to Layer 1's output — then Layer 1's output changes, so Layer 2 re-adjusts, so Layer 3 re-adjusts — cascading instability. This makes deep networks slow and fragile.

Normalization fixes this by keeping activations in a stable range after every layer. Each layer always sees a consistent input distribution regardless of what's happening earlier in the network. Result: faster convergence, higher learning rates, less sensitivity to initialization.

### The Math — Both Norms Same Formula, Different Axis

```
Step 1: mean  = average of values along target dimension
Step 2: var   = variance along target dimension
Step 3: x_norm = (x - mean) / sqrt(var + ε)   ← zero mean, unit variance
Step 4: output = γ × x_norm + β               ← learnable rescale
```

`ε (eps=1e-5)` prevents division by zero when variance is near zero.

**Why mean is always zero after step 3:**
Subtracting the mean from every value means positives and negatives always cancel:
```
sum(x - mean) = sum(x) - N×mean = sum(x) - N×(sum(x)/N) = 0
```
Always exactly zero — not a coincidence, guaranteed by the math.

### γ and β — Why They Exist

After normalizing to zero mean and unit variance, you might destroy information the model learned. γ and β let the model undo normalization if needed:

```
γ=1, β=0  → keep normalized (default at init)
γ=2, β=1  → scale up and shift
γ=0, β=0  → kill this feature entirely
```

Both are learned by backprop alongside all other weights. Normalization is a starting point, not a constraint.

---

## BatchNorm vs LayerNorm

### BatchNorm — Normalize Across Batch

```python
mean = x.mean(dim=0)              # across batch dimension
var  = x.var(dim=0, unbiased=False)
```

For tensor `(batch=4, seq_len=6, features=8)`:
- Takes one feature across all 4 examples
- Computes mean and std from those 4 values
- Normalizes that feature to zero mean unit variance

```
BatchNorm slices VERTICALLY — same feature, different examples
```

**Running stats for inference:**
```python
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
running_var  = (1 - momentum) * running_var  + momentum * batch_var
```

At inference you can't compute batch statistics from one sample — use accumulated running stats instead. This is the train/test mismatch problem. Momentum=0.1 means each batch contributes 10% to the running average.

`register_buffer` — saves running stats with the model (state_dict) and moves to GPU with `.to(device)` but is NOT updated by the optimizer.

**Where BatchNorm shines:** CNNs with large batch sizes, image tasks.

**Where BatchNorm breaks:**
- Small batches — statistics too noisy to be meaningful
- Variable sequence lengths — position N may not exist in all sequences
- Inference with single samples — no batch to compute stats from
- Transformers — attention mechanism + variable lengths make it a poor fit

### LayerNorm — Normalize Across Features

```python
mean = x.mean(dim=-1, keepdim=True)   # across feature dimension
var  = x.var(dim=-1, keepdim=True, unbiased=False)
```

For tensor `(batch=4, seq_len=6, features=8)`:
- Takes one token's 8 feature values
- Computes mean and std from those 8 values
- Normalizes that token independently

```
LayerNorm slices HORIZONTALLY — same example, different features
```

**No running stats needed** — same computation at train and test time. Works perfectly at batch size 1. Each token normalized independently regardless of sequence length.

**Why LayerNorm is better for Transformers:**
1. Works at any batch size including 1 (autoregressive inference)
2. Variable sequence lengths — each token normalized independently
3. No train/test mismatch — identical computation both times
4. Attention mixes tokens — batch statistics become entangled
5. Fits `(batch, seq, features)` naturally without transpose gymnastics

### The Transpose Problem

`nn.BatchNorm1d` expects `(batch, features, seq)` — features in middle. Your tensor is `(batch, seq, features)`. Workaround:

```python
x_bn = bn(x_3d.transpose(1, 2)).transpose(1, 2)
# (4,6,8) → transpose → (4,8,6) → BN → (4,8,6) → transpose → (4,6,8)
```

LayerNorm just works directly:
```python
x_ln = ln(x_3d)   # no reshape needed — normalizes last dim naturally
```

---

## The Dataset

```python
X = torch.randn(2000, 20)   # 2000 samples, 20 features — tall thin rectangle
y = ((X[:,0] + X[:,1]*2 - X[:,2] + noise) > 0).long()
```

Plain tabular data — no sequences, no time steps. Think spreadsheet:
```
rows    = 2000 samples
columns = 20 features per sample
label   = binary 0 or 1
```

Only features 0, 1, 2 carry signal. Features 3-19 are pure noise. The model has to learn which 3 columns matter and ignore the other 17.

`manual_seed(0)` vs `manual_seed(42)` — no meaningful difference. Both produce reproducible random sequences. The number is arbitrary — 42 is a convention from pop culture.

---

## MLP Architecture — nn.Sequential Pattern

```python
def make_model(norm_type, input_dim=20, hidden=128, num_layers=6):
    layers = []
    in_dim = input_dim
    for i in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden))
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(hidden))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        in_dim = hidden
    layers.append(nn.Linear(hidden, 2))
    return nn.Sequential(*layers)
```

`nn.Sequential(*layers)` chains layers automatically — no custom `forward()` needed. Same result as writing a class with an explicit forward method, just more compact for repetitive blocks.

---

## Pre-LN vs Post-LN

```python
# Post-LN (original Vaswani 2017):
return self.norm(x + self.dropout(self.ff(x)))
# flow: x → FF → dropout → add x → normalize

# Pre-LN (modern GPT-2 style):
return x + self.dropout(self.ff(self.norm(x)))
# flow: x → normalize → FF → dropout → add x
```

One line difference — `self.norm` moves from outside to inside. Same input, same output shape — drop-in replaceable. Difference is training stability only.

**Why Pre-LN is better:** Post-LN can have unstable gradients early in training — the residual path dominates and LayerNorm sees very large values. Pre-LN normalizes before the sublayer so gradients flow more cleanly through deep networks.

**All modern LLMs use Pre-LN. You will use Pre-LN from Day 26 onward.**

### Residual Connection — `x + sublayer(x)`

The layer learns a **correction** rather than a full transformation. If it learns nothing useful, it outputs zero and x passes through unchanged. Gradients can flow directly through the `+x` path without passing through any weights — this is why very deep networks train stably with residuals.

---

## Reflection Questions

**1. Internal covariate shift and why it slows training**

Internal covariate shift is the phenomenon where the distribution of activations at each layer keeps changing during training as earlier layers update their weights. Each layer is trying to learn from a moving target — as Layer 1's weights change, its outputs shift, forcing Layer 2 to re-adjust, which forces Layer 3 to re-adjust, and so on. In deep networks this cascading instability means layers spend most of their capacity chasing each other's changes rather than learning the actual task. Training slows dramatically and becomes sensitive to initialization and learning rate. Normalization fixes this by clamping activations to a stable range after every layer, so each layer always sees a consistent input distribution.

**2. BatchNorm vs LayerNorm on a (4, 6, 8) tensor**

```
tensor: (batch=4, seq_len=6, features=8)

BatchNorm: take feature 0 across all 4 examples × 6 positions = 24 values
           compute mean and std from those 24 values
           normalize feature 0 across the batch
           → slices VERTICALLY through the batch

LayerNorm: take example 0, position 0, all 8 features = 8 values
           compute mean and std from those 8 values
           normalize that one token's representation
           → slices HORIZONTALLY across features within each token
```

BatchNorm needs all examples in the batch to compute statistics. LayerNorm only needs one token's features.

**3. Why BatchNorm breaks with small batch sizes**

BatchNorm computes mean and variance from the current batch. With batch size 1, mean = the single sample's value and variance = 0 — completely meaningless statistics. With batch size 2-4, the estimates are extremely noisy — one unusual sample dominates the statistics and the normalization is essentially random. The fundamental issue is statistical — you need enough samples to estimate a distribution. BatchNorm requires batch size ≥ 16 to produce stable statistics, ideally 32-128. At smaller sizes the running statistics never converge to the true population statistics and the normalization hurts more than it helps.

**4. Why LayerNorm is better for Transformers — three reasons**

First, autoregressive inference runs with batch size 1 — you generate one token at a time. BatchNorm with batch size 1 produces meaningless statistics, but LayerNorm normalizes across features within that one sample so it works perfectly. Second, Transformers handle variable-length sequences — a sentence of 10 tokens and a sentence of 50 tokens in the same batch means position 11 only exists in one sequence. BatchNorm can't compute meaningful statistics for that position. LayerNorm normalizes each token independently so sequence length is irrelevant. Third, LayerNorm has no train/test mismatch — it uses the same computation at both times. BatchNorm uses running statistics at inference that only approximate the true distribution, introducing a systematic error that can accumulate across deep networks.

**5. γ and β — why they exist**

After normalizing to zero mean and unit variance, the model might need a different scale or center to represent certain features effectively. Forcing all activations to zero mean and unit variance is too restrictive — it might destroy learned representations that happen to work best at a different scale. γ (scale) and β (shift) give the model a way to undo normalization if needed, learned through backprop just like any other weight. At initialization γ=1 and β=0 so the output is normalized. As training proceeds the model adjusts them — it might set γ=2 to amplify a useful feature, β=0.5 to shift the distribution, or γ≈0 to effectively ignore a feature entirely. Normalization is a starting point that the model can move away from, not a permanent constraint.

**6. Pre-LN vs Post-LN — which one and why**

Post-LN (original 2017 paper): `norm(x + sublayer(x))` — normalize after the residual. Pre-LN (GPT-2 onward): `x + sublayer(norm(x))` — normalize before the sublayer. I will use Pre-LN in my Transformer because it trains more stably for deep networks. In Post-LN the residual path can dominate early in training, causing LayerNorm to see very large input values and producing unstable gradients. Pre-LN normalizes before the sublayer so the sublayer always receives well-conditioned input and gradients flow cleanly. Every modern LLM uses Pre-LN for this reason.

**7. Training curves — which converged fastest, did results match expectations**

BatchNorm was the only one that converged — it dropped from 0.7 loss to ~0.04 over 40 epochs while No Norm and LayerNorm flatlined at 0.7 the entire time. The validation accuracy confirmed it: BatchNorm reached 85.5% while the other two oscillated around 50% (random guessing). The results matched expectations for No Norm — without normalization, a high learning rate on a 20-layer network produces exploding or vanishing gradients and training collapses. LayerNorm collapsing was the more interesting result — it confirms that LayerNorm normalizes across features per sample, which destroys the relative scale differences between features on tabular data. Those relative scales are useful information for the model. BatchNorm per-feature normalization is the right choice here because it preserves the relationship between different samples for each feature, which is what a tabular classifier needs.

---

## Key Code Patterns

```python
# BatchNorm manual implementation
mean = x.mean(dim=0)                    # across batch
var  = x.var(dim=0, unbiased=False)
x_norm = (x - mean) / torch.sqrt(var + self.eps)
output = self.gamma * x_norm + self.beta

# LayerNorm manual implementation
mean = x.mean(dim=-1, keepdim=True)     # across features
var  = x.var(dim=-1, keepdim=True, unbiased=False)
x_norm = (x - mean) / torch.sqrt(var + self.eps)
output = self.gamma * x_norm + self.beta

# BatchNorm on 3D tensor — transpose workaround
x_bn = bn(x_3d.transpose(1, 2)).transpose(1, 2)

# LayerNorm on 3D tensor — just works
x_ln = ln(x_3d)
```

---

