# Day 23 — ResNets & Skip Connections
**Date:** March 21, 2026
**Notebook:** `day23_resnets.ipynb`
**Dataset:** Synthetic 4-class classification (1000 × 32)
**Machine:** Mac MPS
**Paper:** Deep Residual Learning — He et al. (2015)

---

## What Was Built

- PlainBlock and ResidualBlock from scratch (1D Linear version)
- PlainNetwork vs ResNet — depth experiment at 10 and 50 blocks
- Gradient flow diagnostic — measured gradient norms across all layers
- ConvResidualBlock — full image version with identity + projection shortcuts
- 4-plot visualization: loss curves, accuracy, gradient norms, architecture diagram

---

## Results

```
10 blocks:
  Plain:  84.5%   ResNet: 82.0%  ← both work, task too easy to show difference

50 blocks:
  Plain:  23.0%   ResNet: 80.0%  ← plain collapses to random guessing
```

Gradient flow at 10 blocks:
```
Plain  input_proj gradient:  10.28  ← 26x larger than output (exploding)
ResNet input_proj gradient:   4.16  ← ~1x consistent throughout (stable)
```

---

## Core Concepts

### The Degradation Problem

Before ResNets, deeper networks were assumed to be strictly better — more layers = more capacity. But experiments showed:

```
20-layer network: training error = 0.36
56-layer network: training error = 0.54  ← worse on training data
```

This isn't overfitting. The deeper model has worse training error — it's a pure optimization problem. Two things cause it:

**Vanishing gradients** — gradients multiply through 50+ layers during backprop. Each multiplication by a small number shrinks the signal. By the time gradients reach early layers they're essentially zero — early layers stop learning.

**Degradation** — even with BatchNorm and good initialization, the loss landscape of a very deep plain network is so complex that optimizers struggle. Extra layers don't just fail to help — they actively hurt.

### The Residual Formulation

Standard layer asks: "learn the full mapping H(x)"

Residual formulation asks: "learn only what to ADD — F(x) = H(x) - x"

```
without skip:  output = F(x)           learn full transformation
with skip:     output = F(x) + x       learn the residual only
```

**Why F(x) = 0 is easier than H(x) = x:**

If the optimal transformation is identity (don't change anything), the standard layer must learn `H(x) = x` from random initialization — hard. The residual layer just needs to learn `F(x) = 0` — push weights toward zero, which gradient descent does naturally.

### The Residual Block

```
input x
├──────────────────────────────┐
↓                              │ (skip connection — identity or projection)
Conv3×3 → BN → ReLU            │
↓                              │
Conv3×3 → BN  (NO ReLU here)   │
↓                              │
(+) ←──────────────────────────┘
↓
ReLU
output: F(x) + x
```

**Why no ReLU before the add:**
The second conv produces new values including negatives. Those negatives mean "subtract from input" — useful information. ReLU before the add would clip them to zero, destroying that signal before x is added back. ReLU after the add activates the combined signal.

**Why not sigmoid instead of ReLU:**
Sigmoid gradient max = 0.25. Stacked 50 layers: `0.25^50 ≈ 10^-31` — catastrophic vanishing. ReLU gradient = 1 for positive inputs — no shrinking.

### The Gradient Highway

During backprop the gradient flows through the addition:

```
∂loss/∂x = ∂loss/∂output × (∂F(x)/∂x + 1)
                                          ↑ skip path always contributes 1
```

That `+1` means even if `∂F(x)/∂x` vanishes completely, the gradient still reaches early layers via the skip path. No weight multiplication on the skip path = no shrinking.

### Identity vs Projection Shortcut

**Identity shortcut** — same channels, same spatial size (stride=1, in_channels=out_channels):
```python
self.shortcut = nn.Sequential()   # empty — does nothing
out + self.shortcut(x)            # = out + x
```

**Projection shortcut** — dimensions change (stride≠1 or channels change):
```python
self.shortcut = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
    nn.BatchNorm2d(out_channels)
)
out + self.shortcut(x)   # = out + resized_x
```

The condition:
```python
if stride != 1 or in_channels != out_channels:
    # use projection shortcut
else:
    # use identity — nn.Sequential() passes x through unchanged
```

A 1×1 conv used for projection does only channel mixing — no spatial mixing, minimal parameters, just resizes the skip path to match dimensions.

---

## Conv2d Parameter Reference

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

# output_size = (input_size - kernel_size + 2×padding) / stride + 1

# standard conv — same spatial size:
nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
# (N, 64, H, W) → (N, 64, H, W)

# downsampling — halves spatial, doubles channels:
nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
# (N, 64, 56, 56) → (N, 128, 28, 28)

# projection shortcut — channel resize only:
nn.Conv2d(64, 128, kernel_size=1, stride=2)
# (N, 64, 56, 56) → (N, 128, 28, 28)
```

---

## Dataset

```python
X = torch.randn(1000, 32)         # 1000 samples, 32 features
y = X[:, :4].argmax(dim=1)        # label = which of first 4 features is largest
```

4-class classification — only features 0-3 carry signal, features 4-31 are noise. Label is the argmax of the first 4 features. Clean deterministic rule — model should approach 100% if training correctly.

---

## Gradient Flow Diagnostic

```python
def measure_gradient_flow(model, x, y, criterion):
    model.zero_grad()
    out  = model(x)
    loss = criterion(out, y)
    loss.backward()           # computes gradients
    # NO optimizer.step()     ← diagnostic only, weights never update

    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            grad_norms.append((name, param.grad.norm().item()))
```

This is a **diagnostic tool** not training. Reads gradient magnitudes after one backward pass without updating weights — like taking a blood pressure reading without administering medicine.

---

## Connection to Transformers

The residual pattern from ResNets is used identically in every Transformer block:

```python
# ResNet block:
output = F.relu(conv_block(x) + x)

# Transformer block (Pre-LN):
x = x + self.attention(self.norm1(x))   # residual after attention
x = x + self.ff(self.norm2(x))          # residual after feedforward
```

Same `F(x) + x` — just the sublayer is attention/FFN instead of conv. Without residuals, GPT-3's 96 layers would be untrainable. The skip connection is the thread running from Day 23 through all of Phase 3 and 4.

```
ResNet (2015):    F(x) + x through conv blocks
Transformer:      F(x) + x through attention + FFN
GPT-3 (2020):     96 layers of F(x) + x
```

---

## Reflection Questions

**1. The degradation problem**

The degradation problem is the counterintuitive finding that adding more layers to a plain network can make it perform worse — even on training data, ruling out overfitting. A 56-layer network had higher training error than a 20-layer network. Theoretically the deeper network could just learn identity functions for the extra layers and match the shallower one, but in practice it can't. The optimization landscape of a very deep plain network is so complex that gradient descent gets lost. Combined with vanishing gradients — where the error signal effectively disappears before reaching early layers — the extra depth actively hurts rather than helps.

**2. Residual learning formulation**

Instead of asking a layer to learn the full desired mapping H(x) from input to output, you ask it to learn only the residual F(x) = H(x) - x — the difference between what you want and what you already have. Then you add the original input back: output = F(x) + x. This is easier because if the optimal answer is "don't change anything" (identity mapping), a standard layer must figure out H(x) = x from random initialization which is surprisingly hard. A residual layer just needs F(x) = 0 — push the weights toward zero — which gradient descent does naturally since weight decay and small gradients both push weights toward zero. The layer only needs to learn corrections to the input, not full transformations.

**3. Skip connections and vanishing gradients**

During backprop the gradient of the loss with respect to the input x flows through the addition operation: `∂loss/∂x = ∂loss/∂output × (∂F(x)/∂x + 1)`. The key is that `+1` — it comes directly from the skip path where x passes through unchanged. No matter what happens to `∂F(x)/∂x` — even if it vanishes to zero through weight multiplications — the `+1` ensures a gradient signal always reaches x. Trace it: gradient arrives at the output, flows back through ReLU, hits the addition, splits into two paths. One path goes backward through the conv layers (potentially shrinking). The other path travels directly to x with no weight multiplication — gradient magnitude unchanged. Early layers always receive a clean signal via this highway.

**4. Projection shortcut and when it's needed**

The skip connection adds F(x) + x — element-wise addition requiring identical shapes. When the main path changes dimensions (different channels or different spatial size due to stride=2), the skip path still carries the old shape and the addition fails. A projection shortcut applies a 1×1 convolution to the skip path to resize it. A 1×1 conv has kernel size 1 so it does no spatial mixing — it only recombines channels via learned weights. Add stride=2 and it simultaneously halves spatial size. Minimal parameters, minimal transformation, just enough to make shapes match. The condition: `if stride != 1 or in_channels != out_channels` — use projection, otherwise use identity (empty Sequential).

**5. Gradient flow plot observations**

The plain network showed wildly oscillating gradient norms across layers — zigzagging between ~12 at early layers and near zero at others. The gradient was actually larger at early layers than late layers, meaning exploding gradients rather than vanishing. This instability means early layers are receiving massive, inconsistent gradient updates — training is chaotic rather than dead. ResNet's gradient norms were much flatter and consistent — hovering around 0-4 throughout with no wild swings. The skip connections absorbed the gradient instability. The 50-block accuracy result confirmed it: plain network collapsed to 23% (random guessing), ResNet reached 80%.

**6. Transformer residual connection**

```
x = x + Attention(LayerNorm(x))   # after attention sublayer
x = x + FFN(LayerNorm(x))         # after feedforward sublayer
```

Seen this before on Day 22 in both `PostLNBlock` and `PreLNBlock` — the `x +` at the start of each return statement is exactly this. Also saw the same `F(x) + x` pattern conceptually in every residual block today. The Transformer just applies the same idea twice per block — once after attention, once after the feedforward network.

**7. What made 8 → 152 layers possible**

The jump from VGG's ~8 layers to ResNet-152 was made possible purely by the architectural innovation of skip connections — not more compute, not more data, not better hardware. The exact same hardware and datasets that produced 8-layer VGGs could train 152-layer ResNets once the residual formulation was introduced. This tells you that architectural innovation is often more valuable than raw compute scaling. A clever design change that costs zero FLOPs — one addition operation — unlocked 20× deeper networks. The lesson for LLMs is the same: the Transformer's architectural choices (attention, residuals, layer norm placement) are what made scaling to billions of parameters possible, not just throwing more compute at the problem.

---

## Bugs / Notes

- `_` in `for _ in range(num_blocks)` — loop variable intentionally discarded, just repeating N times
- `nn.Sequential()` with nothing in it = identity function — returns input unchanged
- `bias=False` in conv layers when followed by BatchNorm — BN has its own shift parameter (β) so conv bias is redundant and wastes parameters

---
