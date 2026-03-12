# Day 11 — Overfitting, Underfitting & Regularization

---

## Day 11 Questions

### 1. What is overfitting in one sentence and why is it a problem?

Overfitting is when a model learns the training data **too well** — including its
noise and random flukes — and loses the ability to generalize to new unseen data.

It's a problem because the entire point of training a model is to make good
predictions on data it hasn't seen before. A model that only works on training
data is useless in production.

```
Signal to watch:
  Training loss:    0.05  ←  very low
  Validation loss:  0.45  ←  much higher = overfitting
```

The gap between training and validation loss IS overfitting. The bigger the gap,
the worse the overfitting.

---

### 2. What does L2 regularization do to the weights mathematically?

L2 adds a penalty term to the loss function that grows with the size of the weights:

$$\text{Loss} = \text{CrossEntropy}(predictions, labels) + \lambda \sum w^2$$

When you compute the gradient and update the weights, this penalty term adds an
extra shrink toward zero at every step:

$$w = w - \alpha \cdot \nabla L - \alpha \cdot \lambda \cdot w$$

$$w = w(1 - \alpha\lambda) - \alpha \cdot \nabla L$$

**In plain terms:** every single update step, each weight gets slightly pulled
back toward zero — regardless of what the gradient is doing. This prevents any
single weight from growing large enough to memorize individual training examples.

**Why squaring the weights?**
- Smooth and differentiable everywhere (easy gradient computation)
- Penalizes large weights disproportionately: w=10 → penalty=100, w=1 → penalty=1
- Large weights that "memorize" are far more expensive than small distributed ones

**The λ tradeoff:**

| λ value | Effect |
|---------|--------|
| 0 | No regularization, model free to overfit |
| 1e-4 | Light regularization, good default for CNNs |
| 1e-2 | Strong regularization, AdamW default for transformers |
| 10+ | Too strong, forces all weights to ~0, causes underfitting |

**In PyTorch:** L2 lives in the optimizer, not the model:

```python
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
```

---

### 3. Why does Dropout force more robust learning?

Without Dropout, neurons can **co-adapt** — neuron A relies on neuron B always
being there to correct its mistakes. They memorize patterns together as a team.

```python
nn.Dropout(p=0.5)  # randomly kills 50% of neurons each forward pass
```

With Dropout, each neuron can't rely on its neighbors showing up next step.
Every neuron is forced to learn **independently useful features**:

```
Without Dropout:
  Neuron 5: "I detect noses. Neuron 8 handles everything else."
  → fragile co-dependence, memorizes together

With Dropout:
  Neuron 5: "I need to be useful alone — I'll learn full face structure."
  → each neuron becomes robust and general ✅
```

**The Ensemble Interpretation:**
Every training step uses a different random subnetwork. Over thousands of steps,
you're effectively training thousands of thinned-out networks that share weights.
At inference, using the full network approximates averaging all those subnetworks —
like a free ensemble of models.

**The scaling fix (inverted dropout):**
During training, outputs are divided by `p` to compensate for dropped neurons,
so the expected magnitude stays consistent at inference when all neurons are active.
PyTorch handles this automatically — you never see it.

---

### 4. Why must Dropout be OFF during evaluation?

During training, Dropout randomly kills neurons → different output every forward pass.
If left ON at inference, the same input would give **different predictions** every
time you run it — completely non-deterministic and unreliable.

```python
model.train()   # Dropout ACTIVE   — random neurons killed each step
model.eval()    # Dropout INACTIVE — all neurons on, deterministic output
```

There's also a magnitude problem:

```
Training:  only 50% of neurons active → outputs are scaled to compensate
Inference: all 100% of neurons active
           → without eval(), output magnitude would be ~2x too large
```

`model.eval()` switches Dropout off AND disables the scaling compensation
simultaneously, so outputs are at the correct magnitude.

> ⚠️ **Most common beginner mistake:** forgetting `model.eval()` before inference.
> Your model will give different results every run and produce wrong magnitudes.

---

### 5. What does Batch Normalization actually normalize and why does it help?

**What it normalizes:**
BatchNorm normalizes the **output of each layer** — the values flowing between
`Linear → BatchNorm → ReLU` — so each feature has zero mean and unit variance
within each mini-batch.

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$

Where:
- $\mu_B$ = mean of the current mini-batch for this feature
- $\sigma_B^2$ = variance of the current mini-batch for this feature
- $\varepsilon$ = 1e-5, prevents divide-by-zero

**Why it helps — Internal Covariate Shift:**
Without BatchNorm, each layer's output has a wildly different distribution.
As you go deeper, distributions shift and scale out of control:

```
Input:      [-0.3, 0.8, 1.2, ...]    ← nicely normalized
Layer 1:    [-4.2, 11.3, 0.001, ...]  ← already all over the place
Layer 2:    [820, -0.0001, 440, ...]  ← exploding
Layer 3:    ??? → gradients vanish or explode
```

BatchNorm resets distributions back to controlled ranges after every layer,
so gradients flow cleanly all the way through.

**The learnable γ and β:**
Pure normalization would force outputs to always have mean=0, std=1 — which
could destroy learned representations. So BatchNorm adds two learnable parameters
per feature that can "undo" normalization if needed:

$$y_i = \gamma \cdot \hat{x}_i + \beta$$

- **γ (gamma):** learnable scale — "how spread should this layer's output be?"
- **β (beta):**  learnable shift — "what should the mean be?"

These are trained by backprop just like regular weights.

**Layer order matters:**

```python
# ✅ Correct
nn.Linear(784, 256),
nn.BatchNorm1d(256),
nn.ReLU()

# ❌ Wrong — ReLU zeros half the values before BatchNorm sees them
nn.Linear(784, 256),
nn.ReLU(),
nn.BatchNorm1d(256)

# ❌ Wrong — never BatchNorm the output layer
nn.Linear(128, 10),
nn.BatchNorm1d(10)
```

---

---

## Crucial Concepts (Tricky Parts)

---

### Overfitting vs Underfitting — The Full Picture

```
UNDERFITTING             JUST RIGHT              OVERFITTING
(High Bias)              (Balanced)              (High Variance)

Too simple to learn      Learned the pattern     Memorized noise too
  Train loss: high         Train loss: low         Train loss: very low
  Val loss:   high         Val loss:   low         Val loss:   high
```

- **Bias** = model too rigid, wrong assumptions, misses the real pattern
- **Variance** = model too sensitive, chases noise, fails on new data

The three regularization techniques all attack overfitting (high variance) from
different angles:

```
L2 Weight Decay  → attacks the WEIGHTS       → keeps weights small, distributed
Dropout          → attacks the NEURONS        → forces independent, redundant learning
Batch Norm       → attacks the DISTRIBUTIONS  → stabilizes layer outputs
```

---

### The Two Formulas Inside BatchNorm (Easy to Confuse)

These look similar but do completely different jobs:

**Formula 1 — Data Normalization (transforms data RIGHT NOW):**

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$

- Transforms the actual neuron values flowing through the network
- Uses the CURRENT batch's mean and variance
- Fires every single forward pass, affects training directly

**Formula 2 — Running Average Update (saves stats for LATER):**

$$\mu_{\text{running}} = 0.9 \times \mu_{\text{running}} + 0.1 \times \mu_B$$

- Does NOT touch the data at all
- Quietly updates a stored number inside the BatchNorm layer
- Builds up a long-term estimate of the true data distribution
- Used at inference when there's no batch to compute from

```
Formula 1  →  "process data RIGHT NOW"
Formula 2  →  "save stats for LATER (inference)"
```

At inference (`model.eval()`), Formula 1 still runs but plugs in `running_mean`
from Formula 2 instead of computing a fresh batch mean.

> **Note:** Running stats are NOT learnable parameters — they are buffers.
> They don't appear in the parameter count and the optimizer never updates them.
> PyTorch updates them automatically on every training forward pass.

---

### Why `model.train()` / `model.eval()` Affects TWO Layers

Both Dropout and BatchNorm behave differently in train vs eval mode:

| Layer | `model.train()` | `model.eval()` |
|-------|-----------------|----------------|
| Dropout | Randomly kills p% of neurons | All neurons active, no dropping |
| BatchNorm | Uses current batch μ and σ² | Uses stored running_mean and running_var |

> ⚠️ Forgetting `model.eval()` breaks BOTH simultaneously.

---

### Where Each Regularization Technique Lives in Code

```python
# L2 — lives in the OPTIMIZER, not the model
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)

# Dropout — lives in the MODEL as a layer
nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),      # ← inserted as a layer in the architecture
    nn.Linear(256, 1)
)

# BatchNorm — lives in the MODEL as a layer, BEFORE ReLU
nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),    # ← inserted as a layer, before activation
    nn.ReLU()
)
```

---

### The BatchNorm1d Output — All Parameters Explained

```
BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```

| Parameter | What it means |
|-----------|---------------|
| `256` | Normalize each of the 256 features independently across the batch |
| `eps=1e-05` | Prevents divide-by-zero when variance ≈ 0 |
| `momentum=0.1` | Running average speed: 90% old history + 10% new batch |
| `affine=True` | γ and β exist and are trained by backprop |
| `track_running_stats=True` | Stores running_mean and running_var for inference |

**Parameter count from BatchNorm:**

```
BatchNorm1d(256): 256 γ + 256 β = 512 parameters
BatchNorm1d(128): 128 γ + 128 β = 256 parameters
Total BatchNorm params in model:    768  (tiny but crucial)
```

---

### Why Tanh > ReLU for Shallow Networks Fitting Smooth Data

The `good_model` used `nn.Tanh()` not `nn.ReLU()`:

| Property | ReLU | Tanh | Best for sin(x) |
|----------|------|------|-----------------|
| Output range | [0, ∞) | (-1, 1) | **Tanh** — matches sin(x) range |
| Handles negatives | ✗ | ✓ | **Tanh** |
| Zero-centered | ✗ | ✓ | **Tanh** — balanced gradients |
| Good for deep nets | ✓ | ✗ | ReLU — only matters at 20+ layers |
| Smooth curve | ✗ piecewise | ✓ smooth | **Tanh** — sin is smooth |

**Rule:** match the activation function to the shape of your problem.
- Shallow network + smooth bounded output → **Tanh**
- Deep network + general classification → **ReLU**
