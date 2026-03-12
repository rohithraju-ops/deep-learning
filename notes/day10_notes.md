# Day 10 — Adam: A Method for Stochastic Optimization
**Paper:** Kingma & Ba, 2015 — arxiv.org/abs/1412.6980

---

## While Reading

### What two things does Adam combine?
Adam combines **AdaGrad** and **RMSProp**:
- **AdaGrad** → handles sparse gradients well (rare features still get meaningful updates)
- **RMSProp** → handles non-stationary objectives well (noisy gradients from dropout/minibatches)

Adam inherits the strength of both in a single algorithm.

---

### What is `m_t` (first moment) and `v_t` (second moment)?

**`m_t` — First Moment (mean of gradients)**
$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

- An exponential moving average of the gradient direction
- Answers: *"which direction have I consistently been going?"*
- Smooths out noisy gradient signals from dropout and minibatches
- β₁ = 0.9 → keeps 90% of old average, adds 10% of new gradient

**`v_t` — Second Moment (mean of squared gradients)**
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

- An exponential moving average of the squared gradient magnitude
- Answers: *"how large have gradients historically been for this weight?"*
- Tracks volatility/activity per parameter
- β₂ = 0.999 → very long memory of gradient sizes

Together they power the final update:
$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

---

### What is "bias correction" and why is it needed?

Both `m` and `v` are initialized at **zero** before training. This causes them to be
heavily biased toward zero in the early steps — they haven't seen enough gradients yet
to be accurate estimates.

**The fix — divide by `(1 - β^t)`:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The `t` superscript is β raised to the power of the current step number.

**What it looks like in practice (for v̂ with β₂=0.999):**

| Step  | β₂^t  | 1 - β₂^t | Correction |
|-------|-------|----------|------------|
| t=1   | 0.999 | 0.001    | ×1000      |
| t=10  | 0.990 | 0.010    | ×100       |
| t=100 | 0.905 | 0.095    | ×10.5      |
| t=∞   | →0    | →1.0     | ×1 (gone)  |

The correction is largest at the start and automatically fades away as the moving
averages accumulate enough real history. Without it, high β₂ values cause training
to diverge completely in early epochs (proven in the paper's Section 6.4).

---

### What are the default hyperparameter values recommended?

| Hyperparameter | Default  | What it controls                         |
|----------------|----------|------------------------------------------|
| `α` (lr)       | `0.001`  | Global step size ceiling                 |
| `β₁`           | `0.9`    | Momentum — smoothing of gradient direction |
| `β₂`           | `0.999`  | Memory of gradient magnitude history     |
| `ε`            | `1e-8`   | Prevents divide-by-zero when v̂ is tiny  |

These are the exact defaults used in PyTorch's `torch.optim.Adam`.

---

---

## Day 10 Notes

### 1. What two concepts does Adam combine and what problem does each solve?

| Concept   | Problem it solves                                                                 |
|-----------|-----------------------------------------------------------------------------------|
| **AdaGrad** | Sparse gradients — rare features (e.g. uncommon words) barely get updated with SGD because their gradient is 0 most steps. AdaGrad gives them larger steps when they do appear. |
| **RMSProp** | Non-stationary objectives — dropout changes the effective model every step, minibatches produce noisy gradients. RMSProp adapts step sizes to gradient volatility over time. |

Adam combines both into one algorithm by tracking:
- `m_t` (mean of gradients) → handles non-stationarity via smoothing
- `v_t` (mean of squared gradients) → handles sparsity via per-parameter adaptive LR

---

### 2. What is bias correction and why is it especially important early in training?

Bias correction is a mathematical fix for the fact that `m₀ = 0` and `v₀ = 0`.

At step 1 with β₂ = 0.999:
v₁ = 0.999 × 0 + 0.001 × g₁² = 0.001 × g₁²

This is 1000× smaller than the true gradient magnitude — the estimate is almost
entirely pulled down by the zero initialization. Without correction, the step size
would be 1000× too large at the very start.

It is especially critical early because:
1. The correction factor `1/(1-β^t)` is enormous at t=1 and shrinks automatically
2. High β₂ (needed for sparse gradients) causes even more initialization bias
3. Without it, the paper shows training diverges completely when β₂ is close to 1

---

### 3. Why do parameters with large historical gradients get a smaller effective LR?

The update rule divides by `√v̂_t`:
$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

`v_t` tracks the running average of **squared** gradients. A parameter that
consistently receives large gradients accumulates a large `v_t`. Taking the square
root still leaves a large denominator — which shrinks the effective step:

Weight A — gets big gradients every step:
v̂ is large → √v̂ is large → step = α × (direction / large number) = SMALL step

Weight B — rarely gets gradients (sparse):
v̂ is tiny → √v̂ is tiny → step = α × (direction / tiny number) = LARGE step


This is the adaptive learning rate in action. Frequently-updated weights self-regulate
to avoid overshooting. Rarely-updated weights get compensated with larger steps
whenever they do appear.

The paper calls `m̂_t / √v̂_t` the **signal-to-noise ratio (SNR)**. Near convergence,
gradients become small and noisy, SNR → 0, and steps automatically shrink —
a built-in form of learning rate annealing.

---

### 4. What is the difference between Adam and AdamW?

Both use the same core update rule. The difference is **how weight decay is applied**.

**Weight decay** shrinks all weights slightly toward zero every step to prevent
overfitting — stops any single weight from growing huge.

**Adam (broken weight decay):**
- Adds decay to the gradient *before* the adaptive step
- The decay signal goes through `m_t` and `v_t` and gets scaled by the adaptive mechanism
- Active weights (large `v_t`) get their decay accidentally scaled down → under-regularized
- Inactive weights (small `v_t`) get their decay scaled up → over-regularized

**AdamW (correct weight decay):**
- Applies decay *after* the Adam update, completely separately
$$\theta_t = \underbrace{\theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}}_{\text{Adam step}} - \underbrace{\alpha \cdot \lambda \cdot \theta_{t-1}}_{\text{independent decay}}$$
- Every weight gets the exact same proportional shrink regardless of gradient history
- `decoupled_weight_decay: True` in PyTorch

**When to use which:**
- `Adam` → general prototyping, CNNs, quick experiments
- `AdamW` → transformers, BERT, GPT, LLMs (industry standard)

---

### 5. Three Most Surprising Things from the Paper

1. **Bias correction is what separates Adam from RMSProp.** Removing those two
   lines (`m̂` and `v̂` corrections) literally converts Adam back into RMSProp with
   momentum. The paper proves this causes complete divergence at high β₂ values —
   two lines of division are doing more work than they look like.

2. **Adam automatically anneals its own learning rate.** Near convergence, gradients
   become small and inconsistent → SNR = `m̂/√v̂` → 0 → effective step → 0.
   No learning rate scheduler needed for the algorithm to slow down near a minimum.
   This emergent behavior wasn't designed in — it falls out of the math naturally.

3. **The adaptive step size is bounded by α.** No matter how large the gradient is,
   Adam's step never exceeds approximately `α = 0.001`. This "trust region" property
   means the model literally cannot take a catastrophically large step even if a
   single batch has extreme gradients — a safety property SGD completely lacks.


---

## Extra Concepts (From Session)

---

### What are Sparse Gradients?

A sparse gradient means most gradient values are **zero** for a given training step —
only a few weights receive a meaningful update signal at a time.

**Real example — word embeddings:**

Vocabulary: 10,000 words
One review: "The movie was absolutely fantastic"

- Gradient for "fantastic" weight = 0.83 ← appeared, gets updated
- Gradient for "elephant" weight = 0.00 ← never appeared, gradient = 0
- Gradient for "brilliant" weight = 0.00 ← never appeared, gradient = 0
...9,995 other weights = 0.00 ← all zero this batch


**Why SGD fails:** Same global LR for every weight. Rare words get gradient=0 for
thousands of steps and barely learn. Common words dominate every batch.

**How Adam fixes it:** `v_t` for a rarely-updated weight stays near zero →
small `√v̂_t` → large compensating step the few times that weight does appear.
Rare features automatically "catch up."

---

### What are Non-Stationary Objectives?

Non-stationary means the loss landscape **keeps changing** during training —
gradients don't come from a fixed, stable distribution.

**Three causes in a typical neural network:**

1. **Minibatching** — every batch is a different random sample, so every gradient
   is a noisy estimate. The "true" gradient direction shifts every step.

2. **Dropout** — randomly kills neurons each forward pass, so the effective model
   is different every single step. The loss function is literally different each step
   because a different subnetwork computed it.

3. **Shifting loss landscape** — as weights update, the shape of the loss surface
   changes. A direction that was downhill in epoch 1 may not be downhill in epoch 10.

**How Adam fixes it:** `m_t` acts as a smoothing buffer:
β₁ = 0.9 → m_t = 90% old direction + 10% new noisy gradient

Noisy batch gradients: ↑ ↗ ↓ → ↑ ↘
Smoothed m_t: → consistent direction wins, noise gets averaged out

---

### SGD vs SGD+Momentum — The Math

**Plain SGD:**
$$\theta_t = \theta_{t-1} - \alpha \cdot g_t$$

No memory. Every step only uses the current gradient. If one batch is noisy,
the optimizer chases it at full force.

**SGD + Momentum:**
$$v_t = 0.9 \cdot v_{t-1} - \alpha \cdot g_t$$
$$\theta_t = \theta_{t-1} + v_t$$

Maintains a `velocity` variable that accumulates across steps.
Physical analogy: a ball rolling downhill — it builds up speed as it rolls,
doesn't just react to the current slope angle.

- Step 1: gradient pushes right → velocity = small
- Step 2: gradient pushes right again → velocity = bigger
- Step 3: gradient pushes right again → velocity = even bigger

Ball is now flying toward the minimum


**The tradeoff:**
- Momentum converges faster in consistent directions
- But overshoots the minimum because it can't brake instantly
- Oscillates back and forth before settling

---

### How to Read the Convergence Graph (Adam vs SGD vs Momentum)

Starting point: `theta_init = 5.0`, true minimum at `x = 0`, loss = `f(x) = x²`

**Phase 1 — Steps 0–20 (The Rush Down):**
- SGD plods steadily. Each step = `lr × 2x`. As x shrinks, gradient shrinks,
  steps get smaller automatically.
- Momentum rockets down — velocity accumulates with every consistent step downward.
- Adam barely moves — effective step is bounded by `α = 0.001` (10× smaller LR
  than the others in this code — unfair comparison).

**Phase 2 — Steps 20–40 (The Overshoot):**
- Momentum crosses 0 (the minimum) and goes to −1.3.
- Built up too much velocity to stop at the minimum — flies right past it.
- SGD never overshoots — no velocity accumulation, always reactive to current slope.

**Phase 3 — Steps 40–80 (The Oscillation):**
- Momentum bounces back and forth: −1.3 → +0.4 → −0.2 → ...
- Each bounce is smaller because gradient on the opposite side pushes back.
- SGD continues steady plodding, now around x = 1.5 → 0.8.

**Phase 4 — Steps 80–100 (The Settling):**
- Momentum oscillations die out, converges to ~0 ✅ (wins the race)
- SGD still at ~0.7, hasn't reached 0 yet
- Adam at ~4.9, barely moved (LR mismatch, not a real failure)

**Key insight — overshoot is not always bad:**
In a real loss landscape with local minima, momentum's built-up speed can carry
the optimizer *over* a bad local minimum that would trap SGD completely.

---

### What All Those Extra Adam Params Mean

When you print `optimizer.defaults`, PyTorch shows every configuration field.
Most are internal flags you don't need to touch:

| Param | What it is | Do you need it? |
|---|---|---|
| `lr`, `betas`, `eps` | The actual Adam algorithm params | Yes — these are Adam |
| `weight_decay` | L2 regularization strength | Yes — tune this |
| `amsgrad` | A variant of Adam with a max operator on v̂ | No — rarely used |
| `maximize` | Maximize the objective instead of minimize (for RL) | No |
| `foreach` | Use faster vectorized PyTorch ops internally | No — auto |
| `capturable` | For CUDA graph capture optimization | No |
| `differentiable` | For computing second-order gradients through optimizer | No |
| `fused` | Use fused CUDA kernel (faster on GPU) | No — auto |
| `decoupled_weight_decay` | Whether weight decay is separated from gradient path | Auto — False for Adam, True for AdamW |

**Rule of thumb:** In any project, you only ever actively set `lr` and
`weight_decay`. Everything else is PyTorch internals.
