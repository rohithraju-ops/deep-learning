# Day 16 — LSTMs & GRUs

**Date:** March 14, 2026
**Status:** Complete
**Notebook:** `day16_lstm_gru.ipynb`
**Paper:** LSTM — Hochreiter & Schmidhuber (1997)

---

## What I Built
- LSTM cell from scratch using `nn.Module`
- GRU cell from scratch using `nn.Module`
- Full sequence models using `nn.LSTM` and `nn.GRU`
- Trained both on a sine wave next-step prediction task
- Debugged overfitting down to root cause

---

## Core Concepts

### LSTM — Long Short-Term Memory
The key idea is the **cell state** `c` — a conveyor belt that carries memory across timesteps with minimal interference. Three gates control it:

| Gate | Activation | Job |
|------|-----------|-----|
| Forget `f` | sigmoid → (0,1) | How much of old `c` to erase |
| Input `i` | sigmoid → (0,1) | How much new info to write |
| Candidate `g` | tanh → (-1,1) | What new value to write |
| Output `o` | sigmoid → (0,1) | How much of `c` to expose as `h` |

**The two update equations:**
```
c_next = f * c_prev + i * g       # update memory
h_next = o * tanh(c_next)         # compute output
```
Everything is element-wise multiply. Each of the 20 (or N) memory slots gets its own independent gate decisions.

### GRU — Gated Recurrent Unit
Simpler than LSTM — 2 gates, no separate cell state:

| Gate | Job |
|------|-----|
| Reset `r` | How much of `h_prev` to use when computing candidate |
| Update `z` | Blend between old `h` and new candidate |

**The key equation:**
```
h_next = (1 - z) * h_prev + z * n
```
One gate does the job of both forget and input gates. If `z=1` → fully replace. If `z=0` → fully keep. Elegant.

### LSTM vs GRU

| | LSTM | GRU |
|---|---|---|
| Gates | 3 (f, i, o) | 2 (r, z) |
| States | h + c | h only |
| Parameters | More | ~25% fewer |
| Convergence | Slower | Faster |
| Peak accuracy | Higher (given time) | Slightly lower |
| Stability | More sensitive to init | More robust |

---

## Key Code Patterns

### Device detection (always use this)
```python
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
```

### torch.cat for combining h and x
```python
combined = torch.cat([h_prev, x], dim=1)  # (batch, hidden+input)
```
`dim=1` glues along the feature axis — same rows, wider columns. The (batch, 30) intermediate shape is just a packaging step so one Linear layer can see both vectors at once.

### Gradient clipping — always use with RNNs
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
RNNs multiply gradients across timesteps → exploding gradients. This caps the norm at 1.0.

### nn.LSTM dropout gotcha
```python
# dropout is SILENTLY IGNORED with num_layers=1
nn.LSTM(input_size, hidden_size, num_layers=1, dropout=0.2)  # no effect

# dropout works between layers when num_layers >= 2
nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.2)  # works
```

### Mini-batch slider loop
```python
permutation = torch.randperm(X_train.size(0))   # shuffle indices
for i in range(0, X_train.size(0), batch_size): # i = 0, 32, 64, ...
    idx = permutation[i:i+batch_size]            # grab this batch's indices
    X_batch, y_batch = X_train[idx], y_train[idx]
```
`range(0, N, batch_size)` — i jumps by batch_size each step, slicing a different random chunk each time.

---

## Debugging Journal — The Sine Wave Problem

### Attempt 1 — Baseline (20 epochs)
- Both models: training loss ~0.0008 ✅
- Test MSE: LSTM ~0.45, GRU ~0.0003 ❌
- **Root cause:** Sequential train/test split — model trained on first 80% of the wave, tested on last 20% (unseen phase). Not a model problem — a data problem.

### Attempt 2 — Added shuffle + more epochs (50)
- Still failing on LSTM
- **Root cause:** 2 layers overkill for sine wave. Dropout destabilizing training. Random init sensitivity.

### Attempt 3 — Final fix ✅
Changes made:
```python
torch.manual_seed(42)                          # stable initialization
indices = torch.randperm(len(X))               # shuffle before split
X, y = X[indices], y[indices]
lstm_model = LSTMModel(1, 32, 1, 1)           # 1 layer, no dropout
lstm_losses = train_model(..., lr=0.0005)      # lower lr for stability
```

**Final Test MSE:**
```
LSTM: 0.000008   (4,513 params)
GRU:  0.000102   (9,729 params)
```
LSTM won — 12x more accurate with half the parameters. Both predictions visually indistinguishable from the actual sine curve.

### Lessons from debugging

| Symptom | Root cause | Fix |
|---|---|---|
| Good train loss, terrible test MSE | Sequential split = data leakage | Shuffle before splitting |
| LSTM flatlines near 0 | Stuck in bad local minimum | Set manual seed, lower lr |
| Wobbly training loss | Dropout + small model = too aggressive | Remove dropout or reduce rate |
| 2-layer worse than 1-layer | Overcapacity for simple task | Match model size to task complexity |

---

## Final Results

| Model | Test MSE | Params | Notes |
|---|---|---|---|
| LSTM | 0.000008 | 4,513 | 1 layer, no dropout, lr=0.0005 |
| GRU | 0.000102 | 9,729 | 2 layers, faster convergence |

**Winner for this task:** LSTM — more accurate, fewer parameters.
**General rule:** GRU converges faster and is more robust. LSTM can be more precise given the right setup.

---

## Watch Out Fors
- Always shuffle before train/test split on time series — sequential split = guaranteed data leakage
- `nn.LSTM` dropout is silently ignored with `num_layers=1`
- 2-layer RNNs need careful tuning — 1 layer is often enough for simple tasks
- LSTM is more sensitive to random initialization than GRU — set a seed
- Gradient clipping (`max_norm=1.0`) is non-negotiable for RNN training

---

## Paper Notes — Hochreiter & Schmidhuber (1997)

### The problem they were solving (Section 1)
Vanilla RNNs backpropagate errors through time by multiplying gradients across every timestep. The gradient at step `t` looking back `q` steps gets scaled by roughly this factor `q` times:

```
f'(net) * w
```

- If this > 1 at every step → gradient **explodes** → unstable training
- If this < 1 at every step → gradient **vanishes** → model can't learn long-range patterns

With sigmoid activations, the max derivative `f'` is 0.25. So even with large weights, the signal shrinks exponentially with sequence length. A vanilla RNN simply cannot learn "what happened 50 steps ago matters now" — the gradient from step 1 is essentially zero by the time it reaches the weights responsible for step 50.

**This is exactly why you used gradient clipping in your training loop** — `clip_grad_norm_(max_norm=1.0)`. You were defending against the exploding side of this same problem.

---

### The fix — Constant Error Carousel (Section 4)
Their core idea: create a unit connected to itself with weight exactly **1.0**. A self-connection of 1.0 means:

```
error flowing back one step = error * 1.0 = error unchanged
```

This is the **Constant Error Carousel (CEC)** — error can flow backwards through it indefinitely without shrinking or exploding. This is your cell state `c`. The reason `c` can carry memory across 50+ timesteps while `h` in a vanilla RNN can't is precisely this — `c` is architecturally protected from gradient decay.

---

### Paper equations → your code (direct mapping)

The paper's equation (9) is the heart of the architecture:

| Paper notation | Your code | What it does |
|---|---|---|
| `s_c(t) = s_c(t-1) + y^in * g(net_c)` | `c_next = f*c_prev + i*g` | Cell state update |
| `y^c(t) = y^out * h(s_c(t))` | `h_next = o * tanh(c_next)` | Hidden state output |
| `y^in` (input gate) | `i = sigmoid(...)` | Controls what gets written |
| `y^out` (output gate) | `o = sigmoid(...)` | Controls what gets read |
| `g(net_c)` | `g = tanh(...)` | Candidate value to write |

One important difference — **the original 1997 paper has no forget gate**. The forget gate `f` in your code was added by Gers et al. in 1999. The original LSTM could only write and read — it couldn't selectively erase. The forget gate made LSTM significantly more practical and is why modern implementations always include it.

---

### Why gates exist — the conflict problem
The paper identifies exactly why a raw CEC isn't enough. Without gates, the same input weight `w` has to simultaneously:
1. Store useful inputs into the cell when relevant
2. Ignore irrelevant inputs when not relevant

These two jobs conflict — the weight can't do both at once. The input gate resolves this by learning *when* to write. Same logic for the output gate — it learns *when* to expose the cell's contents to the rest of the network.

**This is why your gates use sigmoid** — sigmoid outputs (0, 1), acting as a soft switch. 0 = fully closed, 1 = fully open, anything in between = partial. The network learns the right switch positions from data.

---

### What the paper does NOT have that you coded
- No forget gate (added 1999) — your `f * c_prev` term
- No GRU (proposed by Cho et al. 2014, 17 years later)
- No `batch_first=True` — the paper predates modern tensor conventions entirely
- No Adam optimizer — they used a variant of RTRL

---

## Day 16 — Reflection Questions

1. **What are the three gates in an LSTM and what does each one control?**

   The three gates are the forget gate `f`, input gate `i`, and output gate `o`.
   - Forget gate — looks at the current input and previous hidden state and outputs a value between 0 and 1 for each memory slot. 0 means completely erase that slot, 1 means keep it fully. It decides what to throw away from the cell state.
   - Input gate — decides how much of the new candidate value `g` to actually write into the cell state. Works alongside the candidate (tanh) to control what new information gets stored.
   - Output gate — decides how much of the current cell state to expose as the hidden state `h`. The cell state might hold a lot of information but the output gate controls what fraction is actually passed forward.

2. **What are the two state vectors in an LSTM — and what is the conceptual difference between them?**

   The two states are the cell state `c` and the hidden state `h`.
   - `c` is the long-term memory — it flows through the network with minimal interference, protected by the gates. Gradients can travel through it without shrinking, which is the whole point of the architecture. It's internal and never directly used as output.
   - `h` is the working memory / output — it's what gets passed to the next timestep AND what you actually use for predictions. It's a filtered, squashed version of `c` controlled by the output gate.
   - One way to think about it: `c` is what the network remembers, `h` is what it chooses to say.

3. **GRU has fewer parameters than LSTM — why? What does it give up?**

   GRU has fewer parameters because it merges the forget and input gates into a single update gate `z`, and removes the separate cell state entirely — everything runs through `h`. This means two weight matrices instead of four, which is roughly 25% fewer parameters.
   What it gives up is expressiveness. The update gate enforces a hard constraint: the amount of new information written in equals exactly the amount of old information forgotten `(1-z)`. LSTM's separate forget and input gates can make independent decisions — it's possible to simultaneously keep most of the old memory AND write a lot of new information. GRU can't do that. In practice this rarely matters, but on complex long-range tasks LSTM can occasionally edge ahead.

4. **Why is gradient clipping important specifically for RNNs, and what does clip_grad_norm_ actually do under the hood?**

   RNNs process sequences by applying the same weight matrix at every timestep. During backprop through time, gradients get multiplied by that weight matrix repeatedly — once per timestep. If any eigenvalue of the matrix is slightly above 1, multiplying it across 50 timesteps causes the gradient to explode exponentially. This doesn't happen in feedforward networks because there's no repeated multiplication.
   `clip_grad_norm_(parameters, max_norm=1.0)` computes the global L2 norm across all gradients in the model — essentially treating every gradient as a single giant vector and measuring its length. If that length exceeds `max_norm`, it scales all gradients down proportionally so the total norm equals exactly `max_norm`. It doesn't clip individual gradients independently — it rescales the whole gradient vector, preserving direction but capping magnitude.

5. **In PyTorch's nn.LSTM, the output is (out, (h_n, c_n)). When would you use out[:, -1, :] vs the full out tensor?**

   - `out[:, -1, :]` — takes only the last timestep's hidden state. Use this for sequence-to-one tasks: next value prediction, sentiment classification, any task where you only need one output for the whole sequence. This is what the sine wave notebook used.
   - `out` (full tensor, shape `batch, seq_len, hidden`) — use this for sequence-to-sequence tasks where you need an output at every timestep: translation, named entity tagging, time series forecasting for every future point, or when feeding into an attention mechanism. Every row is the hidden state at that timestep.
   - `h_n` is equivalent to `out[:, -1, :]` for a single-layer LSTM but gives the last hidden state per layer for multi-layer models. Mostly used to pass state between batches in stateful RNNs.

6. **After looking at your training curves — did LSTM or GRU converge faster? Was the final MSE meaningfully different? What does that tell you?**

   GRU converged faster in the early epochs — it had lower loss around epoch 5-10 consistently across all three training runs. This matches the theory: fewer parameters means less to tune, so GRU finds a decent solution more quickly.
   In the final run after fixing the data split and tuning LSTM properly, the final MSE was meaningfully different — LSTM reached `0.000008` vs GRU's `0.000102`, a 12x gap. But this only showed up after fixing the sequential split bug and reducing LSTM to 1 layer. Before those fixes, LSTM looked like it was losing badly.
   The lesson: raw training curves can be misleading. GRU's robustness makes it easier to get working out of the box. LSTM's higher ceiling only shows once the setup is right. For production use cases where engineering time matters, GRU is often the safer default starting point.

7. **(Paper reflection) After skimming Hochreiter & Schmidhuber — what was the core problem they were solving, in your own words?**

   Vanilla RNNs can't learn from events that happened many timesteps in the past because the gradient signal shrinks exponentially as it travels backward through time. At every timestep, the gradient gets multiplied by the derivative of the activation function times the weight — with sigmoid activations this factor is at most 0.25, so looking back 50 steps means the gradient has been multiplied by something ≤ 0.25 fifty times, which is essentially zero. The network literally cannot tell its early weights that they did something wrong.
   Hochreiter and Schmidhuber's fix was to create a protected memory unit — the cell state — whose gradient path has a self-connection of exactly 1.0, meaning the error signal can flow backwards through it unchanged regardless of sequence length. The gates then control access to this memory, solving the secondary problem of conflicting weight updates. What I coded on Day 16 is a direct implementation of this idea, with the one addition of the forget gate (which came two years after the original paper).
