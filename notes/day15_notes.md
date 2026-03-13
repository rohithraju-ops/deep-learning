# Day 15 Notes — Recurrent Neural Networks (RNNs)
**Date:** March 13, 2026
**Topic:** RNNs, Hidden State, Vanishing Gradients, Sine Wave Prediction

---

### 1. What is a hidden state and what information does it carry?

The hidden state `h` is the RNN's memory — a fixed-size vector that gets passed
from one timestep to the next. At every step it carries a compressed summary of
everything the RNN has seen so far in the sequence.

```
h_t = tanh(W_xh · x_t  +  W_hh · h_(t-1))
         ↑ current input      ↑ memory from previous step
```

After processing 6 timesteps, the final `h` is a single vector that encodes
the "meaning" of all 6 inputs combined. It is the only thing passed forward —
the raw inputs are discarded. Shape in our code: `torch.Size([2, 8])`
→ 2 sequences in the batch, each with 8 memory values.

---

### 2. What does `batch_first=True` change about tensor shapes?

PyTorch RNNs have two shape conventions:

| Setting           | Expected Input Shape      | Meaning                        |
|-------------------|---------------------------|--------------------------------|
| `batch_first=False` (default) | `(seq_len, batch, features)` | sequence length comes first |
| `batch_first=True`            | `(batch, seq_len, features)` | batch size comes first      |

In our sine wave code, `X_train` has shape `(784, 20, 1)` → batch first.
So we must use `batch_first=True` in the RNN definition, otherwise PyTorch
interprets the 784 as the sequence length and 20 as the batch size — wrong.

```python
# Correct for our data shape (batch, seq_len, features)
self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
```

---

### 3. Why do we only use the last timestep's output for prediction?

The RNN produces an output at EVERY timestep, but we only care about the
final one for next-value prediction:

```
Input:  [sin(t0), sin(t1), ..., sin(t19)]
         ↓        ↓               ↓
        out0     out1    ...    out19  ← we only use this one
```

`out19` = the hidden state AFTER seeing all 20 inputs. It has processed and
compressed the entire sequence into one vector. Taking any earlier output
means the model has not seen the full context yet.

```python
out, _ = self.rnn(x)        # out shape: (batch, seq_len, hidden)
pred   = self.fc(out[:, -1, :])  # out[:, -1, :] = last timestep only
```

`-1` in the middle dimension means "take the last timestep" — standard
Python negative indexing applied to the sequence dimension.

---

### 4. Describe the vanishing gradient problem in plain English — no math

Imagine you are playing a game of telephone across 20 people.
You whisper a message at one end, and by the time it reaches person 20,
it has become a faint whisper — almost nothing.

That is the vanishing gradient. When the RNN learns, it sends an error signal
BACKWARDS through all 20 timesteps. Each step weakens the signal slightly.
By the time it reaches timestep 0 (the oldest input), the signal is so tiny
that the weights connected to early inputs get almost no update.

Result: the RNN effectively ignores old information and only truly learns
from the last few timesteps. For a sine wave this does not matter much —
but for a sentence like "The cat, which had been sitting all morning, was ___",
the RNN forgets "cat" by the time it needs to predict the verb. That is why
LSTMs were invented.

---

### 5. At what timestep does the gradient basically die?

Looking at the gradient bar chart from our trained sine wave model:

```
Timesteps 0–4  (RED bars):  gradient magnitude ≈ 0.0003 – 0.0009
Timesteps 5+   (BLUE bars): gradient magnitude ≈ 0.0022 – 0.0147
```

**The gradient effectively dies by timestep 4.**

From t=5 onward the gradient starts recovering. The peak signal is at t=18
(magnitude ≈ 0.0147), which is ~49× stronger than the signal at t=0
(magnitude ≈ 0.0003). The first 5 timesteps contributed almost nothing
to learning — the model was trained almost entirely on the last 15 steps.

---

## ⚠️ Must-Remember Concepts

### The 7-Step PyTorch Training Checklist
```python
model.train()                                          # 1. before training loop
optimizer.zero_grad()                                  # 2. before every backward
loss = criterion(pred, target)                         # 3. compute loss
loss.backward()                                        # 4. backprop
torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # 5. clip grads (RNNs always)
optimizer.step()                                       # 6. update weights
model.eval() + torch.no_grad()                         # 7. before every evaluation
```

### RNN Input Shape — Never Forget
```
(batch_size, seq_len, input_features)
     ↑            ↑         ↑
  how many    how long   how many numbers
  sequences  is each     per timestep
             sequence
```

### The Fixed Weights Clarification
- W does NOT change during a forward pass
- W is shared across ALL timesteps (same W applied 20 times)
- The hidden state h is what CHANGES at each step
- W is updated only during backprop (optimizer.step())

### unsqueeze(-1) vs unsqueeze(1) on 2D tensors
```
X shape: (5, 3)
.unsqueeze(1)  → (5, 1, 3)  ← new dim in MIDDLE  ❌ wrong for RNN
.unsqueeze(-1) → (5, 3, 1)  ← new dim at END      ✅ correct for RNN
```
RNN needs (batch, seq_len, features) — features dimension must be last.

### Sliding Window Math
```
data_points = N
seq_len     = L
sequences   = N - L          (NOT N)
train_size  = int(0.8 × sequences)
```
Always verify X_train.shape BEFORE training starts.

### Seeds — Three Separate Systems
```python
random.seed(42)           # Python's built-in random
np.random.seed(42)        # NumPy
torch.manual_seed(42)     # PyTorch CPU
torch.cuda.manual_seed(42)# PyTorch GPU
```
Setting one does NOT affect the others.

---

## 🤔 Concepts That Were Tricky — Re-read These

### 1. unsqueeze with negative index
- Negative index = count from the end (same as Python list indexing)
- `unsqueeze(-1)` = add new dimension at the very last position
- On a 1D tensor (5,): unsqueeze(1) and unsqueeze(-1) give the SAME result (5,1)
- The difference ONLY shows on 2D+ tensors

### 2. Why the same W is a feature, not a limitation
- W is shared across timesteps → learns PATTERNS not POSITIONS
- Like a CNN filter detecting edges anywhere in an image
- W learns "subject before verb" at position 3 AND position 47
- This is called temporal invariance

### 3. Hidden state norm fluctuating
- Norm fluctuates because torch.randn() gives different inputs each step
- random seed makes it REPRODUCIBLE (same random noise every run)
- random seed does NOT make it STABLE
- Stability comes from real structured data or a trained model

### 4. Test loss lower than train loss in sine wave
- This is NOT a bug — it is expected for perfectly periodic data
- Sine wave test portion is just as predictable as train portion
- Real-world data: usually test loss > train loss (overfitting risk)

### 5. grad_norms chain: `.abs().squeeze().cpu().numpy()`
```
.grad      → access gradient tensor (after backward())
.abs()     → take magnitude (remove +/- sign)
.squeeze() → remove size-1 dimensions for clean shape
.cpu()     → move from GPU VRAM → CPU RAM
.numpy()   → convert to NumPy array so matplotlib can plot it
```
This chain is STANDARD boilerplate for visualising any PyTorch tensor.

---

## 📈 Results from Today's Sine Wave Training

| Epoch | Train Loss | Test Loss | Error (approx) |
|-------|-----------|-----------|----------------|
| 0     | 0.4836    | 0.4607    | ±0.69 units    |
| 30    | 0.0281    | 0.0264    | ±0.17 units    |
| 90    | 0.0030    | 0.0009    | ±0.05 units    |
| 135   | 0.0024    | 0.0006    | ±0.02 units    |

Wave range is -1 to +1. Final error of ±0.02 = ~1% error. ✅

---

## 🔗 The Bigger Picture — Where This Leads

```
RNNs added memory to neural nets          ✅ solved sequences
Vanishing gradients broke long memory     ❌ RNNs forget the distant past
LSTMs added gates to control memory       ✅ Day 16 — solves this
Fixed hidden state bottleneck remains     ❌ can't hold everything
Attention lets model look back directly   ✅ Day 17/18 — solves this
Transformers replaced RNNs entirely       → how GPT works
```

---

