# Day 05 Notes — PyTorch, Autograd & XOR with Neural Networks

---

## Day 4 Quick Recall Check


 1. What does a gradient tell you?
 It tells you the direction and magnitude of the steepest increase in loss.
 We move in the OPPOSITE direction to reduce loss (gradient descent).

 2. What is the chain rule in one sentence?
 The chain rule says: to find how a change in an early variable affects the final output,
 multiply all the small derivatives along the path from that variable to the output.

 3. What order do we compute gradients — forward or backward?
 Forward pass first (compute predictions & loss),
 then BACKWARD pass (compute gradients from output → input, using the chain rule).

 4. What does sigmoid_derivative equal?
 sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
 If s = sigmoid(x), then ds/dx = s * (1 - s)


---

## Core Concepts

### NumPy Array vs PyTorch Tensor

| Feature            | `np.array`       | `torch.tensor`          |
|--------------------|-----------------|--------------------------|
| Purpose            | General math     | Neural network training  |
| Backprop           |    Manual        |  Automatic (`autograd`)|
| GPU support        |   CPU only       |  `.to('cuda')`         |
| Gradient tracking  |    None          |  `requires_grad=True`  |

Both hold the same numbers in the same shapes.
A PyTorch tensor is essentially a NumPy array **that knows calculus**.

---

## Autograd — The Core Engine

### What `requires_grad=True` does
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x   # y = x³ + 2x
y.backward()
print(x.grad)         # → 14.0
```

- PyTorch **records every operation** on watched tensors into a **computational graph**
- `.backward()` walks the graph in reverse, applying the chain rule at each node
- The result is stored in `.grad`

### The Computational Graph (behind the scenes)
```
x  →  [x³]  →  [+ 2x]  →  y
       ↑            ↑
  PowBackward   AddBackward   ← PyTorch records these silently
```

### Manual Verification
```
y  =  x³ + 2x
dy/dx  =  3x² + 2
At x=2:  3(4) + 2 = 14  -->  matches PyTorch
```

> **Tape recorder analogy**: Forward pass = recording. `.backward()` = playing in reverse. `.grad` = final written result.

---

## PyTorch XOR Network — Full Code Breakdown

### NumPy → PyTorch Mapping

```
NUMPY CODE                       PYTORCH EQUIVALENT
─────────────────────────────────────────────────────────
W1, b1, W2, b2 = ...         →   nn.Linear layers (auto-initialized)
sigmoid(Z)                   →   nn.Sigmoid()
mse_loss(pred, y)            →   nn.MSELoss()
All manual backprop math     →   loss.backward()
w -= lr * dw  (×4 lines)     →   optimizer.step()
```

### The Training Loop — 3 Lines Replace 30

```python
# ============ NUMPY — manual everything ============
Z1 = X @ W1 + b1;  A1 = sigmoid(Z1)
Z2 = A1 @ W2 + b2; A2 = sigmoid(Z2)
loss = mse_loss(A2, y)
# ... 12 lines of manual chain rule ...
W2 -= lr * dL_dW2; b2 -= lr * dL_db2
W1 -= lr * dL_dW1; b1 -= lr * dL_db1

# =========== PYTORCH — automated ============
predictions = model(X_t)           # full forward pass
loss        = loss_fn(predictions, y_t)
optimizer.zero_grad()              # clear old gradients
loss.backward()                    # full backward pass ← THIS IS ALL YOUR BACKPROP
optimizer.step()                   # update all weights ← THIS IS GRADIENT DESCENT
```

---

## Conceptual Questions

### 1. What does `loss.backward()` actually do under the hood?
It traverses the computational graph **in reverse** from the loss node back to every
leaf tensor (weights & biases), applying the chain rule at each step automatically.
It computes ∂loss/∂w for every single weight in the network simultaneously and stores
the result in each tensor's `.grad` attribute. This is the entire manual backprop
math you wrote in Day 4 — just automated.

---

### 2. What does `optimizer.zero_grad()` do and why is it necessary?
PyTorch **accumulates** (adds) gradients by default — every `.backward()` call
adds to existing `.grad` values rather than replacing them.

Without zeroing first:
```
Epoch 1: gradient = 0.5
Epoch 2: gradient = 0.5 + 0.5 = 1.0   ← wrong, doubled!
Epoch 3: gradient = 1.0 + 0.5 = 1.5   ← exploding!
```
`optimizer.zero_grad()` wipes all `.grad` values to zero before the next backward pass.

**Always follow the 3-step dance every epoch:**
```python
optimizer.zero_grad()   # 1. wipe the slate
loss.backward()         # 2. calculate fresh gradients
optimizer.step()        # 3. apply them
```

---

### 3. Why is tracking tensor shapes so critical in backpropagation?
Matrix multiplication requires exact shape alignment. During backprop, every gradient
must have the exact same shape as its corresponding weight matrix, or the chain rule
math breaks. For example:

- `W1` shape `(2, 3)` → gradient `dL/dW1` must also be `(2, 3)`
- If shapes mismatch, PyTorch throws a runtime error OR silently gives wrong gradients
- `A1.T @ dL_dZ2` — the transpose is required to make shapes align for `dL/dW2`

Shape errors are the #1 debugging task in real neural network code.

---

### 4. What was the hardest part of manual backprop vs PyTorch?
**Manual backprop**: Keeping track of every intermediate variable (`Z1`, `A1`, `Z2`, `A2`)
and ensuring every chain rule step was in the correct order with correct shapes.
One wrong transpose or wrong axis in `np.sum()` and the whole network silently
produces garbage gradients.

**PyTorch**: The hardest part shifts from *computing* gradients to *understanding*
what is happening inside `loss.backward()`. The code is simple but you must know
why `zero_grad()` is needed, what the graph looks like, and how to debug shape errors.

Days 1–4 manual backprop was essential — it's what makes PyTorch debuggable.

---

### 5. XOR can't be solved by a single neuron — why not? What did the hidden layer do?
A single neuron draws **one straight line** through the input space. XOR is not
linearly separable — no single line can separate the 4 XOR points into correct groups:

```
(0,0)→0    (0,1)→1
(1,0)→1    (1,1)→0
```

Plot these 4 points — you cannot draw one straight line to correctly separate 0s from 1s.

**The hidden layer's job**: Each hidden neuron draws its own line. Together, 3 hidden
neurons create a region (bounded by multiple lines) that correctly captures the XOR pattern.
The hidden layer **transforms** the input into a new space where the problem *is* linearly
separable — then the output neuron can solve it with a single line in that new space.

This is the fundamental power of depth in neural networks: **learned feature transformation**.

---

## Key Insight from Today

> You built the same XOR network **twice** — once in 80 lines of manual NumPy,
> once in 15 lines of PyTorch. The math is 100% identical. PyTorch just automates
> the painful parts. Knowing the manual version means you truly understand what
> every PyTorch line is doing under the hood. That is the difference between
> a practitioner and someone who just calls `.fit()`.

---

## Vocab Quick Reference

| Term | Meaning |
|------|---------|
| `requires_grad=True` | Tell PyTorch to watch & record this tensor |
| Computational graph | The map of operations PyTorch builds during forward pass |
| `.backward()` | Traverse graph in reverse, compute all gradients |
| `.grad` | Where PyTorch stores ∂loss/∂w for each tensor |
| `optimizer.zero_grad()` | Reset all gradients to 0 before next epoch |
| `optimizer.step()` | Apply all computed gradients to update weights |
| `nn.Linear(in, out)` | Creates W and b, performs Z = XW + b |
| `nn.Sequential` | Run layers in order, top to bottom |
| Autograd | PyTorch's automatic differentiation engine |