# Day 08 Notes — PyTorch Internals: Tensors, Autograd & Computational Graph

---

## Step 8 — Written Answers

### 1. What is `requires_grad=True` and what does it enable?

`requires_grad=True` is the on/off switch that tells PyTorch:
**"watch every operation done to this tensor and record it."**

```python
x = torch.tensor(3.0, requires_grad=True)   # PyTorch starts watching
y = x ** 2                                   # recorded: y = x²
z = y + 5                                    # recorded: z = x² + 5

z.backward()       # walk the recording backwards
print(x.grad)      # → 6.0  (d/dx of x²+5 = 2x = 2×3 = 6)
```

**What it enables:**
- `.backward()` — computes all gradients automatically via chain rule
- `x.grad` — stores the computed gradient on the tensor itself
- The entire backprop of your neural network in one line

**Who needs it and who doesn't:**
```python
# Weights — YES, optimizer must update them
nn.Linear(784, 128)   # requires_grad=True by default for all parameters

# Input data — NO, we never update pixels
X_train = torch.tensor(data, dtype=torch.float32)  # requires_grad=False
```

> `torch.no_grad()` is the global off switch — stops recording for
> everything inside the block. Used during evaluation to save
> memory and speed, since you never call `.backward()` there.

---

### 2. What is a computational graph — describe it in your own words?

A computational graph is PyTorch's **live recording of every math
operation** that happens during a forward pass.

```
Input X
   ↓
X @ W1 + b1     ← node recorded
   ↓
ReLU(...)       ← node recorded
   ↓
... @ W2 + b2   ← node recorded
   ↓
CrossEntropyLoss ← node recorded
   ↓
loss.backward() → walks ALL nodes backwards, applies chain rule at each
```

Each node stores:
1. The result of the operation
2. How to compute the gradient flowing back through it

**Key property — it's dynamic:**
The graph is built fresh every single forward pass and thrown away
after `.backward()`. This means loops, conditionals, and variable-length
inputs all work naturally — the graph just rebuilds differently each time.

```python
for epoch in range(100):
    output = model(X)    # new graph built here every epoch
    loss.backward()      # graph walked backwards and discarded
    optimizer.step()     # weights updated using the gradients
```

---

### 3. Why do gradients accumulate and why is `zero_grad()` critical?

PyTorch **adds new gradients on top of existing ones** by default
instead of replacing them. So if you don't clear them between steps:

```python
# Without zero_grad():
# Step 1: x.grad = 6.0
# Step 2: x.grad = 6.0 + 6.0 = 12.0  ← wrong! doubled
# Step 3: x.grad = 12.0 + 6.0 = 18.0 ← getting worse every step
```

This is intentional design — some advanced use cases need to accumulate
gradients across multiple batches. But for standard training it breaks
everything if you forget to clear.

```python
# Correct training order — every single step:
optimizer.zero_grad()   # 1. clear old gradients
loss.backward()         # 2. compute fresh gradients
optimizer.step()        # 3. update weights with fresh gradients
```

> `zero_grad()` must come BEFORE `loss.backward()` every step.
> Forgetting it is one of the most silent and destructive bugs in
> PyTorch — model trains but learns incorrectly with no error thrown.

---

### 4. What's the difference between `model.eval()` and `torch.no_grad()`?

They do completely different things and you need BOTH during evaluation:

| | `model.eval()` | `torch.no_grad()` |
|---|---|---|
| What it does | Changes layer behaviour | Stops graph recording |
| Affects | Dropout, BatchNorm | Memory & compute |
| Dropout | Turned OFF | Not affected |
| BatchNorm | Uses stored running stats | Not affected |
| Gradient tracking | Not affected | Fully disabled |
| Memory saved | No | Yes — no graph stored |

```python
# Correct evaluation block — needs BOTH:
model.eval()                    # flip layer behaviour
with torch.no_grad():           # stop building graph
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
```

**The analogy:**
- `model.eval()` = switching from "practice mode" to "exam mode"
- `torch.no_grad()` = putting down the pencil (no need to write anything down)

Forgetting `model.eval()` → Dropout randomly drops neurons during
testing → inconsistent, lower accuracy.
Forgetting `torch.no_grad()` → PyTorch builds a full graph for every
test batch → wastes memory for no reason.

---

### 5. What does `.item()` do and when do you need it?

`.item()` extracts a **single Python number** out of a tensor:

```python
loss = torch.tensor(0.4231)   # this is a tensor
loss.item()                   # → 0.4231  (plain Python float)

correct = torch.tensor(95)
correct.item()                # → 95  (plain Python int)
```

**Why you need it:**
Tensors live on GPU and carry gradient tracking machinery.
You can't use them directly in regular Python operations:

```python
# Won't work or gives wrong output
total_loss += loss            # adds tensor to float — type mismatch
print(f"Loss: {loss}")        # prints "tensor(0.4231)" not "0.4231"

# Correct
total_loss += loss.item()     # plain float addition
print(f"Loss: {loss.item():.4f}")   # prints "0.4231"
```

**The two places you always use it:**
```python
running_loss += loss.item()                       # logging loss
correct += (predicted == labels).sum().item()     # counting correct
```

`.sum()` gives a tensor, `.item()` pulls the number out of it.
Always use `.item()` whenever you want a plain number for printing,
storing in a list, or doing regular Python math.

---

## The Three PyTorch Pillars

### Tensors
Multi-dimensional arrays that can live on GPU:
```python
scalar  = torch.tensor(42)              # 0D
vector  = torch.tensor()       # 1D — shape (3,)
matrix  = torch.tensor([,])   # 2D — shape (2,2)
batch   = torch.zeros(64, 1, 28, 28)   # 4D — shape (64,1,28,28)
```

### Autograd
Records operations on `requires_grad=True` tensors and computes
gradients automatically:
```
forward pass  → builds the computational graph
.backward()   → walks it in reverse, applies chain rule at each node
.grad         → stores the result on each watched tensor
```

### Computational Graph
A live map of every operation during the forward pass.
Dynamic — rebuilt fresh every forward pass, discarded after `.backward()`.

---

## Useful Functions from Today

| Function | What it does |
|---|---|
| `torch.manual_seed(42)` | Fix random seed — same results every run |
| `torch.linspace(a, b, n)` | n evenly spaced numbers from a to b |
| `tensor.unsqueeze(1)` | Add a dimension → (100,) becomes (100,1) |
| `tensor.item()` | Extract plain Python number from tensor |
| `torch.zeros(shape)` | Tensor filled with 0.0 |
| `torch.ones(shape)` | Tensor filled with 1.0 |
| `torch.randn(shape)` | Random tensor, mean=0, std=1 |
| `requires_grad=True` | Tell PyTorch to track this tensor |
| `optimizer.zero_grad()` | Clear gradients before each backward pass |

---

## Doubts & Clarifications — Day 08

---

### Doubt 1 — What does `unsqueeze(1)` actually do to a tensor?

**Question:** What is `X = torch.linspace(-1, 1, 100).unsqueeze(1)` doing
and how do I visualize it as a matrix?

**Answer:**
- `torch.linspace(-1, 1, 100)` creates a **flat 1D tensor** of 100 numbers, shape `(100,)`
- `.unsqueeze(1)` adds a new dimension at position 1, turning it into shape `(100, 1)`

```
BEFORE unsqueeze → shape (100,)     = flat list, no rows/columns
AFTER  unsqueeze → shape (100, 1)   = 100 rows, 1 column each

[[-1.0000],
 [-0.9798],
 [-0.9596],
    ...
 [ 0.9596],
 [ 0.9798],
 [ 1.0000]]
```

**Why it's needed:**
`nn.Linear` expects input shaped as `(batch_size, features)`.
Without `unsqueeze`, PyTorch sees 100 numbers but doesn't know
if that means 100 samples or 100 features. The `(100, 1)` shape
makes it unambiguous — **100 samples, 1 feature each**.

---

### Doubt 2 — What does `torch.manual_seed(42)` do?

**Question:** What is `torch.manual_seed(42)` doing?

**Answer:**
Sets the **starting point of PyTorch's random number generator** so
every run produces the exact same "random" numbers.

```python
torch.manual_seed(42)
torch.randn(3)  # → tensor([ 0.3367,  0.1288,  0.2345]) — always the same

# Without seed:
torch.randn(3)  # → different every run
```

**Why 42?**
It's a joke from *The Hitchhiker's Guide to the Galaxy* where 42 is
"the answer to life, the universe, and everything."
Any number works — 42 is just the ML community's default by convention.

**Full reproducibility setup:**
```python
torch.manual_seed(42)   # PyTorch randomness
np.random.seed(42)      # NumPy randomness
random.seed(42)         # Python randomness
```
Each library has its own random engine — seed all three for full
reproducibility.

---

### Key Takeaway from Today's Doubts

| Concept | The one-liner |
|---|---|
| `unsqueeze(1)` | Turns a flat list into a proper column — shape `(n,)` → `(n, 1)` |
| `torch.manual_seed(42)` | Freeze randomness so every run gives identical results |
