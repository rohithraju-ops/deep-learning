# Day 06 Notes — Custom Neural Networks, Iris Classification & PyTorch Deep Dive

---

### 1. What is `nn.Module` and why is it better than loose variables?

`nn.Module` is PyTorch's base class for every neural network. When you inherit
from it, you get for free:
- Automatic parameter tracking — it finds every `nn.Linear`, every weight & bias
- `.parameters()` works — so the optimizer can update everything in one call
- `.state_dict()` — save and load your model
- `.train()` / `.eval()` mode switching

Without it, you'd manually track every weight like Day 4:
```python
# Old way — loose variables, manual everything
W1 = np.random.randn(4, 8)
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 3)
b2 = np.zeros((1, 3))
# and manually update each one...

# nn.Module way — everything tracked automatically
model = NeuralNet(4, 8, 3)
optimizer.step()  # updates ALL weights in one line
```

### 2. What does model.train() vs model.eval() do? 
They flip a mode switch that changes behaviour of specific layers:

| model.train() | model.eval() |
|---------------|--------------|
| When to use | During training loop | During evaluation/testing |
| Dropout | ON — randomly drops neurons | OFF — all neurons active |
| BatchNorm | Uses batch statistics | Uses running statistics |
| Gradients | Not affected | Not affected (use no_grad() for that) |
```python
# Training loop
model.train()
outputs = model(X_train)
loss.backward()

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
```

### 3. Why do we need `torch.no_grad()` during evaluation?

Because it tells PyTorch: “Don’t build the computation graph, don’t store intermediate
values, don’t compute gradients.”

Without `no_grad()`:
- Memory usage explodes (stores everything for backprop)
- Computation slows down (extra graph-building overhead)
- No benefit — you don’t need gradients during inference

With `no_grad()`:
- Uses minimal memory
- Runs fast
- Pure inference mode

```python
# WRONG — wastes memory, slows down
model.eval()
outputs = model(X_test)  # graph built, gradients computed

# RIGHT — fast, no memory waste
model.eval()
with torch.no_grad():
    outputs = model(X_test)  # no graph, no gradients
```

### 4. Why CrossEntropyLoss for classification instead of MSE?

MSE measures distance between numbers — it treats your output like a value
on a number line. Classification is about which category wins, not distance.

True class: setosa (class 0)

- Model A: [0.90, 0.05, 0.05]  ← very confident, correct
- Model B: [0.40, 0.35, 0.25]  ← unsure, but still picks correct

MSE scores both roughly the same ← wrong

CrossEntropyLoss heavily rewards Model A ← correct
CrossEntropyLoss asks: "How much probability did you give the correct class?"
It punishes overconfident wrong answers extremely hard.

It also includes Softmax internally — so you never add Softmax to your output layer
manually when using CrossEntropyLoss. Adding it would apply Softmax twice and
flatten class differences → broken training.

### 5. What test accuracy did you achieve and what would improve it?

Model	Architecture	Test Accuracy
- NeuralNet (Sigmoid bug)	4 → 8 → 3	96.7%
- NeuralNet (bug fixed)	4 → 8 → 3	100%
- DeeperNet	4 → 16 → 8 → 3	~97–100%

What improved it:
Removing the Sigmoid from the output layer when using CrossEntropyLoss stopped the double-Softmax problem and gave cleaner gradients.

What could improve it further (on harder datasets):
- More epochs — loss was still decreasing at epoch 180
- Larger hidden layer — more capacity to learn patterns
- Dropout — prevents overfitting on small datasets
- Learning rate scheduler — reduce lr as training converges

### Class & nn.Module — How It Works

```python
class NeuralNet(nn.Module):                                   # inherit from nn.Module
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()   # run nn.Module's setup first
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu    = nn.ReLU()
```

### Output Layer Activation — Golden Rule

| Task | Loss Function | Output Activation |
|------|---------------|-------------------|
| Multi-class classification | CrossEntropyLoss | ❌ None (raw logits) |
| Binary classification | BCELoss | ✅ Sigmoid |
| Regression | MSELoss | ❌ None |


### StandardScaler

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)   # fit = learn mean & std | transform = apply formula
# z = (x - mean) / std_dev
# Result: every feature has mean=0, std=1
```

### Why StandardScaler Matters

| Feature | Without Scaling | With StandardScaler |
|---------|-----------------|----------------------|
| Range | 0.01–1000 | -3 to +3 |
| Gradient Flow | Unstable, explodes | Stable, smooth |
| Convergence | Slow, erratic | Fast, reliable |
| Loss Landscape | Twisted, bowl-shaped | Circular, easy to descend |

Without scaling, features with large values (like 1000) dominate the loss and
prevent small-value features (like 0.01) from ever learning.
