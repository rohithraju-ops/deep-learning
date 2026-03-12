# Day 02 — Linear Algebra Essentials Notes
**Date:** February 27, 2026

---

## Experiment Answers

### 1. Change W1 = np.random.randn(2, 5) — what else do you need to change?
When W1 changes from shape (2, 3) to (2, 5), the hidden layer now has 5 neurons
instead of 3. This means:
- b1 must change from np.zeros(3) → np.zeros(5) — one bias per neuron
- W2 must change its input dimension from (3, 1) → (5, 1) — it now receives
  5 inputs from the hidden layer instead of 3
- b2 stays the same shape (1,) — output layer is unchanged

Rule: whenever you change the number of neurons in a layer, you must update
the bias for that layer AND the weight matrix of the NEXT layer.

### 2. What happens if the shapes don't match?
You get a ValueError like:
"matmul: Input operand 1 has a mismatch in its core dimension"
NumPy refuses to multiply matrices where the inner dimensions don't match.
e.g. (2,) · (3, 3) → CRASH because 2 ≠ 3.
This is actually useful — shape errors are your first debugging tool.
Always check .shape on your arrays when something breaks.

### 3. Why is W1 shape (2, 3) and not (3, 2)?
Because np.dot(X, W1) requires the inner dimensions to match:
  X is shape (2,) and W1 must be (2, something)
  (2,) · (2, 3) → (3,)  
  (2,) · (3, 2) → CRASH 
The rule is: W1 shape = (number of inputs, number of neurons)
Rows = inputs coming IN, Columns = neurons going OUT.

---

## Notes in My Own Words

### 1. What is a dot product and what does it represent in a neuron?
A dot product multiplies two arrays element-by-element and sums the results.
e.g. [1,2] · [3,4] = (1×3) + (2×4) = 11
In a neuron, it represents the weighted sum of all inputs — how much total
signal is flowing into that neuron. A high dot product means the inputs
strongly aligned with that neuron's weights, causing it to activate more.

### 2. Why do we use matrix multiplication instead of looping through neurons?
A for-loop computing each neuron one at a time is slow — it processes
sequentially. Matrix multiplication does ALL neurons simultaneously in one
operation, and NumPy/PyTorch runs it on optimized hardware (CPU SIMD / GPU).
A single np.dot() call is 10-100x faster than the equivalent Python loop.
This is why GPUs are so powerful for deep learning — they're built for
exactly this kind of parallel matrix math.

### 3. What does the shape of a weight matrix tell you about a layer?
W shape = (inputs, neurons) tells you everything about that layer:
- First number = how many inputs it expects to receive
- Second number = how many neurons (outputs) it produces
e.g. W shape (784, 128) means: takes 784 inputs, produces 128 outputs.
You can read the entire architecture of a network just from its weight shapes.

### 4. What is a transpose and where will it matter later?
A transpose flips a matrix along its diagonal — rows become columns and
columns become rows. W shape (2, 3) transposed → W.T shape (3, 2).
It matters later in:
- Backpropagation: gradients flow backwards, requiring W.T to reverse
  the matrix multiplication direction
- Different frameworks use W vs W.T conventions — knowing this prevents
  silent shape bugs

### 5. What shape would W need to be to connect 4 inputs to 7 neurons?
W shape = (4, 7)
- 4 rows = 4 inputs
- 7 columns = 7 neurons
np.dot(X, W) → (4,) · (4, 7) → (7,) 
Bias b would be shape (7,) — one per neuron.

---

## Key Rules to Remember
- W shape = (inputs_in, neurons_out)
- b shape = (neurons_out,) — always matches columns of W
- Inner dimensions MUST match for np.dot() to work
- Change neurons in layer N → update b[N] AND W[N+1]
- Matrix multiply >> for loops, always

---

