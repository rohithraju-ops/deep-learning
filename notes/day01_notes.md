# Day 01 — Neural Network Intuition Notes
**Date:** February 26, 2026

---

## Conceptual Questions

### 1. What is a neural network in one sentence?
A neural network is a system of layered mathematical functions that learns
to map inputs to outputs by adjusting weights through repeated exposure to data.

### 2. What does sigmoid do and why use it?
Sigmoid squishes any number into a value between 0 and 1 using the formula
1 / (1 + e^(-x)). It's used as an activation function to introduce non-linearity
into the network — without it, stacking layers would be pointless because the
whole network would collapse into a single linear equation.

### 3. What is a forward pass?
A forward pass is the process of feeding input data through the network layer
by layer — multiplying by weights, adding biases, applying activations — until
you get a final prediction. It's the network making its current best guess.

### 4. What makes a network "deep"?
A network is "deep" when it has more than one hidden layer. Depth allows the
network to learn hierarchical features — early layers detect simple patterns,
later layers combine them into complex ones.

---

## Core Concepts

### What is a neuron? What does "activation" mean?
A neuron takes multiple inputs, multiplies each by a weight, sums them up,
adds a bias, and passes the result through an activation function.
"Activation" refers to the output value of that neuron after the activation
function is applied — it represents how strongly that neuron "fired."

### What does a "layer" represent?
A layer is a group of neurons that all operate on the same input in parallel.
Each layer transforms its input into a new representation that the next layer
can use. Input layer receives raw data, hidden layers extract features,
output layer produces the final prediction.

### What is a weight and what is a bias?
A weight controls the strength of the connection between two neurons —
how much influence one neuron has on the next. A bias is an extra adjustable
number added to the weighted sum that lets the neuron shift its activation
independently of the input. Together: output = (weights · inputs) + bias.

### What does it mean for a network to "learn"?
Learning means adjusting the weights and biases to reduce the error between
the network's predictions and the actual correct answers. This is done through
backpropagation + gradient descent — the network measures how wrong it was,
figures out which weights caused the error, and nudges them in the right direction.
Repeat thousands of times → the network gets better.

---

## Key Formulas
- Forward pass: `output = sigmoid(np.dot(X, W) + b)`
- Matrix rule: `(2,) · (2, 3) → (3,)` — inner dims must match
- Sigmoid: `σ(x) = 1 / (1 + e^(-x))`

---