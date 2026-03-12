# Day 12 Notes — Convolutional Neural Networks (CNNs)

**Date:** March 10, 2026
**Topic:** CNNs — how they work, what they see, and why they beat normal NNs on images

---

## 1. What is a Convolution? (No Jargon)

A convolution is just **sliding a small grid of numbers (a filter) across an image and asking "how much does this patch look like my filter?" at every position.**

```
Image patch:        Filter (3×3):       Output (one number):
[ 0  0  0 ]        [-1 -1 -1]
[ 1  1  1 ]   ×    [ 0  0  0]   →   sum of all multiplications = activation
[ 0  0  0 ]        [ 1  1  1]
```

- If the patch matches the filter → **large positive number** (strong activation)
- If the patch is the opposite → **large negative number**
- If no match → **near zero**

The filter slides to every position, producing one activation per location → this output grid is called a **feature map**.

---

## 2. What is Weight Sharing and Why Does it Make CNNs Efficient?

**Weight sharing = the same filter (same 9 numbers) is reused at every single pixel position in the image.**

In a fully connected layer, every pixel gets its own unique weight connecting it to every neuron. In a conv layer, the same 3×3 filter slides everywhere — it doesn't get a new set of weights per position.

```
Fully Connected:  784 pixels × 256 neurons = 200,704 unique weights
                  (each pixel has its own connection to every neuron)

Conv Layer:       32 filters × 3×3 = 288 weights
                  (same 9 numbers used at all 784 positions)
```

**Why this is powerful:**
- The network learns to detect a vertical edge once — then detects it everywhere
- Massively fewer parameters needed for spatial feature detection
- Forces the network to learn patterns that are **position-independent**

> ⚠️ Important nuance: The CNN in Day 12 actually has MORE total parameters (421,642) than
> the Day 7 FC net (234,752) — but the conv layers themselves only use 18,816 params.
> The bulk of CNN params are still in the FC tail at the end. Weight sharing makes the
> *feature detection* part efficient, not the whole network.

---

## 3. What Does MaxPooling Do and Why Is It Used?

**MaxPooling takes a 2×2 window, keeps only the largest value, and throws the rest away — then slides to the next non-overlapping window.**

```
Input (4×4):          After MaxPool(2×2):
[ 1  3  2  4 ]        [ 3  4 ]
[ 5  6  1  2 ]   →    [ 6  8 ]
[ 3  2  8  1 ]
[ 1  4  2  7 ]
```

**What it achieves:**
- **Spatial compression** — halves width and height (28×28 → 14×14 → 7×7)
- **Translation robustness** — if a feature shifts by 1 pixel, MaxPool still keeps the same max value
- **Keeps the strongest signal** — if an edge was detected anywhere in that 2×2 region, it survives
- **Reduces computation** — fewer numbers flowing into deeper layers

> ⚠️ MaxPool has NO learnable parameters — it's a fixed operation, not trained by backprop.
> It's purely a downsampling + robustness mechanism.

---

## 4. What Does `padding=1` Do to Output Spatial Dimensions?

Without padding, a 3×3 filter sliding over a 28×28 image produces a **26×26 output** — the filter can't sit with its center on the edge pixels, so the edges get cut off.

`padding=1` adds a border of zeros around the image before the convolution:

```
Original image:    28×28
After padding=1:   30×30  (one row/col of zeros on each side)
After conv (3×3):  28×28  ← same size as input ✅
```

**Why this matters:**
- Preserves spatial dimensions through conv layers
- Edge pixels get to participate fully in convolutions
- Without it, the image shrinks after every conv layer and you lose edge information

> Formula for output size (no stride):
> output = input + 2×padding - kernel_size + 1
> = 28 + 2(1) - 3 + 1 = 28 ✅

---

## 5. What Patterns Do the Learned Filters Detect?

Nobody told the filters what to look for — **backprop discovered these patterns entirely from data** over 300,000 training passes.

From the Day 12 filter visualizations (32 filters, 3×3 each):

| Filter Type | What it looks like | What it detects |
|---|---|---|
| Horizontal detector | Bright top row, dark bottom | Horizontal edges / top strokes |
| Diagonal detector | Bright top-right to dark bottom-left | ╱ diagonal strokes |
| Corner / junction detector | Bright center, mixed surroundings | Where two strokes meet |
| Presence detector | Uniformly positive weights | "Is there any ink here?" |
| Endpoint detector | Bright center, negative ring | Stroke termination points |
| Suppression detector | Negative background weights | Inverted — highlights strokes by suppressing background |
| Blur / low-freq detector | All small positive weights | Overall shape, no edge specificity |
| Near-blank filter | Near-zero weights everywhere | Redundant — not needed for this dataset |

**Key insight:** The filter weights are the *questions*. The feature maps are the *answers for a specific image*. Bright yellow in a feature map = "YES, this pattern is strongly present here."

---

## 6. Why Do CNNs Outperform Fully Connected Nets on Images?

Three fundamental reasons:

### Reason 1 — Spatial Awareness
```
FC net sees:   [0, 0, 0.8, 0.9, 0.7, 0, 0 ...]  ← flat list, no structure
CNN sees:      "horizontal stroke at top, diagonal slash below" ← actual geometry
```
FC nets treat pixel position 42 and pixel position 43 as completely unrelated inputs. CNNs explicitly model that neighboring pixels form structures.

### Reason 2 — Translation Invariance
```
FC net: "7" shifted left 2 pixels = DIFFERENT input, potentially wrong prediction
CNN:    "7" shifted left 2 pixels = same filter fires at a slightly different position
         → same feature maps → same correct prediction
```
The sliding filter naturally handles shifts, rotations, and scale variations that completely fool FC nets.

### Reason 3 — Hierarchical Feature Learning
```
Conv Layer 1:  detects edges, strokes, corners (simple primitives)
Conv Layer 2:  detects combinations of primitives (angles, curves, junctions)
FC Layer:      combines all detected features into a digit class
```
FC nets can only do step 3. CNNs do all three.

**Day 12 result:**
```
Day 7  FC Net:  ~97–98% accuracy  (234,752 params)
Day 12 CNN:      99.11% accuracy  (421,642 params total / 18,816 conv params)

On 10,000 test images:
  FC  misclassified: ~200–300 digits
  CNN misclassified: ~89 digits
```

---

## 🔑 Crucial Concepts to Remember

### The CNN Architecture Pipeline
```
Input (1×28×28)
  → Conv1 (32 filters, 3×3, padding=1) → ReLU → MaxPool(2×2)   [→ 32×14×14]
  → Conv2 (64 filters, 3×3, padding=1) → ReLU → MaxPool(2×2)   [→ 64×7×7]
  → Flatten                                                       [→ 3136]
  → Linear(3136 → 128) → ReLU
  → Linear(128 → 10)
  → Softmax → predicted digit
```

### ReLU After Every Conv — Why?
Without ReLU, stacking conv layers is mathematically equivalent to a single conv layer (linear transforms compose into one linear transform). ReLU introduces non-linearity so each layer can learn genuinely new representations.

### The `model.eval()` vs `model.train()` Switch
- `model.train()` — enables dropout + batchnorm in training mode
- `model.eval()` — disables them for inference; essential before evaluating accuracy
- Always pair `model.eval()` with `torch.no_grad()` during evaluation to save memory and speed

### Feature Maps vs Filter Weights — Don't Confuse These
```
Filter weights  = the 3×3 grid of learned numbers (what the filter looks for)
Feature map     = the 28×28 output after sliding that filter over the image (what it found)
```
You visualized both in Day 12. The weights tell you the filter's "specialty." The feature maps show its response to a specific image.

### Why `torch.max(outputs, 1)`?
The model outputs 10 raw scores (logits), one per digit class. `torch.max(..., 1)` picks the index of the highest score along dimension 1 — that index IS the predicted digit (0–9).

---

## ⚠️ Things Easy to Gloss Over — Don't Forget These

### 1. The Parameter Paradox
The CNN has MORE total parameters than the Day 7 FC net, yet wins. This trips people up. The efficiency of weight sharing is in the *conv layers only* (18,816 params). The FC tail still dominates total count. On larger images (e.g. 224×224 in ImageNet), the gap would be enormous.

### 2. The Diagonal Direction of a "7"
The stroke of a "7" runs **top-right → bottom-left (╱)**, not top-left → bottom-right (╲). Filters that detect this slash are tuned to a **negative-slope** (forward-slash) direction. Getting directional details wrong when analyzing feature maps leads to wrong conclusions about what a filter learned.

### 3. Depth Dimension in Conv2
```
conv2_params = 64 × 32 × 3 × 3 + 64
```
Conv2's filters are **not** 3×3 — they are **32×3×3** (depth × height × width). Each of the 64 filters in layer 2 has to look across all 32 feature maps from layer 1 simultaneously. The depth dimension is easy to forget but crucial for understanding parameter counts.

### 4. MaxPool Has No Learnable Parameters
MaxPool is not "trained." It's a fixed rule: take the max. When you count model parameters, MaxPool contributes zero. Only Conv and Linear layers have learnable weights.

### 5. `unsqueeze(0)` When Running Single Images
```python
sample_image = test_data[0][0].unsqueeze(0)  # shape: (1, 1, 28, 28)
```
The model always expects a **batch dimension** first. A single image has shape (1, 28, 28). `unsqueeze(0)` adds the batch dimension → (1, 1, 28, 28). Forgetting this causes a shape mismatch error.

### 6. Filter Visualization Used `RdBu`, Feature Maps Used `viridis`
- `RdBu` (red-blue diverging) for filter weights — shows positive vs negative weights symmetrically around zero
- `viridis` (dark purple → yellow) for feature maps — shows activation strength from 0 upward
Using the wrong colormap would misrepresent what you're looking at.

---

## 📌 One-Line Summaries

| Concept | One Line |
|---|---|
| Convolution | Slide a small pattern-detector across the whole image |
| Weight sharing | One filter, reused at every pixel position |
| MaxPooling | Keep the strongest signal, throw away the rest, halve the size |
| padding=1 | Add zeros around the border so the output stays the same size |
| Feature map | The activation grid showing where a filter fired in this specific image |
| CNN vs FC | CNNs understand spatial structure; FC nets just see flat numbers |
| ReLU after conv | Prevents multiple conv layers from collapsing into one |

---

*Day 12 complete — CNNs from scratch, filter visualization, feature maps, and parameter analysis. Next up: deeper architectures, batch normalization, and dropout.*
