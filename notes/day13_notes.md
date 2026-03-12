# Day 13 Notes — CIFAR-10 CNN with Advanced Training Techniques

**Date:** March 11, 2026
**Topic:** Real-world image classification — data augmentation, BatchNorm, Dropout2d, cosine annealing
**Result:** 🏆 82.45% accuracy on CIFAR-10 (target was 75–80%)

---

## 1. Why is Data Augmentation Important for CIFAR-10 but Less Critical for MNIST?

**Short answer:** CIFAR-10 has far more visual complexity and far less data per class.

```
MNIST:
  - 60,000 images, 10 classes = 6,000 per class
  - Grayscale, 28×28, white digit on black background
  - Every "7" looks roughly like every other "7"
  - Variation is minimal → network sees enough diversity naturally
  - Augmentation helps but isn't critical

CIFAR-10:
  - 50,000 images, 10 classes = 5,000 per class
  - Colour, 32×32, real-world photos with backgrounds
  - A "cat" can be any breed, pose, lighting, background, distance
  - 5,000 examples is NOT enough to cover all that variation
  - Without augmentation → network memorises training images
  - With augmentation → same 5,000 images shown in infinite variations
```

**The three augmentations used and what each prevents:**

| Transform | Simulates | Prevents memorising |
|---|---|---|
| `RandomHorizontalFlip(p=0.5)` | Animal/object facing either direction | "This dog faces right = dog" |
| `RandomCrop(32, padding=4)` | Subject slightly off-centre | "Dog is always centred = dog" |
| `ColorJitter(0.2, 0.2, 0.2)` | Different lighting, time of day | "Dog is always this brown = dog" |

> ⚠️ Augmentation is ONLY applied during training, never during testing.
> Test transform = ToTensor() + Normalize() only — clean, consistent, reproducible.

---

## 2. What is Dropout2d and How is it Different from Regular Dropout?

Both randomly zero out values during training to prevent overfitting. The difference is **what unit gets zeroed.**

```
Regular nn.Dropout — zeros individual pixels:
Feature map channel:
[ 0.8  0.0  0.3 ]   ← random individual pixels zeroed
[ 0.0  0.5  0.9 ]
[ 0.2  0.0  0.6 ]

nn.Dropout2d — zeros entire channels:
Channel 1: [ 0.8  0.7  0.3 ]   ← kept entirely
           [ 0.9  0.6  0.5 ]
Channel 2: [ 0.0  0.0  0.0 ]   ← entire channel zeroed
           [ 0.0  0.0  0.0 ]
Channel 3: [ 0.5  0.2  0.9 ]   ← kept entirely
           [ 0.3  0.8  0.4 ]
```

**Why Dropout2d is correct for conv layers:**

In a feature map, neighbouring pixels are spatially correlated — if you drop pixel (3,4), the network can still recover that information from pixels (3,3), (3,5), (4,4) etc. Regular dropout on feature maps barely hurts the network because adjacent pixels carry the same information.

Dropout2d forces the network to work without entire feature detectors at once — much stronger regularisation for spatial data.

```
Use nn.Dropout   → after Linear (FC) layers
Use nn.Dropout2d → after Conv layers
```

**Dropout rates used in Day 13 (increasing with depth):**
```
Block 1 (early, simple features):   Dropout2d(0.2)  ← light
Block 2 (mid, complex features):    Dropout2d(0.3)  ← medium
Block 3 (deep, abstract features):  Dropout2d(0.4)  ← heavier
Classifier FC layer:                Dropout(0.5)    ← heaviest
```
Deeper layers need more regularisation because they can overfit faster.

---

## 3. What is Cosine Annealing — Plain English?

**Cosine Annealing is a learning rate schedule that starts high, drops following a cosine curve, and reaches near-zero by the final epoch.**

```
lr
│
0.001 ┤●
      │  ╲
      │    ╲
0.0005┤      ╲___
      │           ╲___
      │                ╲___●
~0.00 └─────────────────────── epochs
      0         10        20
```

- Drops **slowly** at first (gentle slope early on)
- Drops **steeply** through the middle
- **Flattens near zero** at the end

**Why this works better than a fixed learning rate:**

```
Fixed lr = 0.001 for all 20 epochs:
  Epochs 1–10:  ✅ good — big steps help explore loss landscape
  Epochs 11–20: ❌ bad  — big steps overshoot the minimum, bounces around

Cosine annealing:
  Epochs 1–10:  ✅ high lr — fast exploration
  Epochs 11–20: ✅ low lr  — precise fine-tuning near the solution
```

Think of it like parking a car — drive fast to get close, then slow down to park precisely.

**How to use it in code:**
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# At the END of each epoch (after both train AND eval):
scheduler.step()   # ← one call per epoch, not per batch
```

> ⚠️ `scheduler.step()` must go AFTER the eval loop, not inside the training loop.
> Calling it per batch would drop the LR way too fast.

**What Day 13 results showed:**
```
Epochs 1–8 (LR high):    accuracy jumped 5–8% per epoch
Epochs 15–20 (LR low):   accuracy moved <0.5% per epoch
→ schedule matched the learning curve perfectly
```

---

## 4. Which Classes Had the Lowest Accuracy and Why?

**Day 13 per-class results:**

| Class | Accuracy | Verdict |
|---|---|---|
| automobile | 93.3% | Easy — consistent silhouette |
| truck | 92.8% | Easy — consistent silhouette |
| frog | 89.9% | Easy — distinctive colour/shape |
| ship | 89.5% | Easy — horizontal, water bg |
| horse | 85.4% | Medium — varied poses |
| deer | 83.4% | Medium — varied poses |
| airplane | 82.3% | Medium — context-dependent |
| dog | 76.2% | Hard — visually similar to cat |
| bird | 70.1% | Hard — extreme pose/species variation |
| **cat** | **61.6%** | **Hardest — confused with dog/deer** |

**Why cat is consistently the hardest class in CIFAR-10:**
- Cat and dog share the same body shape, fur texture, size at 32×32
- Cat faces, dog faces, and deer faces are nearly indistinguishable at low resolution
- Cats appear in more varied postures than any other class
- This is a known benchmark challenge — even state-of-the-art models in 2012 struggled

**The pattern: objects beat animals**
```
Vehicles/objects → predictable shape, silhouette, orientation → easy
Animals          → huge variation in pose, lighting, species  → hard
```

**The one error in the prediction grid:**
- Airplane on a billboard → predicted as ship
- The model saw a horizontal shape with text/grey background and guessed ship
- Context (it's a sign) is invisible to a CNN — it only sees pixel patterns

---

## 5. What Would You Change to Push Accuracy Above 85%?

**From the training curves, the model was still improving at epoch 20** — it hadn't plateaued. The most impactful changes:

### Change 1 — More Epochs (Easiest win)
```python
# Change T_max too so cosine schedule stretches across all epochs
for epoch in range(30):   # was 20
scheduler = CosineAnnealingLR(optimizer, T_max=30)
```
The accuracy curve was still rising at epoch 20. Epochs 21–30 would likely add 1–2%.

### Change 2 — Add a 4th Conv Block
```python
self.block4 = nn.Sequential(
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout2d(0.4)
)
# Update classifier input: 256*2*2 = 1024
```
More depth = more abstract feature combinations = better discrimination.

### Change 3 — Stronger Augmentation
```python
transforms.RandomRotation(15),          # cats appear at all angles
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
transforms.RandomGrayscale(p=0.1),
```
Especially helps cat and bird which appear in highly varied orientations.

### Change 4 — Proper Train/Val/Test Split *(Day 14 side quest)*
```python
train_subset, val_subset = random_split(train_set, [45000, 5000])
# Save best based on val, report final score on test once
```
More honest accuracy reporting and better hyperparameter decisions.

### Change 5 — Lower Initial LR + Warmup
```python
# Start with tiny lr, warm up to 0.001 over first 5 epochs
# Prevents unstable gradients in the first few batches
```

---

## 🔑 Key Concepts from Day 13

### AdamW vs Adam
```python
optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```
AdamW adds `weight_decay` — a small penalty that pushes weights toward zero each update. This prevents any single weight from dominating and improves generalisation. It's the default choice for modern networks.

### BatchNorm + batch_size = 1 → Error
BatchNorm computes mean and std **from the current batch**. With batch_size=1, std is undefined (can't compute spread from one sample). Always use `model.eval()` before running shape traces or single-image inference.

```python
model.eval()                  # ← required for batch_size=1
with torch.no_grad():
    out = model(dummy)
```

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
If the total gradient norm exceeds 1.0, scale all gradients down proportionally. Preserves direction, prevents explosive updates. Goes AFTER `loss.backward()` and BEFORE `optimizer.step()`.

### Why Test Accuracy > Train Accuracy
In this model, test accuracy consistently exceeded train accuracy. Reason: Dropout is ON during training (randomly disabling neurons makes training harder → inflated train loss, lower train accuracy). During eval, dropout is OFF → full network capacity → higher test accuracy. **This is healthy, not a bug.**

### Test Set ≠ Validation Set
- Current code saves best model based on test accuracy → mild data leakage (20 peeks at test set)
- Proper approach: val set for saving decisions, test set touched only ONCE at the end
- Day 14 side quest: implement proper train/val/test split

### The `state_dict()` Clone Pattern
```python
best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
```
Without `.clone()`, `best_model_state` is just a reference to the model's current weights — it would keep updating as training continues. `.clone()` makes a deep copy, freezing the weights at their best point.

### imshow() Unnormalization
```python
img = img * std + mean          # reverse normalize
img = torch.clamp(img, 0, 1)   # clip to valid display range
img = img.permute(1,2,0)        # CHW → HWC for matplotlib
```
Always required before displaying a normalized image. The model trains on normalized values but humans need [0,1] RGB.

---

## 📌 One-Line Summaries

| Concept | One Line |
|---|---|
| Data augmentation | Show the same image in infinite variations so the network generalises |
| Dropout2d | Drop entire feature map channels, not individual pixels |
| Cosine Annealing | Start lr high, lower it smoothly — fast early, precise late |
| AdamW | Adam + weight decay penalty to prevent oversized weights |
| Gradient clipping | Cap gradient size so one bad batch can't derail training |
| BatchNorm eval mode | Use running stats instead of batch stats → works with batch_size=1 |
| Cat accuracy (61.6%) | Hardest CIFAR class — visually overlaps with dog at 32×32 |

---

## 🗓️ Day 14 Side Quest

Implement proper train/val/test split on CIFAR-10:
```python
from torch.utils.data import random_split
train_subset, val_subset = random_split(train_set, [45000, 5000])
# Save best model on val_loader
# Report final accuracy on test_loader ONCE after all training
```

---

*Day 13 complete — 82.45% on CIFAR-10, beat the target by 2.45 points.
Next: deeper architectures, skip connections, ResNet-style networks.*
