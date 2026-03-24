# Day 24 — Transfer Learning & Fine-tuning Intuition
**Date:** March 21, 2026
**Notebook:** `day24_transfer_learning.ipynb`
**Dataset:** Synthetic Task A (2000 samples, 6-class) + Task B (120 samples, binary)
**Machine:** Mac MPS
**Paper:** How transferable are features? — Yosinski et al. (2014)

---

## What Was Built

- Backbone pretrained on Task A (2000 samples, 6-class argmax)
- Four transfer strategies on Task B (96 train, 24 val, binary classification):
  - From scratch (wiped weights)
  - Frozen backbone (head only)
  - Fine-tune last layers
  - Fine-tune all (differential LRs)
  - Progressive unfreezing
- PCA feature space visualization — pretrained vs random init
- Training loss + validation accuracy comparison across all strategies

---

## Results

```
scratch:            87.5%  ← memorized 96 samples, got lucky
frozen:             62.5%  ← features don't perfectly transfer
finetune_last:      87.5%  ← good, fewer updates needed
finetune_all:       91.7%  ← best ✅ differential LRs preserved pretrained knowledge
progressive:        91.7%  ← also best, slower to get there, more controlled
```

---

## Core Concepts

### Why Transfer Learning Works — The Feature Hierarchy

Deep networks build features in layers:

```
layers 1-3:  edges, corners, textures, gradients     ← universal
layers 4-6:  object parts, shapes, spatial patterns  ← semi-universal
layers 7+:   "this is a golden retriever"            ← task-specific
```

The core assumption: **early layer features are universal**. A network trained on ImageNet to classify dogs and cars has learned to detect edges and textures — not dog-specific or car-specific patterns. Those features are useful for any visual task. The same applies to language — a GPT-2 trained on internet text has learned grammar, syntax, and semantic relationships that transfer to almost any language task.

Earlier layers = more universal = safer to freeze and reuse.
Later layers = more task-specific = may need adaptation.

### The Three Strategies

**Strategy 1 — Feature Extraction (frozen backbone)**
```
[Pretrained backbone — FROZEN] → [New head — trainable]
```
Use when: small dataset, task similar to pretraining.
Why it works: frozen features are good enough, training only the head prevents overfitting.
Risk: if pretrained features don't align with new task, head can't compensate.

**Strategy 2 — Fine-tune Last Layers**
```
[Early layers — FROZEN] → [Late layers — trainable, small LR] → [Head — trainable]
```
Use when: moderate dataset, task somewhat different from pretraining.
Sweet spot for most real-world fine-tuning scenarios.

**Strategy 3 — Full Fine-tuning (differential LRs)**
```
[All layers — trainable, tiny LR on early, small LR on late] → [Head — normal LR]
```
Use when: larger dataset, task differs significantly from pretraining.
Critical: use much smaller LR for backbone — nudge, don't destroy.

**From Scratch**
```
[Random init — all trainable] → train everything from zero
```
Use when: completely different data domain, massive data, no relevant pretrained model.

### Differential Learning Rates

```python
optimizer = torch.optim.Adam([
    {'params': bb.early_layers.parameters(), 'lr': 1e-5},  # barely touch
    {'params': bb.late_layers.parameters(),  'lr': 1e-4},  # nudge
    {'params': head.parameters(),            'lr': 1e-3},  # learn fast
])
```

Early layers already encode universal features — huge updates destroy them. Head starts from random — needs fast learning. The 10-100x LR difference between backbone and head is what makes fine-tuning work rather than just wiping pretrained knowledge.

### Progressive Unfreezing

Instead of unfreezing all layers at once:

```
epoch 1-10:   train head only (backbone frozen)
epoch 11-20:  unfreeze late layers + head (small LR)
epoch 21+:    unfreeze early layers + late + head (tiny LR on early)
```

Each unfreeze event shows as a jump in the validation accuracy curve. More controlled than unfreezing everything at once — the head stabilizes first before the backbone starts adapting.

### requires_grad = False

The mechanism that freezes layers:

```python
for p in backbone.parameters():
    p.requires_grad = False   # gradients don't flow here
```

When PyTorch builds the computation graph during backprop, it skips any parameter with `requires_grad=False`. Gradients literally don't reach those weights — they're locked in place.

### copy.deepcopy

```python
bb = copy.deepcopy(backbone)
```

Creates a completely independent copy with its own weights in memory. Without deepcopy all strategies would share the same backbone object — modifying one would break all others.

---

## PCA Feature Space Visualization

```python
pca = PCA(n_components=2)
feat_2d = pca.fit_transform(features)   # (120, 64) → (120, 2)
```

**PCA (Principal Component Analysis)** — finds the two directions of maximum variance in 64D space and projects onto them. Compresses 64 dimensions to 2 for visualization while preserving as much structure as possible.

Not the same as 1×1 conv projection — PCA is computed from data statistics (unsupervised, no learning). 1×1 conv is learned through backprop to minimize task loss. Both are linear projections but driven by completely different objectives.

**Pretrained feature space:** partial class separation visible — clusters forming in some regions. Backbone learned representations that partially separate Task B classes even though it was trained on Task A.

**Random init feature space:** uniform blob — no structure, blue and orange completely mixed. A classifier on top has to discover everything from 96 samples.

---

## Code Patterns

### Building Transfer Models

```python
def build_transfer_model(strategy, backbone):
    bb   = copy.deepcopy(backbone)
    head = nn.Linear(64, 2).to(device)

    if strategy == 'scratch':
        # wipe weights — random init
        nn.init.xavier_uniform_(module.weight)

    elif strategy == 'frozen':
        for p in bb.parameters():
            p.requires_grad = False

    elif strategy == 'finetune_last':
        for p in bb.early_layers.parameters():
            p.requires_grad = False
        for p in bb.late_layers.parameters():
            p.requires_grad = True

    elif strategy == 'finetune_all':
        for p in bb.parameters():
            p.requires_grad = True   # differential LRs set at optimizer

    return bb, head
```

### Combining Parameters From Two Models

```python
optimizer = optim.Adam(
    list(backbone.parameters()) + list(head.parameters()),
    lr=0.005
)
```

When you have two separate model objects you combine their parameter lists manually with `list() + list()`.

---

## Reflection Questions

**1. Core assumption that makes transfer learning work**

The core assumption is that features learned in early layers of a deep network are universal — not specific to the task they were trained on. A network trained on ImageNet learns to detect edges, textures, and shapes in layers 1-3. These aren't "dog detector" features or "car detector" features — they're general visual primitives that appear in any natural image. Yosinski et al. confirmed this empirically — features in early layers transfer almost perfectly across completely different tasks, while later layers become increasingly task-specific. Early layers generalize because the low-level statistics of natural images (edges, gradients, local patterns) are consistent across all visual domains. The same logic applies to language — grammatical structure and semantic relationships learned from internet text transfer to almost any downstream task.

**2. Three strategies and when to use each**

Frozen backbone is the right choice when you have a small dataset and your task is similar to what the backbone was pretrained on. You only train the new head — prevents overfitting because only a tiny fraction of parameters update. Fine-tune last layers is the sweet spot for most real-world scenarios — moderate data, task somewhat different. You freeze the universal early features and let the more task-specific late layers adapt. Full fine-tuning with differential learning rates is for larger datasets where the task differs significantly from pretraining. Everything trains but backbone gets 10-100x smaller LR than the head — nudges pretrained features toward the new task without destroying them. From scratch is reserved for completely different domains with massive data and no relevant pretrained model.

**3. Risk of full fine-tuning on small data**

Catastrophic forgetting — the pretrained representations get overwritten. With only 96 samples and 46k parameters, the model has far more capacity than needed. The optimizer will find a solution that perfectly fits the 96 training samples by memorizing them, and in the process destroys the general features the backbone spent 2000 samples learning. After fine-tuning the backbone no longer encodes useful general representations — it encodes noise patterns specific to the 96 training samples. The validation accuracy drops because the model hasn't generalized — it memorized. This is exactly why the scratch model in our experiment showed near-zero training loss but only 87.5% val accuracy — classic overfitting on a tiny dataset.

**4. Differential learning rates intuition**

Pretrained weights are already in a good region of parameter space — they encode useful representations built from 2000 samples. A standard fine-tuning learning rate (1e-3) would move them so far from that good region in a single step that the pretrained knowledge is lost. The early layers need only tiny adjustments — a rate of 1e-5 moves them 100x less per step, preserving the core representations while allowing slight adaptation. The head starts from random initialization and needs to learn quickly from scratch — it gets the full 1e-3 rate. The intuition is: the further a layer is from random (the more it's already learned), the smaller the step you want to take when updating it.

**5. Progressive unfreezing**

Progressive unfreezing starts with only the head trainable, then gradually unfreezes layer groups from late to early, each time with a smaller LR. The advantage over unfreezing everything at once is stability — the head stabilizes first before the backbone starts adapting. If you unfreeze everything immediately, the randomly initialized head produces large gradients early in training that flow into the backbone and destabilize the pretrained features before the head has had a chance to converge. Progressive unfreezing lets the head find a reasonable solution first, then the late backbone layers fine-tune on top of that, then the early layers make tiny final adjustments. Each unfreeze event shows as a visible accuracy jump in the training curve — the staircase pattern in the progressive unfreezing line.

**6. Feature space plots — pretrained vs random**

The pretrained backbone's feature space showed partial class separation — visible clustering in some regions with class 0 and class 1 points showing some tendency to group separately. The random init feature space was a uniform blob with no structure — blue and orange completely mixed with no separability. This tells you the pretraining on Task A taught the backbone to extract representations that carry some signal about the input structure even for Task B — even though the tasks are different. The backbone learned which input dimensions matter and how to combine them in ways that generalize. The random backbone produces arbitrary transformations of the input that carry no structured information about class membership.

**7. Connection to LLMs — GPT-2 fine-tuning and LoRA**

When you call `GPT2Model.from_pretrained('gpt2')` on Day 35 and fine-tune on a task, you're using Strategy 3 — full fine-tuning with differential learning rates. The GPT-2 backbone has learned language representations from internet text (equivalent to Task A on a massive scale). Your downstream task has far less data. You unfreeze the backbone but use a very small LR (1e-5 or less) to nudge it toward your task without destroying the language representations it spent enormous compute learning.

LoRA on Day 46 changes the picture fundamentally — it's an extreme version of Strategy 1. Instead of updating any pretrained weights at all, LoRA adds tiny trainable matrices alongside the frozen pretrained weights. The pretrained GPT-2 weights never change — you inject small low-rank updates that get added to the frozen weights at inference time. This means you never risk catastrophic forgetting, you can train with far less memory since most parameters have no gradients, and you can easily swap between different LoRA adapters for different tasks by just loading different small matrices. LoRA works because the effective changes needed to adapt a large pretrained model to a specific task live in a low-dimensional subspace — you don't need to update all 175B parameters to teach GPT-3 a new task.

---
