# Lab 07 — Grad-CAM on a Pretrained CNN

A detailed walkthrough of `lab07.ipynb`: what each section does, why it was done that way, what the results mean, and what the alternatives were at each step (and why they were not picked).

---

## Contents

1. [The assignment in one paragraph](#1-the-assignment-in-one-paragraph)
2. [Files and how to run](#2-files-and-how-to-run)
3. [Background: how Grad-CAM works](#3-background-how-grad-cam-works)
4. [Core walkthrough (§1 – §8)](#4-core-walkthrough-1--8)
5. [Learning component walkthrough (§9 – §12)](#5-learning-component-walkthrough-9--12)
6. [Results summary](#6-results-summary)
7. [Design decisions and rejected alternatives](#7-design-decisions-and-rejected-alternatives)
8. [Alternatives to Grad-CAM](#8-alternatives-to-grad-cam)
9. [Limitations and how to improve](#9-limitations-and-how-to-improve)
10. [Glossary](#10-glossary)
11. [Likely viva questions](#11-likely-viva-questions)
12. [References](#12-references)

---

## 1. The assignment in one paragraph

Take a CNN that is already trained on ImageNet (VGG-16 or AlexNet), show it **one simple image**, and use **Grad-CAM** to draw a heatmap of *where in the image* the network found evidence for its answer. Then change one thing at a time — the **target class**, the **layer**, the **model** — and watch how the heatmap reacts. The notebook uses VGG-16 as the main model and AlexNet as the comparison model, so both options from the sheet are covered.

On top of the required tasks, the **learning component** goes further in three directions:
- **Verify** — prove the hand-written code is correct by matching a reference library.
- **Extend** — implement two stronger relatives of Grad-CAM from scratch: **Grad-CAM++** and **Guided Grad-CAM**.
- **Evaluate** — stop judging heatmaps by eye and *measure* them: **deletion / insertion** faithfulness curves and the **model-randomisation sanity check**.

| Task from the sheet | Where in notebook |
|---|---|
| Load a pretrained VGG-16 / AlexNet | §3 (VGG-16), §8 (AlexNet) |
| Check what it predicts for one image | §3 (top-5 table) |
| Implement Grad-CAM | §4 (`GradCAM` class, from scratch with hooks) |
| Show heatmap + overlay for the predicted class | §5 |
| Experiment: a different class | §6 (dog → cat) |
| Experiment: a different model | §8 (VGG-16 vs AlexNet) |
| **Learning component** | |
| Check our code against `pytorch-grad-cam` | §5.1 |
| Grad-CAM for the top-5 classes | §6.1 |
| Changing the layer (the aim mentions it) | §7 |
| Grad-CAM++ from scratch (+ library check) | §9 |
| Guided Backprop + Guided Grad-CAM from scratch | §10 |
| Deletion / insertion faithfulness metrics | §11 |
| Model-randomisation sanity check | §12 |

---

## 2. Files and how to run

```
lab07/
├── lab07.ipynb            # the notebook (already executed, outputs saved)
├── lab07.pdf              # PDF export of the executed notebook (A4, 16 pages)
├── lab07_explanation.md   # this file
└── data/
    └── both.png           # the input image (bulldog + tabby cat, 224×224)
```

**Image source.** `both.png` is the example image from the [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) repository (`examples/both.png`). The same dog+cat picture is used in the original Grad-CAM paper, so results can be compared with published figures.

**Environment.** The repository's `.venv` (Python 3.13) already had `torch 2.14`, `torchvision 0.29` (CUDA), `scipy` and `matplotlib`. One package was added:

```bash
uv pip install --python .venv/bin/python grad-cam
```

This also pulled `opencv-python-headless` and `ttach`. As in Lab 06, `uv pip` was used instead of `uv add` so the environment is not re-synced to `uv.lock` (which would remove packages installed outside the lockfile, e.g. TensorFlow from Lab 04). The library is used **only to check** our own code (§5.1 and §9); every method in the notebook is implemented from scratch in plain PyTorch.

**Run.** Open in Jupyter and *Run All*, or headless:

```bash
cd lab07
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab07.ipynb
```

The first run downloads VGG-16 weights (528 MB) and AlexNet weights (233 MB) into `~/.cache/torch/hub/checkpoints/`. After that the whole notebook takes about 30 seconds on the GTX 1650 Ti (the deletion/insertion section is the slowest: 4 × 51 × 3 = 612 forward passes). It also works on CPU, just slower. The results are the same on every run.

**GPU memory note.** The GPU has 4 GB. The first version ran out of memory because every `backward()` also filled `.grad` buffers for all 138 M VGG weights (~530 MB per model). Grad-CAM only needs gradients with respect to **activations**, never the weights, so every model is loaded with `.requires_grad_(False)` and the input is marked `requires_grad=True` instead. That keeps the whole notebook at about 1.5 GB of GPU memory.

---

## 3. Background: how Grad-CAM works

### 3.1 The problem

A CNN takes an image and outputs 1000 class scores. It does not tell us *why*. Grad-CAM (Selvaraju et al., 2017) produces a **coarse heatmap** showing which regions pushed the score of a chosen class up.

### 3.2 The intuition

- The **last convolutional layer** of a CNN has K feature maps (512 for VGG-16, 256 for AlexNet), each a small grid (14×14 or 13×13). Each map is a detector for some high-level pattern ("floppy ear", "striped fur", "wet nose"…), and *where* it fires tells us *where* that pattern is.
- Not every pattern matters for every class. To know how important map *k* is for class *c*, look at the **gradient** of the class score with respect to that map: if increasing the map would increase the score a lot, the map is important.
- So: weight each map by its importance and add them up. The result is a heatmap of "evidence for class *c*".

### 3.3 The formula

For class *c* with raw score (logit) *y<sup>c</sup>* and feature maps *A<sup>k</sup>* of the target layer:

1. **Importance weight** of map *k* (global-average-pool of the gradient):

   α<sub>k</sub><sup>c</sup> = (1 / Z) · Σ<sub>i,j</sub> ∂y<sup>c</sup> / ∂A<sup>k</sup><sub>ij</sub>      (Z = H × W)

2. **Heatmap** (weighted sum, then ReLU):

   L<sup>c</sup> = ReLU( Σ<sub>k</sub> α<sub>k</sub><sup>c</sup> · A<sup>k</sup> )

3. **Upsample** L<sup>c</sup> from H×W to the input size (224×224) and min-max normalise to 0–1 for display.

### 3.4 Why each piece is there

| Piece | Why |
|---|---|
| Use the **logit** *y<sup>c</sup>*, not the softmax probability | Softmax couples all classes: the gradient of p<sub>c</sub> also includes "make other classes smaller". The logit gives the cleanest "evidence for *c*" signal. This is what the paper does. |
| **Average** the gradient over space | Turns a per-pixel gradient into one number per feature map — "how much does this detector matter overall for class *c*". |
| **ReLU** at the end | We only want regions that *increase* the class score. Negative values are evidence for *other* classes; without ReLU they would pollute the map. |
| **Last** conv layer | Deepest layer that still has a spatial grid. Deeper (FC) layers have no "where"; earlier layers have "where" but not "what". Tested in §7. |
| **Upsample** | The 14×14 map is too coarse to read on its own; bilinear upsampling makes it overlay smoothly on the image. |

### 3.5 Relation to CAM

Grad-CAM generalises **CAM** (Zhou et al., 2016). CAM needs a special architecture (global average pooling directly followed by one linear layer) and uses that layer's weights as α. VGG-16 and AlexNet have several fully-connected layers, so CAM cannot be applied without retraining. Grad-CAM gets the weights from gradients instead, so it works on **any** CNN unchanged.

---

## 4. Core walkthrough (§1 – §8)

### Setup cell

Imports, `torch.manual_seed(0)`, picks `cuda` if available, and sets the figure DPI. Warnings are silenced only to keep the output readable.

*Why seed?* Inference in eval mode is already deterministic; the seed matters for §12, where `reset_parameters()` draws random weights. With the seed the sanity-check numbers are the same on every run.

### §1 Load the image

`data/both.png` — a 224×224 photo of a bulldog sitting behind a tabby cat.

*Why this image?* The sheet says "one simple image", and the experiments ask for "a different class". With a single-object image, a different class would just give a weak or random-looking map and the experiment would be hard to judge. With **two clearly separated objects** (dog on top, cat bottom-right), the question "does the heatmap follow the class?" has an obvious visual answer — and it also makes the faithfulness tests in §11 much more interesting (the two objects compete).

*Rejected:* a random photo from the internet (licensing, and it may contain only one object); a synthetic image (ImageNet models behave oddly on non-natural images).

### §2 Pre-process

Uses `VGG16_Weights.IMAGENET1K_V1.transforms()` — resize to 256, center-crop to 224, convert to tensor, normalise with ImageNet mean/std.

*Why the weights' own transform?* It is exactly what the model was trained with; a hand-written pipeline risks a mismatch (forgetting normalisation drops accuracy badly). The same transform is reused for AlexNet because both torchvision ImageNet models use identical preprocessing.

*Side effect:* the image is already 224×224, so resize-to-256 then crop-to-224 slightly zooms in and trims ~16 px off each edge. Both objects stay in frame. The `rgb` array is the **de-normalised model input**, so overlays line up pixel-for-pixel with what the model actually saw.

### §3 Load VGG-16 and check its prediction

```
1. pug              p = 0.366
2. bull mastiff     p = 0.325
3. boxer            p = 0.094
4. French bulldog   p = 0.040
5. tiger cat        p = 0.014
```

*Is "pug" wrong?* The dog is an **English bulldog**, which is **not one of the 1000 ImageNet classes**. The top-4 are all flat-faced / bulldog-like breeds, so the model makes the most sensible choice it can. The cat appears at rank 5 — the model "sees" it but the dog dominates.

`model.eval()` turns off dropout so outputs are deterministic. `.requires_grad_(False)` freezes the weights (see the memory note in §2).

### §4 Implement Grad-CAM

The `GradCAM` class is the core of the lab:

```python
self.handle = layer.register_forward_hook(self._save)       # grab activations A on the way forward
if out.requires_grad:
    out.register_hook(lambda g: ...)                         # grab dy/dA on the way back
logits = self.model(x.detach().requires_grad_(True))        # gradients flow to activations, not weights
logits[0, class_idx].backward()                             # gradient of the raw class score
alpha = self.weights(A, G)                                  # Grad-CAM: G.mean(dim=(1, 2))
cam = F.relu((alpha[:, None, None] * A).sum(0))             # weighted sum, keep positive
cam = F.interpolate(..., size=224, mode="bilinear")
cam = (cam - min) / (max - min)
```

Design points, each with its reason:

| Point | Why |
|---|---|
| **Hooks** instead of splitting the model by hand | Hooks read any layer of an unmodified torchvision model. Splitting into "up to layer L" + "the rest" must be re-coded for every model and layer, which would make §7, §8 and §12 clumsy. |
| **Tensor hook** (`out.register_hook`) instead of `register_full_backward_hook` | VGG and AlexNet use `ReLU(inplace=True)`. Module backward hooks around in-place ops raise errors or give wrong gradients in recent PyTorch. A hook on the output tensor avoids this — it's also what the `pytorch-grad-cam` library does. |
| `if out.requires_grad:` guard | The hook also fires on plain `torch.no_grad()` forward passes (the top-k cell, the 600 forward passes in §11). Registering a gradient hook there raises an error, so we skip it. |
| Hook the **ReLU** (`features[29]`), not the conv (`features[28]`) | The conv output is overwritten in place by the following ReLU, so hooking the conv captures a tensor that later changes. The ReLU output is the final, rectified activation. Same layer as the paper and library. |
| `weights()` is a **separate method** | Grad-CAM++ (§9) differs only in how the weights are computed, so it can subclass `GradCAM` and override one method instead of copying the whole class. |
| `remove()` | Each hook stays on the model until removed. Loops that create many `GradCAM` objects (§7) remove theirs so hooks don't pile up. |
| Written **from scratch** | The sheet says *implement*. The library is used only to check (§5.1). |

### §5 Grad-CAM for the predicted class

Prints `vgg.features` so the reader can see that index 29 is the last ReLU before the final max-pool (14×14×512 output), then shows image / heatmap / overlay for **pug**.

**Result:** a tight hot spot on the dog's **face and upper chest**; the cat, the floor and the background are cold. The model's dog decision is based on the dog's face — the part that tells breeds apart — which is what we would hope.

*Overlay design:* `jet` colormap blended 50/50 with the image. Jet is the convention in the Grad-CAM paper and the library, so figures can be compared directly with published ones. (A perceptually uniform map like `inferno` is better for reading exact values, but here we care about *where*, not exact size.)

### §5.1 Learning component — check against `pytorch-grad-cam`

Runs `pytorch_grad_cam.GradCAM` on the same model, layer and class and computes the Pearson correlation between the two 224×224 maps.

**Result: r = 1.0000** — the hand-written version is numerically identical to the reference implementation.

*Why?* Grad-CAM bugs are silent: a wrong axis in the mean, a missing ReLU, or softmax instead of logits all still produce a plausible-looking blob. Comparing with a well-tested implementation is the cheapest proof of correctness. The library runs in a `with` block so its hooks are released afterwards. Because our models are frozen, the library is given `x.clone().requires_grad_(True)` so gradients can still reach the activations.

### §6 Experiment 1 — a different target class

Same image, model and layer — only the class changes: **pug** (rank 1) vs **tiger cat** (rank 5, the best of ImageNet's five cat classes 281–285).

To make the comparison **quantitative** rather than "looks different", the cell defines two rough rectangles on the 224×224 input (white = dog, yellow = cat, drawn on the first panel) and reports what share of each heatmap's total weight falls inside each. The same `share()` function is reused in §8 and §9.

| Target class | Probability | Weight on dog region | Weight on cat region |
|---|---|---|---|
| pug | 0.366 | **69%** | 16% |
| tiger cat | 0.014 | 13% | **68%** |

**What this shows:** Grad-CAM is **class-discriminative**. The heatmap is not "where is something interesting" — it answers a question about one class. Even though "tiger cat" has only 1.4% probability, its gradient points to exactly the right object. The hot area covers the cat's **striped body**, not just the face — stripes are what defines "tiger cat" in ImageNet.

*Why rectangles and not segmentation masks?* For two objects that barely overlap, rectangles are good enough, need no extra model or labelling, and are drawn on screen so the reader can judge them. The boxes include a bit of floor, so numbers are approximate — but a 69/13 vs 16/68 split is far bigger than that error.

*Why choose the cat class automatically (`max` over 281–285)?* So the notebook does not hard-code a guess; it uses whichever cat class VGG believes most.

### §6.1 Learning component — Grad-CAM for the top-5 classes

One overlay per top-5 class. The four dog breeds all highlight roughly the **same face/chest area**, with small shifts (bull mastiff and boxer spread a bit more over the body). Only "tiger cat" moves.

*Lesson:* Grad-CAM's 14×14 maps separate **very different** classes well but can't show the fine differences between **similar** classes. That is the motivation for Guided Grad-CAM in §10.

### §7 Learning component — changing the layer

Grad-CAM (pug, VGG-16) on the last ReLU of block 3 (`features[15]`, 56×56), block 4 (`features[22]`, 28×28) and block 5 (`features[29]`, 14×14).

| Layer | Map size | What it looks like |
|---|---|---|
| block3 | 56×56 | Many tiny scattered dots on edges — door frame, window, floor. No focus on the object. |
| block4 | 28×28 | Fewer, still scattered spots; weak hints near the dog, plus spots on the background. |
| block5 | 14×14 | One clean blob on the dog's face/chest. |

**Why:** early layers detect low-level things (edges, colours, textures) that appear everywhere and aren't specific to "pug", and their gradients are noisy because many layers lie between them and the output. Only the final block has features that mean "dog-face-ish". This **resolution vs. meaning** trade-off is why Grad-CAM normally uses the last conv layer.

### §8 Experiment 2 — a different model (AlexNet)

AlexNet (2012) has 5 conv layers and a big early stride (11×11 kernel, stride 4), vs VGG-16's 13 small 3×3 conv layers. Grad-CAM runs on `features[11]`, the ReLU after its last conv (13×13×256), for the **same two classes** as Experiment 1 so the comparison is fair.

**Prediction:** AlexNet's top-1 is **Rhodesian ridgeback** (0.20), then malinois, bull mastiff, bloodhound, redbone — a much less sensible list than VGG's bulldog-like breeds, and less confident. (ImageNet top-1 accuracy: AlexNet ≈ 56.5%, VGG-16 ≈ 71.6%.)

| Model | Target | Dog region | Cat region |
|---|---|---|---|
| VGG-16 | pug | 69% | 16% |
| AlexNet | pug | 53% | 15% |
| VGG-16 | tiger cat | 13% | 68% |
| AlexNet | tiger cat | **26%** | 53% |

Correlation between VGG and AlexNet heatmaps: **0.38** (pug) and **0.42** (tiger cat).

**What this shows:**
- Both models get the **broad** location right (dog class → dog, cat class → cat).
- AlexNet's maps are **less focused**: for pug the peak is on the top of the dog's head and more weight leaks outside the dog; for tiger cat a quarter of the weight is on the **dog's head** — it partly confuses the two animals.
- For tiger cat AlexNet's peak is on the cat's **striped flank**, not the face — consistent with AlexNet relying more on texture.
- Correlation ~0.4 means the explanations are only loosely similar. **A Grad-CAM heatmap explains one specific model**, not the image.

*Why AlexNet (and not ResNet)?* The sheet names VGG-16 and AlexNet, and both have the same `features` → `classifier` structure, so the same `GradCAM` class works with only the layer index changed. That keeps "model" as the only thing that changed.

---

## 5. Learning component walkthrough (§9 – §12)

The core lab answers *"what does the heatmap look like?"*. The learning component asks the harder questions: *"can we do better than Grad-CAM?"* (§9, §10) and *"how do we know a heatmap is telling the truth?"* (§11, §12).

### §9 Grad-CAM++

#### The idea

Grad-CAM's weight for map *k* is a **plain average** of its gradient, so a map that is strongly positive in one place and negative elsewhere can average out to almost zero. Grad-CAM++ (Chattopadhay et al., 2018) uses a **weighted** average that only counts **positive** gradients, with per-pixel weights α<sub>ij</sub> derived from higher-order derivatives:

α<sub>ij</sub><sup>k</sup> = g<sup>2</sup> / ( 2g<sup>2</sup> + Σ<sub>ab</sub>A<sup>k</sup><sub>ab</sub> · g<sup>3</sup> ),     w<sub>k</sub> = Σ<sub>ij</sub> α<sub>ij</sub><sup>k</sup> · ReLU(g<sub>ij</sub>)

where g = ∂y<sup>c</sup>/∂A<sup>k</sup><sub>ij</sub>. (The paper's exact form uses exp(y<sup>c</sup>) as the score; with that choice the 2nd and 3rd derivatives become powers of the 1st, which is why only g², g³ appear — the standard implementation does the same.) The heatmap is then built exactly as in Grad-CAM. The claimed benefit is better coverage of **whole objects** and of **several instances** of the same class.

#### The code

Because `GradCAM` keeps the weight formula in its own `weights()` method, Grad-CAM++ is a 7-line subclass. Two details:
- `+ 1e-6` in the denominator avoids division by zero.
- `torch.where(G != 0, a, 0)` sets the weight to zero where there is no gradient — the same rule the reference library uses, so the outputs can match exactly.

**Check:** our Grad-CAM++ vs `pytorch_grad_cam.GradCAMPlusPlus` → **r = 1.0000**.

#### The result

| Method | Class | Dog region | Cat region | Coverage (pixels > 0.5) |
|---|---|---|---|---|
| Grad-CAM | pug | 69% | 16% | 5% |
| Grad-CAM | tiger cat | 13% | 68% | 14% |
| Grad-CAM++ | pug | 54% | 32% | 9% |
| Grad-CAM++ | tiger cat | 42% | 43% | 19% |

- Grad-CAM++ does light up **more** of the image (coverage nearly doubles) — the dog's whole face and chest instead of one spot.
- But here it is **less class-discriminative**: the tiger-cat map is split almost evenly between dog and cat, and the visual shows a strong hot spot on the dog's face for "tiger cat".
- *Why?* Grad-CAM++ throws away negative gradients. In Grad-CAM, the "dog-face" feature maps get a **negative** average gradient for "tiger cat" (a dog face is evidence *against* cat), so they are subtracted out. Grad-CAM++ ignores those negatives, so any feature map with *some* positive gradient — even a dog-face detector — gets added in.

*Lesson:* "newer" doesn't mean "better for every image". Whether Grad-CAM++ is better here can't be settled by looking — which is exactly what §11 measures.

*Alternatives not implemented:* **Score-CAM** (gradient-free; 512 forward passes per heatmap — too slow to add value here), **XGrad-CAM**, **Layer-CAM**. Grad-CAM++ was chosen because it is the most widely used successor and changes only the weighting step, so the comparison with Grad-CAM is clean.

### §10 Guided Grad-CAM

#### The idea

Grad-CAM has 14×14 resolution — it can say "the face" but not "the eyes and nose". **Guided Backpropagation** (Springenberg et al., 2015) gives pixel-level detail: back-propagate the class score all the way to the input pixels, but at every ReLU only pass **positive** gradients through, and only where the forward activation was positive:

grad<sub>in</sub> = ReLU(grad<sub>out</sub>) · [activation > 0]

(A normal ReLU passes `grad_out · [activation > 0]`, positive or negative.) Guided Backprop's problem is that it is **not class-specific**. The Grad-CAM paper's fix is **Guided Grad-CAM = Guided Backprop × upsampled Grad-CAM**: sharp detail from one, class focus from the other.

#### The code

- `GuidedReLU` is a custom `torch.autograd.Function` whose `backward` applies the rule above. It saves the **output** (not the input) for backward; output > 0 and input > 0 are the same condition, and it is what the forward produced.
- A **fresh copy** of VGG-16 is loaded and every `nn.ReLU` in `features` and `classifier` is replaced by `GuidedReLUModule`. A copy is used so the original model (and its Grad-CAM hooks) is untouched.
- `guided_backprop()` makes the input require gradients, back-props the class logit, and returns `x.grad` as a 224×224×3 image.
- `to_img()` standardises the gradient and centres it on grey (0.5), the usual way to display it: grey = zero, colour = positive/negative per channel.
- The copy is deleted at the end of the cell to free ~530 MB of GPU memory (the loop variables `seq`, `m` still point into it, so they are deleted too).

*Why replace modules instead of using hooks?* A backward hook on every in-place ReLU runs into the same in-place problem as in §4. A custom autograd function is the cleanest way to change a backward pass in PyTorch.

#### The result

| | Dog-vs-cat similarity of the two maps |
|---|---|
| Guided Backprop | **0.998** |
| Guided Grad-CAM | **−0.043** |

- **Guided Backprop** for "pug" and for "tiger cat" are **practically the same picture** (0.998) — a sharp edge map of both animals, door frame and floor. It shows *what edges exist*, not *what the class is*.
- **Guided Grad-CAM** is completely different per class (−0.04): for pug it shows a sharp outline of the dog's **eyes, nose, jowls and collar**; for tiger cat it shows the cat's **face and stripes**.
- So the combination gets both properties: class-specific (from Grad-CAM) and high-resolution (from Guided Backprop).

### §11 Faithfulness: deletion and insertion

#### The problem

Everything so far was judged by eye: "the heatmap is on the dog, so it's good". But a heatmap can look plausible and still not reflect what the model uses. We need a test that asks **the model itself**.

#### The idea (Petsiuk et al., 2018 — the RISE paper)

Sort all 224×224 pixels from most to least important according to the heatmap, then:

- **Deletion:** remove the pixels in that order (most important first) in 50 steps, and record the class probability after each step. If the heatmap is right, the probability should **crash quickly**. Score = area under the curve (AUC) → **lower is better**.
- **Insertion:** start from an image with *everything* removed and put pixels back in the same order. If the heatmap is right, the probability should **climb quickly**. Score = AUC → **higher is better**.
- A **random** heatmap is the baseline. A method that can't beat random is useless.

#### Code choices

| Choice | Why | Rejected alternative |
|---|---|---|
| "Remove" = replace with a **heavily blurred** copy (Gaussian, σ = 20) | Keeps colours and rough shapes natural; the model sees "less detail", not a strange object. | Black/zero pixels: a sharp black hole is an unnatural shape that itself changes the prediction, so the test would measure the hole, not the evidence. |
| Same blurred image for deletion and insertion | One baseline for both curves keeps them comparable. | RISE uses zero for deletion and blur for insertion — mixes two effects. |
| 50 steps (2% of pixels each) | Smooth enough curve; 51 × 2 images per method. | Per-pixel steps: 50 000 forward passes — far too slow. |
| Tiny random noise added to the heatmap before sorting | After ReLU many pixels are exactly 0. Without noise, `argsort` would order them by index (top rows first), which is a hidden bias. | — |
| Softmax probability (not logit) | Probabilities are bounded 0–1, so AUCs are comparable between methods. | Logit is unbounded and hard to compare. |
| Batches of 8 under `torch.no_grad()` | Fits the 4 GB GPU. | One batch of 51 → out of memory. |
| `np.trapezoid` for the area | Simple trapezoid rule on the 51 points. | — |

#### The result

| Class | Method | Deletion AUC ↓ | Insertion AUC ↑ |
|---|---|---|---|
| pug | **Grad-CAM** | **0.0087** | **0.6037** |
| pug | Grad-CAM++ | 0.0136 | 0.4281 |
| pug | Random | 0.0407 | 0.0262 |
| tiger cat | **Grad-CAM** | **0.0004** | **0.1167** |
| tiger cat | Grad-CAM++ | 0.0121 | 0.0108 |
| tiger cat | Random | 0.0025 | 0.0014 |

How to read it:

1. **Grad-CAM beats random on every score** — 4–5× lower deletion AUC and 20–80× higher insertion AUC. Its heatmaps really point at what the model uses.
2. **Grad-CAM beats Grad-CAM++ on all four scores.** This confirms §9 with numbers: on this image, Grad-CAM++'s wider maps include regions that don't help (or even hurt) the class.
3. **Grad-CAM++ is worse than random for tiger-cat deletion** (0.012 vs 0.003). Its tiger-cat map puts the dog's face first, so deletion **removes the dog first** — and with the dog gone, the cat probability *rises* to ~0.17 (the orange spike on the plot). The method highlights the competitor.
4. **Insertion goes above the full-image probability.** Inserting only the top ~20% of Grad-CAM "pug" pixels gives p(pug) ≈ **0.95**, far above the 0.37 on the full image. For tiger cat it reaches ~0.19, vs 0.014 on the full image. *Why?* On the full image the dog and cat **compete** for the softmax probability. Show the model only the dog and it is very sure it's a pug; show it only the cat and the cat score goes up 14×. This also explains why tiger cat had only 1.4% at the start — not because the model can't recognise it, but because the dog takes most of the probability.
5. The insertion curves are **bumpy**, not smooth. Adding pixels doesn't always help: half-revealed objects can look like other classes for a few steps. This is normal and why AUC (an average) is used instead of a single point.

### §12 Sanity check: model randomisation

#### The problem

Adebayo et al. (2018), *"Sanity Checks for Saliency Maps"*, showed that some popular methods — Guided Backprop among them — produce **almost the same picture even when the model's weights are random**. Such a method isn't explaining the model; it's acting like an edge detector on the input. An explanation that doesn't depend on what the model learned is worthless.

#### The test (cascading randomisation)

1. Load a **fresh copy** of VGG-16 (so the original stays intact).
2. Randomise layers **from the top down**, keeping earlier randomisations: classifier → conv5_3 → conv5_2 → conv5_1 → conv4_3. Randomise = call each layer's `reset_parameters()`, which re-draws the weights from PyTorch's default initialisation.
3. After each step, compute Grad-CAM for "pug" and compare it with the original map using the **Spearman rank correlation** ρ.

A trustworthy method should **change a lot** (ρ → 0) as soon as the layers it depends on are randomised.

*Why Spearman, not Pearson?* We care whether the **ranking** of important pixels stays the same (that's what someone reading a heatmap uses), not whether the values scale linearly. Spearman is also what the original paper uses.
*Why cascading (top-down) rather than one layer at a time?* It follows the paper and shows how quickly the explanation breaks as more of the model is destroyed.

#### The result

| Randomised so far | ρ with original |
|---|---|
| nothing (original) | 1.000 |
| + classifier | 0.571 |
| + conv5_3 | **0.011** |
| + conv5_2 | 0.138 |
| + conv5_1 | 0.110 |
| + conv4_3 | 0.153 |

- **Randomising only the classifier** keeps ρ = 0.57: the map still sits roughly on the dog but moves to the chest/legs. The conv5 features are still the trained ones and fire most strongly on the dog, so a random classifier still "reads" from there — but it no longer picks the face.
- **Randomising conv5_3 too** drops ρ to **0.01** — no relation to the original. From then on the maps are scattered patterns that don't follow the dog.
- The small rise to 0.11–0.15 afterwards is noise between random maps, not a recovery.
- **Grad-CAM passes the sanity check**: its heatmap depends on the learned weights, so it is a genuine explanation of the model. This matches Adebayo et al., where Grad-CAM passed while Guided Backprop failed. (In that paper Guided Grad-CAM was also only partly sensitive, because its Guided Backprop half barely changes. So the sharp detail in §10 should be read as "edges inside the region Grad-CAM picked", not as extra evidence from the model.)

---

## 6. Results summary

| # | Experiment | Key number(s) | Conclusion |
|---|---|---|---|
| 1 | VGG-16 prediction | pug 0.37, bull mastiff 0.33 | Sensible: English bulldog isn't an ImageNet class. |
| 2 | Grad-CAM, predicted class | hot spot on face/chest | Model uses the dog's face — the right evidence. |
| 3 | **Different class** | pug: 69% on dog; tiger cat: 68% on cat | Grad-CAM is class-discriminative, even for a 1.4% class. |
| 4 | **Different model** | AlexNet top-1 ridgeback (0.20); map corr ≈ 0.4 with VGG; 26% of cat map on dog | Weaker model → weaker, less focused evidence. Heatmaps are model-specific. |
| LC | Ours vs library | r = 1.0000 (Grad-CAM and Grad-CAM++) | Both implementations are correct. |
| LC | Top-5 classes | 4 dog breeds ≈ same map | Coarse maps can't separate similar classes. |
| LC | Different layer | block3/4 noisy, block5 clean | Last conv layer = best balance of "what" and "where". |
| LC | Grad-CAM++ | coverage 5% → 9%; cat map 42% dog / 43% cat | Wider maps but less class-specific on this image. |
| LC | Guided Grad-CAM | GB dog-vs-cat sim 0.998 → GGC −0.04 | Guided Backprop alone isn't class-specific; × Grad-CAM fixes it and adds detail. |
| LC | Deletion / insertion | Grad-CAM best on all 4 scores; random worst | Grad-CAM is faithful; dog and cat compete for probability. |
| LC | Sanity check | ρ: 1.00 → 0.57 → 0.01 | Grad-CAM depends on learned weights — passes. |

(LC = learning component.)

---

## 7. Design decisions and rejected alternatives

| Decision | Chosen | Alternatives considered | Why not |
|---|---|---|---|
| Main model | VGG-16 | AlexNet as main | VGG-16 is more accurate (≈72% vs ≈57% top-1) and its deeper stack of small 3×3 filters gives more specific last-layer features, so it gives a clearer "reference" map. AlexNet is used as the comparison. |
| Image | Dog + cat (`both.png`) | Single-object image | The class experiment needs two objects to show the map *moving*; the competition between them also explains the §11 results. |
| Implementation | From scratch with hooks | Only use `pytorch-grad-cam` | The sheet asks to *implement*. Library used only to verify. |
| Capturing activations | Forward hook on the layer | Splitting `model.features` by hand | Splitting must be re-coded per model/layer; hooks work unchanged for §7, §8, §12. |
| Capturing gradients | `tensor.register_hook` | `register_full_backward_hook`; `torch.autograd.grad` | Module backward hooks misbehave with `ReLU(inplace=True)`. `autograd.grad` would work but needs the activation tensor passed around explicitly. |
| Weight gradients | Frozen (`requires_grad_(False)`), input requires grad | Leave weights trainable | Weight gradients are never used but cost ~530 MB per model → out of memory on a 4 GB GPU. |
| Grad-CAM++ | Subclass overriding `weights()` | Separate copy of the class | Only the weight formula differs; one overridden method keeps the difference obvious. |
| Guided Backprop | Custom autograd ReLU on a model copy | Backward hooks on each ReLU | Same in-place problem; a custom `Function` is the standard way to change a backward pass. |
| Score to backprop | Raw logit | Softmax probability, log-probability | Paper uses the logit; softmax mixes in other classes and saturates for confident predictions. |
| Target layer (VGG) | `features[29]` (last ReLU) | `features[28]` (conv), `features[30]` (max-pool) | Conv output is overwritten in place by the ReLU. Max-pool halves the grid to 7×7. |
| Target layer (AlexNet) | `features[11]` | `features[12]` (max-pool, 6×6) | Same reason — keep the larger grid. |
| Upsampling | Bilinear | Nearest, bicubic | Nearest is blocky; bicubic can go below 0. Bilinear matches the library. |
| Normalisation | Min-max to [0,1] per map | Divide by max; none | Needed for a colormap; shows *relative* importance. (Loses absolute strength — see §9 of this file.) |
| Overlay colormap | `jet`, 50% blend | `inferno`/`viridis`, contours | Matches the paper and library. |
| Measuring "where" | Share of heatmap mass in boxes | Visual only; IoU with masks; pointing game | Visual is subjective; masks need another model or labelling. Boxes are simple and drawn on the figure. |
| Comparing maps | Pearson correlation | SSIM | Simple, well known; maps share the same 224×224 grid. |
| Faithfulness metric | Deletion + insertion AUC | Pointing game; ROAR (retrain-and-remove) | Pointing game needs ground-truth boxes and only checks the peak. ROAR needs retraining the model — impossible for one image. |
| Removal baseline | Gaussian blur | Zeros (black), mean colour, noise | Black holes and noise are unnatural shapes that change the prediction on their own. |
| Sanity-check metric | Spearman rank correlation | Pearson, SSIM, HOG similarity | Ranking is what matters for a heatmap; matches the paper. |
| Randomisation order | Cascading, top-down | Independent single-layer | Follows the paper; shows how fast the map breaks. |
| Preprocessing | `weights.transforms()` | Hand-written transform | Guarantees the training preprocessing. |

---

## 8. Alternatives to Grad-CAM

| Method | Idea | Pros vs Grad-CAM | Cons vs Grad-CAM | In notebook? |
|---|---|---|---|---|
| **CAM** (2016) | Use weights of a GAP→linear head as α. | No gradients needed. | Only works on GAP architectures — not VGG/AlexNet without retraining. | No |
| **Grad-CAM++** (2018) | Weighted average of *positive* gradients. | Covers whole objects / several instances better. | Can lose class-specificity (seen in §9); more complex. | **Yes, §9** |
| **Guided Grad-CAM** (2017) | Guided Backprop × Grad-CAM. | High-resolution *and* class-specific. | Needs a second, modified model; GB part fails sanity checks on its own. | **Yes, §10** |
| **Guided Backprop** (2015) | Gradients to pixels, positive paths only. | Pixel-level detail. | Not class-specific (0.998 similarity in §10); fails sanity checks. | **Yes, §10** |
| **Score-CAM** (2020) | Weight each map by the score rise when the image is masked by it. | Gradient-free, often less noisy. | One forward pass per channel (512 for VGG) — much slower. | No |
| **Eigen-CAM** | First principal component of the activations. | Class-free, no gradients, cheap. | Not class-specific — would fail Experiment 1. | No |
| **Layer-CAM** (2021) | Per-pixel gradient weights. | Works on earlier layers, finer maps. | Noisier on the last layer. | No |
| **Integrated Gradients** (2017) | Average gradients along a path from a baseline image. | Satisfies "completeness"; pixel-level. | Needs a baseline and 50+ backward passes. | No |
| **LIME** (Lab 06) | Perturb superpixels, fit a linear model. | Black-box (no internals needed). | Slow, random, depends on segmentation. | No (Lab 06) |
| **Occlusion** | Slide a grey patch, record the score drop. | Very intuitive, black-box. | One forward pass per patch position. | No |
| **RISE** (2018) | Average thousands of random masks weighted by the score. | Black-box, faithful by design. | Thousands of forward passes. | Its *metric* (deletion/insertion) is used in §11 |

Grad-CAM was the required method. Its strengths for this lab are **one forward + one backward pass**, **no settings** except the layer, and being **class-specific** — exactly what Experiment 1 tests. §11 and §12 also show it is faithful and passes the sanity check on this image.

---

## 9. Limitations and how to improve

1. **Coarse resolution.** The map is 14×14 / 13×13 upsampled to 224×224. → *Partly addressed* by Guided Grad-CAM (§10). Layer-CAM is another option.
2. **Min-max normalisation hides strength.** Every map is stretched to 0–1, so the 1.4% "tiger cat" map looks as bright as the 37% "pug" map. → Show raw maps on a shared colour scale, or print the max value.
3. **Gradients can saturate** for very confident predictions, making maps noisy. → Score-CAM (gradient-free).
4. **Plausible ≠ faithful.** → *Addressed* by deletion/insertion (§11) and the sanity check (§12). A stronger but costly test is ROAR (remove pixels, **retrain**, measure accuracy drop) — not possible with one image.
5. **One image.** Conclusions are about this image only; e.g. "Grad-CAM beats Grad-CAM++" may not hold on images with several dogs (Grad-CAM++'s claimed strength). → Repeat §11 over a few hundred ImageNet validation images and average the AUCs.
6. **Hand-drawn regions.** The dog/cat boxes are approximate. → Use a segmentation mask.
7. **Deletion/insertion depends on the baseline.** Blur was chosen as the most natural, but zero or noise baselines would give different numbers. → Report several baselines.
8. **Sanity check covers only the model-randomisation test.** Adebayo et al. also propose a **data-randomisation** test (retrain on shuffled labels). → Needs retraining; out of scope.
9. **Hooks** stay on `vgg` for `cam_vgg` and `campp_vgg` until the kernel restarts. Harmless here (small notebook, run top to bottom), and the loops that create many objects do call `.remove()`.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Activation / feature map** | Output of a conv layer for one filter — a 2-D grid showing where that filter's pattern appears. |
| **Logit** | Raw class score before softmax. |
| **Gradient ∂y<sup>c</sup>/∂A** | How much the class score would change if a feature-map value changed slightly. |
| **Global average pooling (GAP)** | Averaging a 2-D grid to a single number. |
| **Hook (PyTorch)** | A function PyTorch calls automatically during forward or backward, letting us read internals without changing the model. |
| **In-place ReLU** | `ReLU(inplace=True)` overwrites its input tensor instead of creating a new one (saves memory). |
| **`torch.autograd.Function`** | A way to define a custom operation with your own forward and backward pass (used for `GuidedReLU`). |
| **Class-discriminative** | The explanation changes depending on which class you ask about. |
| **Guided Backpropagation** | Back-propagating to the input while letting only positive gradients through each ReLU. |
| **Faithfulness** | How well an explanation reflects what the model actually uses (vs. what merely looks plausible). |
| **Deletion / insertion curve** | Class probability as pixels are removed / added in order of importance. |
| **AUC** | Area under a curve; here, the average probability over the deletion or insertion process. |
| **Sanity check (model randomisation)** | Randomise model weights; a real explanation must change. |
| **Spearman's ρ** | Correlation between the *rankings* of two sets of values (1 = same order, 0 = unrelated). |
| **Receptive field** | The region of the input image that one cell of a feature map "sees". |
| **ImageNet** | 1.2 M image dataset with 1000 classes that both models were trained on. |

---

## 11. Likely viva questions

**Q: What does Grad-CAM compute, in one sentence?**
A weighted sum of the last conv layer's feature maps, where each map's weight is the average gradient of the class score with respect to it, followed by ReLU and upsampling.

**Q: Why the last convolutional layer?**
It is the deepest layer that still has spatial layout. FC layers lose "where"; earlier layers have "where" but low-level features — §7 shows those maps are noise.

**Q: Why ReLU? Why the logit, not softmax?**
ReLU keeps only features that *increase* the class score. The logit is direct evidence for one class; softmax mixes in the other classes and saturates.

**Q: How did you check your implementation is correct?**
Compared both Grad-CAM and Grad-CAM++ with the `pytorch-grad-cam` library: correlation 1.0000 for both.

**Q: VGG-16 predicted "pug" — is the model wrong?**
The dog is an English bulldog, which isn't an ImageNet class. The top-4 are all similar flat-faced breeds, and the heatmap shows it looked at the face.

**Q: What happened when you changed the class? The model?**
Class: the heatmap moved from the dog (69%) to the cat (68%). Model: AlexNet predicted a different breed and gave blurrier, less focused maps (26% of the cat map on the dog), with only ~0.4 correlation to VGG.

**Q: How is Grad-CAM++ different, and was it better?**
It uses a weighted average of only the positive gradients. It covered more of the object but was less class-specific on this image and scored worse than Grad-CAM on all four deletion/insertion scores, because it ignores the negative gradients that let Grad-CAM subtract out the competing dog.

**Q: Why is Guided Backprop alone not enough?**
It isn't class-specific — its dog and cat maps had 0.998 similarity — and it is known to fail the randomisation sanity check. Multiplying by Grad-CAM makes it class-specific (−0.04 similarity) while keeping the pixel detail.

**Q: What are deletion and insertion?**
Remove (or add back) pixels in order of heatmap importance and track the class probability. A faithful map makes the probability drop fast on deletion (low AUC) and rise fast on insertion (high AUC). Grad-CAM beat both Grad-CAM++ and a random map.

**Q: Why did insertion reach 0.95 for pug when the full image gives only 0.37?**
On the full image the dog and cat compete for the softmax probability. With only the dog visible, the model is much more confident. The same happened for the cat (0.014 → ~0.19).

**Q: Why blur instead of black pixels?**
Black holes are unnatural shapes that change the prediction by themselves, so the test would partly measure the hole, not the evidence.

**Q: What is the sanity check and did Grad-CAM pass?**
Randomise the model's layers from the top down and check if the heatmap changes. Grad-CAM's rank correlation fell from 1.0 to 0.57 (classifier randomised) to 0.01 (conv5_3 randomised), so it depends on what the model learned — it passes.

**Q: Why freeze the model weights?**
Grad-CAM only needs gradients with respect to activations. Computing weight gradients wasted ~530 MB per model and ran out of memory on the 4 GB GPU.

**Q: Why use `tensor.register_hook` instead of a module backward hook?**
VGG and AlexNet use in-place ReLUs, which break module backward hooks. A hook on the output tensor avoids that.

**Q: How is Grad-CAM different from CAM? From LIME?**
CAM needs a GAP + single linear layer architecture; Grad-CAM works on any CNN. LIME is black-box (perturb and fit, many forward passes, random); Grad-CAM is white-box (one forward + one backward, deterministic) but needs the model's internals and gives coarser maps.

---

## 12. References

1. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
2. Chattopadhay et al., *Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks*, WACV 2018.
3. Springenberg et al., *Striving for Simplicity: The All Convolutional Net* (Guided Backpropagation), ICLR Workshop 2015.
4. Petsiuk, Das, Saenko, *RISE: Randomized Input Sampling for Explanation of Black-box Models* (deletion/insertion metrics), BMVC 2018.
5. Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018.
6. Zhou et al., *Learning Deep Features for Discriminative Localization* (CAM), CVPR 2016.
7. Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG), ICLR 2015.
8. Krizhevsky, Sutskever, Hinton, *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet), NeurIPS 2012.
9. Jacob Gildenblat et al., `pytorch-grad-cam`, https://github.com/jacobgil/pytorch-grad-cam
