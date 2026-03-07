# CIFAR-10 Image Classification — CNN · PyTorch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-0.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Compatible-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-74.71%25-brightgreen?style=flat-square)

<br/>

A clean, well-documented **Convolutional Neural Network** implementation for **CIFAR-10** image classification, built with **PyTorch**.  
Covers the complete deep learning pipeline — from raw image tensors to evaluated predictions — with a focus on **clarity, correctness, and reproducibility**.

</div>

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **74.71%** |
| Final Training Loss | **0.1135** |
| Epochs | 10 |
| Batch Size | 64 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |

**Training loss progression:**

```
Epoch  1/10  →  loss: 0.7556  ██████████████████████████████
Epoch  2/10  →  loss: 0.6322  ████████████████████████
Epoch  3/10  →  loss: 0.5293  ████████████████████
Epoch  4/10  →  loss: 0.4282  ████████████████
Epoch  5/10  →  loss: 0.3469  █████████████
Epoch  6/10  →  loss: 0.2791  ██████████
Epoch  7/10  →  loss: 0.2118  ████████
Epoch  8/10  →  loss: 0.1725  ██████
Epoch  9/10  →  loss: 0.1350  █████
Epoch 10/10  →  loss: 0.1135  ████
```

---

## Architecture

```
Input (3 × 32 × 32)
        │
        ▼
┌─────────────────┐
│  Conv2D  32ch   │  3×3, padding=1  →  32 × 32 × 32
│  ReLU           │
│  MaxPool 2×2    │                  →  32 × 16 × 16
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Conv2D  64ch   │  3×3, padding=1  →  64 × 16 × 16
│  ReLU           │
│  MaxPool 2×2    │                  →  64 × 8 × 8
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Conv2D  128ch  │  3×3, padding=1  →  128 × 8 × 8
│  ReLU           │
│  MaxPool 2×2    │                  →  128 × 4 × 4
└────────┬────────┘
         │
         ▼
     Flatten      →  2048
         │
         ▼
┌─────────────────┐
│  Linear 2048→256│
│  ReLU           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Linear  256→10 │  (logits, no softmax — handled by CrossEntropyLoss)
└────────┬────────┘
         │
         ▼
   Output (10 classes)
```

**Total learnable parameters:** ~562K

---

## Why This Implementation

Most introductory CIFAR-10 notebooks use two convolutional layers and SGD. This implementation makes deliberate choices that improve both performance and learning value:

**Three convolutional stages instead of two.**  
The added third stage (128 channels) allows the network to learn more abstract, discriminative features before the classifier head. This directly contributes to the ~74% test accuracy without any regularization techniques.

**Adam optimizer.**  
Adaptive learning rate gives smooth, robust convergence out of the box — visible in the monotonically decreasing loss curve — without requiring manual LR scheduling or warmup.

**`CrossEntropyLoss` used correctly.**  
No softmax in the final layer. `nn.CrossEntropyLoss` fuses log-softmax and negative log-likelihood internally, which is both numerically more stable and the idiomatic PyTorch pattern.

**Tensor shapes documented inline.**  
Every layer transition is annotated with input → output dimensions, making the notebook a readable reference for understanding how spatial dimensions change through convolution and pooling.

**Extendable structure.**  
Adding `BatchNorm2d`, `Dropout`, or residual connections requires no architectural refactoring. The model serves as a clean baseline from which deeper networks (VGG-style, ResNet-style) can be incrementally derived.

---

## Dataset

**CIFAR-10** — 60,000 RGB images (32×32) across 10 classes.

| Split | Images |
|---|---|
| Training | 50,000 |
| Test | 10,000 |

Classes: `airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

---

## Project Structure

```
cnn-cifar10-pytorch/
├── CNN_for_CIFAR10.ipynb   # Main notebook — data loading, training, evaluation
├── data/                   # CIFAR-10 downloaded here on first run
└── README.md
```

---

## Quickstart

**1. Clone**
```bash
git clone https://github.com/yourusername/cnn-cifar10-pytorch.git
cd cnn-cifar10-pytorch
```

**2. Install dependencies**
```bash
pip install torch torchvision notebook
```

**3. Run**
```bash
jupyter notebook CNN_for_CIFAR10.ipynb
```

CIFAR-10 data downloads automatically on first run (`download=True`). No manual setup required.

---

## Training Pipeline

```python
for epoch in range(epochs):
    for images, labels in trainloader:
        optimizer.zero_grad()          # clear accumulated gradients
        output = model(images)         # forward pass
        loss = criterion(output, labels)  # cross-entropy loss
        loss.backward()                # backpropagation
        optimizer.step()               # update weights (Adam)
```

Evaluation is performed with `model.eval()` and `torch.no_grad()` to disable gradient tracking and ensure deterministic inference.

---

## Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.13 | Runtime |
| PyTorch | 2.x | Deep learning framework |
| torchvision | 0.x | Dataset loading & transforms |
| NumPy | — | Numerical operations |
| Jupyter Notebook | — | Interactive development |

---

## Possible Extensions

- Add `BatchNorm2d` after each conv layer for faster convergence
- Add `Dropout` before the final linear layer to reduce overfitting
- Extend to 15–20 epochs and add a learning rate scheduler
- Implement per-class accuracy breakdown
- Add data augmentation (`RandomHorizontalFlip`, `RandomCrop`)

---

## Author

Minor Project — Deep Learning  
CNN-based Image Classification using PyTorch on CIFAR-10
