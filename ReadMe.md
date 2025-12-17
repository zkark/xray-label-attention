# XRay-ML: Label-Attention ConvNeXt for Chest X-ray Diagnosis

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-red)

## Overview

**XRay-ML** is a complete, reproducible deep-learning pipeline developed for the **Grand X-ray Slam (Division A)** Kaggle competition. The system performs **multi-label diagnosis of 14 thoracic diseases** from chest X-ray images using a **ConvNeXt-Tiny backbone enhanced with a Label-Attention Head** to model medical context and disease co-occurrence.

The pipeline covers **data preprocessing, efficient storage, training, evaluation, exploratory data analysis (EDA), and submission generation**. It achieves a **mean AUROC of 0.925–0.927** across the 14 disease classes.

---

## Key Results

* **Task**: Multi-label chest X-ray classification (14 diseases)
* **Backbone**: ConvNeXt-Tiny
* **Input**: 320×320 grayscale X-ray images
* **Modeling strategy**:

  * Patient-wise data splits (leakage-free)
  * Label-attention head to capture inter-disease correlation
  * Exponential Moving Average (EMA) of weights
* **Performance**: Mean AUROC **0.925–0.927**

---

## Methodology

### Data preprocessing

* Raw X-ray images are resized and converted to grayscale.
* Images are stored in **memory-mapped NumPy arrays (`.npy`)**, allowing training on large datasets with minimal RAM usage.
* This preprocessing strategy reduces training I/O overhead and **accelerates training by ~40%**.

### Model architecture

* A **ConvNeXt-Tiny** backbone (via `timm`) is used for feature extraction.
* The default classification head is replaced with a **Label-Attention Head**, which explicitly models correlations among diseases.
* This design reflects real-world medical context where certain conditions frequently co-occur.

### Training strategy

* **Patient-wise splitting** prevents data leakage between train and validation folds.
* Optimizer: AdamW with **layer-wise learning rate decay (LLRD)**.
* Scheduler: Warmup followed by cosine decay.
* Mixed precision training (`torch.amp`) for speed and memory efficiency.
* Multiple loss options supported (default: asymmetric loss).

---

## Repository Structure

```
xray-ml/
├─ preprocess_train_and_generate_submission.ipynb  # preprocessing + training + submission
├─ analyze_data.ipynb                              # EDA and label correlation analysis
├─ requirements.txt                                # dependencies
├─ README.md                                       # documentation
├─ LICENSE                                         # MIT license
├─ .gitignore                                      # ignored files
└─ Runs/                                           # model checkpoints (generated)
```

---

## File Descriptions

### `preprocess_train_and_generate_submission.ipynb`

This notebook contains the full training pipeline.

**Preprocessing**

* Validates dataset paths and CSV integrity
* Packs images into memory-mapped `.npy` arrays using multiprocessing
* Generates:

  * `images_train_320_c1_uint8.npy`
  * `labels_train.npy`
  * `index_train.csv`
  * corresponding test files

**Training & inference**

* Custom PyTorch datasets for memmap access
* Patient-wise group splitting
* ConvNeXt + Label-Attention model
* EMA, gradient clipping, AMP, and checkpointing
* Ensemble inference and generation of `submission.csv`

### `analyze_data.ipynb`

* Computes per-disease label counts
* Calculates Pearson correlation matrix across labels
* Visualizes disease co-occurrence patterns
* Saves correlation heatmap as `disease_corr_matrix.png`

---

## Reproducibility

* Random seed fixed via `SEED`
* Patient-wise splits eliminate data leakage
* Deterministic behavior can be enforced by disabling CuDNN benchmarking

---

## Hardware Requirements

* **GPU recommended** (≥8 GB VRAM)
* Tested on Kaggle GPUs (T4 / P100 / A100)
* CPU-only execution is possible but significantly slower

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage (Kaggle)

1. Create a new Kaggle Notebook
2. Attach the dataset: `grand-xray-slam-division-a`
3. Upload repository files or copy notebooks
4. Ensure all paths point to `/kaggle/input/` and outputs to `/kaggle/working/`
5. Enable GPU acceleration
6. Run `preprocess_train_and_generate_submission.ipynb`
7. Retrieve `submission.csv` from `/kaggle/working/`

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Citation

If you use this work for research or academic purposes, please cite:

```text
Zain Soliman. XRay-ML: Label-Attention ConvNeXt for Chest X-ray Diagnosis, 2025.
```

---

## Acknowledgements

* Kaggle Grand X-ray Slam organizers
* PyTorch and timm communities
* ConvNeXt authors
