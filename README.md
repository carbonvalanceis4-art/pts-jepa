# PTS-JEPA

**Petrographic Thin Section JEPA** — a lightweight research codebase for testing whether self-supervised Joint-Embedding Predictive Architectures can learn useful representations for mineral identification in petrographic thin sections.

## Research question

> Can JEPA-style self-supervised pretraining improve mineral-grain classification, especially when labeled petrographic training data are limited?

## Initial benchmark plan

### Phase 1: inexpensive proof of concept
Dataset: MUMDMC2025-compatible image folders.

Compare:
- ResNet-18 supervised baseline
- ViT-Tiny supervised baseline
- I-JEPA-Tiny pretraining + linear probe

Primary metrics:
- accuracy
- macro-F1
- per-class F1
- confusion matrix
- training time
- parameter count

### Phase 2: label-efficiency experiment
Evaluate downstream classifiers using 1%, 5%, 10%, 25%, 50%, and 100% of available labels.

### Phase 3: rotation-aware experiment
Use specimen-aware splits and rotational metadata to test whether multi-orientation observations improve classification.

## Important experimental rule

Do **not** randomly split near-duplicate rotational images from the same specimen across training and test sets. Splits should be specimen/crystal aware whenever metadata permit this.

## Repository status

This repository is being built as a small, reproducible research prototype. It intentionally does not attempt to reproduce the large-scale compute configuration of the original I-JEPA work.

## Planned layout

```text
pts-jepa/
├── configs/
├── data/
├── models/
├── training/
├── evaluation/
├── experiments/
├── tests/
└── notebooks/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Dataset preparation and training commands will be added as the prototype is implemented.

## Intended datasets

The code will support adapters rather than bundling datasets:
- MUMDMC2025 for mineral classification and rotation experiments
- LITHOS for larger-scale grain-level benchmarking
- DeepCarbonate as a separate petrographic/lithological generalization task

Dataset licenses and access requirements remain the responsibility of the user.

## Research principle

The goal is not to prove that JEPA is automatically superior. The goal is to run a fair benchmark in which JEPA can succeed or fail against strong baselines.
