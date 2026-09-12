# PTS-JEPA

Research codebase for self-supervised computer vision on petrographic thin-section imagery.

## Current benchmark

The first controlled benchmark compares:

- ResNet-18 (supervised CNN baseline)
- ViT-Tiny (supervised transformer baseline)
- I-JEPA-Tiny (self-supervised representation-learning model)

The first dataset is the public 2,500-image XPL subset of MUMDMC2025. The dataset paper describes five mineral classes and a public cropped subset of 500 images per class. The full collection contains 14,400 images acquired at 72 rotations under PPL and XPL. We keep the images out of git and provide reproducible acquisition/manifest code instead.

## Get the MUMDMC2025 subset

See [`data/README.md`](data/README.md). Two supported workflows are implemented:

### Automatic Figshare download

```bash
python data/prepare_mumdmc.py --download
```

### Use a downloaded/uploaded archive

```bash
python data/prepare_mumdmc.py --local-source /path/to/MUMDMC2025_DataSet_sample.zip
```

The preparation step creates `data/manifests/mumdmc2025.csv` and a JSON summary. Dataset files are ignored by git.

## Research integrity

The benchmark must avoid data leakage. Repeated views of the same specimen/crystal must not cross train/validation/test boundaries. When reliable specimen metadata are available, the split should be group-aware. Without such metadata, the preparation step uses conservative filename-based grouping and flags the limitation.

## Provenance

Amer, B. G., Mousa, H. M., Dawoud, M., & Youssef, A. (2025). *A Photomicrographic Dataset of Rocks for the Accurate Classification of Minerals*. Scientific Data, 12, 1775. DOI: 10.1038/s41597-025-05879-9.

Dataset: Amer, B. G. *MUMDMC2025_DataSet_sample*. Figshare (2025). DOI: 10.6084/m9.figshare.29483204.v1.
