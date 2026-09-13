# PTS-JEPA

Research codebase for self-supervised computer vision on petrographic thin-section imagery.

## Research question

Can a lightweight JEPA-style self-supervised representation improve mineral-grain classification in petrographic thin sections, especially when labeled data are limited?

## E1 benchmark

The first controlled comparison is:

- ResNet-18 — supervised CNN baseline
- ViT-Tiny — supervised transformer baseline
- I-JEPA-Tiny — self-supervised representation learning + frozen linear probe

The benchmark uses the public 2,500-image XPL subset of MUMDMC2025. The images stay outside GitHub; the repository contains reproducible acquisition, auditing, splitting, training, and evaluation code. The MUMDMC paper describes the full dataset as 14,400 images across 72 rotation positions under PPL/XPL and documents the filename metadata needed for grouping. citeturn636697search0

## Windows/local quick start

From a fresh clone on Windows PowerShell:

```powershell
cd pts-jepa
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

If you already downloaded the ~4 GB cropped-subset archive:

```powershell
python data\prepare_mumdmc.py --local-source "C:\path\to\MUMDMC2025_DataSet_sample.zip"
python data\audit_mumdmc.py
python experiments\run_e1.py --epochs 5 --ijepa-pretrain-epochs 10 --probe-epochs 20
```

Or download the public subset directly from Figshare:

```powershell
python data\prepare_mumdmc.py --download
```

## Research safeguards

Before training, audit the manifest and create a group-aware split. Rotation angle is excluded from the research grouping so repeated views of the same crystal/photomicrograph cannot cross train/validation/test boundaries. Any failed parsing should stop the benchmark rather than silently falling back to a random image split.

## Outputs

E1 writes:

- `outputs/e1/metrics.csv` — accuracy and macro-F1
- `outputs/e1/results.json` — metrics plus confusion matrices
- `outputs/e1/split_sizes.json` — split sizes, group count, elapsed time

Generated data, checkpoints, and result artifacts are ignored by git.

## Repository layout

```text
pts-jepa/
├── data/
│   ├── prepare_mumdmc.py
│   ├── audit_mumdmc.py
│   ├── manifest.py
│   └── splits.py
├── models/
│   ├── baselines.py
│   └── ijepa_tiny.py
├── experiments/
│   └── run_e1.py
├── scripts/
│   └── setup_windows.ps1
├── tests/
└── requirements.txt
```

## Provenance

Amer, B. G., Mousa, H. M., Dawoud, M., & Youssef, A. (2025). *A Photomicrographic Dataset of Rocks for the Accurate Classification of Minerals*. Scientific Data, 12, 1775. DOI: 10.1038/s41597-025-05879-9.

Dataset: Amer, B. G. *MUMDMC2025_DataSet_sample*. Figshare (2025). DOI: 10.6084/m9.figshare.29483204.v1.
