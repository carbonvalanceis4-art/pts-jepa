# Online execution

## Kaggle

The recommended execution environment for E1 is Kaggle Notebooks with GPU enabled.

Open `notebooks/mumdmc_e1_kaggle.ipynb` from this repository in Kaggle, enable Internet and GPU in the notebook settings, then run the cells in order.

The notebook:

1. clones the current `pts-jepa` repository;
2. installs `requirements.txt`;
3. downloads the public 2,500-image MUMDMC2025 XPL subset from Figshare;
4. audits the downloaded dataset;
5. derives research groups from the documented MUMDMC filename convention;
6. creates a group-aware train/validation/test split;
7. trains ResNet-18 and ViT-Tiny supervised baselines;
8. pretrains I-JEPA-Tiny without mineral labels;
9. trains a linear probe on the frozen I-JEPA representation;
10. writes E1 metrics to `outputs/e1/metrics.csv` and `outputs/e1/results.json`.

The notebook uses short default runs as a development benchmark. Once the pipeline completes successfully, increase the epoch counts in the notebook for the formal experiment and record the exact configuration in the research log.

## Local execution

The same repository can be run locally. The 4 GB MUMDMC ZIP does not need to be uploaded to GitHub; point `data/prepare_mumdmc.py --local-source` at the downloaded archive.
