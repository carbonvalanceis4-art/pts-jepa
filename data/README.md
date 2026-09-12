# MUMDMC2025 data workflow

The images are **not stored in GitHub**. `pts-jepa` supports two reproducible ways to obtain the public cropped XPL subset.

## A. Automatic download from Figshare

The paper identifies the public subset as 2,500 cropped XPL images, 500 per mineral class, and gives the Figshare record `MUMDMC2025_DataSet_sample` (DOI 10.6084/m9.figshare.29483204.v1). The direct file endpoint used by the downloader is the public Figshare file `55998200`.

```bash
python data/prepare_mumdmc.py --download
```

This downloads the archive into `data/raw/`, extracts it, scans the images, and creates:

- `data/manifests/mumdmc2025.csv`
- `data/manifests/mumdmc2025.json`

The downloader prints a SHA-256 checksum for provenance.

## B. Use a downloaded/uploaded copy

This is useful when the environment cannot reach Figshare, or when a collaborator has already downloaded the data.

```bash
python data/prepare_mumdmc.py --local-source /path/to/MUMDMC2025_DataSet_sample.zip
```

An already extracted directory also works:

```bash
python data/prepare_mumdmc.py --local-source /path/to/MUMDMC2025_DataSet_sample/
```

In a notebook/ChatGPT environment, upload the ZIP containing the cropped subset, then pass its local path to `--local-source`.

## Important leakage rule

The MUMDMC2025 full dataset contains repeated views across 72 rotation angles and both PPL/XPL. The public 2,500-image subset is XPL-only. When specimen/crystal metadata are available, the benchmark must split by specimen/crystal rather than randomly splitting individual views. If no trustworthy grouping metadata are present, the preparation script uses a conservative filename-based fallback and reports that in the manifest summary.

## Dataset provenance

Amer et al. (2025), *A Photomicrographic Dataset of Rocks for the Accurate Classification of Minerals*, Scientific Data 12, 1775. DOI: 10.1038/s41597-025-05879-9.

Dataset citation: Amer, B. G. *MUMDMC2025_DataSet_sample*. Figshare (2025), DOI 10.6084/m9.figshare.29483204.v1.
