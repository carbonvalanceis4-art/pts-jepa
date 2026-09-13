# Optional notebooks

The project is intentionally platform-independent. The primary workflow is the local command-line pipeline in the repository root.

Use notebooks only for exploratory analysis, visualization, or presentation of saved experiment outputs. Training should run through `experiments/run_e1.py` so experiments are reproducible from the command line.

## E1 workflow

```powershell
python data\audit_mumdmc.py
python experiments\run_e1.py --epochs 5 --ijepa-pretrain-epochs 10 --probe-epochs 20
```

The benchmark writes machine-readable results under `outputs/e1/`.
