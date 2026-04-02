# RAGNet Multimodal

A configurable repository for multimodal missing-data imputation with support for:
- text, categorical, and numerical attributes
- dataset-specific column schemas
- configurable missingness ratio and scenario
- train/evaluate entry points

## Supported datasets
- HIAD 2.1
- Adult
- Australian Credit
- Airbnb
- NYC Crash

Each dataset is defined by a YAML file in `configs/datasets/`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train:

```bash
python -m src.main train --config configs/experiments/train_hiad.yaml
```

Evaluate:

```bash
python -m src.main evaluate --config configs/experiments/eval_hiad.yaml
```

Override from CLI:

```bash
python -m src.main train \
  --config configs/experiments/train_hiad.yaml \
  --dataset adult \
  --missing-ratio 0.2 \
  --missing-scenario mar
```

## Repository layout

```text
ragnet-multimodal/
├── configs/
├── data/
├── docs/
├── scripts/
├── src/
└── outputs/
```

## Notes
- Put raw CSVs in `data/raw/`.
- Update dataset YAML files with your real column names and file paths.
- The model scaffold is adapted from your research code and cleaned into modules.
- This repo is designed to be extended; some dataset-specific text embedding work is intentionally lightweight by default.




