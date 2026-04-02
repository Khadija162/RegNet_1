from pathlib import Path
from typing import Dict

from src.datasets.base import DatasetSpec
from src.utils.config import load_yaml


DATASET_REGISTRY: Dict[str, str] = {
    "hiad_2_1": "configs/datasets/hiad_2_1.yaml",
    "adult": "configs/datasets/adult.yaml",
    "australian_credit": "configs/datasets/australian_credit.yaml",
    "airbnb": "configs/datasets/airbnb.yaml",
    "nyc_crash": "configs/datasets/nyc_crash.yaml",
}


def load_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    payload = load_yaml(Path(DATASET_REGISTRY[name]))
    return DatasetSpec(**payload)
