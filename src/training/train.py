from pathlib import Path

import pandas as pd
import torch

from src.datasets.embeddings import build_simple_modality_embeddings
from src.datasets.preprocessing import DataPreprocessor
from src.datasets.registry import load_dataset_spec
from src.models.ragnet import RAGNet
from src.training.trainer import Trainer
from src.training.utils import build_modality_based_edge_index
from src.utils.config import load_yaml
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed


def _merge_overrides(cfg, args):
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.missing_ratio is not None:
        cfg.setdefault("missingness", {})["ratio"] = args.missing_ratio
    if args.missing_scenario is not None:
        cfg.setdefault("missingness", {})["scenario"] = args.missing_scenario
    return cfg


def run_training(args) -> None:
    cfg = _merge_overrides(load_yaml(args.config), args)
    set_seed(cfg["training"]["seed"])

    spec = load_dataset_spec(cfg["dataset"])
    df = pd.read_csv(spec.file_path)
    prepared = DataPreprocessor(spec).prepare(df)
    embeddings = build_simple_modality_embeddings(prepared)
    edge_index = build_modality_based_edge_index(embeddings, prepared.modalities, top_k=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAGNet(
        d_llm=384,
        d_num=32,
        d_cat=16,
        d_shared=384,
        num_numerical_attrs=len(spec.numerical_columns),
        num_categorical_attrs=len(spec.categorical_columns),
        num_classes_per_cat_attr=prepared.num_classes_per_cat_attr or [2],
    ).to(device)

    output_root = Path(cfg["output"]["root_dir"]) / cfg["output"]["run_name"]
    trainer = Trainer(
        model=model,
        device=device,
        lr=cfg["training"]["lr"],
        batch_size=cfg["training"]["batch_size"],
        epochs=cfg["training"]["epochs"],
        output_dir=output_root,
    )
    result = trainer.fit(embeddings, prepared, edge_index, test_size=cfg["training"]["test_size"], seed=cfg["training"]["seed"])
    ensure_dir(output_root)
    save_json(output_root / "train_summary.json", result.__dict__)
    print(result)
