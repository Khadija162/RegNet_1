from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.datasets.embeddings import build_simple_modality_embeddings
from src.datasets.preprocessing import DataPreprocessor
from src.datasets.registry import load_dataset_spec
from src.evaluation.metrics import safe_mean, summarize_categorical, summarize_numeric
from src.models.ragnet import RAGNet
from src.training.utils import build_modality_based_edge_index
from src.utils.config import load_yaml
from src.utils.io import ensure_dir, save_json


def _merge_overrides(cfg, args):
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.missing_ratio is not None:
        cfg.setdefault("missingness", {})["ratio"] = args.missing_ratio
    if args.missing_scenario is not None:
        cfg.setdefault("missingness", {})["scenario"] = args.missing_scenario
    if args.checkpoint:
        cfg.setdefault("model", {})["checkpoint_path"] = args.checkpoint
    return cfg


def run_evaluation(args) -> None:
    cfg = _merge_overrides(load_yaml(args.config), args)
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

    checkpoint = cfg.get("model", {}).get("checkpoint_path")
    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        print("checkpoint not found; evaluating untrained weights")

    indices = list(range(len(embeddings)))
    _, test_idx = train_test_split(indices, test_size=cfg["training"]["test_size"], random_state=cfg["training"]["seed"])

    model.eval()
    embeddings = embeddings.to(device)
    edge_index = edge_index.to(device)

    num_true, num_pred, cat_true, cat_pred = [], [], [], []
    with torch.no_grad():
        for idx in test_idx:
            emb = embeddings[idx]
            cat = torch.tensor(prepared.categorical_indices[idx], dtype=torch.long, device=device)
            num = torch.tensor(prepared.numerical_values[idx], dtype=torch.float32, device=device)
            gt = prepared.ground_truth[idx]
            _, outputs = model(emb, num, cat, edge_index, prepared.modalities, gt)
            out_ptr = 0
            for attr_idx, modality in enumerate(prepared.modalities):
                if modality == "txt":
                    out_ptr += 1
                    continue
                if modality == "cat":
                    cat_true.append(int(gt[attr_idx]))
                    cat_pred.append(int(outputs[out_ptr].argmax().item()))
                if modality == "num":
                    num_true.append(float(gt[attr_idx]))
                    num_pred.append(float(outputs[out_ptr].squeeze().item()))
                out_ptr += 1

    metrics = {
        "num": summarize_numeric(num_true, num_pred) if num_true else {},
        "cat": summarize_categorical(cat_true, cat_pred) if cat_true else {},
        "num_samples": len(test_idx),
    }
    output_root = Path(cfg["output"]["root_dir"]) / cfg["output"]["run_name"]
    ensure_dir(output_root)
    save_json(output_root / "eval_metrics.json", metrics)
    print(metrics)
