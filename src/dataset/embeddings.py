from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class TextEmbedder:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            if not str(text).strip():
                embeddings.append(np.zeros(self.model.config.hidden_size, dtype=np.float32))
                continue
            batch = self.tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=128)
            out = self.model(**batch)
            pooled = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            embeddings.append(pooled)
        return np.stack(embeddings, axis=0)


def build_simple_modality_embeddings(prepared_data, d_model: int = 384) -> torch.Tensor:
    n = len(prepared_data.dataframe)
    m = len(prepared_data.modalities)
    result = np.zeros((n, m, d_model), dtype=np.float32)

    # lightweight deterministic placeholders; replace with stronger encoders later if needed
    idx = 0
    for j in range(prepared_data.text_values.shape[1]):
        lengths = np.array([len(str(x).split()) for x in prepared_data.text_values[:, j]], dtype=np.float32)
        result[:, idx, 0] = lengths
        idx += 1

    for j in range(prepared_data.categorical_indices.shape[1]):
        vals = prepared_data.categorical_indices[:, j].astype(np.float32)
        result[:, idx, 0] = vals
        result[:, idx, 1] = vals % 7
        idx += 1

    for j in range(prepared_data.numerical_values.shape[1]):
        vals = prepared_data.numerical_values[:, j].astype(np.float32)
        result[:, idx, 0] = vals
        idx += 1

    tensor = torch.tensor(result, dtype=torch.float32)
    non_zero = tensor.abs().sum(dim=2) > 0
    for attr in range(m):
        mask = non_zero[:, attr]
        if mask.any():
            mu = tensor[mask, attr].mean(dim=0)
            sigma = tensor[mask, attr].std(dim=0)
            tensor[mask, attr] = (tensor[mask, attr] - mu) / (sigma + 1e-8)
    return tensor
