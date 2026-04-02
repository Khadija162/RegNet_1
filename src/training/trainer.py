from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.io import ensure_dir


@dataclass
class TrainResult:
    checkpoint_path: str
    train_size: int
    test_size: int


class Trainer:
    def __init__(self, model, device, lr: float, batch_size: int, epochs: int, output_dir: Path):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = output_dir

    def fit(self, embeddings, prepared_data, edge_index, test_size=0.2, seed=42):
        indices = np.arange(len(embeddings))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
        embeddings = embeddings.to(self.device)
        edge_index = edge_index.to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for idx in tqdm(train_idx, desc=f"Epoch {epoch+1}/{self.epochs}"):
                self.optimizer.zero_grad()
                emb = embeddings[idx]
                cat = torch.tensor(prepared_data.categorical_indices[idx], dtype=torch.long, device=self.device)
                num = torch.tensor(prepared_data.numerical_values[idx], dtype=torch.float32, device=self.device)
                gt = prepared_data.ground_truth[idx]
                loss, _ = self.model(emb, num, cat, edge_index, prepared_data.modalities, gt)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())
            avg = total_loss / max(1, len(train_idx))
            print(f"epoch={epoch+1} train_loss={avg:.4f}")

        ensure_dir(self.output_dir)
        checkpoint = self.output_dir / "model.pt"
        torch.save(self.model.state_dict(), checkpoint)
        return TrainResult(str(checkpoint), len(train_idx), len(test_idx))
