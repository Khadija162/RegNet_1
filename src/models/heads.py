import torch.nn as nn
import torch.nn.functional as F


class NumericalHead(nn.Module):
    def __init__(self, d_h: int):
        super().__init__()
        self.fc1 = nn.Linear(d_h, d_h // 2)
        self.norm = nn.LayerNorm(d_h // 2)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(d_h // 2, 1)

    def forward(self, u):
        x = F.relu(self.fc1(u))
        x = self.norm(x)
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


class CategoricalHead(nn.Module):
    def __init__(self, d_h: int, num_classes: int):
        super().__init__()
        self.cls_head = nn.Linear(d_h, num_classes)

    def forward(self, u):
        return self.cls_head(u)
