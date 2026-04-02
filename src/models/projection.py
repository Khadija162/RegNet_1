import torch
import torch.nn as nn


class RAGNetProjectionModule(nn.Module):
    def __init__(self, d_llm=384, d_num=32, d_cat=16, d_shared=384,
                 num_numerical_attrs=1, num_categorical_attrs=1,
                 num_classes_per_cat_attr=None):
        super().__init__()
        num_classes_per_cat_attr = num_classes_per_cat_attr or [10] * num_categorical_attrs
        self.num_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(1, d_num), nn.ReLU(), nn.LayerNorm(d_num))
            for _ in range(num_numerical_attrs)
        ])
        self.cat_emb = nn.ModuleList([
            nn.Embedding(n_cls, d_cat) for n_cls in num_classes_per_cat_attr
        ])
        self.shared_proj_num = nn.ModuleList([
            nn.Linear(d_llm + d_num, d_shared) for _ in range(num_numerical_attrs)
        ])
        self.shared_proj_cat = nn.ModuleList([
            nn.Linear(d_llm + d_cat, d_shared) for _ in range(num_categorical_attrs)
        ])

    def forward_numerical(self, attr_idx, llm_emb, raw_value):
        if llm_emb.dim() == 1:
            llm_emb = llm_emb.unsqueeze(0)
        raw_value = torch.as_tensor(raw_value, dtype=torch.float32, device=llm_emb.device).view(-1, 1)
        num_vector = self.num_proj[attr_idx](raw_value)
        return self.shared_proj_num[attr_idx](torch.cat([llm_emb, num_vector], dim=-1)).squeeze(0)

    def forward_categorical(self, attr_idx, llm_emb, class_index):
        if llm_emb.dim() == 1:
            llm_emb = llm_emb.unsqueeze(0)
        class_index = torch.as_tensor(class_index, dtype=torch.long, device=llm_emb.device).view(-1)
        cat_vector = self.cat_emb[attr_idx](class_index)
        return self.shared_proj_cat[attr_idx](torch.cat([llm_emb, cat_vector], dim=-1)).squeeze(0)
