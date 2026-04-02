from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.losses.marip_loss import MARIPLoss
from src.memory.fusion import AttributeMemoryFusion
from src.memory.memory_bank import PerAttributeMemoryBank
from src.models.decoder import TextualDecoder
from src.models.gnn import CrossModalGNNLayer
from src.models.heads import CategoricalHead, NumericalHead
from src.models.projection import RAGNetProjectionModule


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class RAGNet(nn.Module):
    def __init__(self, d_llm=384, d_num=32, d_cat=16, d_shared=384,
                 num_numerical_attrs=1, num_categorical_attrs=1,
                 num_classes_per_cat_attr=None, vocab_size=None,
                 decoder_dim=384, max_text_len=64, memory_size=300):
        super().__init__()
        num_classes_per_cat_attr = num_classes_per_cat_attr or [10] * max(1, num_categorical_attrs)
        self.projection = RAGNetProjectionModule(
            d_llm=d_llm,
            d_num=d_num,
            d_cat=d_cat,
            d_shared=d_shared,
            num_numerical_attrs=max(1, num_numerical_attrs),
            num_categorical_attrs=max(1, num_categorical_attrs),
            num_classes_per_cat_attr=num_classes_per_cat_attr,
        )
        self.gnn = CrossModalGNNLayer(d_shared)
        self.memory_bank = PerAttributeMemoryBank(
            num_attributes=num_numerical_attrs + num_categorical_attrs + 64,
            d_h=d_shared,
            max_memory=memory_size,
        )
        self.memory_fusion = AttributeMemoryFusion(d_shared, self.memory_bank)
        self.num_heads = nn.ModuleList([NumericalHead(d_shared) for _ in range(num_numerical_attrs)])
        self.cat_heads = nn.ModuleList([CategoricalHead(d_shared, n) for n in num_classes_per_cat_attr])
        self.txt_decoder = TextualDecoder(d_shared, vocab_size or tokenizer.vocab_size, max_len=max_text_len, decoder_dim=decoder_dim)
        self.loss_module = MARIPLoss()

    def forward(self, llm_embeddings, raw_numericals, cat_indices, edge_index, modality_types: List[str], targets=None):
        device = llm_embeddings.device
        z = []
        num_idx = 0
        cat_idx = 0
        for i, modality in enumerate(modality_types):
            if modality == "txt":
                emb = llm_embeddings[i]
            elif modality == "cat":
                emb = self.projection.forward_categorical(cat_idx, llm_embeddings[i], cat_indices[cat_idx])
                cat_idx += 1
            elif modality == "num":
                emb = self.projection.forward_numerical(num_idx, llm_embeddings[i], raw_numericals[num_idx])
                num_idx += 1
            else:
                raise ValueError(f"Unknown modality: {modality}")
            z.append(emb)
        z = torch.stack(z, dim=0)
        z_gnn = self.gnn(z, edge_index)
        z_mem = self.memory_fusion(z_gnn)

        outputs = []
        targets_out = []
        modality_out = []
        num_idx = 0
        cat_idx = 0
        for i, modality in enumerate(modality_types):
            if modality == "txt":
                if targets is not None:
                    tok = tokenizer(str(targets[i]), return_tensors="pt", truncation=True, padding="max_length", max_length=16).input_ids.to(device)
                    outputs.append(self.txt_decoder(z_mem[i], tok))
                    targets_out.append(targets[i])
                    modality_out.append(modality)
            elif modality == "cat":
                outputs.append(self.cat_heads[cat_idx](z_mem[i].unsqueeze(0)).squeeze(0))
                if targets is not None:
                    targets_out.append(targets[i])
                    modality_out.append(modality)
                cat_idx += 1
            elif modality == "num":
                outputs.append(self.num_heads[num_idx](z_mem[i].unsqueeze(0)))
                if targets is not None:
                    targets_out.append(targets[i])
                    modality_out.append(modality)
                num_idx += 1

        if targets is None:
            return None, outputs
        loss = self.loss_module(outputs, targets_out, modality_out)
        return loss, outputs
