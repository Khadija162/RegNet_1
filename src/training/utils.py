from collections import defaultdict

import torch
import torch.nn.functional as F


def build_modality_based_edge_index(normalized_embeddings, modality_types, top_k=5):
    n, m, d = normalized_embeddings.shape
    attr_embeddings = []
    for attr in range(m):
        emb = normalized_embeddings[:, attr, :]
        mask = emb.abs().sum(dim=1) != 0
        mean = emb[mask].mean(dim=0) if mask.any() else torch.zeros(d, dtype=normalized_embeddings.dtype)
        attr_embeddings.append(mean)
    attr_embeddings = F.normalize(torch.stack(attr_embeddings, dim=0), dim=1)
    groups = defaultdict(list)
    for i, mod in enumerate(modality_types):
        groups[mod].append(i)
    edges = []
    for _, idxs in groups.items():
        if len(idxs) < 2:
            continue
        sim = torch.matmul(attr_embeddings[idxs], attr_embeddings[idxs].T)
        for i, src in enumerate(idxs):
            top = torch.topk(sim[i], k=min(top_k + 1, len(idxs))).indices.tolist()
            for t in top:
                dst = idxs[t]
                if src != dst:
                    edges.append((src, dst))
    edges += [(j, i) for i, j in edges]
    edges = sorted(set(edges))
    return torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)
