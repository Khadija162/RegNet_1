import torch


class PerAttributeMemoryBank:
    def __init__(self, num_attributes, d_h, max_memory=300):
        self.num_attributes = num_attributes
        self.d_h = d_h
        self.max_memory = max_memory
        self.bank = {i: [] for i in range(num_attributes)}

    def add(self, attr_id, embedding, modality, value=None):
        item = {"embedding": embedding.detach().cpu(), "modality": modality, "value": value}
        self.bank[attr_id].append(item)
        if len(self.bank[attr_id]) > self.max_memory:
            self.bank[attr_id].pop(0)

    def get(self, attr_id, return_values=False):
        items = self.bank.get(attr_id, [])
        if not items:
            return (None, None) if return_values else None
        embeddings = torch.stack([x["embedding"] for x in items], dim=0)
        if not return_values:
            return embeddings
        values = [x.get("value") for x in items]
        return embeddings, values
