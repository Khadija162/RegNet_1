import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeMemoryFusion(nn.Module):
    def __init__(self, d_h, memory_bank):
        super().__init__()
        self.memory_bank = memory_bank
        self.W_g = nn.Linear(d_h, d_h)
        self.U_g = nn.Linear(d_h, d_h)
        self.b_g = nn.Parameter(torch.zeros(d_h))

    def forward(self, h_tilde):
        u = torch.zeros_like(h_tilde)
        for i in range(h_tilde.size(0)):
            mem = self.memory_bank.get(i)
            if mem is None or mem.size(0) == 0:
                u[i] = h_tilde[i]
                continue
            mem = mem.to(h_tilde.device)
            scores = torch.matmul(mem, h_tilde[i])
            attn = F.softmax(scores, dim=0)
            r_i = torch.sum(attn[:, None] * mem, dim=0)
            g_i = torch.sigmoid(self.W_g(h_tilde[i]) + self.U_g(r_i) + self.b_g)
            u[i] = g_i * r_i + (1 - g_i) * h_tilde[i]
        return u
