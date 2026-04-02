import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalGNNLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.W_g = nn.Linear(d, d, bias=False)
        self.a = nn.Parameter(torch.randn(2 * d))
        self.W_c = nn.Linear(d, d, bias=False)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        m, d = z.size()
        wg = self.W_g(z)
        alpha = torch.zeros(m, m, device=z.device)
        for idx in range(edge_index.size(1)):
            i, j = edge_index[0, idx], edge_index[1, idx]
            alpha[i, j] = F.leaky_relu(torch.dot(self.a, torch.cat([wg[i], wg[j]], dim=-1)))
        for i in range(m):
            nbrs = edge_index[1][edge_index[0] == i]
            if len(nbrs) > 0:
                alpha[i, nbrs] = F.softmax(alpha[i, nbrs], dim=0)
        local = torch.matmul(alpha, wg)
        q = self.W_q(z)
        k = self.W_k(z)
        beta = F.softmax(torch.matmul(q, k.T) / (d ** 0.5), dim=-1)
        global_msg = torch.matmul(beta, self.W_c(z))
        return F.leaky_relu(local + global_msg + z)
