import torch
import torch.nn as nn


class TextualDecoder(nn.Module):
    def __init__(self, d_h, vocab_size, max_len=64, decoder_dim=384, nhead=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, decoder_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, decoder_dim))
        self.fc_in = nn.Linear(d_h, decoder_dim)
        layer = nn.TransformerDecoderLayer(d_model=decoder_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(decoder_dim, vocab_size)
        self.max_len = max_len

    def forward(self, u, target):
        if u.dim() == 1:
            u = u.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        tgt_emb = self.embedding(target) + self.pos_embedding[:, :target.size(1), :]
        memory = self.fc_in(u).unsqueeze(1).expand(-1, tgt_emb.size(1), -1)
        out = self.transformer_decoder(tgt=tgt_emb, memory=memory)
        return self.out_proj(out)
