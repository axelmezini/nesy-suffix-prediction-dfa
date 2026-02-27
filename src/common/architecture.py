import os
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        logits = self.output_layer(output)
        return logits, (hn, cn)

    def forward_from_state(self, x, state):
        output, (hn, cn) = self.lstm(x, state)
        logits = self.output_layer(output)
        return logits, (hn, cn)

    def export(self, folder, run_id, model):
        torch.save(self.state_dict(), os.path.join(folder, f'r{run_id}_lstm_{model}.pt'))
        del self
        torch.cuda.empty_cache()


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len

        self.in_proj = nn.Linear(vocab_size, d_model) #bias = False
        self.embedding = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x):
        bsz, seq_len, v = x.size()
        if v != self.vocab_size:
            raise ValueError(f"Expected vocab_size={self.vocab_size}, got {v}")
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")

        h = self.in_proj(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = h + self.embedding(pos)

        mask = self._causal_mask(seq_len, x.device)
        h = self.transformer_encoder(h, mask=mask)
        h = self.norm(h)

        logits = self.out_proj(h)
        return logits, x

    def forward_from_state(self, x, state):
        if state is None:
            new_state = x
        else:
            new_state = torch.cat((state, x), dim=1)
        logits, _ = self.forward(new_state)
        return logits, new_state

    def export(self, folder, run_id, model):
        torch.save(self.state_dict(), os.path.join(folder, f'r{run_id}_transformer_{model}.pt'))
        del self
        torch.cuda.empty_cache()