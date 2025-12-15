import torch
import torch.nn as nn

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hid_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.2, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out = nn.Linear(hid_dim, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.rnn(e)          # (B, T, H)
        return self.out(h)          # (B, T, V)

class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, nhead: int = 8,
                 ff_dim: int = 1024, num_layers: int = 6, dropout: float = 0.1,
                 max_len: int = 256, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, emb_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len={self.max_len}.")
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        key_padding_mask = (x == self.pad_id)

        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.out(h)
