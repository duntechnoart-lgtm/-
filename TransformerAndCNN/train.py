import argparse
import json
import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import RNNLanguageModel, CausalTransformerLM

class NpzDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.x = data["x"].astype(np.int64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]

def loss_fn(logits, targets, pad_id: int):
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(B*T, V),
        targets.reshape(B*T),
        ignore_index=pad_id
    )

@torch.no_grad()
def evaluate(model, loader, device, pad_id: int):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x in loader:
        x = x.to(device)
        inp = x[:, :-1]
        tgt = x[:, 1:]
        logits = model(inp)
        loss = loss_fn(logits, tgt, pad_id)
        tokens = (tgt != pad_id).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    avg = total_loss / max(1, total_tokens)
    ppl = math.exp(avg) if avg < 20 else float("inf")
    return avg, ppl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts")
    ap.add_argument("--model", choices=["rnn", "transformer"], default="rnn")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hid_dim", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--ff_dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    pad_id = vocab["<pad>"]
    vocab_size = len(vocab)

    train_ds = NpzDataset(os.path.join(args.data_dir, "train.npz"))
    valid_ds = NpzDataset(os.path.join(args.data_dir, "valid.npz"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "rnn":
        model = RNNLanguageModel(
            vocab_size=vocab_size, emb_dim=args.emb_dim, hid_dim=args.hid_dim,
            num_layers=max(1, min(4, args.layers)), dropout=args.dropout, pad_id=pad_id
        )
    else:
        model = CausalTransformerLM(
            vocab_size=vocab_size, emb_dim=args.emb_dim, nhead=args.nhead, ff_dim=args.ff_dim,
            num_layers=args.layers, dropout=args.dropout, max_len=args.max_len, pad_id=pad_id
        )

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_valid = float("inf")
    best_path = os.path.join(args.ckpt_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x in pbar:
            x = x.to(device)
            inp = x[:, :-1]
            tgt = x[:, 1:]
            logits = model(inp)
            loss = loss_fn(logits, tgt, pad_id)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        v_loss, v_ppl = evaluate(model, valid_loader, device, pad_id)
        print(f"[Valid] loss/token={v_loss:.4f} ppl={v_ppl:.2f}")

        if v_loss < best_valid:
            best_valid = v_loss
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "vocab": vocab},
                best_path
            )
            print("Saved best checkpoint:", best_path)

    print("Training finished. Best valid loss/token:", best_valid)

if __name__ == "__main__":
    main()
