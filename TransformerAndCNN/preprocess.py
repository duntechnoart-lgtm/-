import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from tokenizer import tokenize_smiles, build_vocab, encode

def read_smiles(path: str) -> List[str]:
    path_low = path.lower()
    if path_low.endswith((".csv", ".tsv")):
        sep = "," if path_low.endswith(".csv") else "\t"
        df = pd.read_csv(path, sep=sep)
        candidate_cols = [c for c in df.columns if c.lower() in ("smiles", "smile", "canonical_smiles", "canon_smiles")]
        if not candidate_cols:
            candidate_cols = [c for c in df.columns if "smile" in c.lower()]
        col = candidate_cols[0] if candidate_cols else df.columns[0]
        smiles = df[col].astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            smiles = [line.strip() for line in f if line.strip()]
    return smiles

def canonicalize(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def split_train_valid(smiles: List[str], valid_ratio: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(smiles))
    rng.shuffle(idx)
    n_valid = int(len(smiles) * valid_ratio)
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    train = [smiles[i] for i in train_idx]
    valid = [smiles[i] for i in valid_idx]
    return train, valid

def pad_sequences(seqs: List[List[int]], pad_id: int, max_len: int) -> np.ndarray:
    arr = np.full((len(seqs), max_len), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        arr[i, :len(s)] = np.asarray(s, dtype=np.int64)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw = read_smiles(args.input)
    raw = [s.strip() for s in raw if s]
    print(f"Loaded raw SMILES: {len(raw)}")

    canon = []
    for s in tqdm(raw, desc="Canonicalize"):
        cs = canonicalize(s)
        if cs is not None:
            canon.append(cs)
    canon = list(dict.fromkeys(canon))
    print(f"Valid & canonical unique SMILES: {len(canon)}")

    tokenized = []
    kept_smiles = []
    for s in tqdm(canon, desc="Tokenize"):
        toks = tokenize_smiles(s)
        if not toks:
            continue
        if len(toks) + 2 <= args.max_len:
            tokenized.append(toks)
            kept_smiles.append(s)

    print(f"Kept (<= max_len): {len(kept_smiles)}")

    vocab = build_vocab(tokenized, min_freq=args.min_freq)
    with open(os.path.join(args.outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    pad_id = vocab["<pad>"]
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]

    encoded = []
    for toks in tokenized:
        ids = [bos_id] + encode(toks, vocab) + [eos_id]
        encoded.append(ids)

    train_sm, valid_sm = split_train_valid(kept_smiles, valid_ratio=args.valid_ratio, seed=args.seed)
    sm_to_ids = {s: ids for s, ids in zip(kept_smiles, encoded)}
    train_ids = [sm_to_ids[s] for s in train_sm]
    valid_ids = [sm_to_ids[s] for s in valid_sm]

    train_arr = pad_sequences(train_ids, pad_id, args.max_len)
    valid_arr = pad_sequences(valid_ids, pad_id, args.max_len)

    np.savez_compressed(os.path.join(args.outdir, "train.npz"), x=train_arr)
    np.savez_compressed(os.path.join(args.outdir, "valid.npz"), x=valid_arr)

    with open(os.path.join(args.outdir, "train_smiles.txt"), "w", encoding="utf-8") as f:
        for s in train_sm:
            f.write(s + "\n")

    print("Done. Saved artifacts to:", args.outdir)

if __name__ == "__main__":
    main()
