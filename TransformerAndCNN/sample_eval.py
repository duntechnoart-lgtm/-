import argparse
import json
import os
from typing import List, Tuple, Set

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, DataStructs
from tqdm import tqdm

from models import RNNLanguageModel, CausalTransformerLM

def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    vocab = ckpt["vocab"]
    inv_vocab = {i: t for t, i in vocab.items()}
    args = ckpt["args"]
    return ckpt["model"], vocab, inv_vocab, args

def build_model(model_type: str, vocab_size: int, pad_id: int, args: dict):
    if model_type == "rnn":
        return RNNLanguageModel(
            vocab_size=vocab_size, emb_dim=args.get("emb_dim", 256),
            hid_dim=args.get("hid_dim", 512),
            num_layers=max(1, min(4, args.get("layers", 2))),
            dropout=args.get("dropout", 0.1), pad_id=pad_id
        )
    return CausalTransformerLM(
        vocab_size=vocab_size, emb_dim=args.get("emb_dim", 256),
        nhead=args.get("nhead", 8), ff_dim=args.get("ff_dim", 1024),
        num_layers=args.get("layers", 6), dropout=args.get("dropout", 0.1),
        max_len=args.get("max_len", 128), pad_id=pad_id
    )

@torch.no_grad()
def sample_one(model, bos_id: int, eos_id: int, pad_id: int, inv_vocab: dict,
               max_len: int = 128, temperature: float = 1.0, topk: int = 0, device="cpu") -> str:
    model.eval()
    seq = [bos_id]
    for _ in range(max_len - 1):
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[:, -1, :]
        logits = logits / max(1e-6, temperature)

        if topk and topk > 0:
            v, ix = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)
            probs = torch.softmax(v, dim=-1)
            next_id = ix[0, torch.multinomial(probs[0], 1).item()].item()
        else:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs[0], 1).item()

        if next_id == eos_id:
            break
        if next_id == pad_id:
            continue
        seq.append(next_id)

    tokens = [inv_vocab[i] for i in seq[1:]]
    tokens = [t for t in tokens if t not in ("<bos>", "<eos>", "<pad>")]
    return "".join(tokens)

def is_valid_smiles(smiles: str) -> Tuple[bool, str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, ""
        can = Chem.MolToSmiles(mol, canonical=True)
        return True, can
    except Exception:
        return False, ""

def compute_diversity(smiles_list: List[str], radius=2, nbits=2048, max_pairs=20000) -> float:
    mols = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
    if len(mols) < 2:
        return 0.0
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in mols]
    n = len(fps)
    total_possible = n * (n - 1) // 2
    k = min(max_pairs, total_possible)

    rng = np.random.default_rng(0)
    seen = set()
    while len(seen) < k:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        seen.add((i, j))

    sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j]) for (i, j) in seen]
    return float(1.0 - float(np.mean(sims)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--outdir", default="outputs")
    args_cli = ap.parse_args()

    os.makedirs(args_cli.outdir, exist_ok=True)

    state_dict, vocab, inv_vocab, train_args = load_checkpoint(args_cli.ckpt)
    pad_id = vocab["<pad>"]
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    vocab_size = len(vocab)

    model_type = train_args.get("model", "rnn")
    model = build_model(model_type, vocab_size, pad_id, train_args)
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_sm_path = os.path.join(args_cli.data_dir, "train_smiles.txt")
    train_set: Set[str] = set()
    if os.path.exists(train_sm_path):
        with open(train_sm_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    train_set.add(s)

    generated_raw = []
    generated_can = []
    for _ in tqdm(range(args_cli.num_samples), desc="Sampling"):
        s = sample_one(
            model, bos_id, eos_id, pad_id, inv_vocab,
            max_len=args_cli.max_len, temperature=args_cli.temperature, topk=args_cli.topk, device=device
        )
        generated_raw.append(s)
        ok, can = is_valid_smiles(s)
        if ok:
            generated_can.append(can)

    validity = len(generated_can) / max(1, len(generated_raw))
    unique_valid = list(dict.fromkeys(generated_can))
    uniqueness = len(unique_valid) / max(1, len(generated_can))
    novelty = None
    if train_set:
        novelty = sum(1 for s in unique_valid if s not in train_set) / max(1, len(unique_valid))

    diversity = compute_diversity(unique_valid) if len(unique_valid) >= 2 else 0.0

    with open(os.path.join(args_cli.outdir, "generated_raw.txt"), "w", encoding="utf-8") as f:
        for s in generated_raw:
            f.write(s + "\n")
    with open(os.path.join(args_cli.outdir, "generated_valid_unique.txt"), "w", encoding="utf-8") as f:
        for s in unique_valid:
            f.write(s + "\n")

    metrics = {
        "num_samples": args_cli.num_samples,
        "validity": validity,
        "uniqueness_on_valid": uniqueness,
        "novelty_on_unique_valid": novelty,
        "diversity_est": diversity,
        "num_valid": len(generated_can),
        "num_unique_valid": len(unique_valid),
        "train_set_size": len(train_set),
    }
    with open(os.path.join(args_cli.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # properties
    props = []
    for s in unique_valid:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        props.append({
            "smiles": s,
            "mw": float(Descriptors.MolWt(m)),
            "logp": float(Descriptors.MolLogP(m)),
            "qed": float(QED.qed(m)),
            "hbd": int(Descriptors.NumHDonors(m)),
            "hba": int(Descriptors.NumHAcceptors(m)),
            "tpsa": float(Descriptors.TPSA(m)),
        })
    if props:
        import pandas as pd
        pd.DataFrame(props).to_csv(os.path.join(args_cli.outdir, "properties_unique_valid.csv"), index=False)

    print("Done.")
    print(f"Validity: {validity:.3f}")
    print(f"Uniqueness (on valid): {uniqueness:.3f}")
    if novelty is not None:
        print(f"Novelty (vs train set): {novelty:.3f}")
    print(f"Diversity (1-mean_tanimoto): {diversity:.3f}")

if __name__ == "__main__":
    main()
