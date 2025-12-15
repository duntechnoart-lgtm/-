import argparse
import os
import json
from collections import Counter

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from tokenizer import tokenize_smiles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles_file", required=True, help="每行一个 canonical SMILES（例如 train_smiles.txt）")
    ap.add_argument("--outdir", default="eda_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.smiles_file, "r", encoding="utf-8") as f:
        smiles = [line.strip() for line in f if line.strip()]

    lengths = []
    token_freq = Counter()
    mws = []

    for s in tqdm(smiles, desc="EDA"):
        toks = tokenize_smiles(s)
        lengths.append(len(toks))
        token_freq.update(toks)
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mws.append(Descriptors.MolWt(m))

    plt.figure()
    plt.hist(lengths, bins=50)
    plt.title("Token length distribution")
    plt.xlabel("length")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "length_hist.png"), dpi=200)

    if mws:
        plt.figure()
        plt.hist(mws, bins=50)
        plt.title("Molecular weight distribution")
        plt.xlabel("MolWt")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "mw_hist.png"), dpi=200)

    top = token_freq.most_common(50)
    with open(os.path.join(args.outdir, "top_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(top, f, ensure_ascii=False, indent=2)

    print("Saved EDA outputs to:", args.outdir)

if __name__ == "__main__":
    main()
