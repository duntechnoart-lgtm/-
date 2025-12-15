import argparse
import os

from rdkit import Chem
from rdkit.Chem import Draw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles_file", required=True)
    ap.add_argument("--out_png", default="mols_grid.png")
    ap.add_argument("--max_mols", type=int, default=64)
    ap.add_argument("--mol_per_row", type=int, default=8)
    ap.add_argument("--sub_img_size", type=int, default=200)
    args = ap.parse_args()

    with open(args.smiles_file, "r", encoding="utf-8") as f:
        smiles = [line.strip() for line in f if line.strip()]

    mols, legends = [], []
    for s in smiles[:args.max_mols]:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
            legends.append(s)

    if not mols:
        raise RuntimeError("No valid molecules to draw.")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=args.mol_per_row,
        subImgSize=(args.sub_img_size, args.sub_img_size),
        legends=legends
    )
    out_dir = os.path.dirname(args.out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    img.save(args.out_png)
    print("Saved:", args.out_png)

if __name__ == "__main__":
    main()
