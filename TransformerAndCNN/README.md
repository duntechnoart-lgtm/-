# 任务C：生成式化学结构设计（SMILES 生成）项目模板

本项目提供一个“从零跑通”的 SMILES 生成管线：
- 数据清洗/规范化（RDKit）
- SMILES 分词（token-level，比逐字符更稳健）
- 基线模型：GRU 语言模型（next-token prediction）
- 进阶模型：Causal Transformer（GPT-style，from scratch）
- 采样生成 + 评估指标（有效性/唯一性/新颖性/多样性/性质分布）
- 分子结构图批量输出（PNG）

## 0. 安装
建议新建虚拟环境后安装依赖：
```bash
pip install -r requirements.txt
```

## 1. 数据准备
将 Dataset3 放在任意位置，支持：
- `.csv/.tsv`：包含列名 `smiles`（推荐）；若没有，会尝试自动寻找类似列名
- `.txt`：每行一个 SMILES

## 2. 预处理（清洗 + 分词 + 构建词表）
```bash
python preprocess.py --input data/Dataset3.csv --outdir artifacts --max_len 128
```

输出：
- `artifacts/train.npz` / `artifacts/valid.npz`
- `artifacts/vocab.json`
- `artifacts/train_smiles.txt`（用于 novelty 判定）

## 3. 训练（基线 GRU）
```bash
python train.py --data_dir artifacts --model rnn --epochs 10 --batch_size 256 --lr 3e-4
```

## 4. 训练（进阶 Transformer）
```bash
python train.py --data_dir artifacts --model transformer --epochs 10 --batch_size 128 --lr 3e-4
```

## 5. 采样生成 + 指标评估 + 出图
```bash
python sample_eval.py --data_dir artifacts --ckpt checkpoints/best.pt --num_samples 2000 --temperature 1.0 --topk 50
python draw_samples.py --smiles_file outputs/generated_valid_unique.txt --out_png outputs/mols_grid.png --max_mols 64
```

## 6. 指标说明
- Validity：生成 SMILES 里能被 RDKit 解析并通过 sanitize 的比例
- Uniqueness：有效 SMILES 去重后的比例
- Novelty：有效且唯一 SMILES 中，不在训练集 canonical SMILES 集合里的比例
- Diversity：基于 Morgan 指纹的平均 Tanimoto 距离（可选）

提示：若你希望“天然保证有效性”，可将表示改成 SELFIES（需要额外依赖 `selfies`），再转回 SMILES 评估。
