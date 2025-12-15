import re
from typing import List, Dict

# 常用 SMILES token 正则：优先匹配括号原子、Cl/Br 等双字符元素、环编号、手性、键类型等
SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]"          # bracket atom, e.g. [nH+], [C@@H]
    r"|Br|Cl|Si|Se|Na|Ca|Li|Mg|Al|Sn|Ag|Zn|As|Fe|Cu|Mn|Hg|Pb|Bi|Pt|Au|Co|Ni"
    r"|@@?"                # chirality
    r"|%\d{2}"             # ring numbers >= 10
    r"|\d"                 # ring numbers 0-9
    r"|==|=|#|-|\+"        # bonds/charges
    r"|\\|/"               # bond direction
    r"|\(|\)"              # branches
    r"|\."                 # disconnected
    r"|:|~|@|\*|\$"        # other
    r"|[A-Za-z])"          # atoms/aromatic
)

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def tokenize_smiles(smiles: str) -> List[str]:
    # Tokenize SMILES into a list of tokens.
    tokens = SMILES_TOKEN_PATTERN.findall(smiles.strip())
    if not tokens:
        return []
    return tokens

def build_vocab(tokenized: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    freq = {}
    for seq in tokenized:
        for t in seq:
            freq[t] = freq.get(t, 0) + 1
    vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    for t, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and t not in vocab:
            vocab[t] = len(vocab)
    return vocab

def encode(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab["<unk>"]
    return [vocab.get(t, unk) for t in tokens]

def decode(ids: List[int], inv_vocab: Dict[int, str]) -> List[str]:
    return [inv_vocab[i] for i in ids]
