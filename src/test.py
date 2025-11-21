import os
import glob
from typing import List, Tuple
from core.config import Config
def load_corpus(corpus_dir: str) -> Tuple[List[str], List[List[str]]]:
    print(corpus_dir)
    docs = []
    tokens = []
    for fp in glob.glob(os.path.join(corpus_dir, "*.txt")):
        print(fp)
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            t = f.read().strip()
        docs.append(t)
        # 简易分词（后续可替换为 jieba/nltk）
        tokens.append(t.split())
    return docs, tokens

if __name__ == "__main__":
    cfg = Config()
    # corpus_dir = cfg.data["corpus_dir"]
    # docs, tokens = load_corpus(corpus_dir)
    # print(len(docs))
    # print(len(tokens))
    train_csv = cfg.get("kaggle", "train_csv")
    out_csv = cfg.get("kaggle", "output_submission")
    print(train_csv)
    print(out_csv)