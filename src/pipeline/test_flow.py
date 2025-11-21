import os
import csv
from core.config import Config
from index.faiss_store import FaissIndex
from retrieval.embedding_jina_gguf import JinaEmbeddingGGUF


def load_csv_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def split_train_val(rows, val_ratio=0.1):
    n = len(rows)
    k = max(1, int(n * val_ratio))
    return rows[k:], rows[:k]


def main():
    cfg = Config()

    train_csv = cfg.get("kaggle", "train_csv")
    test_csv = train_csv.replace("train.csv", "test.csv")

    train_rows = load_csv_rows(train_csv)
    test_rows = load_csv_rows(test_csv)
    train_rows, val_rows = split_train_val(train_rows)

    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    index_path = cfg.get("indices", "jina", "index_path")
    meta_path = cfg.get("indices", "jina", "meta_path")
    assert os.path.exists(index_path), "索引不存在，请先运行 build_chunks_and_indices.py"
    assert os.path.exists(meta_path), "元数据不存在，请先运行 build_chunks_and_indices.py"

    emb = JinaEmbeddingGGUF(
        repo_id=cfg.embedding["repo_id"],
        filename=cfg.embedding["filename"],
        n_ctx=cfg.embedding.get("n_ctx", 8192),
        n_threads=cfg.embedding.get("n_threads", 8),
        n_batch=cfg.embedding.get("n_batch", 32),
    )

    index = FaissIndex(index_type="flat_ip_fp16", save_path=index_path)
    index.load()

    sample_q = val_rows[0]["prompt"] if val_rows else train_rows[0]["prompt"]
    q_vec = emb.embed([sample_q])
    I, D = index.search(q_vec, top_k=5)
    print("Search top-5 IDs:", I[0].tolist())
    print("Scores:", [float(x) for x in D[0].tolist()])

    print("Flow test completed: CSV读取、索引加载、检索正常")


if __name__ == "__main__":
    main()