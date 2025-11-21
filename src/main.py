import os
import glob
import json
from typing import List, Tuple

from core.config import Config
from retrieval.embedding_jina_gguf import JinaEmbeddingGGUF
from retrieval.bm25 import BM25Retrieval
from retrieval.fusion import rrf_fusion
from index.faiss_store import FaissIndex
from rerank.jina_reranker_gguf import JinaRerankerGGUF
from model.qwen_zero_shot import QwenZeroShotClassifier

def load_corpus(corpus_dir: str) -> Tuple[List[str], List[List[str]]]:
    docs = []
    tokens = []
    for fp in glob.glob(os.path.join(corpus_dir, "*.txt")):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            t = f.read().strip()
        docs.append(t)
        # 简易分词（后续可替换为 jieba/nltk）
        tokens.append(t.split())
    return docs, tokens

def build_index(cfg: Config, docs: List[str]) -> FaissIndex:
    emb = JinaEmbeddingGGUF(
        repo_id=cfg.embedding["repo_id"],
        filename=cfg.embedding["filename"],
        n_ctx=cfg.embedding.get("n_ctx", 8192),
        n_threads=cfg.embedding.get("n_threads", 8),
        n_batch=cfg.embedding.get("n_batch", 32),
    )
    vectors = emb.embed(docs)
    index = FaissIndex(index_type=cfg.index["type"], save_path=cfg.index["save_path"], dim=cfg.index["dim"])
    index.build(vectors)
    index.save()
    return index

def search_pipeline(cfg: Config, query: str, docs: List[str], bm25_tokens: List[List[str]]):
    # 1) embedding 检索
    emb = JinaEmbeddingGGUF(
        repo_id=cfg.embedding["repo_id"],
        filename=cfg.embedding["filename"],
        n_ctx=cfg.embedding.get("n_ctx", 8192),
        n_threads=cfg.embedding.get("n_threads", 8),
        n_batch=cfg.embedding.get("n_batch", 32),
    )
    index = FaissIndex(index_type=cfg.index["type"], save_path=cfg.index["save_path"], dim=cfg.index["dim"])
    index.load()
    q_vec = emb.embed([query])
    I, D = index.search(q_vec, cfg.retrieval["top_k_embedding"])
    emb_rank = [(int(i), float(D[0][rank])) for rank, i in enumerate(I[0])]

    # 2) BM25（辅助）
    bm25 = BM25Retrieval(bm25_tokens)
    bm25_rank = bm25.search(query.split(), cfg.retrieval["bm25_top_k"])

    # 3) Rank Fusion (RRF)
    fused = rrf_fusion({
        "embedding": emb_rank,
        "bm25": bm25_rank
    }, k=cfg.retrieval["fusion"]["rrf_k"], top_k=max(cfg.retrieval["top_k_embedding"], cfg.retrieval["bm25_top_k"]))

    # 4) Reranker（可选，提示式打分）
    top_docs = [docs[doc_id] for doc_id, _ in fused]
    if cfg.reranker["enabled"]:
        rer = JinaRerankerGGUF(repo_id=cfg.reranker["repo_id"], filename=cfg.reranker["filename"])
        rer_scores = rer.score(query, top_docs)
        reranked = [top_docs[i] for i, _ in rer_scores[:cfg.reranker["max_docs"]]]
    else:
        reranked = top_docs[:cfg.reranker["max_docs"]]

    return reranked

def run_qwen_zero_shot(cfg: Config, question: str, answer: str):
    clf = QwenZeroShotClassifier(
        model_id=cfg.qwen["model_id"],
        device_map=cfg.qwen.get("device_map", "auto"),
        trust_remote_code=cfg.qwen.get("trust_remote_code", True)
    )
    # 1) 文本输出校验
    clf.test_text_output()
    # 2) Zero-shot 二分类（仅输出 0/1）
    y = clf.classify(question, answer, use_kv_cache_prefix=cfg.zero_shot["use_kv_cache_prefix"])
    print(f"Zero-shot classify => {y}")
    return y

if __name__ == "__main__":
    cfg = Config()

    # corpus_dir = cfg.data["corpus_dir"]
    # docs, bm25_tokens = load_corpus(corpus_dir)

    # # 首次构建索引（如已构建，可跳过）
    # if not os.path.exists(cfg.index["save_path"]):
    #     print("Building FAISS index...")
    #     build_index(cfg, docs)
    #     print("Index built and saved.")

    # # 简单检索演示
    # demo_query = "细胞膜的主要组成成分是什么？"
    # evidence = search_pipeline(cfg, demo_query, docs, bm25_tokens)
    # print(f"Top Evidence (after fusion & rerank): {len(evidence)}")
    # print(evidence[0][:200] if evidence else "No evidence.")

    # Zero-shot 基线（问题相同，回答变化）
    question = "水在标准气压下的沸点是几摄氏度？"
    answer = "100°C"  # 正确答案示例
    run_qwen_zero_shot(cfg, question, answer)