"""验证 FAISS 索引有效性的脚本
从 CSV 文件读取文本，在已保存的向量索引中检索，验证索引是否正确构建
"""
import os
import sys
import csv
import json
from typing import List, Tuple
from core.config import Config
from index.faiss_store import FaissIndex
from retrieval.embedding_hf import BGEEmbeddingHF


def load_csv_samples(csv_path: str, max_samples: int = 10) -> List[Tuple[int, str]]:
    """从 CSV 加载样本文本用于检索测试"""
    samples = []
    try:
        csv.field_size_limit(min(sys.maxsize, 1_000_000_000))
    except Exception:
        pass
    
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        
        # 选择文本列
        text_col = None
        for candidate in ["text", "content", "article", "body", "paragraph"]:
            if candidate in header:
                text_col = candidate
                break
        if not text_col and header:
            text_col = header[0]
        
        if not text_col:
            print(f"[error] 无法找到文本列在 {csv_path}")
            return []
        
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            text = row.get(text_col, "").strip()
            if text:
                samples.append((i, text))
    
    return samples


def verify_index(cfg: Config, csv_path: str, top_k: int = 5, max_samples: int = 10):
    """验证索引有效性
    
    Args:
        cfg: 配置对象
        csv_path: CSV 文件路径（包含待检索文本）
        top_k: 检索返回的 top-k 结果数
        max_samples: 从 CSV 读取的样本数
    """
    # 加载嵌入模型（使用配置中启用的第一个模型）
    model_name = None
    model = None
    index_path = None
    meta_jsonl_path = None
    
    if cfg.get("indices", "bge_small", "enabled", default=False):
        model_name = "bge_small"
        model_id = cfg.get("embedding_bge_small", "model_id")
        index_path = cfg.get("indices", "bge_small", "index_path")
        meta_jsonl_path = cfg.get("indices", "bge_small", "meta_jsonl_path")
        model = BGEEmbeddingHF(
            model_id=model_id,
            device=cfg.get("embedding_bge_small", "device", default=None),
            max_length=cfg.get("embedding_bge_small", "max_length", default=256),
            dtype=cfg.get("embedding_bge_small", "dtype", default=None),
        )
    elif cfg.get("indices", "bge", "enabled", default=False):
        model_name = "bge"
        model_id = cfg.get("embedding_bge", "model_id")
        index_path = cfg.get("indices", "bge", "index_path")
        model = BGEEmbeddingHF(
            model_id=model_id,
            device=cfg.get("embedding_bge", "device", default=None),
            max_length=cfg.get("embedding_bge", "max_length", default=256),
            dtype=cfg.get("embedding_bge", "dtype", default=None),
        )
    
    if not model or not index_path:
        print("[error] 未找到启用的嵌入模型或索引路径")
        return
    
    print(f"[verify] 使用模型: {model_name}")
    print(f"[verify] 索引路径: {index_path}")
    print(f"[verify] Meta JSONL: {meta_jsonl_path or 'N/A'}")
    
    # 检查索引文件是否存在
    if not os.path.exists(index_path):
        print(f"[error] 索引文件不存在: {index_path}")
        return
    
    # 加载索引
    print(f"[verify] 加载索引...")
    index = FaissIndex(index_type="flat_ip_fp16", save_path=index_path)
    index.load()
    index_size = index.index.ntotal if hasattr(index.index, 'ntotal') else 0
    print(f"[verify] 索引向量数量: {index_size}")
    
    # 加载 Meta JSONL（如果存在）
    chunk_meta = {}
    if meta_jsonl_path and os.path.exists(meta_jsonl_path):
        print(f"[verify] 加载 Meta JSONL...")
        with open(meta_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    chunk_id = obj.get("chunk_id")
                    if chunk_id is not None:
                        chunk_meta[chunk_id] = obj
                except Exception:
                    continue
        print(f"[verify] Meta 记录数: {len(chunk_meta)}")
    
    # 加载 CSV 样本
    print(f"[verify] 从 CSV 加载样本: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"[error] CSV 文件不存在: {csv_path}")
        return
    
    samples = load_csv_samples(csv_path, max_samples)
    if not samples:
        print(f"[error] 未能从 CSV 加载样本")
        return
    
    print(f"[verify] 加载了 {len(samples)} 个样本")
    print("=" * 80)
    
    # 对每个样本进行检索
    for sample_idx, query_text in samples:
        print(f"\n[样本 {sample_idx}] 查询文本 (前100字): {query_text[:100]}...")
        
        # 嵌入查询文本
        query_vec = model.embed([query_text])
        
        # 检索
        distances, indices = index.search(query_vec, top_k)
        
        print(f"[检索结果] Top-{top_k}:")
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if idx == -1:
                continue
            
            # 获取 meta 信息
            meta_info = chunk_meta.get(int(idx), {})
            doc_id = meta_info.get("doc_id", "unknown")
            chunk_text = meta_info.get("text", "")[:80]
            
            print(f"  [{rank}] 距离={dist:.4f} | chunk_id={idx} | doc_id={doc_id}")
            if chunk_text:
                print(f"      文本片段: {chunk_text}...")
        
        print("-" * 80)
    
    print("\n[verify] 验证完成！")
    print(f"[总结] 索引包含 {index_size} 个向量，检索了 {len(samples)} 个样本")


def main():
    """主函数"""
    cfg = Config()
    
    # 默认使用 data/raw/articles/0.csv
    csv_path = "data/raw/articles/0.csv/0.csv"
    
    # 支持命令行参数指定 CSV 路径
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    print(f"[verify] CSV 路径: {csv_path}")
    
    verify_index(
        cfg=cfg,
        csv_path=csv_path,
        top_k=5,
        max_samples=10
    )


if __name__ == "__main__":
    main()
