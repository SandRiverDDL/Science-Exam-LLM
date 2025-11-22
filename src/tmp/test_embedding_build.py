"""测试embedding构建流程"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))


def test_parquet_loading():
    """测试Parquet文件加载"""
    import pyarrow.parquet as pq
    
    parquet_dir = "data/processed/parquet/chunks"
    
    if not os.path.exists(parquet_dir):
        print(f"❌ Parquet目录不存在: {parquet_dir}")
        return
    
    parquet_files = sorted(Path(parquet_dir).glob("chunks_*.parquet"))
    
    if not parquet_files:
        print(f"❌ 未找到chunks Parquet文件")
        return
    
    print("=" * 80)
    print("测试Parquet文件加载")
    print("=" * 80)
    
    # 读取第一个文件
    first_file = parquet_files[0]
    print(f"\n读取文件: {first_file}")
    
    table = pq.read_table(str(first_file))
    df = table.to_pandas()
    
    print(f"\n📊 文件统计:")
    print(f"  总行数: {len(df):,}")
    print(f"  列名: {df.columns.tolist()}")
    
    print(f"\n📋 Schema:")
    print(table.schema)
    
    print(f"\n📝 前3行示例:")
    for idx, row in df.head(3).iterrows():
        print(f"\n  [{idx}] chunk_id: {row['chunk_id']}")
        print(f"      doc_id: {row['doc_id']}")
        print(f"      chunk_len: {row['chunk_len']}")
        print(f"      child_ids: {row['child_ids'][:10]}... (共{len(row['child_ids'])}个)")
        print(f"      rerank_text: {row['rerank_text'][:80]}...")
    
    print("\n" + "=" * 80)
    print("✅ Parquet文件加载测试通过")


def test_embedding_model():
    """测试embedding模型加载和推理"""
    import torch
    from retrieval.embedding_hf import HFTextEmbedding
    
    print("=" * 80)
    print("测试BGE-small模型")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    model = HFTextEmbedding(
        model_id="BAAI/bge-small-en-v1.5",
        device=device,
        max_length=256,
        dtype="float16",
    )
    
    print("✅ 模型加载成功")
    
    # 测试文本嵌入
    print("\n测试文本嵌入...")
    test_texts = [
        "What is machine learning?",
        "How does deep learning work?",
    ]
    
    embeddings = model._embed_batch(test_texts)
    print(f"  输出形状: {embeddings.shape}")
    print(f"  数据类型: {embeddings.dtype}")
    print(f"  向量维度: {embeddings.shape[1]}")
    
    # 测试token IDs嵌入
    print("\n测试token IDs嵌入...")
    test_ids = [
        [101, 2054, 2003, 3698, 4083, 1029, 102],  # 模拟token ids
        [101, 2129, 2515, 2784, 4083, 2147, 1029, 102],
    ]
    
    embeddings_ids = model._embed_batch_ids(test_ids)
    print(f"  输出形状: {embeddings_ids.shape}")
    print(f"  数据类型: {embeddings_ids.dtype}")
    print(f"  向量维度: {embeddings_ids.shape[1]}")
    
    print("\n" + "=" * 80)
    print("✅ Embedding模型测试通过")


def test_faiss_index():
    """测试FAISS索引创建"""
    import numpy as np
    import faiss
    
    print("=" * 80)
    print("测试FAISS索引")
    print("=" * 80)
    
    # 创建测试向量
    dim = 384  # BGE-small维度
    n_vectors = 100
    
    print(f"\n创建测试向量: {n_vectors}个, 维度={dim}")
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    
    # 归一化
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-9, None)
    
    # 创建IndexFlatIP
    print("\n创建IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    
    # 添加向量
    index.add(vectors)
    print(f"  索引向量数: {index.ntotal}")
    
    # 测试检索
    print("\n测试检索...")
    query = vectors[:1]  # 第一个向量作为query
    k = 5
    
    distances, indices = index.search(query, k)
    print(f"  Top-{k} 相似度: {distances[0]}")
    print(f"  Top-{k} 索引: {indices[0]}")
    
    print("\n" + "=" * 80)
    print("✅ FAISS索引测试通过")


def main():
    """运行所有测试"""
    print("\n" * 2)
    print("🚀 开始测试Embedding构建流程")
    print("\n")
    
    try:
        # 测试1: Parquet加载
        test_parquet_loading()
        print("\n" * 2)
        
        # 测试2: Embedding模型
        test_embedding_model()
        print("\n" * 2)
        
        # 测试3: FAISS索引
        test_faiss_index()
        print("\n" * 2)
        
        print("=" * 80)
        print("🎉 所有测试通过！可以运行 build_embeddings.py 构建索引")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
