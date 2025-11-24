"""构建BM25索引

从chunks.parquet和documents_cleaned.parquet构建BM25索引并保存到data/processed/
"""
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from retrieval.bm25 import BM25Retriever
from transformers import AutoTokenizer


def main():
    """构建BM25索引"""
    
    print("\n" + "="*80)
    print("BM25索引构建")
    print("="*80)
    
    # 文件路径
    chunks_parquet = project_root / 'data' / 'processed' / 'chunks.parquet'
    docs_parquet = project_root / 'data' / 'processed' / 'documents_cleaned.parquet'
    bm25_index_path = project_root / 'data' / 'processed' / 'bm25_index.pkl'
    
    # 检查文件是否存在
    if not chunks_parquet.exists():
        print(f"错误: chunks.parquet不存在: {chunks_parquet}")
        return
    
    if not docs_parquet.exists():
        print(f"错误: documents_cleaned.parquet不存在: {docs_parquet}")
        return
    
    # 加载tokenizer
    print("\n[1/2] 加载tokenizer...")
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"  Tokenizer加载完成: {model_id}")
    
    # 构建BM25索引
    print("\n[2/2] 构建BM25索引...")
    bm25 = BM25Retriever(
        chunks_parquet=str(chunks_parquet),
        docs_parquet=str(docs_parquet),
        tokenizer=tokenizer
    )
    
    # 保存索引
    print(f"\n保存索引到: {bm25_index_path}")
    bm25.save(str(bm25_index_path))
    
    print("\n" + "="*80)
    print("BM25索引构建完成")
    print("="*80)
    print(f"\n索引文件: {bm25_index_path}")
    print(f"文件大小: {bm25_index_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()


if __name__ == "__main__":
    main()
