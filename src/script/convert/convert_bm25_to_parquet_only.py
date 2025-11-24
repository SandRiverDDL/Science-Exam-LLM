"""BM25索引格式转换脚本 (pkl+parquet → 纯parquet)

将旧的混合格式索引转换为新的纯parquet格式：
- 旧格式：tokenized_chunks.pkl, inverted_index.pkl + chunk_len/idf_table/metadata.parquet
- 新格式：全部改为parquet格式，支持高效部分读取

使用方式：
    python convert_bm25_to_parquet_only.py <source_index_dir> <target_index_dir>

示例：
    python convert_bm25_to_parquet_only.py data/processed/bm25_index data/processed/bm25_index_v2
"""

import sys
import pickle
from pathlib import Path
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def convert_bm25_index(source_dir: str, target_dir: str):
    """转换BM25索引格式
    
    Args:
        source_dir: 源索引目录（混合格式）
        target_dir: 目标索引目录（纯parquet格式）
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.is_dir():
        raise ValueError(f"源目录不存在: {source_path}")
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[转换] 从 {source_path} 转换到 {target_path}")
    print()
    
    # ===== 1. 加载源索引 =====
    print("[1/5] 加载元数据...")
    metadata_df = pd.read_parquet(source_path / 'metadata.parquet')
    avgdl = float(metadata_df.loc[0, 'avgdl'])
    k1 = float(metadata_df.loc[0, 'k1'])
    b = float(metadata_df.loc[0, 'b'])
    num_docs = int(metadata_df.loc[0, 'num_docs'])
    print(f"  avgdl={avgdl:.2f}, k1={k1}, b={b}, num_docs={num_docs}")
    
    # 加载tokenized_chunks（pkl）
    print("[2/5] 加载tokenized_chunks（从pkl）...")
    with open(source_path / 'tokenized_chunks.pkl', 'rb') as f:
        chunks_data = pickle.load(f)
    chunk_ids = chunks_data['chunk_ids']
    tokenized_chunks = chunks_data['tokenized_chunks']
    print(f"  加载了 {len(chunk_ids)} 个chunks")
    
    # 加载chunk_len（parquet）
    print("[3/5] 加载chunk_len...")
    chunk_len_df = pd.read_parquet(source_path / 'chunk_len.parquet')
    chunk_len = chunk_len_df['chunk_len'].tolist()
    
    # 加载idf_table（parquet）
    print("[4/5] 加载idf_table...")
    idf_df = pd.read_parquet(source_path / 'idf_table.parquet')
    idf_table = dict(zip(idf_df['token'], idf_df['idf']))
    print(f"  词汇表大小: {len(idf_table)}")
    
    # 加载inverted_index（pkl）
    print("[5/5] 加载inverted_index（从pkl）...")
    with open(source_path / 'inverted_index.pkl', 'rb') as f:
        inverted_dict = pickle.load(f)
    inverted_index = defaultdict(list, inverted_dict)
    print(f"  倒排索引大小: {len(inverted_index)}")
    
    # ===== 2. 保存到纯parquet格式 =====
    print()
    print("=" * 60)
    print("保存到新的纯parquet格式")
    print("=" * 60)
    
    # 1. 保存tokenized_chunks（parquet）
    print("[1/5] 保存tokenized_chunks.parquet...")
    chunks_data = {
        'chunk_id': chunk_ids,
        'tokens': [' '.join(tokens) for tokens in tokenized_chunks]
    }
    chunks_df = pd.DataFrame(chunks_data)
    chunks_df.to_parquet(
        target_path / 'tokenized_chunks.parquet',
        index=False,
        compression='snappy'
    )
    
    # 2. 保存chunk_len（parquet）
    print("[2/5] 保存chunk_len.parquet...")
    chunk_len_data = {
        'chunk_id': chunk_ids,
        'chunk_len': chunk_len
    }
    chunk_len_df = pd.DataFrame(chunk_len_data)
    chunk_len_df.to_parquet(
        target_path / 'chunk_len.parquet',
        index=False,
        compression='snappy'
    )
    
    # 3. 保存idf_table（parquet）
    print("[3/5] 保存idf_table.parquet...")
    idf_data = {
        'token': list(idf_table.keys()),
        'idf': list(idf_table.values())
    }
    idf_df = pd.DataFrame(idf_data)
    idf_df.to_parquet(
        target_path / 'idf_table.parquet',
        index=False,
        compression='snappy'
    )
    
    # 4. 保存倒排索引（parquet）
    print("[4/5] 保存inverted_index.parquet...")
    inverted_data = []
    for token, doc_indices in inverted_index.items():
        inverted_data.append({
            'token': token,
            'doc_indices': ','.join(map(str, doc_indices))
        })
    inverted_df = pd.DataFrame(inverted_data)
    inverted_df.to_parquet(
        target_path / 'inverted_index.parquet',
        index=False,
        compression='snappy'
    )
    
    # 5. 保存metadata（parquet）
    print("[5/5] 保存metadata.parquet...")
    metadata = {
        'avgdl': [avgdl],
        'k1': [k1],
        'b': [b],
        'num_docs': [num_docs]
    }
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_parquet(
        target_path / 'metadata.parquet',
        index=False,
        compression='snappy'
    )
    
    print()
    print("=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"目标目录: {target_path}")
    print()
    print("文件清单 (全部为parquet格式):")
    print("  ✓ tokenized_chunks.parquet  (支持部分读取)")
    print("  ✓ inverted_index.parquet    (支持按token查询)")
    print("  ✓ chunk_len.parquet         (必须全量读入)")
    print("  ✓ idf_table.parquet         (必须全量读入)")
    print("  ✓ metadata.parquet          (索引元数据)")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    
    try:
        convert_bm25_index(source_dir, target_dir)
    except Exception as e:
        print(f"[错误] {e}")
        sys.exit(1)
