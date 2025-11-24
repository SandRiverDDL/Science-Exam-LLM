"""将BM25索引从pkl格式转换为parquet格式

支持从现有的pkl索引转换为parquet格式，避免重新构建
"""
import sys
from pathlib import Path
import pickle
import pandas as pd

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))


def convert_pkl_to_parquet():
    """将pkl格式的BM25索引转换为parquet格式"""
    
    print("\n" + "="*80)
    print("BM25索引格式转换 (pkl -> parquet)")
    print("="*80)
    
    # 文件路径
    pkl_path = project_root / 'data' / 'processed' / 'bm25_index.pkl'
    parquet_dir = project_root / 'data' / 'processed' / 'bm25_index'
    
    # 检查pkl文件是否存在
    if not pkl_path.exists():
        print(f"\n错误: pkl文件不存在: {pkl_path}")
        print("请先运行 build_bm25_index.py 构建索引")
        return False
    
    print(f"\n[1/4] 加载pkl索引: {pkl_path}")
    try:
        with open(pkl_path, 'rb') as f:
            index_data = pickle.load(f)
        print("  加载完成")
    except Exception as e:
        print(f"  错误: {e}")
        return False
    
    # 创建输出目录
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: 保存tokenized_chunks
    print(f"\n[2/4] 保存tokenized_chunks...")
    try:
        tokenized_chunks = index_data['tokenized_chunks']
        # 将列表列表转换为字符串（用空格分隔）
        chunks_data = {
            'chunk_id': index_data['chunk_ids'],
            'tokens': [' '.join(tokens) for tokens in tokenized_chunks]
        }
        chunks_df = pd.DataFrame(chunks_data)
        chunks_df.to_parquet(
            parquet_dir / 'tokenized_chunks.parquet',
            index=False,
            compression='snappy'
        )
        print(f"  保存完成: {len(chunks_df)} 条记录")
    except Exception as e:
        print(f"  错误: {e}")
        return False
    
    # Step 3: 保存chunk_len, avgdl, idf_table
    print(f"\n[3/4] 保存chunk_len和IDF权重...")
    try:
        # chunk_len和avgdl
        chunk_len_data = {
            'chunk_id': index_data['chunk_ids'],
            'chunk_len': index_data['chunk_len']
        }
        chunk_len_df = pd.DataFrame(chunk_len_data)
        chunk_len_df.to_parquet(
            parquet_dir / 'chunk_len.parquet',
            index=False,
            compression='snappy'
        )
        print(f"  chunk_len保存完成: {len(chunk_len_df)} 条记录")
        
        # IDF表
        idf_data = {
            'token': list(index_data['idf_table'].keys()),
            'idf': list(index_data['idf_table'].values())
        }
        idf_df = pd.DataFrame(idf_data)
        idf_df.to_parquet(
            parquet_dir / 'idf_table.parquet',
            index=False,
            compression='snappy'
        )
        print(f"  idf_table保存完成: {len(idf_df)} 个词")
    except Exception as e:
        print(f"  错误: {e}")
        return False
    
    # Step 4: 保存倒排索引
    print(f"\n[4/4] 保存倒排索引...")
    try:
        # 倒排索引：token -> [doc_idx1, doc_idx2, ...]
        inverted_data = []
        for token, doc_indices in index_data['inverted_index'].items():
            inverted_data.append({
                'token': token,
                'doc_indices': ','.join(map(str, doc_indices))  # 以逗号分隔的字符串存储
            })
        inverted_df = pd.DataFrame(inverted_data)
        inverted_df.to_parquet(
            parquet_dir / 'inverted_index.parquet',
            index=False,
            compression='snappy'
        )
        print(f"  inverted_index保存完成: {len(inverted_df)} 个词")
    except Exception as e:
        print(f"  错误: {e}")
        return False
    
    # 保存元数据
    print(f"\n保存元数据...")
    try:
        metadata = {
            'avgdl': [index_data['avgdl']],
            'k1': [index_data['k1']],
            'b': [index_data['b']],
            'num_docs': [len(index_data['chunk_ids'])]
        }
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_parquet(
            parquet_dir / 'metadata.parquet',
            index=False,
            compression='snappy'
        )
        print(f"  元数据保存完成")
    except Exception as e:
        print(f"  错误: {e}")
        return False
    
    print("\n" + "="*80)
    print("格式转换完成")
    print("="*80)
    print(f"\n输出目录: {parquet_dir}")
    print(f"文件列表:")
    for file in sorted(parquet_dir.glob('*.parquet')):
        size_mb = file.stat().st_size / 1024 / 1024
        print(f"  - {file.name}: {size_mb:.2f} MB")
    
    # 计算总大小
    total_size = sum(f.stat().st_size for f in parquet_dir.glob('*.parquet'))
    print(f"\n总大小: {total_size / 1024 / 1024:.2f} MB")
    
    print("\n提示: 可以删除旧的pkl文件")
    print(f"  rm {pkl_path}")
    print()
    
    return True


if __name__ == "__main__":
    success = convert_pkl_to_parquet()
    sys.exit(0 if success else 1)
