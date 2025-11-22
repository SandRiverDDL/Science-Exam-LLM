"""数据预处理主流程 V2

核心改进：
1. 支持CSV和Parquet格式（自动识别）
2. 使用text_cleaner和title_cleaner进行清洗
3. 使用tiktoken进行tokenization（模型解耦）
4. 基于句子的动态chunking
5. 清洗后的文档保存为单个Parquet文件（几百MB，可以全部载入内存）
6. chunks单独保存为Parquet文件
"""
import os
import sys
import glob
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from core.config import Config
from processing.data_loader import load_files
from processing.text_cleaner import full_text_cleaning, filter_short_text
from processing.title_cleaner import clean_title_conservative, is_good_title
from chunking.sentence_chunker import SentenceChunker


def process_documents(
    data_files: List[str],
    output_dir: str,
    min_text_chars: int = 100,
    target_lang: str = 'en',
    child_chunk_size: int = 128,
    parent_chunk_size: int = 512
) -> Dict[str, int]:
    """处理文档并输出为Parquet
    
    Args:
        data_files: 数据文件路径列表（CSV或Parquet）
        output_dir: 输出目录
        min_text_chars: 最小文本字符数
        target_lang: 目标语言
        child_chunk_size: 子chunk大小（tokens）
        parent_chunk_size: 父chunk大小（tokens）
    
    Returns:
        统计信息字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化chunker（使用tiktoken）
    chunker = SentenceChunker(
        model_name="cl100k_base",  # GPT-3.5/4使用的编码
        child_size=child_chunk_size,
        parent_size=parent_chunk_size,
        min_chunk_chars=50
    )
    
    # 统计信息
    stats = {
        "total_docs": 0,
        "kept_docs": 0,
        "total_chunks": 0,
        "filtered_short": 0,
        "filtered_title": 0,
        "filtered_language": 0,
    }
    
    # 存储清洗后的文档
    doc_records = []
    chunk_records = []
    
    # 处理文档
    print("\n开始处理文档...")
    for doc_id, title_raw, text_raw in load_files(data_files):
        stats["total_docs"] += 1
        
        # 1. 清洗标题
        title_cleaned = ""
        if title_raw:
            if is_good_title(title_raw):
                title_cleaned = clean_title_conservative(title_raw)
            else:
                stats["filtered_title"] += 1
        
        # 2. 清洗正文
        text_cleaned = full_text_cleaning(text_raw, target_lang=target_lang)
        
        if text_cleaned is None or len(text_cleaned) < min_text_chars:
            stats["filtered_short"] += 1
            continue
        
        # 3. 生成文档记录（保存清洗后的文档）
        doc_record = {
            'doc_id': doc_id,
            'title': title_cleaned,
            'text': text_cleaned,
            'text_len': len(text_cleaned),
        }
        doc_records.append(doc_record)
        stats["kept_docs"] += 1
        
        # 4. 切分chunks
        chunks = chunker.chunk_document(
            doc_id=doc_id,
            text=text_cleaned,
            title=title_cleaned
        )
        
        # 5. 保存chunks
        for chunk in chunks:
            chunk_records.append(chunk)
            stats["total_chunks"] += 1
        
        # 6. 进度显示
        if stats["total_docs"] % 100 == 0:
            print(f"\r已处理 {stats['total_docs']:,} 文档 | "
                  f"保留 {stats['kept_docs']:,} 文档 | "
                  f"生成 {stats['total_chunks']:,} chunks", end='', flush=True)
    
    print(f"\n\n处理完成，正在保存文件...")
    
    # 7. 保存清洗后的文档（单个Parquet文件）
    if doc_records:
        docs_path = os.path.join(output_dir, "documents_cleaned.parquet")
        df_docs = pd.DataFrame(doc_records)
        df_docs.to_parquet(docs_path, index=False, compression='snappy')
        file_size_mb = os.path.getsize(docs_path) / (1024 * 1024)
        print(f"✅ 文档已保存: {docs_path}")
        print(f"   大小: {file_size_mb:.2f} MB | 记录数: {len(doc_records):,}")
    
    # 8. 保存chunks（单个Parquet文件）
    if chunk_records:
        chunks_path = os.path.join(output_dir, "chunks.parquet")
        df_chunks = pd.DataFrame(chunk_records)
        df_chunks.to_parquet(chunks_path, index=False, compression='snappy')
        file_size_mb = os.path.getsize(chunks_path) / (1024 * 1024)
        print(f"✅ Chunks已保存: {chunks_path}")
        print(f"   大小: {file_size_mb:.2f} MB | 记录数: {len(chunk_records):,}")
    
    # 9. 保存统计信息
    stats_path = os.path.join(output_dir, "preprocess_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✅ 统计信息已保存: {stats_path}")
    
    return stats


def main():
    """主函数"""
    
    # 配置
    cfg = Config()
    
    # 输入：Parquet数据目录
    data_dir = "data/raw/archive"
    
    # 输出目录
    output_dir = "data/processed"
    
    # 查找所有数据文件（支持CSV和Parquet）
    file_patterns = ["*.parquet", "*.csv"]
    data_files = []
    
    for pattern in file_patterns:
        files = glob.glob(os.path.join(data_dir, pattern))
        data_files.extend(files)
    
    if not data_files:
        print(f"❌ 未找到数据文件: {data_dir}")
        print(f"   支持的格式: {', '.join(file_patterns)}")
        return
    
    print(f"\n找到 {len(data_files)} 个数据文件:")
    for f in data_files[:10]:
        print(f"  - {f}")
    if len(data_files) > 10:
        print(f"  ... 还有 {len(data_files) - 10} 个文件")
    
    # 处理文档
    stats = process_documents(
        data_files=data_files,
        output_dir=output_dir,
        min_text_chars=100,
        target_lang='en',
        child_chunk_size=128,
        parent_chunk_size=512
    )
    
    # 显示统计
    print("\n" + "="*60)
    print("处理统计:")
    print(f"  总文档数: {stats['total_docs']:,}")
    print(f"  保留文档: {stats['kept_docs']:,} ({stats['kept_docs']/max(stats['total_docs'],1)*100:.1f}%)")
    print(f"  总chunks: {stats['total_chunks']:,}")
    print(f"  平均chunks/文档: {stats['total_chunks']/max(stats['kept_docs'],1):.1f}")
    print(f"\n过滤统计:")
    print(f"  过短文本: {stats['filtered_short']:,}")
    print(f"  标题质量: {stats['filtered_title']:,}")
    print("="*60)


if __name__ == "__main__":
    main()
