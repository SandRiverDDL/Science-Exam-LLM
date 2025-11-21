"""数据预处理主脚本 (2025 SOTA版本 + 父文档索引)
读取CSV文件，进行清洗、预tokenization、父文档chunking，输出为Parquet格式

核心改进：
1. 使用ftfy修复乱码
2. 保留Unicode符号（货币、数学、单位等）
3. 保留HTML结构
4. 标题保留停用词和型号，直接返回token IDs
5. 父文档索引 (Parent Document): 子chunk 128 tokens，父chunk 512 tokens
6. 输出两个Parquet：文档级 + chunk级
"""
import os
import sys
import csv
import glob
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from core.config import Config
from processing.text_cleaner import full_text_cleaning, filter_short_text
from processing.title_cleaner import process_title
from chunking.parent_chunker import ParentDocumentChunker


def load_csv_iterator(csv_path: str, text_columns: List[str] = None):
    """迭代器方式加载CSV文件
    
    Yields:
        (row_id, title, text) tuples
    """
    try:
        csv.field_size_limit(min(sys.maxsize, 1_000_000_000))
    except Exception:
        pass
    
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        
        # 自动检测文本列
        if text_columns is None:
            text_candidates = ["text", "content", "article", "body", "paragraph"]
            text_col = None
            for candidate in text_candidates:
                if candidate in header:
                    text_col = candidate
                    break
            if not text_col and header:
                text_col = header[0]
            text_columns = [text_col] if text_col else []
        
        # 检测标题列
        title_col = None
        for candidate in ["title", "heading", "subject", "name"]:
            if candidate in header:
                title_col = candidate
                break
        
        for i, row in enumerate(reader):
            # 提取文本
            text_parts = []
            for col in text_columns:
                val = row.get(col, "")
                if isinstance(val, str) and val.strip():
                    text_parts.append(val.strip())
            
            text = "\n\n".join(text_parts).strip()
            
            # 提取标题
            title = row.get(title_col, "") if title_col else ""
            
            # 生成行ID
            row_id = f"{os.path.basename(csv_path)}:row:{i}"
            
            yield (row_id, title, text)


def process_documents(
    csv_files: List[str],
    tokenizer,
    output_dir: str,
    chunk_size_mb: int = 256,
    min_text_tokens: int = 32,
    title_max_tokens: int = 16,
    target_lang: str = 'en',
    child_chunk_size: int = 128,
    parent_chunk_size: int = 512
) -> Dict[str, int]:
    """处理文档并输出为Parquet
    
    Args:
        csv_files: CSV文件路径列表
        tokenizer: BGE-small tokenizer
        output_dir: 输出目录
        chunk_size_mb: 每个Parquet文件的大小（MB）
        min_text_tokens: 最小文本token数
        title_max_tokens: 标题最大token数
        target_lang: 目标语言
        child_chunk_size: 子chunk大小（用于检索）
        parent_chunk_size: 父chunk大小（用于LLM）
    
    Returns:
        统计信息字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    docs_dir = os.path.join(output_dir, "documents")
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    
    # 初始化chunker
    chunker = ParentDocumentChunker(
        tokenizer=tokenizer,
        child_size=child_chunk_size,
        parent_size=parent_chunk_size,
        min_chunk_tokens=32
    )
    
    # 统计信息
    stats = {
        "total_docs": 0,
        "kept_docs": 0,
        "total_chunks": 0,
        "filtered_short": 0,
        "filtered_language": 0,
        "filtered_quality": 0,
        "doc_parquet_files": 0,
        "chunk_parquet_files": 0,
    }
    
    # 当前批次的数据
    doc_batch_data = []  # 文档级数据
    chunk_batch_data = []  # chunk级数据
    doc_batch_size = 0
    chunk_batch_size = 0
    target_size_bytes = chunk_size_mb * 1024 * 1024
    
    def write_doc_parquet_batch():
        """写入文档级Parquet文件"""
        nonlocal doc_batch_data, doc_batch_size
        
        if not doc_batch_data:
            return
        
        # 构建PyArrow表（文档级）
        schema = pa.schema([
            ('doc_id', pa.string()),
            ('title', pa.string()),
            ('title_ids', pa.list_(pa.int32())),
            ('doc_ids', pa.list_(pa.int32())),  # 原文档token IDs
            ('length_tokens', pa.int32()),
        ])
        
        table = pa.table({
            'doc_id': [d['doc_id'] for d in doc_batch_data],
            'title': [d['title'] for d in doc_batch_data],
            'title_ids': [d['title_ids'] for d in doc_batch_data],
            'doc_ids': [d['doc_ids'] for d in doc_batch_data],
            'length_tokens': [d['length_tokens'] for d in doc_batch_data],
        }, schema=schema)
        
        # 写入文件
        output_file = os.path.join(docs_dir, f"docs_{stats['doc_parquet_files'] + 1}.parquet")
        pq.write_table(table, output_file, compression='snappy')
        
        print(f"[write] 文档Parquet {stats['doc_parquet_files'] + 1}: {len(doc_batch_data)} docs, {doc_batch_size / (1024*1024):.2f}MB")
        
        stats['doc_parquet_files'] += 1
        doc_batch_data = []
        doc_batch_size = 0
    
    def write_chunk_parquet_batch():
        """写入chunk级Parquet文件"""
        nonlocal chunk_batch_data, chunk_batch_size
        
        if not chunk_batch_data:
            return
        
        # 构建PyArrow表（chunk级）
        schema = pa.schema([
            ('chunk_id', pa.string()),
            ('doc_id', pa.string()),
            ('rerank_text', pa.string()),  # 拼接了标题的文本，reranker输入
            ('child_ids', pa.list_(pa.int32())),  # 子chunk token IDs (128)
            ('parent_start', pa.int32()),  # 父chunk在doc_ids中的起始位置
            ('parent_end', pa.int32()),    # 父chunk在doc_ids中的结束位置
            ('chunk_len', pa.int32()),     # 子chunk长度
        ])
        
        table = pa.table({
            'chunk_id': [c['chunk_id'] for c in chunk_batch_data],
            'doc_id': [c['doc_id'] for c in chunk_batch_data],
            'rerank_text': [c['rerank_text'] for c in chunk_batch_data],
            'child_ids': [c['child_ids'] for c in chunk_batch_data],
            'parent_start': [c['parent_start'] for c in chunk_batch_data],
            'parent_end': [c['parent_end'] for c in chunk_batch_data],
            'chunk_len': [c['chunk_len'] for c in chunk_batch_data],
        }, schema=schema)
        
        # 写入文件
        output_file = os.path.join(chunks_dir, f"chunks_{stats['chunk_parquet_files'] + 1}.parquet")
        pq.write_table(table, output_file, compression='snappy')
        
        print(f"[write] Chunk Parquet {stats['chunk_parquet_files'] + 1}: {len(chunk_batch_data)} chunks, {chunk_batch_size / (1024*1024):.2f}MB")
        
        stats['chunk_parquet_files'] += 1
        chunk_batch_data = []
        chunk_batch_size = 0
    
    # 处理每个CSV文件
    for csv_path in csv_files:
        print(f"\n[process] 正在处理: {csv_path}")
        
        for row_id, title_raw, text_raw in load_csv_iterator(csv_path):
            stats['total_docs'] += 1
            
            # 每1000条打印一次进度
            if stats['total_docs'] % 1000 == 0:
                print(f"  已处理 {stats['total_docs']:,} 条文档，保留 {stats['kept_docs']:,} 条")
            
            # 1. 文本清洗
            text_clean = full_text_cleaning(text_raw, target_lang=target_lang)
            if text_clean is None:
                stats['filtered_quality'] += 1
                continue
            
            # 2. 预Tokenization
            try:
                text_tokens = tokenizer.encode(text_clean, add_special_tokens=False)
            except Exception:
                stats['filtered_quality'] += 1
                continue
            
            # 3. 短文本过滤
            if len(text_tokens) < min_text_tokens:
                stats['filtered_short'] += 1
                continue
            
            # 4. 标题处理（直接返回token IDs）
            title_ids = process_title(
                title_raw,
                tokenizer,
                max_tokens=title_max_tokens
            )
            
            # 5. 准备数据
            final_title_ids = title_ids if title_ids else []
            final_title_text = tokenizer.decode(final_title_ids, skip_special_tokens=True) if final_title_ids else ""
            
            # 6. 添加文档级数据
            doc_data = {
                'doc_id': row_id,
                'title': final_title_text,
                'title_ids': final_title_ids,
                'doc_ids': text_tokens,  # 保存原文档token IDs
                'length_tokens': len(text_tokens),
            }
            
            doc_size = len(row_id) + len(final_title_text) + len(text_tokens) * 4 + len(final_title_ids) * 4
            doc_batch_data.append(doc_data)
            doc_batch_size += doc_size
            stats['kept_docs'] += 1
            
            # 7. 生成chunks（父文档索引）
            chunks = chunker.chunk_document(
                doc_id=row_id,
                title_ids=final_title_ids,
                doc_ids=text_tokens,
                title_text=final_title_text
            )
            
            # 8. 添加chunk级数据
            for chunk in chunks:
                chunk_size = (
                    len(chunk['chunk_id']) + 
                    len(chunk['doc_id']) + 
                    len(chunk['rerank_text']) + 
                    len(chunk['child_ids']) * 4 + 
                    16  # parent_start, parent_end, chunk_len
                )
                chunk_batch_data.append(chunk)
                chunk_batch_size += chunk_size
                stats['total_chunks'] += 1
            
            # 9. 如果达到目标大小，写入Parquet
            if doc_batch_size >= target_size_bytes:
                write_doc_parquet_batch()
            
            if chunk_batch_size >= target_size_bytes:
                write_chunk_parquet_batch()
    
    # 写入最后一个批次
    if doc_batch_data:
        write_doc_parquet_batch()
    
    if chunk_batch_data:
        write_chunk_parquet_batch()
    
    return stats


def main():
    """主函数"""
    cfg = Config()
    
    # 从配置文件读取参数
    corpus_dir = cfg.data.get("corpus_dir", "data/raw/articles")
    output_dir = cfg.get("preprocessing", "output_dir", default="data/processed/parquet")
    min_text_tokens = cfg.get("preprocessing", "min_text_tokens", default=32)
    title_max_tokens = cfg.get("preprocessing", "title_max_tokens", default=16)
    target_lang = cfg.get("preprocessing", "target_language", default="en")
    chunk_size_mb = cfg.get("preprocessing", "parquet_chunk_size_mb", default=256)
    
    # Chunking 参数
    child_chunk_size = cfg.get("chunking", "child_size", default=128)
    parent_chunk_size = cfg.get("chunking", "parent_size", default=512)
    
    # 查找所有CSV文件
    csv_files = []
    for pattern in ["**/*.csv"]:
        for fp in glob.glob(os.path.join(corpus_dir, pattern), recursive=True):
            if os.path.isfile(fp):
                csv_files.append(fp)
    
    print(f"[init] 找到 {len(csv_files)} 个CSV文件")
    
    if not csv_files:
        print("[error] 未找到CSV文件")
        return
    
    # 加载tokenizer
    print(f"[init] 加载tokenizer...")
    model_id = cfg.get("embedding_bge_small", "model_id", default="BAAI/bge-small-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # 处理文档
    print(f"\n[process] 开始处理文档...")
    print(f"  - 目标语言: {target_lang}")
    print(f"  - 最小token数: {min_text_tokens}")
    print(f"  - 标题最大token: {title_max_tokens}")
    print(f"  - 子chunk大小: {child_chunk_size}")
    print(f"  - 父chunk大小: {parent_chunk_size}")
    print(f"  - Parquet文件大小: {chunk_size_mb}MB")
    
    stats = process_documents(
        csv_files=csv_files,
        tokenizer=tokenizer,
        output_dir=output_dir,
        chunk_size_mb=chunk_size_mb,
        min_text_tokens=min_text_tokens,
        title_max_tokens=title_max_tokens,
        target_lang=target_lang,
        child_chunk_size=child_chunk_size,
        parent_chunk_size=parent_chunk_size
    )
    
    # 打印统计
    print("\n" + "=" * 80)
    print("[stats] 处理统计:")
    print(f"  总文档数: {stats['total_docs']:,}")
    print(f"  保留文档数: {stats['kept_docs']:,} ({stats['kept_docs']/max(stats['total_docs'],1)*100:.1f}%)")
    print(f"  生成chunk数: {stats['total_chunks']:,}")
    print(f"  过滤 - 短文本: {stats['filtered_short']:,}")
    print(f"  过滤 - 质量差: {stats['filtered_quality']:,}")
    print(f"  文档Parquet文件: {stats['doc_parquet_files']}")
    print(f"  Chunk Parquet文件: {stats['chunk_parquet_files']}")
    print(f"  文档输出目录: {os.path.join(output_dir, 'documents')}")
    print(f"  Chunk输出目录: {os.path.join(output_dir, 'chunks')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
