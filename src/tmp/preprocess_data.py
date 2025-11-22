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
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from core.config import Config
from processing.text_cleaner import full_text_cleaning, filter_short_text
from processing.title_cleaner import process_title
from chunking.parent_chunker import ParentDocumentChunker


def load_csv_iterator(csv_path: str, text_columns: List[str] = None):
    """迭代器方式加载CSV文件
    
    注意：TBE的警告 “Token indices sequence length is longer than the specified maximum”
    是正常的，它是在tokenizer.encode()時，tokenize整个文档時发出的。
    但在子步骤中，每个chunk都会被截断到128 tokens，不会造成影响。
    
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
    parent_chunk_size: int = 512,
    resume: bool = True
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
        resume: 是否启用断点续跑
    
    Returns:
        统计信息字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    docs_dir = os.path.join(output_dir, "documents")
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    
    # 断点续跑：加载进度
    checkpoint_path = os.path.join(output_dir, "preprocess_checkpoint.json")
    processed_docs = set()
    initial_kept_docs = 0  # 记录恢复前已保留的文档数
    initial_doc_parquet_files = 0  # 恢复已写入的文档Parquet文件数
    initial_chunk_parquet_files = 0  # 恢复已写入的chunk Parquet文件数
    
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                processed_docs = set(checkpoint.get('processed_doc_ids', []))
                checkpoint_stats = checkpoint.get('stats', {})
                initial_kept_docs = checkpoint_stats.get('kept_docs', 0)
                initial_doc_parquet_files = checkpoint_stats.get('doc_parquet_files', 0)
                initial_chunk_parquet_files = checkpoint_stats.get('chunk_parquet_files', 0)
                print(f"[resume] 加载断点：")
                print(f"  已处理文档: {len(processed_docs):,} 个")
                print(f"  已保留文档: {initial_kept_docs:,} 个")
                print(f"  文档Parquet: {initial_doc_parquet_files} 个")
                print(f"  Chunk Parquet: {initial_chunk_parquet_files} 个")
        except Exception as e:
            print(f"[resume] 加载断点失败: {e}，从头开始")
            processed_docs = set()
            initial_kept_docs = 0
            initial_doc_parquet_files = 0
            initial_chunk_parquet_files = 0
    
    # 初始化chunker
    chunker = ParentDocumentChunker(
        tokenizer=tokenizer,
        child_size=child_chunk_size,
        parent_size=parent_chunk_size,
        min_chunk_tokens=32
    )
    
    # 统计信息（断点续跑时从checkpoint恢复）
    stats = {
        "total_docs": 0,
        "kept_docs": 0,
        "total_chunks": 0,
        "filtered_short": 0,
        "filtered_language": 0,
        "filtered_quality": 0,
        "skipped_resume": len(processed_docs),  # 断点续跑跳过的
        "doc_parquet_files": initial_doc_parquet_files,  # 从断点恢复
        "chunk_parquet_files": initial_chunk_parquet_files,  # 从断点恢复
    }
    
    # 当前批次的数据
    doc_batch_data = []  # 文档级数据
    chunk_batch_data = []  # chunk级数据
    doc_batch_size = 0
    chunk_batch_size = 0
    target_size_bytes = chunk_size_mb * 1024 * 1024
    
    # 动态估算系数（根据实际文件大小调整）
    # 初始值：chunks约82MB，docs约95MB（根据你的实际测试）
    doc_size_ratio = 95 / 256  # docs: 95MB 实际/ 256MB 估计 = 0.37
    chunk_size_ratio = 83.4 / 256  # chunks: 83.4MB 实际 / 256MB 估计 = 0.326
    
    # 断点续跑：保存进度
    def save_checkpoint():
        """保存处理进度"""
        try:
            checkpoint = {
                'processed_doc_ids': list(processed_docs),
                'stats': stats.copy(),
            }
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[checkpoint] 保存失败: {e}")
    
    def write_doc_parquet_batch():
        """写入文档级Parquet文件"""
        nonlocal doc_batch_data, doc_batch_size
        
        if not doc_batch_data:
            return
        
        # 构建PyArrow表（文档级）- 使用uint16优化存储
        schema = pa.schema([
            ('doc_id', pa.string()),
            ('title', pa.string()),
            ('title_ids', pa.list_(pa.uint16())),  # uint16节省空间
            ('doc_ids', pa.list_(pa.uint16())),    # uint16节省空间
            ('length_tokens', pa.int32()),
        ])
        
        # 转换为uint16（BGE-small词表大小约30k，uint16足够）
        table = pa.table({
            'doc_id': [d['doc_id'] for d in doc_batch_data],
            'title': [d['title'] for d in doc_batch_data],
            'title_ids': [[np.uint16(x) for x in d['title_ids']] for d in doc_batch_data],
            'doc_ids': [[np.uint16(x) for x in d['doc_ids']] for d in doc_batch_data],
            'length_tokens': [d['length_tokens'] for d in doc_batch_data],
        }, schema=schema)
        
        # 写入文件（使用ZSTD压缩）
        output_file = os.path.join(docs_dir, f"docs_{stats['doc_parquet_files'] + 1}.parquet")
        pq.write_table(table, output_file, compression='zstd', compression_level=3)
        
        # 获取实际文件大小
        actual_file_size = os.path.getsize(output_file)
        print(f"[write] 文档Parquet {stats['doc_parquet_files'] + 1}: {len(doc_batch_data)} docs, {actual_file_size / (1024*1024):.2f}MB")
        
        stats['doc_parquet_files'] += 1
        doc_batch_data = []
        doc_batch_size = 0
        
        # 保存断点
        save_checkpoint()
    
    def write_chunk_parquet_batch():
        """写入chunk级Parquet文件"""
        nonlocal chunk_batch_data, chunk_batch_size
        
        if not chunk_batch_data:
            return
        
        # 构建PyArrow表（chunk级）- 使用uint16优化存储
        schema = pa.schema([
            ('chunk_id', pa.string()),
            ('doc_id', pa.string()),
            ('rerank_text', pa.string()),
            ('child_ids', pa.list_(pa.uint16())),  # uint16节省空间
            ('parent_start', pa.int32()),
            ('parent_end', pa.int32()),
            ('chunk_len', pa.int32()),
        ])
        
        # 转换为uint16
        table = pa.table({
            'chunk_id': [c['chunk_id'] for c in chunk_batch_data],
            'doc_id': [c['doc_id'] for c in chunk_batch_data],
            'rerank_text': [c['rerank_text'] for c in chunk_batch_data],
            'child_ids': [[np.uint16(x) for x in c['child_ids']] for c in chunk_batch_data],
            'parent_start': [c['parent_start'] for c in chunk_batch_data],
            'parent_end': [c['parent_end'] for c in chunk_batch_data],
            'chunk_len': [c['chunk_len'] for c in chunk_batch_data],
        }, schema=schema)
        
        # 写入文件（使用ZSTD压缩）
        output_file = os.path.join(chunks_dir, f"chunks_{stats['chunk_parquet_files'] + 1}.parquet")
        pq.write_table(table, output_file, compression='zstd', compression_level=3)
        
        # 获取实际文件大小
        actual_file_size = os.path.getsize(output_file)
        print(f"[write] Chunk Parquet {stats['chunk_parquet_files'] + 1}: {len(chunk_batch_data)} chunks, {actual_file_size / (1024*1024):.2f}MB")
        
        stats['chunk_parquet_files'] += 1
        chunk_batch_data = []
        chunk_batch_size = 0
        
        # 保存断点
        save_checkpoint()
    
    # 处理每个CSV文件
    for csv_path in csv_files:
        print(f"\n[process] 正在处理: {csv_path}")
        
        for row_id, title_raw, text_raw in load_csv_iterator(csv_path):
            stats['total_docs'] += 1
            
            # 断点续跑：跳过已处理的文档
            if row_id in processed_docs:
                continue
            
            # 进度显示（使用 \r 覆盖前一条）
            if stats['total_docs'] % 100 == 0:
                # 显示累计数量（本次 + 之前）
                total_kept = stats['kept_docs'] + initial_kept_docs
                print(f"\r  已处理 {stats['total_docs']:,} 条文档，保留 {total_kept:,} 条", end='', flush=True)
            
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
            
            # 使用动态估算（uint16 + ZSTD压缩后的实际大小）
            # 注意：这里估算的是“原始数据大小”，不需要乘以ratio
            # 因为target_size_bytes已经是目标压缩后的大小
            # 所以应该除以ratio来得到“需要多少原始数据才能达到1GB”
            doc_size_raw = (
                len(row_id) + 
                len(final_title_text) + 
                len(text_tokens) * 2 +  # uint16 = 2字节
                len(final_title_ids) * 2  # uint16 = 2字节
            )
            doc_size = doc_size_raw * doc_size_ratio  # 估算压缩后大小
            
            doc_batch_data.append(doc_data)
            doc_batch_size += doc_size
            stats['kept_docs'] += 1
            
            # 断点续跑：标记文档已处理
            processed_docs.add(row_id)
            
            # 7. 生成chunks（父文档索引）
            chunks = chunker.chunk_document(
                doc_id=row_id,
                title_ids=final_title_ids,
                doc_ids=text_tokens,
                title_text=final_title_text
            )
            
            # 8. 添加chunk级数据
            for chunk in chunks:
                # 使用动态估算（uint16 + ZSTD压缩后的实际大小）
                chunk_size_raw = (
                    len(chunk['chunk_id']) + 
                    len(chunk['doc_id']) + 
                    len(chunk['rerank_text']) + 
                    len(chunk['child_ids']) * 2 +  # uint16 = 2字节
                    16  # parent_start, parent_end, chunk_len
                )
                chunk_size = chunk_size_raw * chunk_size_ratio  # 估算压缩后大小
                
                chunk_batch_data.append(chunk)
                chunk_batch_size += chunk_size
                stats['total_chunks'] += 1
            
            # 9. 如果达到目标大小，写入Parquet
            # 添加数量阈值作为安全措施，防止内存溢出
            MAX_DOCS_PER_BATCH = 200000  # 最多20万个文档
            MAX_CHUNKS_PER_BATCH = 1000000  # 最多100万个chunks
            
            if doc_batch_size >= target_size_bytes or len(doc_batch_data) >= MAX_DOCS_PER_BATCH:
                if len(doc_batch_data) >= MAX_DOCS_PER_BATCH:
                    print(f"\n[write] 达到数量阈值 ({len(doc_batch_data):,} docs)，强制写入")
                write_doc_parquet_batch()
            
            if chunk_batch_size >= target_size_bytes or len(chunk_batch_data) >= MAX_CHUNKS_PER_BATCH:
                if len(chunk_batch_data) >= MAX_CHUNKS_PER_BATCH:
                    print(f"\n[write] 达到数量阈值 ({len(chunk_batch_data):,} chunks)，强制写入")
                write_chunk_parquet_batch()
    
    # 写入最后一个批次
    if doc_batch_data:
        write_doc_parquet_batch()
    
    if chunk_batch_data:
        write_chunk_parquet_batch()
    
    # 保存最终断点
    save_checkpoint()
    
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
    chunk_size_mb = cfg.get("preprocessing", "parquet_chunk_size_mb", default=1024)  # 默认1GB
    
    # Chunking 参数
    child_chunk_size = cfg.get("chunking", "child_size", default=128)
    parent_chunk_size = cfg.get("chunking", "parent_size", default=512)
    
    # 断点续跑参数
    resume = cfg.get("preprocessing", "resume", default=True)
    
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
        parent_chunk_size=parent_chunk_size,
        resume=resume
    )
    
    # 获取initial_kept_docs用于统计
    checkpoint_path = os.path.join(output_dir, "preprocess_checkpoint.json")
    initial_kept_docs = 0
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                initial_kept_docs = checkpoint.get('stats', {}).get('kept_docs', 0)
        except:
            pass
    
    # 打印统计
    print("\n" + "=" * 80)  # 确保进度信息完全清除（\r会改变游标位置）
    print("[stats] 处理统计:")
    print(f"  总文档数: {stats['total_docs']:,}")
    print(f"  断点跳过: {stats['skipped_resume']:,}")
    print(f"  本次保留: {stats['kept_docs']:,}")
    total_kept = stats['kept_docs'] + initial_kept_docs
    print(f"  累计保留: {total_kept:,} ({total_kept/max(stats['total_docs'],1)*100:.1f}%)")
    print(f"  生成chunk数: {stats['total_chunks']:,}")
    print(f"  过滤 - 短文本: {stats['filtered_short']:,}")
    print(f"  过滤 - 质量差: {stats['filtered_quality']:,}")
    print(f"  文档Parquet文件: {stats['doc_parquet_files']}")
    print(f"  Chunk Parquet文件: {stats['chunk_parquet_files']}")
    print(f"  文档输出目录: {os.path.join(output_dir, 'documents')}")
    print(f"  Chunk输出目录: {os.path.join(output_dir, 'chunks')}")
    print(f"  断点文件: {os.path.join(output_dir, 'preprocess_checkpoint.json')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
