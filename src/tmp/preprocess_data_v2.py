"""数据预处理脚本 V2 - 新架构

核心改进：
1. 文件读取与处理逻辑解耦（支持CSV和Parquet）
2. 使用tiktoken替代transformers tokenizer
3. 基于句子的动态chunking
4. 使用字符offset管理，不存储token IDs
5. 支持多embedding模型ensemble
"""
import os
import glob
from pathlib import Path
from typing import List

from core.config import Config
from processing.data_loader import load_files
from processing.text_cleaner import full_text_cleaning
from chunking.sentence_chunker import SentenceChunker


def main():
    """主函数 - 演示新架构的使用"""
    
    # 加载配置
    cfg = Config()
    
    # 1. 发现数据文件（支持CSV和Parquet）
    data_dir = "data/raw/archive"  # Parquet文件目录
    # data_dir = "data/raw/csv"  # CSV文件目录
    
    # 查找所有支持的文件
    file_patterns = ["*.parquet", "*.csv"]
    data_files = []
    
    for pattern in file_patterns:
        files = glob.glob(os.path.join(data_dir, pattern))
        data_files.extend(files)
    
    if not data_files:
        print(f"❌ 未找到数据文件: {data_dir}")
        return
    
    print(f"找到 {len(data_files)} 个数据文件:")
    for f in data_files[:5]:  # 只显示前5个
        print(f"  - {f}")
    if len(data_files) > 5:
        print(f"  ... 还有 {len(data_files) - 5} 个文件")
    
    # 2. 初始化chunker（使用tiktoken）
    chunker = SentenceChunker(
        model_name="cl100k_base",  # GPT-3.5/4编码
        child_size=128,
        parent_size=512,
        min_chunk_chars=50
    )
    
    # 3. 处理文档
    total_docs = 0
    total_chunks = 0
    
    for doc_id, title, text in load_files(data_files):
        total_docs += 1
        
        # 清洗文本
        cleaned_text = full_text_cleaning(text, target_lang="en")
        
        if cleaned_text is None:
            continue
        
        # 动态chunking
        chunks = chunker.chunk_document(
            doc_id=doc_id,
            text=cleaned_text,
            title=title
        )
        
        total_chunks += len(chunks)
        
        # 显示进度
        if total_docs % 100 == 0:
            print(f"\r已处理 {total_docs} 文档, 生成 {total_chunks} chunks", end='', flush=True)
        
        # 示例：显示第一个文档的chunk结构
        if total_docs == 1:
            print(f"\n示例文档: {doc_id}")
            print(f"标题: {title}")
            print(f"原文长度: {len(cleaned_text)} 字符")
            print(f"生成chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks[:3]):  # 只显示前3个
                print(f"\n  Chunk {i}:")
                print(f"    chunk_id: {chunk['chunk_id']}")
                print(f"    子chunk范围: [{chunk['child_start']}:{chunk['child_end']}]")
                print(f"    父chunk范围: [{chunk['parent_start']}:{chunk['parent_end']}]")
                print(f"    tokens估计: {chunk['chunk_len']}")
                print(f"    文本预览: {chunk['rerank_text'][:100]}...")
    
    print(f"\n\n✅ 处理完成:")
    print(f"  总文档: {total_docs}")
    print(f"  总chunks: {total_chunks}")


if __name__ == "__main__":
    main()
