import sys
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def find_natural_break(text: str, start: int, end: int, backward: bool = False) -> int:
    """
    在给定范围内找到自然的切割点（句号、换行等）
    
    Args:
        text: 完整文本
        start: 搜索起点
        end: 搜索终点
        backward: 是否向前搜索（True为向前）
    
    Returns:
        切割点位置
    """
    breaks = {'.', '!', '?', '\n', '。', '！', '？', ';', '；'}
    
    if backward:
        # 从end往start找
        for i in range(end - 1, start - 1, -1):
            if i < len(text) and text[i] in breaks:
                return i + 1
    else:
        # 从start往end找
        for i in range(start, end):
            if i < len(text) and text[i] in breaks:
                return i + 1
    
    return end if not backward else start


def adjust_parent_chunks(
    chunks_path: str,
    doc_path: str,
    tokenizer,
    target_tokens: int = 256,
    output_path: str = None,
    batch_size: int = 10000
):
    """
    重新调整父chunk大小（批处理版本）
    """
    
    print(f"[加载] chunks.parquet...")
    chunks_df = pd.read_parquet(chunks_path)
    print(f"  共{len(chunks_df)}条chunks")
    print(f"  列名: {list(chunks_df.columns)}")
    
    # 自动检测start/end列名
    start_col = None
    end_col = None
    for col in ['start_char', 'child_start']:
        if col in chunks_df.columns:
            start_col = col
            break
    for col in ['end_char', 'child_end']:
        if col in chunks_df.columns:
            end_col = col
            break
    
    if not start_col or not end_col:
        print(f"❌ 未找到start/end列，可用列: {list(chunks_df.columns)}")
        exit(1)
    
    print(f"  使用列: {start_col}, {end_col}")
    
    print(f"\n[加载] documents_cleaned.parquet...")
    doc_df = pd.read_parquet(doc_path)
    print(f"  共{len(doc_df)}条文档")
    print(f"  列名: {list(doc_df.columns)}")
    
    # 自动检测id列
    id_col = None
    for col in ['id', 'doc_id', 'document_id']:
        if col in doc_df.columns:
            id_col = col
            break
    
    if not id_col:
        print(f"❌ 未找到id列，可用列: {list(doc_df.columns)}")
        exit(1)
    
    # 自动检测text列
    text_col = None
    for col in ['text', 'content', 'document_text']:
        if col in doc_df.columns:
            text_col = col
            break
    
    if not text_col:
        print(f"❌ 未找到text列，可用列: {list(doc_df.columns)}")
        exit(1)
    
    # 构建doc_id -> text映射
    print(f"  构建映射...")
    doc_texts = {}
    for _, row in doc_df.iterrows():
        did = row[id_col]
        txt = row[text_col] if pd.notna(row[text_col]) else ''
        doc_texts[did] = txt
    print(f"  已加载{len(doc_texts)}个文档（id列: {id_col}, text列: {text_col}）")
    
    # 检查parent_start/parent_end列
    print(f"\n[分析] 检查父chunk列...")
    
    if 'parent_start' not in chunks_df.columns or 'parent_end' not in chunks_df.columns:
        print(f"  ❌ 未找到parent_start/parent_end列")
        print(f"  可用列: {list(chunks_df.columns)}")
        exit(1)
    
    print(f"  ✓ 找到parent_start和parent_end列")
    print(f"  当前父chunk范围示例（前5行）:")
    for i in range(min(5, len(chunks_df))):
        ps = chunks_df.iloc[i]['parent_start']
        pe = chunks_df.iloc[i]['parent_end']
        cs = chunks_df.iloc[i][start_col]
        ce = chunks_df.iloc[i][end_col]
        plen = pe - ps if pd.notna(pe) and pd.notna(ps) else 0
        clen = ce - cs if pd.notna(ce) and pd.notna(cs) else 0
        print(f"    [{i}] child:[{cs},{ce}](长{clen}) -> parent:[{ps},{pe}](长{plen})")
    
    # 按文档分组处理
    print(f"\n[处理] 批量调整父chunk索引...")
    
    doc_groups = chunks_df.groupby('doc_id')
    total_chunks = len(chunks_df)
    total_adjusted = 0
    
    # 用总体进度条显示每个chunk的处理
    with tqdm(total=total_chunks, desc="调整chunk", unit="chunk") as pbar:
        for doc_id, group in doc_groups:
            if doc_id not in doc_texts:
                pbar.update(len(group))
                continue
            
            full_text = doc_texts[doc_id]
            
            # 对该文档的每个chunk调整父chunk范围
            for idx, row in group.iterrows():
                child_start = row[start_col]
                child_end = row[end_col]
                
                # 计算中点
                mid_point = (child_start + child_end) // 2
                
                # 估算256 tokens对应的字符数（约1024-1152字符）
                half_chars = int(target_tokens * 2.0)  # 保守估计：1 token ≈ 4字符
                
                # 向前向后扩展
                tentative_start = max(0, mid_point - half_chars)
                tentative_end = min(len(full_text), mid_point + half_chars)
                
                # 在语义边界处调整
                final_start = find_natural_break(full_text, tentative_start, mid_point, backward=True)
                final_end = find_natural_break(full_text, mid_point, tentative_end, backward=False)
                
                # 确保包含子chunk
                final_start = min(final_start, child_start)
                final_end = max(final_end, child_end)
                
                # 验证token数不超过256（使用快速估算）
                estimated_tokens = (final_end - final_start) // 4
                if estimated_tokens > target_tokens * 1.2:  # 允许20%误差
                    # 如果太大，缩小范围
                    final_start = max(child_start, mid_point - int(target_tokens * 2))
                    final_end = min(child_end, mid_point + int(target_tokens * 2))
                
                # 更新
                chunks_df.loc[idx, 'parent_start'] = final_start
                chunks_df.loc[idx, 'parent_end'] = final_end
                total_adjusted += 1
                pbar.update(1)
    
    # 保存结果
    output_path = output_path or chunks_path
    print(f"\n[保存] {output_path}...")
    chunks_df.to_parquet(output_path, index=False, compression='snappy')
    print(f"  ✓ 已保存")
    
    # 统计信息
    print(f"\n[统计]")
    parent_sizes = (chunks_df['parent_end'] - chunks_df['parent_start']).dropna()
    
    if len(parent_sizes) > 0:
        print(f"  父chunk字符长度:")
        print(f"    最小: {int(parent_sizes.min()):,}")
        print(f"    平均: {int(parent_sizes.mean()):,}")
        print(f"    最大: {int(parent_sizes.max()):,}")
        print(f"  调整数: {total_adjusted:,} / {len(chunks_df):,}")
        
        # 估算token数分布
        estimated_tokens = parent_sizes / 4
        print(f"\n  估算token数（假设1 token≈4字符）:")
        print(f"    最小: {int(estimated_tokens.min())}")
        print(f"    平均: {int(estimated_tokens.mean())}")
        print(f"    最大: {int(estimated_tokens.max())}")
    
    print(f"\n✅ 完成！")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='调整父chunk大小到256 tokens')
    parser.add_argument('--chunks-path', type=str, default='data/processed/chunks.parquet',
                       help='chunks.parquet路径')
    parser.add_argument('--doc-path', type=str, default='data/processed/documents_cleaned.parquet',
                       help='documents_cleaned.parquet路径')
    parser.add_argument('--target-tokens', type=int, default=256,
                       help='目标token数')
    parser.add_argument('--tokenizer-id', type=str, default='Qwen/Qwen3-Embedding-0.6B',
                       help='tokenizer模型ID')
    parser.add_argument('--output', type=str, default=None,
                       help='输出路径（默认覆盖原文件）')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='批处理大小')
    
    args = parser.parse_args()
    
    print(f"[初始化] 加载tokenizer: {args.tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    
    adjust_parent_chunks(
        chunks_path=args.chunks_path,
        doc_path=args.doc_path,
        tokenizer=tokenizer,
        target_tokens=args.target_tokens,
        output_path=args.output,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
