"""
检查CSV/Parquet文件中的wiki数据质量

功能：
1. 读取CSV或Parquet文件
2. 应用文本清洗规则
3. 筛选出长度大于32字符的文本
4. 打印前20行供人工检查
"""

import os
import sys
import csv
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from processing.text_cleaner import full_text_cleaning, filter_short_text
from processing.title_cleaner import process_title, is_good_title, clean_title_conservative
from transformers import AutoTokenizer


def read_csv_with_cleaning(
    csv_path: str,
    min_length: int = 32,
    max_rows: int = 20,
    text_columns: List[str] = None
) -> List[Tuple[str, str, str]]:
    """
    读取CSV文件并应用清洗规则
    
    Args:
        csv_path: CSV文件路径
        min_length: 最小字符长度
        max_rows: 最多显示行数
        text_columns: 文本列名（None则自动检测）
    
    Returns:
        [(row_id, title, cleaned_text), ...]
    """
    # 初始化tokenizer（用于token计数）
    tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/bge-small-en-v1.5",
        trust_remote_code=True
    )
    
    results = []
    
    # 提升CSV字段大小限制
    try:
        csv.field_size_limit(min(sys.maxsize, 1_000_000_000))
    except Exception:
        pass
    
    # 读取CSV
    with open(csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        
        # 自动检测文本列
        if text_columns is None:
            candidates = {"text", "content", "article", "body", "paragraph", "desc", "description", "wiki_text"}
            text_columns = [h for h in header if h and h.lower() in candidates]
            if not text_columns:
                text_columns = [h for h in header if "text" in h.lower()]
            if not text_columns:
                text_columns = header[:3]  # 默认前3列
        
        # 检测标题列
        title_columns = [h for h in header if h and h.lower() in {"title", "name", "heading", "subject"}]
        
        print(f"检测到的文本列: {text_columns}")
        print(f"检测到的标题列: {title_columns}")
        print("=" * 80)
        
        for i, row in enumerate(reader):
            if len(results) >= max_rows:
                break
            
            # 提取标题
            title = ""
            for col in title_columns:
                val = row.get(col, "")
                if isinstance(val, str) and val.strip():
                    title = val.strip()
                    break
            
            # 提取文本
            parts = []
            for col in text_columns:
                val = row.get(col, "")
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
            
            text = "\n\n".join(parts).strip()
            
            if not text:
                continue
            
            # 应用文本清洗
            cleaned_text = full_text_cleaning(text, target_lang="en")
            
            # 如果清洗后为None，说明是垃圾文本
            if cleaned_text is None:
                continue
            
            # 过滤长度
            if len(cleaned_text) < min_length:
                continue
            
            # 清洗标题
            cleaned_title = ""
            if title and is_good_title(title):
                cleaned_title = clean_title_conservative(title)
            
            row_id = f"Row {i + 1}"
            results.append((row_id, cleaned_title, cleaned_text))
    
    return results


def read_parquet_with_cleaning(
    parquet_path: str,
    min_length: int = 32,
    max_rows: int = 20,
    text_columns: List[str] = None
) -> List[Tuple[str, str, str]]:
    """
    读取Parquet文件并应用清洗规则
    
    Args:
        parquet_path: Parquet文件路径
        min_length: 最小字符长度
        max_rows: 最多显示行数
        text_columns: 文本列名（None则自动检测）
    
    Returns:
        [(row_id, title, cleaned_text), ...]
    """
    # 初始化tokenizer（用于token计数）
    tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/bge-small-en-v1.5",
        trust_remote_code=True
    )
    
    results = []
    
    # 读取Parquet文件
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    # 获取所有列名
    header = df.columns.tolist()
    
    print(f"\n文件包含的所有列: {header}")
    print(f"总行数: {len(df):,}")
    print("=" * 80)
    
    # 自动检测文本列
    if text_columns is None:
        candidates = {"text", "content", "article", "body", "paragraph", "desc", "description", "wiki_text", "page_content"}
        text_columns = [h for h in header if h and h.lower() in candidates]
        if not text_columns:
            # 查找包含"text"关键字的列
            text_columns = [h for h in header if "text" in h.lower()]
        if not text_columns:
            # 查找包含"content"关键字的列
            text_columns = [h for h in header if "content" in h.lower()]
        if not text_columns:
            # 默认使用前3列（排除明显的ID列）
            non_id_cols = [h for h in header if not any(x in h.lower() for x in ["id", "index", "_id"])]
            text_columns = non_id_cols[:3] if non_id_cols else header[:3]
    
    # 检测标题列
    title_columns = [h for h in header if h and h.lower() in {"title", "name", "heading", "subject", "page_title"}]
    
    print(f"检测到的文本列: {text_columns}")
    print(f"检测到的标题列: {title_columns}")
    print("=" * 80)
    
    # 遍历行
    for i, row in df.iterrows():
        if len(results) >= max_rows:
            break
        
        # 提取标题
        title = ""
        for col in title_columns:
            if col in df.columns:
                val = row[col]
                if isinstance(val, str) and val.strip():
                    title = val.strip()
                    break
                elif val is not None:
                    title = str(val).strip()
                    if title:
                        break
        
        # 提取文本
        parts = []
        for col in text_columns:
            if col in df.columns:
                val = row[col]
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
                elif val is not None:
                    val_str = str(val).strip()
                    if val_str and val_str != "nan":
                        parts.append(val_str)
        
        text = "\n\n".join(parts).strip()
        
        if not text:
            continue
        
        # 应用文本清洗
        cleaned_text = full_text_cleaning(text, target_lang="en")
        
        # 如果清洗后为None，说明是垃圾文本
        if cleaned_text is None:
            continue
        
        # 过滤长度
        if len(cleaned_text) < min_length:
            continue
        
        # 清洗标题
        cleaned_title = ""
        if title and is_good_title(title):
            cleaned_title = clean_title_conservative(title)
        
        row_id = f"Row {i}"
        results.append((row_id, cleaned_title, cleaned_text))
    
    return results


def print_results(results: List[Tuple[str, str, str]]):
    """打印检查结果"""
    print(f"\n找到 {len(results)} 条符合条件的记录（长度 >= 32字符）\n")
    print("=" * 80)
    
    for i, (row_id, title, text) in enumerate(results, 1):
        print(f"\n【{i}】{row_id}")
        
        if title:
            print(f"标题: {title}")
        
        # 显示文本（最多显示500字符）
        text_preview = text[:500] if len(text) > 500 else text
        if len(text) > 500:
            text_preview += "..."
        
        print(f"长度: {len(text)} 字符")
        print(f"文本预览:\n{text_preview}")
        print("-" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查CSV/Parquet文件中的wiki数据质量')
    parser.add_argument('file', help='CSV或Parquet文件路径（如 1.csv 或 0_to_25000.parquet）')
    parser.add_argument('--min-length', type=int, default=32, help='最小字符长度（默认32）')
    parser.add_argument('--max-rows', type=int, default=20, help='最多显示行数（默认20）')
    parser.add_argument('--columns', nargs='+', help='指定文本列名（可选，多列用空格分隔）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file):
        print(f"❌ 错误: 文件不存在: {args.file}")
        return
    
    # 判断文件类型
    file_ext = os.path.splitext(args.file)[1].lower()
    is_parquet = file_ext == '.parquet'
    
    print(f"正在检查文件: {args.file}")
    print(f"文件类型: {'Parquet' if is_parquet else 'CSV'}")
    print(f"筛选条件: 长度 >= {args.min_length} 字符")
    print(f"显示数量: 前 {args.max_rows} 行")
    print("=" * 80)
    
    # 读取并清洗
    if is_parquet:
        results = read_parquet_with_cleaning(
            args.file,
            min_length=args.min_length,
            max_rows=args.max_rows,
            text_columns=args.columns
        )
    else:
        results = read_csv_with_cleaning(
            args.file,
            min_length=args.min_length,
            max_rows=args.max_rows,
            text_columns=args.columns
        )
    
    # 打印结果
    print_results(results)
    
    # 数据主题分析提示
    print("\n" + "=" * 80)
    print("💡 数据主题检查建议：")
    print("1. 查看上述文本是否与科学主题相关（物理、化学、生物、数学等）")
    print("2. 如果大部分文本不相关，建议：")
    print("   - 检查数据来源是否正确")
    print("   - 考虑添加主题过滤逻辑")
    print("   - 使用关键词匹配（science, physics, chemistry, biology等）")
    print("=" * 80)


if __name__ == "__main__":
    main()
