"""检查 context_question.parquet 数据质量

功能：
1. 检查各列是否为空
2. 显示数据统计信息
3. 随机打印几行 prompt、C1、C2、C3 的值，检查相关性
"""
import sys
from pathlib import Path
import random

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def check_parquet(parquet_path: Path):
    """检查 Parquet 文件
    
    Args:
        parquet_path: Parquet 文件路径
    """
    print("=" * 80)
    print(f"检查文件: {parquet_path}")
    print("=" * 80)
    
    # 检查文件是否存在
    if not parquet_path.exists():
        print(f"❌ 文件不存在: {parquet_path}")
        return
    
    # 加载 Parquet
    print("\n[1/4] 加载 Parquet...")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"  ✓ 加载成功")
        print(f"  ✓ 行数: {len(df):,}")
        print(f"  ✓ 列数: {len(df.columns)}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 检查列信息
    print("\n[2/4] 检查列信息...")
    print(f"  列名: {list(df.columns)}")
    print()
    
    # 检查各列是否为空
    print("[3/4] 检查各列数据...")
    print(f"{'列名':<20} {'总数':<10} {'非空':<10} {'空值':<10} {'空值比例':<15}")
    print("-" * 65)
    
    empty_cols = []
    for col in df.columns:
        total = len(df)
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        null_ratio = null_count / total * 100 if total > 0 else 0
        
        status = "✓" if null_count == 0 else "⚠️"
        print(f"{col:<20} {total:<10} {non_null:<10} {null_count:<10} {null_ratio:.2f}% {status}")
        
        if null_count > 0:
            empty_cols.append((col, null_count, null_ratio))
    
    # 提示空值列
    if empty_cols:
        print("\n⚠️  包含空值的列:")
        for col, count, ratio in empty_cols:
            print(f"  - {col}: {count} 个空值 ({ratio:.2f}%)")
    else:
        print("\n✅ 所有列都没有空值")
    
    # 随机打印几行数据
    print("\n[4/4] 随机打印 5 行数据进行相关性检查...")
    print("-" * 80)
    
    # 检查必需列
    required_cols = ['prompt', 'C1', 'C2', 'C3']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  缺少列: {missing_cols}")
        print("\n实际列名:")
        for col in df.columns:
            print(f"  - {col}")
        return
    
    # 随机选择行
    sample_size = min(5, len(df))
    sample_indices = random.sample(range(len(df)), sample_size)
    
    for idx, row_idx in enumerate(sample_indices, 1):
        row = df.iloc[row_idx]
        
        print(f"\n行 {row_idx + 1}:")
        print(f"  【Prompt】({len(str(row['prompt']))} 字)")
        print(f"    {str(row['prompt'])[:150]}{'...' if len(str(row['prompt'])) > 150 else ''}")
        
        for ctx_num in [1, 2, 3]:
            col_name = f'C{ctx_num}'
            content = str(row[col_name]) if pd.notna(row[col_name]) else "[空值]"
            content_len = len(content) if content != "[空值]" else 0
            
            print(f"\n  【C{ctx_num}】({content_len} 字)")
            print(f"    {content[:150]}{'...' if len(content) > 150 else ''}")
        
        print(f"\n  {'─' * 78}")
    
    print()
    print("=" * 80)
    print("✅ 检查完成")
    print("=" * 80)
    
    # 数据统计
    print("\n[补充信息]")
    print(f"  文件大小: {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 计算 prompt 平均长度
    if 'prompt' in df.columns:
        avg_prompt_len = df['prompt'].astype(str).str.len().mean()
        print(f"  Prompt 平均长度: {avg_prompt_len:.0f} 字")
    
    # 计算 contexts 平均长度
    for ctx_num in [1, 2, 3]:
        col_name = f'C{ctx_num}'
        if col_name in df.columns:
            avg_len = df[col_name].astype(str).str.len().mean()
            print(f"  {col_name} 平均长度: {avg_len:.0f} 字")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查 context_question.parquet 数据质量')
    parser.add_argument('--input', type=str,
                       default=r"data\processed\context\val_context.parquet",
                       help='输入 Parquet 文件路径')
    parser.add_argument('--samples', type=int, default=10,
                       help='随机打印的行数')
    
    args = parser.parse_args()
    
    parquet_path = Path(args.input)
    check_parquet(parquet_path)


if __name__ == "__main__":
    main()
