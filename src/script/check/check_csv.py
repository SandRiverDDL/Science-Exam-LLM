"""检查 CSV 文件数据质量

功能：
1. 打印 CSV 文件的总行数
2. 显示各列的空值情况
3. 列出有空值的行号和列名
"""
import sys
from pathlib import Path

import pandas as pd


def check_csv(csv_path: Path):
    """检查 CSV 文件
    
    Args:
        csv_path: CSV 文件路径
    """
    print("=" * 80)
    print(f"检查文件: {csv_path}")
    print("=" * 80)
    
    # 检查文件是否存在
    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return
    
    # 加载 CSV
    print("\n[1/3] 加载 CSV 文件...")
    try:
        df = pd.read_csv(csv_path)
        print(f"  ✓ 加载成功")
        print(f"  ✓ 总行数: {len(df)}")
        print(f"  ✓ 列数: {len(df.columns)}")
        print(f"  ✓ 列名: {list(df.columns)}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 检查空值统计
    print("\n[2/3] 空值统计...")
    null_counts = df.isnull().sum()
    has_nulls = null_counts.sum() > 0
    
    if not has_nulls:
        print("  ✓ 所有列都没有空值")
    else:
        print("  有空值的列:")
        for col, count in null_counts[null_counts > 0].items():
            ratio = (count / len(df)) * 100
            print(f"    - {col}: {count} 个空值 ({ratio:.2f}%)")
    
    # 列出有空值的行
    print("\n[3/3] 有空值的行...")
    rows_with_nulls = df[df.isnull().any(axis=1)]
    
    if len(rows_with_nulls) == 0:
        print("  ✓ 没有包含空值的行")
    else:
        print(f"  共 {len(rows_with_nulls)} 行有空值:")
        print()
        for idx, row in rows_with_nulls.iterrows():
            null_cols = row[row.isnull()].index.tolist()
            print(f"  行号: {idx}")
            print(f"    有空值的列: {null_cols}")
    
    print("\n" + "=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='检查 CSV 文件数据质量')
    parser.add_argument('--input', type=str, default=r'data/raw/6k/6000_train_examples.csv',
                       help='CSV 文件路径')
    
    args = parser.parse_args()
    
    csv_path = Path(args.input)
    check_csv(csv_path)


if __name__ == "__main__":
    main()
