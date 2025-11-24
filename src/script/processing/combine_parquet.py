"""合并两个 Parquet 文件

功能：
1. 读取两个 Parquet 文件
2. 合并成一个 DataFrame
3. 打乱顺序（随机排列）
4. 保存为新的 Parquet 文件
"""
import sys
from pathlib import Path

import pandas as pd


def combine_parquet(input_path1: Path, input_path2: Path, output_path: Path):
    """合并两个 Parquet 文件
    
    Args:
        input_path1: 第一个输入文件路径
        input_path2: 第二个输入文件路径
        output_path: 输出文件路径
    """
    print("=" * 80)
    print("合并 Parquet 文件")
    print("=" * 80)
    
    # 检查文件是否存在
    if not input_path1.exists():
        print(f"❌ 文件不存在: {input_path1}")
        return
    
    if not input_path2.exists():
        print(f"❌ 文件不存在: {input_path2}")
        return
    
    # 加载两个文件
    print(f"\n[1/4] 加载第一个文件: {input_path1}")
    try:
        df1 = pd.read_parquet(input_path1)
        print(f"  ✓ 加载成功")
        print(f"  ✓ 行数: {len(df1)}")
        print(f"  ✓ 列数: {len(df1.columns)}")
        print(f"  ✓ 列名: {list(df1.columns)}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    print(f"\n[2/4] 加载第二个文件: {input_path2}")
    try:
        df2 = pd.read_parquet(input_path2)
        print(f"  ✓ 加载成功")
        print(f"  ✓ 行数: {len(df2)}")
        print(f"  ✓ 列数: {len(df2.columns)}")
        print(f"  ✓ 列名: {list(df2.columns)}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 检查列是否一致
    if set(df1.columns) != set(df2.columns):
        print(f"\n⚠️  警告: 两个文件的列名不一致")
        print(f"  文件1列名: {list(df1.columns)}")
        print(f"  文件2列名: {list(df2.columns)}")
        print(f"  将只保留公共列")
        common_cols = list(set(df1.columns) & set(df2.columns))
        df1 = df1[common_cols]
        df2 = df2[common_cols]
    
    # 合并
    print(f"\n[3/4] 合并数据...")
    df_combined = pd.concat([df1, df2], ignore_index=True)
    print(f"  ✓ 合并后行数: {len(df_combined)}")
    
    # 打乱顺序
    print(f"\n[4/4] 打乱顺序并保存...")
    df_shuffled = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"  ✓ 打乱完成")
    
    # 创建输出目录（如果不存在）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    try:
        df_shuffled.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"  ✓ 保存成功")
        print(f"  ✓ 输出路径: {output_path}")
        print(f"  ✓ 最终行数: {len(df_shuffled)}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return
    
    print("\n" + "=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='合并两个 Parquet 文件')
    parser.add_argument('--input1', type=str, default=r'data\processed\context\tmp\val_context.parquet',
                       help='第一个输入文件路径')
    parser.add_argument('--input2', type=str, default=r'data\processed\context\tmp\val_context2.parquet',
                       help='第二个输入文件路径')
    parser.add_argument('--output', type=str, default=r'data/processed/context/combined_context.parquet',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    input_path1 = Path(args.input1)
    input_path2 = Path(args.input2)
    output_path = Path(args.output)
    
    combine_parquet(input_path1, input_path2, output_path)


if __name__ == "__main__":
    main()
