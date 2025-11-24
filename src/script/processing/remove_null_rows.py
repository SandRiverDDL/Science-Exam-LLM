"""移除包含空值的行

功能：
1. 读取 Parquet 或 CSV 文件
2. 删除任何列包含空值的行
3. 保存为 {源文件名}_without_null.{后缀}
4. 显示删除前后的行数对比
"""
import sys
from pathlib import Path

import pandas as pd


def remove_null_rows(input_path: Path, output_path: Path = None) -> int:
    """移除包含空值的行
    
    Args:
        input_path: 输入文件路径（.parquet 或 .csv）
        output_path: 输出文件路径，如果为None则自动生成
    
    Returns:
        删除的行数
    """
    print("=" * 80)
    print(f"处理文件: {input_path}")
    print("=" * 80)
    
    # 检查文件是否存在
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return 0
    
    # 确定文件类型
    suffix = input_path.suffix.lower()
    if suffix not in ['.parquet', '.csv', '.pq']:
        print(f"❌ 不支持的文件格式: {suffix}（仅支持 .parquet/.pq 或 .csv）")
        return 0
    
    # 加载文件
    print(f"\n[1/3] 加载文件...")
    try:
        if suffix in ['.parquet', '.pq']:
            df = pd.read_parquet(input_path)
            file_type = 'parquet'
        else:
            df = pd.read_csv(input_path)
            file_type = 'csv'
        
        original_rows = len(df)
        print(f"  ✓ 加载成功")
        print(f"  ✓ 原始行数: {original_rows}")
        print(f"  ✓ 列数: {len(df.columns)}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return 0
    
    # 检查空值
    print(f"\n[2/3] 检查空值...")
    null_rows = df.isnull().any(axis=1).sum()
    print(f"  包含空值的行数: {null_rows}")
    
    if null_rows == 0:
        print(f"  ✓ 没有空值，无需处理")
        return 0
    
    # 删除空值行
    print(f"\n[3/3] 删除空值行...")
    df_clean = df.dropna()
    final_rows = len(df_clean)
    deleted_rows = original_rows - final_rows
    
    print(f"  删除行数: {deleted_rows}")
    print(f"  保留行数: {final_rows}")
    print(f"  删除比例: {(deleted_rows/original_rows)*100:.2f}%")
    
    # 生成输出路径
    if output_path is None:
        stem = input_path.stem
        parent = input_path.parent
        if file_type == 'parquet':
            output_path = parent / f"{stem}_without_null.parquet"
        else:
            output_path = parent / f"{stem}_without_null.csv"
    
    # 保存文件
    print(f"\n[4/4] 保存文件...")
    try:
        if file_type == 'parquet':
            df_clean.to_parquet(output_path, index=False, engine='pyarrow')
        else:
            df_clean.to_csv(output_path, index=False)
        
        print(f"  ✓ 保存成功")
        print(f"  ✓ 输出路径: {output_path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return 0
    
    print("\n" + "=" * 80)
    return deleted_rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description='移除包含空值的行')
    parser.add_argument('--input', type=str,
                       help='输入文件路径（.parquet 或 .csv）', default=r"data\processed\context\train_context.parquet")
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径，如不指定则自动生成')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    remove_null_rows(input_path, output_path)


if __name__ == "__main__":
    main()
