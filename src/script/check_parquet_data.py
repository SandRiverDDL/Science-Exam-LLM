"""检查 Parquet 文件内容"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import pyarrow.parquet as pq


def check_parquet_file(parquet_path: str):
    """检查单个 Parquet 文件的内容
    
    Args:
        parquet_path: Parquet 文件路径
    """
    if not os.path.exists(parquet_path):
        print(f"❌ 文件不存在: {parquet_path}")
        return
    
    print("=" * 80)
    print(f"检查文件: {parquet_path}")
    print("=" * 80)
    
    # 1. 文件信息
    file_size = os.path.getsize(parquet_path)
    print(f"\n📁 文件信息:")
    print(f"  大小: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    # 2. Schema 信息
    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema
    print(f"\n📋 Schema:")
    for i, field in enumerate(schema):
        print(f"  {i+1}. {field.name:20s}")
    
    # 3. 行数统计
    df = pd.read_parquet(parquet_path)
    print(f"\n📊 统计:")
    print(f"  总行数: {len(df):,}")
    
    # 4. 列详细信息
    print(f"\n🔍 各列详情:")
    for col in df.columns:
        col_data = df[col]
        print(f"\n  [{col}]")
        print(f"    类型: {col_data.dtype}")
        print(f"    非空: {col_data.notna().sum():,} / {len(col_data):,}")
        
        # 根据类型显示不同的统计
        if col_data.dtype == 'object':
            if isinstance(col_data.iloc[0], str):
                # 字符串列
                lens = col_data.str.len()
                print(f"    长度: min={lens.min()}, max={lens.max()}, avg={lens.mean():.1f}")
            elif isinstance(col_data.iloc[0], list):
                # 列表列
                lens = col_data.apply(len)
                print(f"    列表长度: min={lens.min()}, max={lens.max()}, avg={lens.mean():.1f}")
        elif pd.api.types.is_numeric_dtype(col_data):
            # 数值列
            print(f"    范围: {col_data.min()} ~ {col_data.max()}")
            print(f"    平均: {col_data.mean():.2f}")
    
    # 5. 前3行示例
    print(f"\n📝 前3行示例:")
    for idx in range(min(3, len(df))):
        print(f"\n  --- 行 {idx+1} ---")
        for col in df.columns:
            value = df.iloc[idx][col]
            if isinstance(value, list):
                # 列表类型显示前5个元素
                if len(value) > 5:
                    print(f"    {col}: [{value[0]}, {value[1]}, ..., {value[-1]}] (长度={len(value)})")
                else:
                    print(f"    {col}: {value}")
            elif isinstance(value, str) and len(value) > 100:
                # 长字符串截断显示
                print(f"    {col}: {value[:100]}... (长度={len(value)})")
            else:
                print(f"    {col}: {value}")
    
    print("\n" + "=" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查 Parquet 文件内容')
    parser.add_argument('files', nargs='+', help='Parquet 文件路径')
    parser.add_argument('--dir', help='或者指定目录（检查该目录下所有 .parquet 文件）')
    
    args = parser.parse_args()
    
    files_to_check = []
    
    if args.dir:
        # 检查目录下所有 parquet 文件
        dir_path = args.dir
        if os.path.isdir(dir_path):
            files_to_check = list(Path(dir_path).glob('*.parquet'))
            files_to_check.sort()
        else:
            print(f"❌ 目录不存在: {dir_path}")
            return
    else:
        # 检查指定的文件
        files_to_check = args.files
    
    if not files_to_check:
        print("❌ 没有找到要检查的文件")
        return
    
    print(f"\n找到 {len(files_to_check)} 个文件")
    
    for file_path in files_to_check:
        check_parquet_file(str(file_path))
        print()


if __name__ == "__main__":
    # 默认检查示例文件
    default_files = [
        "data/processed/parquet/documents/docs_1.parquet",
        "data/processed/parquet/chunks/chunks_1.parquet",
    ]
    
    import sys
    if len(sys.argv) == 1:
        # 没有参数时检查默认文件
        print("检查默认文件（如需检查其他文件，请使用命令行参数）")
        print(f"用法: python {sys.argv[0]} <file1.parquet> [file2.parquet ...]")
        print(f"或者: python {sys.argv[0]} --dir <directory>")
        print()
        
        for file_path in default_files:
            if os.path.exists(file_path):
                check_parquet_file(file_path)
            else:
                print(f"⚠️  默认文件不存在: {file_path}\n")
    else:
        main()
