"""查看chunks.parquet的前10行数据"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    import pyarrow.parquet as pq
    import pandas as pd
except ImportError:
    print("❌ 缺少依赖: pyarrow 或 pandas")
    print("   请运行: pip install pyarrow pandas")
    exit(1)

chunks_path = project_root / 'data' / 'processed' / 'chunks.parquet'

print("=" * 100)
print("查看 chunks.parquet 前10行数据")
print("=" * 100)

if not chunks_path.exists():
    print(f"❌ 文件不存在: {chunks_path}")
    exit(1)

print(f"\n📄 文件路径: {chunks_path}")
print(f"   文件大小: {chunks_path.stat().st_size / (1024**2):.2f} MB")

# 加载Parquet文件
print("\n[1] 加载文件...")
try:
    # 使用PyArrow读取获取基本信息
    table = pq.read_table(chunks_path)
    print(f"✅ 加载成功")
    print(f"\n[2] 文件结构信息:")
    print(f"   总行数: {table.num_rows:,}")
    print(f"   列数: {len(table.column_names)}")
    print(f"   列名: {table.column_names}")
    
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 使用pandas显示数据
print(f"\n[3] 前10行数据:")
try:
    df = pd.read_parquet(chunks_path)
    
    # 设置显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', None)
    
    # 显示前10行
    print(df.head(10).to_string())
    
except Exception as e:
    print(f"❌ 显示失败: {e}")
    exit(1)

# 显示数据类型
print(f"\n[4] 数据类型:")
print(df.dtypes)

# 显示统计信息
print(f"\n[5] 统计信息:")
print(f"   总记录数: {len(df):,}")
print(f"   chunk_id 唯一值数: {df['chunk_id'].nunique():,}")
print(f"   doc_id 唯一值数: {df['doc_id'].nunique():,}")

# 显示数据来源
print(f"\n[6] 数据来源分析:")
sources = df['chunk_id'].apply(lambda x: x.split(':')[0]).value_counts()
print(sources)

# 显示样本chunk_id
print(f"\n[7] chunk_id 样本:")
for i, cid in enumerate(df['chunk_id'].head(10)):
    print(f"   [{i}] {cid}")

print("\n" + "=" * 100)
