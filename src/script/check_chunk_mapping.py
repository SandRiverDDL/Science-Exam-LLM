"""检查chunks.parquet和chunk_ids映射的对应关系"""
import json
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent.parent
chunks_parquet = project_root / 'data' / 'processed' / 'chunks.parquet'
chunk_ids_json = project_root / 'data' / 'faiss' / 'qwen3_fp16_ip_chunk_ids.json'

print("=" * 80)
print("检查Chunks映射关系")
print("=" * 80)

# 加载chunks.parquet
print("\n[1] 加载 chunks.parquet...")
try:
    df_chunks = pd.read_parquet(chunks_parquet)
    print(f"✅ 加载成功")
    print(f"   行数: {len(df_chunks):,}")
    print(f"   列名: {list(df_chunks.columns)}")
    print(f"\n   前5个chunk_id:")
    for i, cid in enumerate(df_chunks['chunk_id'].head()):
        print(f"     [{i}] {cid}")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 加载chunk_ids.json
print("\n[2] 加载 qwen3_fp16_ip_chunk_ids.json...")
try:
    with open(chunk_ids_json, 'r', encoding='utf-8') as f:
        chunk_ids_list = json.load(f)
    print(f"✅ 加载成功")
    print(f"   条数: {len(chunk_ids_list):,}")
    print(f"\n   前5个chunk_id:")
    for i, cid in enumerate(chunk_ids_list[:5]):
        print(f"     [{i}] {cid}")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 对比
print("\n[3] 对比分析...")

chunks_parquet_ids = set(df_chunks['chunk_id'].values)
json_ids = set(chunk_ids_list)

print(f"   chunks.parquet 中的chunk_id数: {len(chunks_parquet_ids):,}")
print(f"   JSON映射中的chunk_id数: {len(json_ids):,}")

# 检查是否一致
if chunks_parquet_ids == json_ids:
    print(f"\n✅ 映射完全匹配！")
else:
    missing_in_json = chunks_parquet_ids - json_ids
    extra_in_json = json_ids - chunks_parquet_ids
    
    if missing_in_json:
        print(f"\n⚠️  Parquet中但不在JSON的chunks: {len(missing_in_json):,}")
        print(f"   样本: {list(missing_in_json)[:5]}")
    
    if extra_in_json:
        print(f"\n⚠️  JSON中但不在Parquet的chunks: {len(extra_in_json):,}")
        print(f"   样本: {list(extra_in_json)[:5]}")

# 检查数据来源
print("\n[4] 检查chunk_id的来源...")
parquet_sources = df_chunks['chunk_id'].apply(lambda x: x.split(':')[0]).unique()
json_sources = set()
for cid in chunk_ids_list:
    source = cid.split(':')[0]
    json_sources.add(source)

print(f"   Parquet中的数据源: {sorted(parquet_sources)}")
print(f"   JSON中的数据源: {sorted(json_sources)}")

# 详细对比
print("\n[5] 详细分析...")
if chunks_parquet_ids != json_ids:
    print(f"\n⚠️  发现映射问题！")
    print(f"   建议: JSON映射文件应该来自chunks.parquet")
    print(f"   可能原因:")
    print(f"     1. 重新生成了chunks.parquet")
    print(f"     2. 之前的映射是从其他来源生成的")
    print(f"     3. 索引和映射需要重新同步")
else:
    print(f"\n✅ 映射一致，索引和chunks.parquet相匹配")

print("\n" + "=" * 80)
