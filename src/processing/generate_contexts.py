"""生成 Context 数据脚本

从 CSV 读取题目，批量调用检索 Pipeline，将检索结果与原始数据合并，存储为 Parquet。

流程：
1. 读取 CSV 文件（prompt, A, B, C, D, E, answer）
2. 构造查询字符串（"Question: {prompt}\nRetrieve background scientific knowledge to help answer."）
3. 批处理调用检索 Pipeline（batch_size=64, top_k=3）
4. 合并为 DataFrame（添加 C1, C2, C3 列）
5. 保存为 Parquet 文件

要求：
- 使用 pathlib 处理路径
- tqdm 显示进度条
- 完整日志输出
"""
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retrieval_pipeline import RetrievalPipeline


def load_csv(csv_path: Path) -> pd.DataFrame:
    """加载 CSV 文件
    
    Args:
        csv_path: CSV 文件路径
    
    Returns:
        DataFrame，包含 prompt, A, B, C, D, E, answer 列
    """
    print(f"[1/5] 加载 CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  ✓ 共 {len(df)} 行数据")
    print(f"  ✓ 列名: {list(df.columns)}")
    
    # 验证必需列
    required_cols = ['prompt']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必需列: {missing}")
    
    return df


def build_queries(df: pd.DataFrame) -> List[str]:
    """构造查询字符串
    
    Args:
        df: 包含 prompt 列的 DataFrame
    
    Returns:
        查询字符串列表
    """
    print(f"\n[2/5] 构造查询字符串...")
    queries = []
    for prompt in df['prompt']:
        query = f"Question: {prompt}\nRetrieve background scientific knowledge to help answer."
        queries.append(query)
    
    print(f"  ✓ 生成 {len(queries)} 个查询")
    print(f"  ✓ 示例查询:")
    print(f"    {queries[0][:100]}...")
    
    return queries


def run_retrieval_batch(
    queries: List[str],
    pipeline: RetrievalPipeline,
    top_k: int = 3,
    batch_size: int = 64
) -> List[List[str]]:
    """调用检索 Pipeline
    
    Args:
        queries: 查询字符串列表
        pipeline: RetrievalPipeline 实例
        top_k: 每个查询返回的 context 数量
        batch_size: 批处理大小
    
    Returns:
        每个查询的 contexts 列表: [[C1, C2, C3], [C1, C2, C3], ...]
    """
    print(f"\n[3/5] 调用检索 Pipeline (top_k={top_k}, batch_size={batch_size})...")
    print(f"  共 {len(queries)} 个查询，分 {(len(queries) + batch_size - 1) // batch_size} 个批次处理")
    
    all_contexts = []
    total_queries = len(queries)
    
    # 批处理查询
    for batch_start in tqdm(range(0, total_queries, batch_size), desc="检索批次", unit="batch"):
        batch_end = min(batch_start + batch_size, total_queries)
        batch_queries = queries[batch_start:batch_end]
        
        # 调用 pipeline（会根据 config.yaml 的 batch_size 再一次分batch）
        batch_results = pipeline.retrieve(batch_queries, verbose=False)
        
        # 提取每个查询的 top-k contexts
        for results in batch_results:
            # results 格式: [(chunk_id, score), ...]
            contexts = []
            for chunk_id, score in results[:top_k]:
                # 从 chunks_df 获取文本
                chunk_text = pipeline.reranker.get_chunk_text(chunk_id)
                if chunk_text:
                    contexts.append(chunk_text)
                else:
                    contexts.append("")  # 如果找不到，填充空字符串
            
            # 确保返回正好 top_k 个 contexts（不足时填充空字符串）
            while len(contexts) < top_k:
                contexts.append("")
            
            all_contexts.append(contexts[:top_k])
    
    print(f"  ✓ 检索完成，共 {len(all_contexts)} 个结果")
    
    return all_contexts


def combine_dataframe(
    original_df: pd.DataFrame,
    contexts_list: List[List[str]],
    top_k: int = 3
) -> pd.DataFrame:
    """合并原始 DataFrame 和检索结果
    
    Args:
        original_df: 原始 DataFrame
        contexts_list: 每个查询的 contexts 列表
        top_k: contexts 数量
    
    Returns:
        合并后的 DataFrame，包含 C1, C2, C3 列
    """
    print(f"\n[4/5] 合并数据...")
    
    # 创建副本
    result_df = original_df.copy()
    
    # 添加 C1, C2, C3 列
    for i in range(top_k):
        col_name = f"C{i+1}"
        result_df[col_name] = [contexts[i] for contexts in contexts_list]
    
    print(f"  ✓ 添加列: {[f'C{i+1}' for i in range(top_k)]}")
    print(f"  ✓ 最终列名: {list(result_df.columns)}")
    print(f"  ✓ 数据行数: {len(result_df)}")
    
    return result_df


def save_parquet(df: pd.DataFrame, output_path: Path):
    """保存为 Parquet 文件
    
    Args:
        df: 要保存的 DataFrame
        output_path: 输出文件路径
    """
    print(f"\n[5/5] 保存 Parquet: {output_path}")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    df.to_parquet(output_path, index=False, compression='snappy', engine='pyarrow')
    
    print(f"  ✓ 已保存 ({len(df)} 行)")
    print(f"  ✓ 文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成包含检索 contexts 的数据文件')
    parser.add_argument('--input', type=str, 
                       default=r'data\raw\kaggle-llm-science-exam\train.csv',
                       help='输入 CSV 文件路径')
    parser.add_argument('--output', type=str,
                       default=r'data\processed\context\val_context2.parquet',
                       help='输出 Parquet 文件路径')
    parser.add_argument('--top-k', type=int, default=3,
                       help='每个查询返回的 context 数量')
    parser.add_argument('--retrieval-batch-size', type=int, default=64,
                       help='检索批处理大小')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='检索 Pipeline 配置文件路径')
    
    args = parser.parse_args()
    
    # 转换为 Path
    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)
    
    print("=" * 70)
    print("生成 Context 数据脚本")
    print("=" * 70)
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"Top-K: {args.top_k}")
    print(f"检索批大小: {args.retrieval_batch_size}")
    print("=" * 70 + "\n")
    
    # Step 1: 加载 CSV
    df = load_csv(input_path)
    
    # Step 2: 构造查询
    queries = build_queries(df)
    
    # Step 2.5: 初始化检索 Pipeline
    print(f"\n[2.5/5] 初始化检索 Pipeline...")
    pipeline = RetrievalPipeline(
        config_path=str(config_path) if config_path.exists() else None,
        verbose=True
    )
    
    # Step 3: 调用检索 Pipeline
    contexts_list = run_retrieval_batch(queries, pipeline, args.top_k, args.retrieval_batch_size)
    
    # Step 4: 合并数据
    result_df = combine_dataframe(df, contexts_list, args.top_k)
    
    # Step 5: 保存
    save_parquet(result_df, output_path)
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    
    # 输出示例
    print("\n示例输出（第 1 行）:")
    print("-" * 70)
    first_row = result_df.iloc[0].to_dict()
    for key, value in first_row.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")
    print("-" * 70)


if __name__ == "__main__":
    main()
