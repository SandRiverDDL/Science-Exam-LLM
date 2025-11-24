"""Zero-Shot Answer Generation Pipeline

从离线 context parquet 文件读取 context，然后用 LLM 生成答案
"""
import sys
import pandas as pd
import json
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import argparse

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from modeling.qwen_zeroshot_abcde import QwenZeroShot

def load_config(config_path: str = 'config.yaml') -> Dict:
    """加载config.yaml配置
    
    Args:
        config_path: config.yaml路径
    
    Returns:
        配置字典
    """
    config_file = Path(config_path)
    if not config_file.exists():
        # 尝试今project_root中查找
        config_file = project_root.parent / 'config.yaml'
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_test_data(parquet_path: str) -> pd.DataFrame:
    """加载离线context数据
    
    Args:
        parquet_path: context parquet 路径
    
    Returns:
        DataFrame with columns: id, prompt, A, B, C, D, E, answer, C1, C2, C3
    """
    print(f"[数据加载] 读取离线context数据: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[数据加载] 共 {len(df)} 条数据")
    return df


def load_offline_contexts(
    df: pd.DataFrame,
    context_columns: List[str] = ['C1', 'C2', 'C3']
) -> List[str]:
    """从离线数据加载 contexts
    
    Args:
        df: 包含 context 列的 DataFrame
        context_columns: context 列名列表
    
    Returns:
        context 字符串列表
    """
    print(f"\n[加载] 从离线数据加载 contexts...")
    
    contexts = []
    for idx, row in df.iterrows():
        context_parts = []
        for rank, col in enumerate(context_columns, 1):
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                context_parts.append(f"[Document {rank}]\n{row[col]}")
        
        context = "\n\n".join(context_parts)
        contexts.append(context)
    
    print(f"[加载] 完成，共加载 {len(contexts)} 个 contexts")
    return contexts


def generate_answers_batch(
    questions: List[str],
    options_list: List[Dict[str, str]],
    contexts: List[str],
    generator: QwenZeroShot,
    batch_size: int = 1
) -> List[str]:
    """批量生成答案
    
    Args:
        questions: 问题列表
        options_list: 选项列表
        contexts: context 列表
        generator: Qwen 生成器
        batch_size: 生成批处理大小
    
    Returns:
        答案列表（'A', 'B', 'C', 'D', 'E'）
    """
    print(f"\n[生成] 开始批量生成答案 ({len(questions)} questions, batch_size={batch_size})...")
    
    # 构建所有 prompts
    print(f"  [生成] 构建 prompts...")
    prompts = []
    for question, options, context in zip(questions, options_list, contexts):
        prompt = generator.build_prompt(
            question=question,
            options=options,
            context=context,
        )
        prompts.append(prompt)
    
    # 批量生成
    print(f"  [生成] 调用 LLM 生成...")
    responses = generator.generate(prompts, batch_size=batch_size)
    
    # 解析答案
    print(f"  [生成] 解析答案...")
    answers = []
    invalid_answers = []
    for idx, response in enumerate(responses):
        answer = response if response else 'A'
        # 检查是否为有效答案
        if answer not in ['A', 'B', 'C', 'D', 'E']:
            invalid_answers.append((idx, answer))
        answers.append(answer)
    
    if invalid_answers:
        print(f"  [警告] 发现 {len(invalid_answers)} 个无效答案:")
        for idx, answer in invalid_answers:
            print(f"    样本 {idx}: {repr(answer)}")
    
    print(f"[生成] 完成，共生成 {len(answers)} 个答案")
    return answers


def main():
    """Main pipeline"""
    # 加载config.yaml
    config = load_config()
    qwen_cfg = config.get('qwen', {})
    zero_shot_cfg = config.get('zero_shot', {})
    
    # 所有参数从config.yaml读取
    context_parquet = zero_shot_cfg.get('context_parquet', 'data/processed/context/val_context.parquet')
    output_path = zero_shot_cfg.get('output', 'output/predictions_zeroshot.csv')
    generation_batch_size = zero_shot_cfg.get('generation_batch_size', 1)
    
    class Args:
        pass
    args = Args()
    args.context_parquet = context_parquet
    args.output = output_path
    args.generation_batch_size = generation_batch_size
    args.max_samples = None
    
    print("\n" + "="*80)
    print("Zero-Shot Answer Generation Pipeline")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Context Parquet: {args.context_parquet}")
    print(f"  Output: {args.output}")
    print(f"  Generation Batch Size: {args.generation_batch_size}")
    print(f"  Max Samples: {args.max_samples or 'All'}")
    
    # ===== Step 1: 加载离线数据 =====
    print("\n" + "="*80)
    print("Step 1: 加载离线数据")
    print("="*80)
    test_df = load_test_data(args.context_parquet)
    
    if args.max_samples:
        test_df = test_df.head(args.max_samples)
        print(f"[限制] 仅处理前 {args.max_samples} 条数据")
    
    # 提取问题和选项
    questions = test_df['prompt'].tolist()
    options_list = []
    for _, row in test_df.iterrows():
        options = {
            'A': row['A'],
            'B': row['B'],
            'C': row['C'],
            'D': row['D'],
            'E': row['E']
        }
        options_list.append(options)
    
    # ===== Step 2: 加载离线 Contexts =====
    print("\n" + "="*80)
    print("Step 2: 加载离线 Contexts")
    print("="*80)
    contexts = load_offline_contexts(test_df)
    
    # ===== Step 3: 加载 LLM 模型 =====
    print("\n" + "="*80)
    print("Step 3: 加载 LLM 模型")
    print("="*80)
    generator = QwenZeroShot(
        model_id=qwen_cfg.get('model_id', "Qwen/Qwen3-8B"),
        device_map=qwen_cfg.get('device_map', 'auto')
    )
    
    # ===== Step 4: 批量生成答案 =====
    print("\n" + "="*80)
    print("Step 4: 批量生成答案")
    print("="*80)
    answers = generate_answers_batch(
        questions,
        options_list,
        contexts,
        generator,
        batch_size=args.generation_batch_size
    )
    
    # ===== Step 5: 评估与保存 =====
    print("\n" + "="*80)
    print("Step 5: 评估与保存")
    print("="*80)
    
    # 计算正确率
    correct_mask = test_df['answer'] == answers
    accuracy = correct_mask.sum() / len(test_df)
    
    print(f"\n[评估] 正确率: {accuracy*100:.2f}% ({correct_mask.sum()}/{len(test_df)})")
    
    # 找出所有错误的样本
    error_indices = [i for i, is_correct in enumerate(correct_mask) if not is_correct]
    error_df = test_df.iloc[error_indices].copy()
    error_df['prediction'] = [answers[i] for i in error_indices]
    
    # 保存错误样本到CSV
    error_output_path = Path('output/zeroshot.csv')
    error_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 选择需要的列
    cols_to_save = ['prompt','A', 'B', 'C', 'D', 'E', 'answer', 'prediction', 'C1', 'C2', 'C3']
    error_save_df = error_df[[col for col in cols_to_save if col in error_df.columns]]
    error_save_df.to_csv(error_output_path, index=False)
    
    print(f"[保存] ✅ 错误样本已保存到: {error_output_path}")
    print(f"[保存] 共 {len(error_df)} 条错误结果")
    
    # 显示答案分布
    answer_counts = pd.Series(answers).value_counts()
    print(f"\n[统计] 答案分布:")
    for answer in ['A', 'B', 'C', 'D', 'E']:
        count = answer_counts.get(answer, 0)
        print(f"  {answer}: {count} ({count/len(answers)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("Pipeline 完成 ✅")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
