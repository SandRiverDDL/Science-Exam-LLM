"""Zero-Shot Answer Generation Pipeline

从 test.csv 读取问题，使用检索 pipeline 生成 context，然后用 LLM 生成答案
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

from retrieval.retrieval_pipeline import RetrievalPipeline
from modeling.qwen_generator import QwenGenerator


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


def load_test_data(csv_path: str) -> pd.DataFrame:
    """加载测试数据
    
    Args:
        csv_path: test.csv 路径
    
    Returns:
        DataFrame with columns: id, prompt, A, B, C, D, E
    """
    print(f"[数据加载] 读取测试数据: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[数据加载] 共 {len(df)} 条测试数据")
    return df


def retrieve_contexts_batch(
    questions: List[str],
    retrieval_pipeline: RetrievalPipeline,
    top_k: int = 5,
    batch_size: int = 8
) -> List[str]:
    """批量检索生成 contexts
    
    Args:
        questions: 问题列表
        retrieval_pipeline: 检索 pipeline
        top_k: 每个问题返回 top-k 个 chunks
        batch_size: 批处理大小
    
    Returns:
        context 字符串列表
    """
    print(f"\n[检索] 开始批量检索 ({len(questions)} queries, batch_size={batch_size})...")
    
    all_contexts = []
    
    # 分批检索
    for batch_start in range(0, len(questions), batch_size):
        batch_end = min(batch_start + batch_size, len(questions))
        batch_queries = questions[batch_start:batch_end]
        
        print(f"  [检索] Batch {batch_start//batch_size + 1}/{(len(questions)-1)//batch_size + 1}...", end='', flush=True)
        
        # 批量检索
        batch_results = retrieval_pipeline.retrieve(batch_queries, verbose=False)
        
        # 为每个 query 构建 context
        for query_idx, results in enumerate(batch_results):
            # results 是 [(chunk_id, score), ...] 的列表
            context_chunks = []
            for rank, (chunk_id, score) in enumerate(results[:top_k], 1):
                # 获取 chunk 文本
                try:
                    chunk_text = retrieval_pipeline.reranker.get_parent_chunk_text(chunk_id)
                    context_chunks.append(f"[Document {rank}]\n{chunk_text}")
                except:
                    continue
            
            # 拼接所有 chunks
            context = "\n\n".join(context_chunks)
            all_contexts.append(context)
        
        print(f" ✅")
    
    print(f"[检索] 完成，共生成 {len(all_contexts)} 个 contexts")
    return all_contexts


def generate_answers_batch(
    questions: List[str],
    options_list: List[Dict[str, str]],
    contexts: List[str],
    generator: QwenGenerator,
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
    for response in responses:
        answer = generator.parse_answer(response)
        answers.append(answer if answer else 'A')  # 默认 'A'
    
    print(f"[生成] 完成，共生成 {len(answers)} 个答案")
    return answers


def main():
    """Main pipeline"""
    # 加载config.yaml
    config = load_config()
    qwen_cfg = config.get('qwen', {})
    zero_shot_cfg = config.get('zero_shot', {})
    
    # 所有参数从config.yaml读取
    test_csv = zero_shot_cfg.get('test_csv', 'data/raw/kaggle-llm-science-exam/test.csv')
    output_path = zero_shot_cfg.get('output', 'output/predictions_zeroshot.csv')
    retrieval_batch_size = zero_shot_cfg.get('retrieval_batch_size', 8)
    generation_batch_size = zero_shot_cfg.get('generation_batch_size', 1)
    top_k = zero_shot_cfg.get('top_k', 5)
    
    class Args:
        pass
    args = Args()
    args.test_csv = test_csv
    args.output = output_path
    args.retrieval_batch_size = retrieval_batch_size
    args.generation_batch_size = generation_batch_size
    args.top_k = top_k
    args.max_samples = None
    args.no_reranker = False
    
    print("\n" + "="*80)
    print("Zero-Shot Answer Generation Pipeline")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Test CSV: {args.test_csv}")
    print(f"  Output: {args.output}")
    print(f"  Retrieval Batch Size: {args.retrieval_batch_size}")
    print(f"  Generation Batch Size: {args.generation_batch_size}")
    print(f"  Max Samples: {args.max_samples or 'All'}")
    
    # ===== Step 1: 加载测试数据 =====
    print("\n" + "="*80)
    print("Step 1: 加载测试数据")
    print("="*80)
    test_df = load_test_data(args.test_csv)
    
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
    
    # ===== Step 2: 初始化检索 Pipeline =====
    print("\n" + "="*80)
    print("Step 2: 初始化检索 Pipeline")
    print("="*80)
    
    reranker = None
    if not args.no_reranker:
        try:
            from rerank.jina_reranker import JinaReranker
            print("[Reranker] 加载 Jina Reranker...")
            reranker = JinaReranker()
            print("[Reranker] ✅")
        except Exception as e:
            print(f"[Reranker] ⚠️ 加载失败: {e}")
            reranker = None
    
    print("[Pipeline] 初始化 Retrieval Pipeline...")
    retrieval_pipeline = RetrievalPipeline(reranker_model=reranker)
    print("[Pipeline] ✅")
    
    # ===== Step 3: 批量检索生成 Contexts =====
    print("\n" + "="*80)
    print("Step 3: 批量检索生成 Contexts")
    print("="*80)
    contexts = retrieve_contexts_batch(
        questions,
        retrieval_pipeline,
        top_k=args.top_k,
        batch_size=args.retrieval_batch_size
    )
    
    # ===== Step 4: 加载 LLM 模型 =====
    print("\n" + "="*80)
    print("Step 4: 加载 LLM 模型")
    print("="*80)
    generator = QwenGenerator(
        model_id=qwen_cfg.get('model_id', "ISTA-DASLab/Qwen3-8B-Instruct-FPQuant-QAT-MXFP4-TEMP"),
        device_map=qwen_cfg.get('device_map', 'auto'),
        max_new_tokens=qwen_cfg.get('max_new_tokens', 1),  # 限制为1token
        gen_temperature=qwen_cfg.get('gen_temperature', 0.0),
        do_sample=qwen_cfg.get('do_sample', False),
        trust_remote_code=qwen_cfg.get('trust_remote_code', True)
    )
    
    # ===== Step 5: 批量生成答案 =====
    print("\n" + "="*80)
    print("Step 5: 批量生成答案")
    print("="*80)
    answers = generate_answers_batch(
        questions,
        options_list,
        contexts,
        generator,
        batch_size=args.generation_batch_size
    )
    
    # ===== Step 6: 保存结果 =====
    print("\n" + "="*80)
    print("Step 6: 保存结果")
    print("="*80)
    
    output_df = pd.DataFrame({
        'id': test_df['id'],
        'prediction': answers
    })
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"[保存] ✅ 结果已保存到: {output_path}")
    print(f"[保存] 共 {len(output_df)} 条预测结果")
    
    # 显示答案分布
    answer_counts = output_df['prediction'].value_counts()
    print(f"\n[统计] 答案分布:")
    for answer in ['A', 'B', 'C', 'D', 'E']:
        count = answer_counts.get(answer, 0)
        print(f"  {answer}: {count} ({count/len(output_df)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("Pipeline 完成 ✅")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
