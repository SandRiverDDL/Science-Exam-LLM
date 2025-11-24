"""批处理脚本：对现有的子chunks生成新的父chunks映射

流程：
1. 读取原始文档（parquet或json）
2. 使用ParentChunkGenerator独立生成父chunks (256 tokens)
3. 读取现有的子chunks信息
4. 计算子chunks到父chunks的映射关系
5. 输出新的索引文件
"""
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from chunking.parent_chunk_generator import ParentChunkGenerator


def load_documents(
    doc_path: str,
    sample_size: int = None
) -> List[Dict[str, Any]]:
    """加载原始文档
    
    支持格式：
    - .parquet (pandas)
    - .jsonl (JSON Lines)
    - .json (JSON array)
    
    Args:
        doc_path: 文档文件路径
        sample_size: 如果指定，仅加载前N条（用于测试）
    
    Returns:
        文档列表
    """
    doc_path = Path(doc_path)
    documents = []
    
    print(f"[加载] 读取文档: {doc_path}")
    
    if doc_path.suffix == '.parquet':
        df = pd.read_parquet(doc_path)
        if sample_size:
            df = df.head(sample_size)
        
        # 转换为列表
        for _, row in df.iterrows():
            doc = {
                'id': row.get('id') or row.get('doc_id') or '',
                'title': row.get('title', ''),
                'text': row.get('text') or row.get('content', '')
            }
            documents.append(doc)
    
    elif doc_path.suffix == '.jsonl':
        with open(doc_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                doc = json.loads(line)
                documents.append(doc)
    
    elif doc_path.suffix == '.json':
        with open(doc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                documents = data[:sample_size] if sample_size else data
            elif isinstance(data, dict):
                documents = [data]
    
    else:
        raise ValueError(f"不支持的文件格式: {doc_path.suffix}")
    
    print(f"[加载] 共加载 {len(documents)} 条文档")
    return documents


def load_child_chunks_info(
    chunks_file: str
) -> Dict[str, List[Tuple[int, int]]]:
    """加载现有的子chunks信息
    
    返回格式：
    {
        'doc_id_1': [(start_char_1, end_char_1), (start_char_2, end_char_2), ...],
        'doc_id_2': [...],
        ...
    }
    
    Args:
        chunks_file: chunks json/jsonl文件路径
    
    Returns:
        按doc_id组织的chunks信息
    """
    print(f"[加载] 读取子chunks: {chunks_file}")
    
    doc_chunks = {}
    
    chunks_file = Path(chunks_file)
    
    if chunks_file.suffix == '.jsonl':
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                doc_id = chunk.get('doc_id')
                start = chunk.get('start_char') or chunk.get('child_start', 0)
                end = chunk.get('end_char') or chunk.get('child_end', 0)
                
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                
                doc_chunks[doc_id].append((start, end))
    
    elif chunks_file.suffix == '.json':
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            if isinstance(chunks, list):
                for chunk in chunks:
                    doc_id = chunk.get('doc_id')
                    start = chunk.get('start_char') or chunk.get('child_start', 0)
                    end = chunk.get('end_char') or chunk.get('child_end', 0)
                    
                    if doc_id not in doc_chunks:
                        doc_chunks[doc_id] = []
                    
                    doc_chunks[doc_id].append((start, end))
    
    print(f"[加载] 共加载 {sum(len(v) for v in doc_chunks.values())} 个子chunks")
    return doc_chunks


def generate_parent_chunks_batch(
    documents: List[Dict],
    existing_chunks: List[Dict],
    tokenizer,
    parent_size: int = 256
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """批量生成父chunks并修改持的chunks
    
    流程：
    1. 为每个文档生成不同最排序特序牙消parent_chunks
    2. 根据父chunks手次位置映射子chunks
    3. 修改优祖的chunks：添加parent_chunk_idx字段
    
    Args:
        documents: 文档列表
        existing_chunks: 现有的子chunks列表
        tokenizer: transformers tokenizer
        parent_size: 父chunk目标大小
    
    Returns:
        (parent_chunks, mappings, chunks_with_parent)
    """
    generator = ParentChunkGenerator(tokenizer, parent_size=parent_size)
    
    all_parent_chunks = []
    all_mappings = []
    chunks_with_parent = []
    
    # 为了执行映射，需要查找初始文档数据
    doc_map = {doc.get('id') or doc.get('doc_id'): doc for doc in documents}
    
    # 为了执行映射，按doc_id整理了子chunks
    chunks_by_doc = {}
    for chunk in existing_chunks:
        doc_id = chunk.get('doc_id')
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)
    
    print(f"\n[生成] 开始批量生成父chunks (parent_size={parent_size} tokens)...")
    
    for doc in tqdm(documents, desc="生成父chunks"):
        doc_id = doc.get('id') or doc.get('doc_id', '')
        title = doc.get('title', '')
        text = doc.get('text') or doc.get('content', '')
        
        if not text or not doc_id:
            continue
        
        # 特别注意：不修改text，保证索引一致性
        # 生成这个文档的所有父chunks
        parent_chunks = generator.generate_parent_chunks(
            doc_id=doc_id,
            text=text,
            title=title
        )
        
        all_parent_chunks.extend(parent_chunks)
        
        # 映射这个文档的子chunks到父chunks
        if doc_id in chunks_by_doc:
            child_positions = [
                (c.get('start_char') or c.get('child_start', 0),
                 c.get('end_char') or c.get('child_end', 0))
                for c in chunks_by_doc[doc_id]
            ]
            
            mapping = generator.map_child_to_parent(parent_chunks, child_positions)
            
            # 修改子chunks：添加parent_chunk_idx字段
            parent_chunk_offset = len(all_parent_chunks) - len(parent_chunks)  # 父chunks的开始索引
            
            for child_chunk, parent_idx in zip(chunks_by_doc[doc_id], mapping):
                # 复制原子chunk的所有字段
                new_chunk = child_chunk.copy()
                
                # 添加父chunk索引（绝对索引）
                if parent_idx >= 0:
                    new_chunk['parent_chunk_idx'] = parent_chunk_offset + parent_idx
                else:
                    new_chunk['parent_chunk_idx'] = -1  # 不被任何父chunk包含
                
                chunks_with_parent.append(new_chunk)
                
                # 记录映射（便统计）
                all_mappings.append({
                    'chunk_id': child_chunk.get('chunk_id'),
                    'doc_id': doc_id,
                    'parent_chunk_idx': new_chunk['parent_chunk_idx']
                })
    
    print(f"[生成] 完成！共生成 {len(all_parent_chunks)} 个父chunks")
    
    return all_parent_chunks, all_mappings, chunks_with_parent


def save_results(
    parent_chunks: List[Dict],
    mappings: List[Dict],
    chunks_with_parent: List[Dict],
    output_dir: str
):
    """保存结果
    
    Args:
        parent_chunks: 父chunks列表
        mappings: 映射关系列表（已【写入chunks_with_parent）
        chunks_with_parent: 修改后的chunks列表（包含父chunk索引）
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存父chunks为parquet
    parent_chunks_file = output_dir / "parent_chunks.parquet"
    print(f"\n[保存] 父chunks -> {parent_chunks_file}")
    
    parent_df = pd.DataFrame(parent_chunks)
    parent_df.to_parquet(parent_chunks_file, index=False)
    
    # 2. 保存修改后的chunks为parquet（此为最突出的修改）
    chunks_with_parent_file = output_dir / "chunks_with_parent.parquet"
    print(f"[保存] 修改后的chunks -> {chunks_with_parent_file}")
    
    chunks_df = pd.DataFrame(chunks_with_parent)
    chunks_df.to_parquet(chunks_with_parent_file, index=False)
    
    # 3. 统计信息
    stats = compute_statistics(parent_chunks, chunks_with_parent, mappings)
    
    stats_file = output_dir / "statistics.json"
    print(f"[保存] 统计信息 -> {stats_file}")
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n[统计] 父子Chunk关系统计\n")
    print(f"  数量统计:")
    print(f"    - 总父chunks数: {stats.get('total_parent_chunks', 0)}")
    print(f"    - 总子chunks数: {stats.get('total_child_chunks', 0)}")
    print(f"    - 涉及文档数: {stats.get('unique_docs', 0)}")
    
    print(f"\n  父chunk大小统计 (tokens):")
    print(f"    - 平均: {stats.get('avg_parent_tokens', 0):.2f}")
    print(f"    - 最小: {stats.get('min_parent_tokens', 0)}")
    print(f"    - 最大: {stats.get('max_parent_tokens', 0)}")
    
    print(f"\n  子chunk覆盖统计:")
    print(f"    - 完全包含的子chunks: {stats.get('fully_contained_count', 0)} ({stats.get('fully_contained_ratio', 0)*100:.2f}%)")
    print(f"    - 部分包含的子chunks: {stats.get('total_child_chunks', 0) - stats.get('fully_contained_count', 0) - stats.get('unfound_count', 0)}")
    print(f"    - 未被包含的子chunks: {stats.get('unfound_count', 0)}")
    print(f"    - 总体覆盖率: {stats.get('coverage_ratio', 0)*100:.2f}%")
    
    print(f"\n  父子包含关系统计:")
    print(f"    - 平均每个父chunk包含: {stats.get('avg_children_per_parent', 0):.2f} 个子chunks")
    print(f"    - 最多包含: {stats.get('max_children_per_parent', 0)} 个")
    print(f"    - 最少包含: {stats.get('min_children_per_parent', 0)} 个")
    print(f"    - 有子chunks的父chunks: {stats.get('parent_with_children', 0)}")
    print(f"    - 无子chunks的父chunks: {stats.get('parent_without_children', 0)}")


def compute_statistics(
    parent_chunks: List[Dict],
    chunks_with_parent: List[Dict],
    mappings: List[Dict]
) -> Dict:
    """计算统计信息
    
    Args:
        parent_chunks: 父chunks列表
        chunks_with_parent: 修改后的chunks列表
        mappings: 映射列表
    
    Returns:
        统计信息字典
    """
    stats = {}
    
    # 1. 数量统计
    stats['total_parent_chunks'] = len(parent_chunks)
    stats['total_child_chunks'] = len(chunks_with_parent)
    stats['unique_docs'] = len(set(c['doc_id'] for c in parent_chunks))
    
    # 2. 父chunk大小统计
    if parent_chunks:
        token_counts = [c['token_count'] for c in parent_chunks]
        stats['avg_parent_tokens'] = round(sum(token_counts) / len(token_counts), 2)
        stats['min_parent_tokens'] = min(token_counts)
        stats['max_parent_tokens'] = max(token_counts)
    
    # 3. 映射统计
    parent_child_count = {}  # parent_idx -> count
    fully_contained_count = 0
    unfound_count = 0  # 未被任何父chunk包含的子chunks
    
    for child_chunk in chunks_with_parent:
        parent_idx = child_chunk.get('parent_chunk_idx')
        
        if parent_idx is not None and parent_idx >= 0:
            parent_child_count[parent_idx] = parent_child_count.get(parent_idx, 0) + 1
            
            # 检查是否完全被父chunk包含
            if parent_idx < len(parent_chunks):
                parent = parent_chunks[parent_idx]
                child_start = child_chunk.get('start_char') or child_chunk.get('child_start', 0)
                child_end = child_chunk.get('end_char') or child_chunk.get('child_end', 0)
                
                if parent['start_char'] <= child_start and child_end <= parent['end_char']:
                    fully_contained_count += 1
        else:
            # parent_idx == -1 或 None，未被任何父chunk包含
            unfound_count += 1
    
    # 4. 覆盖比例
    if chunks_with_parent:
        stats['fully_contained_ratio'] = round(fully_contained_count / len(chunks_with_parent), 4)
        stats['fully_contained_count'] = fully_contained_count
        stats['unfound_count'] = unfound_count
        stats['coverage_ratio'] = round((len(chunks_with_parent) - unfound_count) / len(chunks_with_parent), 4)
    
    # 5. 平均每个父chunk包含的子chunks数
    if parent_child_count:
        total_children = sum(parent_child_count.values())
        parent_with_children = len(parent_child_count)
        stats['avg_children_per_parent'] = round(total_children / parent_with_children, 2)
        stats['max_children_per_parent'] = max(parent_child_count.values())
        stats['min_children_per_parent'] = min(parent_child_count.values())
        stats['parent_with_children'] = parent_with_children
        stats['parent_without_children'] = len(parent_chunks) - parent_with_children
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='生成新的父chunks (256 tokens) 并映射到现有子chunks'
    )
    
    parser.add_argument(
        '--doc-path',
        type=str,
        default='data\processed\documents_cleaned.parquet',
        help='原始文档文件路径 (parquet/jsonl/json)'
    )
    
    parser.add_argument(
        '--chunks-path',
        type=str,
        default='data/processed/chunks.parquet',
        help='现有的子chunks文件路径（必须）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/chunks_refined.parquet',
        help='输出目录'
    )
    
    parser.add_argument(
        '--parent-size',
        type=int,
        default=256,
        help='父chunk目标大小（tokens）'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='仅处理前N条文档（用于测试）'
    )

    
    args = parser.parse_args()
    
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    
    # 加载文档
    documents = load_documents(args.doc_path, sample_size=args.sample_size)
    
    # 加载现有的子chunks（必须）
    print(f"[\u52a0\u8f7d] \u8bfb\u53d6\u73b0\u6709\u7684\u5b50chunks: {args.chunks_path}")
    chunks_df = pd.read_parquet(args.chunks_path)
    existing_chunks = chunks_df.to_dict('records')
    print(f"[\u52a0\u8f7d] \u5171\u52a0\u8f7d {len(existing_chunks)} \u4e2a\u5b50chunks")
    
    # 生成父chunks
    parent_chunks, mappings, chunks_with_parent = generate_parent_chunks_batch(
        documents=documents,
        existing_chunks=existing_chunks,
        tokenizer=tokenizer,
        parent_size=args.parent_size
    )
    
    # 保存结果
    save_results(parent_chunks, mappings, chunks_with_parent, args.output_dir)
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
