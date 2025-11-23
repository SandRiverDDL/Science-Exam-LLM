"""检查和解压FAISS索引文件

功能：
1. 解压LZ4压缩的索引文件
2. 验证索引有效性
3. 与chunks.parquet对比，检查chunk_id覆盖情况
4. 生成详细的检查报告
"""
import os
import sys
import json
import lz4.frame
import faiss
import pandas as pd
from pathlib import Path
from typing import Dict, Set, Tuple

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))


def decompress_index(compressed_path: str, output_path: str) -> bool:
    """解压LZ4压缩的索引文件
    
    Args:
        compressed_path: 压缩文件路径(.lz4)
        output_path: 输出路径(.faiss)
    
    Returns:
        是否成功解压
    """
    try:
        print(f"[decompress] 解压索引文件...")
        print(f"  输入: {compressed_path}")
        
        if not os.path.exists(compressed_path):
            print(f"  ❌ 文件不存在: {compressed_path}")
            return False
        
        compressed_size = os.path.getsize(compressed_path) / (1024**2)
        print(f"  压缩文件大小: {compressed_size:.2f} MB")
        
        # 解压
        with lz4.frame.open(compressed_path, 'rb') as f_in:
            data = f_in.read()
        
        # 保存解压后的文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f_out:
            f_out.write(data)
        
        uncompressed_size = os.path.getsize(output_path) / (1024**2)
        ratio = (1 - compressed_size / uncompressed_size) * 100 if uncompressed_size > 0 else 0
        
        print(f"  ✅ 解压成功")
        print(f"  原始文件大小: {uncompressed_size:.2f} MB")
        print(f"  压缩率: {ratio:.1f}%")
        
        return True
    except Exception as e:
        print(f"  ❌ 解压失败: {e}")
        return False


def load_faiss_index(index_path: str) -> Tuple[faiss.Index, bool]:
    """加载FAISS索引
    
    Args:
        index_path: 索引文件路径
    
    Returns:
        (索引对象, 是否成功)
    """
    try:
        print(f"[load_index] 加载FAISS索引...")
        print(f"  路径: {index_path}")
        
        if not os.path.exists(index_path):
            print(f"  ❌ 索引文件不存在")
            return None, False
        
        index = faiss.read_index(index_path)
        
        print(f"  ✅ 索引加载成功")
        print(f"  索引类型: {type(index).__name__}")
        print(f"  向量维度: {index.ntotal if hasattr(index, 'ntotal') else 'N/A'}")
        
        return index, True
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None, False


def load_chunk_id_mapping(mapping_path: str) -> Dict[int, str]:
    """加载chunk_id映射
    
    Args:
        mapping_path: 映射文件路径
    
    Returns:
        chunk_id映射字典 (faiss_id -> chunk_id)
    """
    try:
        print(f"[load_mapping] 加载chunk_id映射...")
        print(f"  路径: {mapping_path}")
        
        if not os.path.exists(mapping_path):
            print(f"  ⚠️  映射文件不存在: {mapping_path}")
            return {}
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            chunk_ids = json.load(f)
        
        # 转换为dict (faiss_id -> chunk_id)
        mapping = {i: cid for i, cid in enumerate(chunk_ids)}
        
        print(f"  ✅ 加载成功")
        print(f"  映射条数: {len(mapping):,}")
        
        return mapping
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return {}


def load_chunks_parquet(chunks_path: str) -> Tuple[pd.DataFrame, bool]:
    """加载chunks.parquet文件
    
    Args:
        chunks_path: chunks.parquet路径
    
    Returns:
        (DataFrame, 是否成功)
    """
    try:
        print(f"[load_chunks] 加载chunks.parquet...")
        print(f"  路径: {chunks_path}")
        
        if not os.path.exists(chunks_path):
            print(f"  ❌ 文件不存在: {chunks_path}")
            return None, False
        
        df = pd.read_parquet(chunks_path)
        
        print(f"  ✅ 加载成功")
        print(f"  总chunks: {len(df):,}")
        print(f"  列名: {list(df.columns)}")
        
        return df, True
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None, False


def check_index_validity(index: faiss.Index, expected_count: int) -> Dict[str, any]:
    """检查索引有效性
    
    Args:
        index: FAISS索引对象
        expected_count: 预期的向量数量
    
    Returns:
        检查结果字典
    """
    print(f"\n[validity_check] 检查索引有效性...")
    
    results = {}
    
    # 检查向量数量
    if hasattr(index, 'ntotal'):
        actual_count = index.ntotal
        results['vector_count'] = actual_count
        results['expected_count'] = expected_count
        
        if actual_count == expected_count:
            print(f"  ✅ 向量数量匹配: {actual_count:,}")
        elif actual_count < expected_count:
            diff = expected_count - actual_count
            print(f"  ⚠️  向量数量不足: {actual_count:,} (缺少 {diff:,})")
            results['missing_vectors'] = diff
        else:
            print(f"  ⚠️  向量数量超出: {actual_count:,}")
            results['extra_vectors'] = actual_count - expected_count
    
    # 检查向量维度
    if hasattr(index, 'd'):
        print(f"  ✅ 向量维度: {index.d}")
        results['dimension'] = index.d
    
    # 尝试检索测试
    try:
        if index.ntotal > 0:
            # 创建一个随机向量进行测试检索
            import numpy as np
            test_vector = np.random.randn(1, index.d).astype(np.float32)
            distances, indices = index.search(test_vector, min(5, index.ntotal))
            
            print(f"  ✅ 检索测试成功 (返回 {indices.shape[1]} 个结果)")
            results['search_test'] = 'passed'
    except Exception as e:
        print(f"  ❌ 检索测试失败: {e}")
        results['search_test'] = f'failed: {e}'
    
    return results


def compare_chunk_coverage(
    chunk_ids_from_mapping: Set[str],
    chunk_ids_from_parquet: Set[str]
) -> Dict[str, any]:
    """比较chunk覆盖情况
    
    Args:
        chunk_ids_from_mapping: 映射文件中的chunk_id集合
        chunk_ids_from_parquet: parquet文件中的chunk_id集合
    
    Returns:
        比较结果字典
    """
    print(f"\n[coverage_check] 检查chunk覆盖情况...")
    
    results = {}
    
    print(f"  映射中的chunks: {len(chunk_ids_from_mapping):,}")
    print(f"  Parquet中的chunks: {len(chunk_ids_from_parquet):,}")
    
    # 完全覆盖？
    if chunk_ids_from_mapping == chunk_ids_from_parquet:
        print(f"  ✅ 完全覆盖 - 所有chunks都被索引")
        results['coverage'] = 'complete'
    else:
        # 计算缺失和多余
        missing = chunk_ids_from_parquet - chunk_ids_from_mapping
        extra = chunk_ids_from_mapping - chunk_ids_from_parquet
        
        if missing:
            print(f"  ⚠️  缺失chunks: {len(missing):,}")
            results['missing_chunks'] = len(missing)
            results['missing_chunk_samples'] = list(missing)[:5]  # 显示前5个
            if len(missing) <= 20:
                print(f"     {list(missing)}")
            else:
                print(f"     {list(missing)[:10]}...")
        
        if extra:
            print(f"  ⚠️  多余chunks: {len(extra):,}")
            results['extra_chunks'] = len(extra)
            results['extra_chunk_samples'] = list(extra)[:5]  # 显示前5个
        
        if not missing and extra:
            print(f"  ⚠️  所有chunks都被索引，但索引中有多余的chunks")
            results['coverage'] = 'over-indexed'
        elif missing and not extra:
            results['coverage'] = 'incomplete'
        else:
            results['coverage'] = 'mismatch'
    
    return results


def main():
    """主函数"""
    print("=" * 80)
    print("FAISS索引检查工具")
    print("=" * 80)
    
    # 配置路径
    base_dir = project_root / 'data' / 'faiss'
    compressed_path = base_dir / 'qwen3_fp16_ip.faiss.lz4'
    decompressed_path = base_dir / 'qwen3_fp16_ip_temp.faiss'
    mapping_path = base_dir / 'qwen3_fp16_ip_chunk_ids.json'
    chunks_path = project_root / 'data' / 'processed' / 'chunks.parquet'
    
    # 全局结果
    all_results = {
        'decompression': {},
        'index_validity': {},
        'chunk_coverage': {},
        'summary': {}
    }
    
    # Step 1: 解压
    print()
    if not decompress_index(str(compressed_path), str(decompressed_path)):
        print("\n❌ 无法继续，索引解压失败")
        return
    
    # Step 2: 加载索引
    print()
    index, success = load_faiss_index(str(decompressed_path))
    if not success or index is None:
        print("\n❌ 无法继续，索引加载失败")
        return
    all_results['index'] = {
        'type': type(index).__name__,
        'vector_count': index.ntotal if hasattr(index, 'ntotal') else None,
        'dimension': index.d if hasattr(index, 'd') else None
    }
    
    # Step 3: 加载chunk_id映射
    print()
    chunk_mapping = load_chunk_id_mapping(str(mapping_path))
    if not chunk_mapping:
        print("\n⚠️  无映射文件，跳过chunk验证")
    all_results['mapping'] = {
        'total_mappings': len(chunk_mapping)
    }
    
    # Step 4: 加载chunks.parquet
    print()
    df_chunks, success = load_chunks_parquet(str(chunks_path))
    if not success or df_chunks is None:
        print("\n⚠️  无chunks.parquet文件")
        df_chunks = None
    
    # Step 5: 检查索引有效性
    expected_count = len(df_chunks) if df_chunks is not None else len(chunk_mapping)
    validity_results = check_index_validity(index, expected_count)
    all_results['index_validity'] = validity_results
    
    # Step 6: 比较chunk覆盖
    if chunk_mapping and df_chunks is not None:
        print()
        chunk_ids_from_mapping = set(chunk_mapping.values())
        chunk_ids_from_parquet = set(df_chunks['chunk_id'].values)
        
        coverage_results = compare_chunk_coverage(
            chunk_ids_from_mapping,
            chunk_ids_from_parquet
        )
        all_results['chunk_coverage'] = coverage_results
    
    # Step 7: 生成总结报告
    print("\n" + "=" * 80)
    print("检查报告总结")
    print("=" * 80)
    
    # 确定整体状态
    status = "✅ 健康"
    if 'missing_vectors' in validity_results:
        status = "⚠️  需要检查"
    if all_results['chunk_coverage'].get('coverage') == 'incomplete':
        status = "❌ 有问题"
    
    print(f"\n整体状态: {status}")
    print(f"\n索引统计:")
    print(f"  向量总数: {validity_results.get('vector_count', 'N/A'):,}")
    print(f"  向量维度: {validity_results.get('dimension', 'N/A')}")
    
    if 'chunk_coverage' in all_results:
        coverage = all_results['chunk_coverage']
        print(f"\nChunk覆盖:")
        print(f"  覆盖状态: {coverage.get('coverage', 'unknown')}")
        print(f"  缺失chunks: {coverage.get('missing_chunks', 0):,}")
        print(f"  多余chunks: {coverage.get('extra_chunks', 0):,}")
    
    # 保存结果
    report_path = base_dir / 'check_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n报告已保存: {report_path}")
    
    # 清理临时文件
    if os.path.exists(decompressed_path):
        try:
            os.remove(decompressed_path)
            print(f"临时文件已清理: {decompressed_path}")
        except:
            print(f"⚠️  无法清理临时文件: {decompressed_path}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
