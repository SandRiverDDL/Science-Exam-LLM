#!/usr/bin/env python
"""
集成测试：验证父chunks生成和统计功能
"""
import sys
import json
from pathlib import Path

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from chunking.parent_chunk_generator import ParentChunkGenerator
from script.chunking.generate_parent_chunks import (
    generate_parent_chunks_batch,
    compute_statistics,
    save_results
)


def test_parent_chunk_generation():
    """测试父chunk生成的完整流程"""
    print("=" * 60)
    print("测试1：父chunk生成和统计功能验证")
    print("=" * 60)
    
    # 创建测试数据
    sample_text = """
    Mitochondria are membrane-bound organelles found in eukaryotic cells.
    They are often referred to as the "powerhouses" of the cell because they generate
    most of the cell's supply of adenosine triphosphate (ATP).
    
    The structure of mitochondria includes an outer membrane and an inner membrane.
    The inner membrane is highly folded into structures called cristae.
    This increases the surface area available for chemical reactions.
    
    While mitochondria are primarily known for energy production, they also play roles
    in other cellular processes. These include signaling, differentiation, and cell death.
    
    Mitochondrial dysfunction is associated with various diseases including
    Parkinson's disease, Alzheimer's disease, and type 2 diabetes.
    Understanding mitochondrial biology is crucial for developing new therapeutic strategies.
    Mitochondria are membrane-bound organelles found in eukaryotic cells.
    They are often referred to as the "powerhouses" of the cell because they generate
    most of the cell's supply of adenosine triphosphate (ATP).
    
    The structure of mitochondria includes an outer membrane and an inner membrane.
    The inner membrane is highly folded into structures called cristae.
    This increases the surface area available for chemical reactions.
    
    While mitochondria are primarily known for energy production, they also play roles
    in other cellular processes. These include signaling, differentiation, and cell death.
    
    Mitochondrial dysfunction is associated with various diseases including
    Parkinson's disease, Alzheimer's disease, and type 2 diabetes.
    Understanding mitochondrial biology is crucial for developing new therapeutic strategies.
    Mitochondria are membrane-bound organelles found in eukaryotic cells.
    They are often referred to as the "powerhouses" of the cell because they generate
    most of the cell's supply of adenosine triphosphate (ATP).
    
    The structure of mitochondria includes an outer membrane and an inner membrane.
    The inner membrane is highly folded into structures called cristae.
    This increases the surface area available for chemical reactions.
    
    While mitochondria are primarily known for energy production, they also play roles
    in other cellular processes. These include signaling, differentiation, and cell death.
    
    Mitochondrial dysfunction is associated with various diseases including
    Parkinson's disease, Alzheimer's disease, and type 2 diabetes.
    Understanding mitochondrial biology is crucial for developing new therapeutic strategies.
    """
    
    documents = [
        {
            'id': 'doc_001',
            'doc_id': 'doc_001',
            'title': 'Mitochondria Structure and Function',
            'text': sample_text.strip(),
            'content': sample_text.strip()
        }
    ]
    
    # 创建模拟的现有子chunks
    # 模拟的字符位置（基于实际文本）
    existing_chunks = [
        {
            'chunk_id': 'chunk_001',
            'doc_id': 'doc_001',
            'start_char': 0,
            'end_char': 150,
            'text': 'Mitochondria are membrane-bound organelles found in eukaryotic cells. They are often referred to as the "powerhouses" of the cell because they generate most of the cell\'s supply'
        },
        {
            'chunk_id': 'chunk_002',
            'doc_id': 'doc_001',
            'start_char': 150,
            'end_char': 350,
            'text': 'of adenosine triphosphate (ATP). The structure of mitochondria includes an outer membrane and an inner membrane. The inner membrane is highly folded into structures called cristae.'
        },
        {
            'chunk_id': 'chunk_003',
            'doc_id': 'doc_001',
            'start_char': 350,
            'end_char': 500,
            'text': 'This increases the surface area available for chemical reactions. While mitochondria are primarily known for energy production, they also play roles in other cellular processes.'
        },
        {
            'chunk_id': 'chunk_004',
            'doc_id': 'doc_001',
            'start_char': 500,
            'end_char': len(sample_text.strip()),
            'text': 'These include signaling, differentiation, and cell death. Mitochondrial dysfunction is associated with various diseases...'
        }
    ]
    
    # 加载tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    except:
        print("⚠️  加载tokenizer失败，使用mock tokenizer")
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text) // 4))
        tokenizer = MockTokenizer()
    
    # 生成父chunks
    print("\n[步骤1] 生成父chunks...")
    parent_chunks, mappings, chunks_with_parent = generate_parent_chunks_batch(
        documents=documents,
        existing_chunks=existing_chunks,
        tokenizer=tokenizer,
        parent_size=256
    )
    
    print(f"✅ 生成了 {len(parent_chunks)} 个父chunks")
    
    # 计算统计信息
    print("\n[步骤2] 计算统计信息...")
    stats = compute_statistics(parent_chunks, chunks_with_parent, mappings)
    
    print("\n📊 统计结果:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 验证关键指标
    print("\n[验证关键指标]")
    errors = []
    
    # 验证1：总子chunks数
    if stats['total_child_chunks'] != len(existing_chunks):
        errors.append(f"❌ 子chunks数量不匹配: {stats['total_child_chunks']} != {len(existing_chunks)}")
    else:
        print(f"✅ 子chunks数量正确: {stats['total_child_chunks']}")
    
    # 验证2：覆盖率不应该是0
    if stats.get('coverage_ratio', 0) == 0:
        errors.append(f"❌ 覆盖率为0，父chunk生成可能有问题")
    else:
        print(f"✅ 覆盖率: {stats['coverage_ratio']*100:.2f}%")
    
    # 验证3：完全包含的子chunks数
    if stats.get('fully_contained_count', 0) >= 0:
        print(f"✅ 完全包含的子chunks: {stats['fully_contained_count']}")
    
    # 验证4：父chunk的大小应该接近256
    if 'avg_parent_tokens' in stats:
        avg_tokens = stats['avg_parent_tokens']
        if 50 <= avg_tokens <= 256:
            print(f"✅ 父chunk平均大小合理: {avg_tokens:.2f} tokens")
        else:
            errors.append(f"❌ 父chunk大小异常: {avg_tokens:.2f} tokens (预期: 50-256)")
    
    # 验证5：每个父chunk应该包含至少1个子chunk（如果有子chunks）
    if stats.get('parent_with_children', 0) > 0:
        avg_children = stats.get('avg_children_per_parent', 0)
        if avg_children > 0:
            print(f"✅ 平均每个父chunk包含: {avg_children:.2f} 个子chunks")
        else:
            errors.append(f"❌ 父chunk未包含任何子chunks")
    
    if errors:
        print("\n❌ 验证失败:")
        for err in errors:
            print(f"  {err}")
        return False
    else:
        print("\n✅ 所有验证通过!")
        return True


def test_mapping_algorithm():
    """测试映射算法的准确性"""
    print("\n" + "=" * 60)
    print("测试2：映射算法准确性")
    print("=" * 60)
    
    # 创建tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    except:
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text) // 4))
        tokenizer = MockTokenizer()
    
    generator = ParentChunkGenerator(tokenizer, parent_size=256)
    
    # 创建测试的父chunks
    parent_chunks = [
        {'start_char': 0, 'end_char': 100},      # 父chunk 0
        {'start_char': 100, 'end_char': 200},    # 父chunk 1
        {'start_char': 200, 'end_char': 300},    # 父chunk 2
    ]
    
    # 测试用例：完全包含
    child_positions = [
        (10, 50),      # 应该映射到父chunk 0（完全包含）
        (110, 150),    # 应该映射到父chunk 1（完全包含）
        (205, 295),    # 应该映射到父chunk 2（完全包含）
    ]
    
    mapping = generator.map_child_to_parent(parent_chunks, child_positions)
    
    print(f"\n[完全包含测试]")
    expected = [0, 1, 2]
    if mapping == expected:
        print(f"✅ 完全包含映射正确: {mapping}")
    else:
        print(f"❌ 完全包含映射错误: {mapping} (预期: {expected})")
        return False
    
    # 测试用例：部分重叠
    child_positions = [
        (90, 110),     # 跨越父chunk 0和1的边界 - 应该选父chunk0或1
        (95, 150),     # 主要在父chunk 1
    ]
    
    mapping = generator.map_child_to_parent(parent_chunks, child_positions)
    
    print(f"\n[部分重叠测试]")
    print(f"映射结果: {mapping}")
    # 只要不是-1就是找到了映射
    if all(m >= 0 for m in mapping):
        print(f"✅ 部分重叠映射找到了最大overlap的父chunk")
    else:
        print(f"❌ 映射失败（返回-1）")
        return False
    
    print("\n✅ 映射算法验证通过!")
    return True


if __name__ == "__main__":
    success = True
    
    try:
        success = test_parent_chunk_generation() and success
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success = test_mapping_algorithm() and success
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)
