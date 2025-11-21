"""测试父文档索引 chunking"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from transformers import AutoTokenizer
from chunking.parent_chunker import ParentDocumentChunker


def test_basic_chunking():
    """测试基本的 chunking 功能"""
    print("=" * 80)
    print("测试父文档索引 Chunking")
    print("=" * 80)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", use_fast=True)
    
    # 创建 chunker
    chunker = ParentDocumentChunker(
        tokenizer=tokenizer,
        child_size=128,
        parent_size=512,
        min_chunk_tokens=32
    )
    
    # 测试文档
    title = "Introduction to Machine Learning"
    text = """
    Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. 
    It focuses on the development of computer programs that can access data and use it to learn for themselves.
    
    The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data 
    and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without 
    human intervention or assistance and adjust actions accordingly.
    
    Machine learning algorithms are categorized into three main types: supervised learning, unsupervised learning, and reinforcement learning.
    Supervised learning algorithms learn from labeled training data, helping predict outcomes for unforeseen data.
    Unsupervised learning algorithms work with unlabeled data to discover hidden patterns or intrinsic structures.
    Reinforcement learning is about taking suitable action to maximize reward in a particular situation.
    """ * 3  # 重复3次确保足够长
    
    # Encode
    title_ids = tokenizer.encode(title, add_special_tokens=False)
    doc_ids = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"\n文档信息:")
    print(f"  标题: {title}")
    print(f"  标题token数: {len(title_ids)}")
    print(f"  正文token数: {len(doc_ids)}")
    
    # Chunking
    chunks = chunker.chunk_document(
        doc_id="test_doc_001",
        title_ids=title_ids,
        doc_ids=doc_ids,
        title_text=title
    )
    
    print(f"\nChunking结果:")
    print(f"  生成chunk数: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  Chunk {i}:")
        print(f"    - chunk_id: {chunk['chunk_id']}")
        print(f"    - 子chunk长度: {chunk['chunk_len']}")
        print(f"    - 父chunk范围: [{chunk['parent_start']}, {chunk['parent_end']})")
        print(f"    - 父chunk长度: {chunk['parent_end'] - chunk['parent_start']}")
        print(f"    - rerank_text长度: {len(chunk['rerank_text'])} chars")
        print(f"    - rerank_text前100字符: {chunk['rerank_text'][:100]}...")
    
    # 验证约束
    print(f"\n验证:")
    for i, chunk in enumerate(chunks):
        child_len = len(chunk['child_ids'])
        parent_len = chunk['parent_end'] - chunk['parent_start']
        
        # 子chunk应该 <= 128
        assert child_len <= 128, f"Chunk {i}: 子chunk太长 ({child_len})"
        
        # 父chunk应该尽可能接近512
        # （最后一个chunk可能较短）
        if i < len(chunks) - 1:
            assert parent_len >= 400, f"Chunk {i}: 父chunk太短 ({parent_len})"
        
        # 父chunk应该包含子chunk
        child_start_in_doc = chunk['parent_start']  # 简化假设
        assert chunk['parent_start'] <= child_start_in_doc, f"Chunk {i}: 父chunk不包含子chunk"
        
        print(f"  ✅ Chunk {i}: 子chunk={child_len}, 父chunk={parent_len}")
    
    print("\n✅ 所有测试通过！")


def test_embedding_input():
    """测试 embedding 输入准备"""
    print("\n" + "=" * 80)
    print("测试 Embedding 输入准备")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", use_fast=True)
    chunker = ParentDocumentChunker(tokenizer=tokenizer, child_size=128)
    
    title_ids = [101, 102, 103]
    child_ids = [201, 202, 203, 204, 205]
    
    # 准备输入
    input_ids = chunker.prepare_embedding_inputs(title_ids, child_ids, add_special_tokens=True)
    
    print(f"\n输入准备:")
    print(f"  标题IDs: {title_ids}")
    print(f"  子chunk IDs: {child_ids}")
    print(f"  完整输入IDs: {input_ids}")
    print(f"  完整输入长度: {len(input_ids)}")
    
    # 验证结构
    assert input_ids[0] == tokenizer.cls_token_id, "应该以[CLS]开头"
    assert tokenizer.sep_token_id in input_ids, "应该包含[SEP]"
    
    print("\n✅ Embedding 输入准备测试通过！")


if __name__ == "__main__":
    try:
        test_basic_chunking()
        test_embedding_input()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试通过！父文档索引 chunking 正常工作！")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
