"""检索管道使用示例

演示如何使用完整的检索管道
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from retrieval.retrieval_pipeline import RetrievalPipeline


def main():
    """使用检索管道"""
    
    print("\n" + "="*80)
    print("检索管道示例 - 从config.yaml读取所有参数")
    print("="*80)
    
    # 初始化管道（自动从config.yaml读取参数）
    print("\n[1] 初始化检索管道...")
    try:
        pipeline = RetrievalPipeline()
        print("✅ 管道初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试查询
    query = "机器学习的基本概念是什么"
    
    print(f"\n[2] 执行检索查询: {query}")
    print("-" * 80)
    
    try:
        # 执行检索（所有参数从config.yaml读取）
        results = pipeline.retrieve(query, verbose=True)
        
        print("\n" + "="*80)
        print(f"检索结果 (共{len(results)}条)")
        print("="*80)
        
        for rank, (chunk_id, score) in enumerate(results, 1):
            # 获取chunk文本
            chunk_text = pipeline.reranker.get_parent_chunk_text(chunk_id)
            
            print(f"\n【Top {rank}】分数: {score:.4f}")
            print(f"ID: {chunk_id}")
            print(f"内容: {chunk_text[:200]}...")
            print("-" * 80)
    
    except Exception as e:
        print(f"\n❌ 检索失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 示例完成\n")


if __name__ == "__main__":
    main()
