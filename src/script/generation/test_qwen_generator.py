"""Quick Test for Qwen Generator

测试单个样本的生成效果
"""
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from modeling.qwen_generator import QwenGenerator
from modeling.qwen_zeroshot_abcde import QwenZeroShot

def main():
    """测试 Qwen Generator"""
    
    print("\n" + "="*80)
    print("Qwen Generator 快速测试")
    print("="*80)
    
    # 初始化生成器
    print("\n[1/3] 初始化 Qwen Generator...")
    # generator = QwenGenerator(
    #     model_id="Qwen/Qwen3-8B",
    #     max_new_tokens=5,
    #     do_sample=False
    # )
    generator = QwenZeroShot()
    print("[1/3] ✅ 初始化完成")
    
    # 准备测试数据
    print("\n[2/3] 准备测试问题...")
    
    question = "What is the primary function of mitochondria in eukaryotic cells?"
    options = {
        'A': 'Cell division',
        'B': 'Protein synthesis',
        'C': 'DNA replication',
        'D': 'Lipid storage',
        'E': 'Energy production through ATP synthesis',
    }
    context = """[Document 1]
Mitochondria are membrane-bound organelles found in eukaryotic cells. They are often referred to as the "powerhouses" of the cell because they generate most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of chemical energy. The process of ATP production occurs through cellular respiration, specifically through oxidative phosphorylation.

[Document 2]
The structure of mitochondria includes an outer membrane and an inner membrane. The inner membrane is highly folded into structures called cristae, which increase the surface area available for chemical reactions. The matrix, the space enclosed by the inner membrane, contains enzymes involved in the citric acid cycle and other metabolic pathways.

[Document 3]
While mitochondria are primarily known for energy production, they also play roles in other cellular processes such as signaling, cellular differentiation, and cell death (apoptosis). However, their main function remains the production of ATP through aerobic respiration."""
    
    
    print(f"  Question: {question}")
    print(f"  Options: {list(options.keys())}")
    print(f"  Context length: {len(context)} characters")
    
    # 构建 prompt
    print("\n[3/3] 生成答案...")
    prompt = generator.build_prompt(
        question=question,
        options=options,
        context=context,
    )
    
    # 生成
    responses = generator.generate([prompt], batch_size=1)
    response = responses[0]
    
    # 解析答案
    # answer = generator.parse_answer(response)
    
    # 显示结果
    print("\n" + "="*80)
    print("生成结果")
    print("="*80)
    print(f"\n{response}")
    print("\n" + "="*80)
    # print(f"解析的答案: {answer}")
    print(f"正确答案应该是: E (Energy production through ATP synthesis)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
