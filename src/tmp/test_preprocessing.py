"""测试预处理功能"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from transformers import AutoTokenizer
from processing.text_cleaner import full_text_cleaning, filter_short_text
from processing.title_cleaner import process_title, is_good_title


def test_text_cleaning():
    """测试文本清洗"""
    print("=" * 80)
    print("测试文本清洗")
    print("=" * 80)
    
    test_cases = [
        "This is a normal text with some URLs: https://example.com and HTML <p>tags</p>",
        "Full-width characters: １２３ＡＢＣ should be normalized to 123ABC",
        "Short text",  # 太短，应该被过滤
        "!@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()@#$%^&*()",  # 纯符号，应该被过滤
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}] 原始文本: {text}")
        cleaned = full_text_cleaning(text, target_lang='en')
        print(f"         清洗后: {cleaned}")


def test_title_cleaning():
    """测试标题清洗"""
    print("\n" + "=" * 80)
    print("测试标题清洗")
    print("=" * 80)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", use_fast=True)
    
    test_titles = [
        "Introduction to Machine Learning",  # 正常标题
        "file_12345_doc_v2.3",  # 垃圾标题
        "A Very Long Title That Should Be Truncated To Fit The Maximum Token Limit",
        "12345",  # 纯数字
        "Good Title with the and of stopwords",  # 包含停用词
    ]
    
    for i, title in enumerate(test_titles, 1):
        print(f"\n[Test {i}] 原始标题: {title}")
        is_good = is_good_title(title)
        print(f"         是否高质量: {is_good}")
        
        if is_good:
            processed = process_title(title, tokenizer, max_tokens=16)
            print(f"         处理后: {processed}")


def test_short_text_filter():
    """测试短文本过滤"""
    print("\n" + "=" * 80)
    print("测试短文本过滤")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", use_fast=True)
    
    test_texts = [
        "This is a very short text.",
        "This is a much longer text that contains enough tokens to pass the minimum token requirement for filtering.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Test {i}] 文本: {text}")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"         Token数: {len(tokens)}")
        keep = filter_short_text(text, tokenizer, min_tokens=32)
        print(f"         是否保留 (min=32): {keep}")


if __name__ == "__main__":
    test_text_cleaning()
    test_title_cleaning()
    test_short_text_filter()
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
