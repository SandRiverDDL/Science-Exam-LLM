"""标题清洗工具模块 (2025 SOTA版本)
保留语义完整性，不删除停用词和产品型号
直接encode标题，返回token IDs用于后续拼接
"""
import regex as re
from typing import Optional, List





def is_good_title(title: str) -> bool:
    """判断标题是否为高质量标题 (2025 SOTA版本)
    
    更宽容，防止误杀技术类标题和产品型号
    
    Args:
        title: 标题文本
    
    Returns:
        True 表示高质量，False 表示垃圾标题
    """
    if not title:
        return False
    
    t = title.strip()
    
    # 1. 长度检查（放宽下限，有些型号如 "X5" 可能就是标题）
    if len(t) < 2 or len(t) > 200:
        return False
    
    # 2. 垃圾模式（更加精准，避免误杀）
    # 只杀纯粹的机器生成日志名，不杀产品型号
    garbage_patterns = [
        r'^file[_-]\d+',          # file_123, file-123 开头（不管后面有什么）
        r'^doc[_-]\d+',           # doc_001, doc-001 开头
        r'^id[_-]\d+$',           # id_999 (仅匹配完整)
        r'^img[_-]\d+$',          # img_002 (仅匹配完整)
        r'^untitled\s*\d*$',      # untitled, untitled 1
        r'^\d{4}-\d{2}-\d{2}$',   # 纯日期 2024-01-01
        r'^\d+$',                 # 纯数字 12345
        r'^[\W_]+$',              # 纯符号 ____
    ]
    
    for pattern in garbage_patterns:
        if re.match(pattern, t.lower()):
            return False
    
    # 3. 字母数字检测（使用 Unicode 属性，兼容多语言）
    # 只要包含至少一个"字母"或"数字"即可
    if not re.search(r'[\p{L}\p{N}]', t):
        return False
    
    # 4. 至少包含一个字母（避免纯数字标题）
    if not re.search(r'\p{L}', t):
        return False
    
    return True





def clean_title_conservative(title: str) -> str:
    """保守清洗：只洗格式，不洗内容
    不删除型号、不删除停用词
    
    Args:
        title: 原始标题
    
    Returns:
        清洗后的标题
    """
    # 1. 移除URL（除非标题本身就是域名）
    if ' ' in title:
        title = re.sub(r'https?://\S+', '', title)
        title = re.sub(r'www\.\S+', '', title)
    
    # 2. 移除文件扩展名（如 .pdf, .docx）
    title = re.sub(r'\.(pdf|docx|txt|md|json|html)$', '', title, flags=re.IGNORECASE)
    
    # 3. 规范化空格
    title = re.sub(r'\s+', ' ', title)
    
    # 4. 移除开头结尾的非语义符号
    title = title.strip(" -_:;,.")
    
    return title.strip()


def encode_title_tokens(title: str, tokenizer, max_tokens: int = 16) -> Optional[List[int]]:
    """将标题encode为token IDs（直接返回IDs用于后续拼接）
    
    Args:
        title: 标题文本
        tokenizer: tokenizer对象
        max_tokens: 最大token数
    
    Returns:
        Token IDs列表，如果失败则返回 None
    """
    if not title:
        return None
    
    try:
        # 直接encode，不添加特殊符号
        tokens = tokenizer.encode(title, add_special_tokens=False)
        
        # 截断到最大长度
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        
        return tokens if tokens else None
    except Exception:
        return None





def process_title(
    title: str,
    tokenizer,
    max_tokens: int = 16,
) -> Optional[List[int]]:
    """完整的标题处理流程 (2025 SOTA版本)
    
    核心改进：
    1. 不去除停用词（保持语义完整性）
    2. 不删除产品型号/ID（高价值实体）
    3. 直接返回token IDs，不解码回文本
    
    Args:
        title: 原始标题
        tokenizer: tokenizer对象
        max_tokens: 最大token数
    
    Returns:
        Token IDs列表，如果标题是垃圾则返回 None
    """
    if not title:
        return None
    
    # 1. 噪声检测
    if not is_good_title(title):
        return None
    
    # 2. 保守清洗（不去停用词，不去型号）
    title = clean_title_conservative(title)
    
    # 3. 直接encode为token IDs
    title_ids = encode_title_tokens(title, tokenizer, max_tokens)
    
    if not title_ids or len(title_ids) == 0:
        return None
    
    return title_ids
