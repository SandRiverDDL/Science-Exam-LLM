"""文本清洗工具模块 (2025 SOTA版本)
包含语义保留、格式规范化、使用 ftfy 修复乱码
"""
import regex as re
import unicodedata
from typing import Optional
from bs4 import BeautifulSoup
import ftfy


def normalize_unicode(text: str) -> str:
    """Unicode标准化，解决全角/半角字符混乱
    使用 NFKC 保证最大兼容性
    """
    return unicodedata.normalize('NFKC', text)


def remove_urls(text: str) -> str:
    """移除URL（使用regex）
    保留域名文本，只删除协议头
    """
    # 移除完整URL
    text = re.sub(r'https?://\S+', '', text)
    # 移除www开头的域名
    text = re.sub(r'www\.\S+', '', text)
    return text


def process_html_content(text: str) -> str:
    """处理HTML内容，保留结构而非暴力删除
    将HTML转换为保留段落结构的文本
    """
    if "<table" not in text.lower() and "<div" not in text.lower() and "<p" not in text.lower():
        return text
    
    try:
        # 将块级标签替换为换行，保持视觉结构
        text = re.sub(r'(</p>|<br>|<br/>|</tr>|</div>|</li>)', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'(<p>|<div>|<tr>|<li>)', '\n', text, flags=re.IGNORECASE)
        
        # 使用 BeautifulSoup 提取文本（保留换行）
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator='\n')  # 使用换行作为分隔符
    except Exception:
        # 如果 BeautifulSoup 失败，使用简单正则
        text = re.sub(r'<[^>]+>', ' ', text)
    
    return text


def remove_control_chars(text: str) -> str:
    """移除控制字符，但保留格式控制符（Tab, Newline）
    许多代码或表格依赖这些字符
    """
    # 只移除真正的控制字符，保留 \t, \n, \r
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def keep_valid_unicode(text: str) -> str:
    """保留有效Unicode字符，移除真正的垃圾字符
    
    保留：
    - 所有字母 (L)
    - 所有数字 (N) 
    - 所有符号 (S) - 包括货币€£、数学±°、单位µ等
    - 所有标点 (P)
    - 空白字符 (Z)
    - Tab 和换行 (\t \n \r)
    
    移除：
    - 控制字符 (C) - 但保留 Tab/换行
    - 私有使用区 (Co)
    - 未定义字符
    """
    # 使用 regex 的 Unicode 属性，移除控制字符和私有区
    # 但保留 \t (0x09), \n (0x0A), \r (0x0D)
    text = re.sub(r'[\p{C}&&[^\t\n\r]]', '', text)
    text = re.sub(r'[\p{Co}]', '', text)
    return text


def clean_whitespace(text: str) -> str:
    """规范化空白字符，保留段落结构
    RAG中段落结构很重要，不要全部合并成一行
    """
    # 合并水平空白（空格、Tab等），但保留换行
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    # 去除行首空白
    text = re.sub(r'\n\s+', '\n', text)
    # 最多保留两个换行（段落分隔）
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


# 英文停用词（用于质量检测，不用于删除）
COMMON_STOPWORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'
}


def detect_language_simple(text: str) -> str:
    """简单的语言检测（基于字符统计）
    返回: 'en', 'zh', 'ja', 'ko', 'other'
    """
    if not text:
        return 'other'
    
    # 统计不同语言字符的比例
    total = len(text)
    ascii_count = sum(1 for c in text if ord(c) < 128)
    
    # 如果80%以上是ASCII，认为是英文
    if ascii_count / total > 0.8:
        return 'en'
    
    return 'other'


def full_text_cleaning(text: str, target_lang: str = 'en') -> Optional[str]:
    """完整的文本清洗流程 (2025 SOTA版本)
    
    核心原则：语义保留 > 格式规范化 > 垃圾过滤
    
    Args:
        text: 原始文本
        target_lang: 目标语言 ('en' 为英文)
    
    Returns:
        清洗后的文本，如果文本被判定为垃圾则返回 None
    """
    if not text or not text.strip():
        return None
    
    # 1. [SOTA核心] 自动修复乱码 (Mojibake fixing)
    # ftfy 能自动修复 "Ã©" -> "é", "â\x80\x99" -> "'" 等编码错误
    text = ftfy.fix_text(text)
    
    # 2. Unicode标准化 (NFKC)
    text = normalize_unicode(text)
    
    # 3. [重要改进] HTML处理：保留结构，不暴力删除
    text = process_html_content(text)
    
    # 4. 移除URL
    text = remove_urls(text)
    
    # 5. 移除控制字符（保留Tab和换行）
    text = remove_control_chars(text)
    
    # 6. [关键调整] 保留有效Unicode，不再只保留ASCII
    # 保留货币符号€£、数学符号±°、单位µ、重音字母等
    text = keep_valid_unicode(text)
    
    # 7. 规范化空白（保留段落结构）
    text = clean_whitespace(text)
    
    text = text.strip()
    
    # --- 质量检查 (基于停用词密度，而非字母密度) ---
    
    if len(text) < 15:
        return None
    
    # 8. [SOTA改进] 基于停用词密度过滤垃圾
    # 如果文本中连一个常见停用词都没有，可能是Base64、Hex Dump等
    tokens = text.lower().split()[:100]
    if len(tokens) > 20:
        stopword_count = sum(1 for t in tokens if t in COMMON_STOPWORDS)
        # 极低的停用词密度 + 足够长 = 可能是垃圾
        if stopword_count == 0 and len(tokens) > 50:
            return None
    
    return text


def filter_short_text(text: str, tokenizer, min_tokens: int = 32) -> bool:
    """过滤过短的文本
    
    Args:
        text: 文本
        tokenizer: tokenizer对象
        min_tokens: 最小token数
    
    Returns:
        True 表示保留，False 表示过滤
    """
    if not text:
        return False
    
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens) >= min_tokens
    except Exception:
        return False
