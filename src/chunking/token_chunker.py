from typing import List, Dict, Any, Tuple

def tokenize_chunks(text, tokenizer, chunk_size: int = 256, overlap: int = 50) -> List[Dict[str, Any]]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n = len(tokens)
    chunks = []
    start = 0
    position = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = tokens[start:end]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        # 过滤空白分块，避免后续嵌入模型报错或无效输入
        if chunk_text and chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "position": position,
            })
        if end == n:
            break
        start = end - overlap
        position += 1
    return chunks

def tokenize_chunks_ids(text, tokenizer, chunk_size: int = 256, overlap: int = 50) -> List[Dict[str, Any]]:
    """按 token 级切片，返回不解码的 token id 分块。
    这样在嵌入阶段可直接使用 ids，避免重复分词。
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n = len(tokens)
    chunks = []
    start = 0
    position = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_ids = tokens[start:end]
        if chunk_ids:  # 保留非空分块
            chunks.append({
                "ids": chunk_ids,
                "position": position,
            })
        if end == n:
            break
        start = end - overlap
        position += 1
    return chunks

def chunk_docs_with_tokenizer(docs: List[Tuple[str, str]], tokenizer, chunk_size: int, overlap: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    all_chunks = []
    meta = []
    chunk_id = 0
    for doc_id, text in docs:
        doc_chunks = tokenize_chunks(text, tokenizer, chunk_size, overlap)
        for ch in doc_chunks:
            all_chunks.append(ch["text"])
            meta.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "position": ch["position"],
            })
            chunk_id += 1
    return all_chunks, meta
