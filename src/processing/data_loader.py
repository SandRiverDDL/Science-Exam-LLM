"""数据加载器 - 支持CSV和Parquet格式

实现文件读取逻辑与处理逻辑的解耦
"""
import os
import sys
import csv
from typing import Iterator, Tuple, List, Optional
import pyarrow.parquet as pq


class DataLoader:
    """统一的数据加载器，支持多种文件格式"""
    
    def __init__(self, file_path: str, text_columns: Optional[List[str]] = None):
        """
        Args:
            file_path: 文件路径
            text_columns: 文本列名（None则自动检测）
        """
        self.file_path = file_path
        self.text_columns = text_columns
        self.file_ext = os.path.splitext(file_path)[1].lower()
    
    def load(self) -> Iterator[Tuple[str, str, str]]:
        """加载数据
        
        Yields:
            (doc_id, title, text) tuples
        """
        if self.file_ext == '.parquet':
            yield from self._load_parquet()
        elif self.file_ext == '.csv':
            yield from self._load_csv()
        else:
            raise ValueError(f"不支持的文件格式: {self.file_ext}")
    
    def _load_parquet(self) -> Iterator[Tuple[str, str, str]]:
        """加载Parquet文件"""
        table = pq.read_table(self.file_path)
        df = table.to_pandas()
        
        # 获取所有列名
        header = df.columns.tolist()
        
        # 自动检测文本列
        text_columns = self.text_columns
        if text_columns is None:
            candidates = {"text", "content", "article", "body", "paragraph", "wiki_text", "page_content"}
            text_columns = [h for h in header if h and h.lower() in candidates]
            
            if not text_columns:
                # 查找包含"text"关键字的列
                text_columns = [h for h in header if "text" in h.lower()]
            
            if not text_columns:
                # 查找包含"content"关键字的列
                text_columns = [h for h in header if "content" in h.lower()]
            
            if not text_columns:
                # 默认使用非ID列
                non_id_cols = [h for h in header if not any(x in h.lower() for x in ["id", "index", "_id", "url"])]
                text_columns = non_id_cols[:1] if non_id_cols else [header[0]]
        
        # 检测标题列
        title_columns = [h for h in header if h and h.lower() in {"title", "name", "heading", "subject", "page_title"}]
        title_col = title_columns[0] if title_columns else None
        
        print(f"[DataLoader] Parquet文件列: {header}")
        print(f"[DataLoader] 使用文本列: {text_columns}")
        print(f"[DataLoader] 使用标题列: {title_col}")
        
        # 遍历行
        for idx, row in df.iterrows():
            # 提取标题
            title = ""
            if title_col and title_col in df.columns:
                val = row[title_col]
                if isinstance(val, str):
                    title = val.strip()
                elif val is not None:
                    title = str(val).strip()
            
            # 提取文本
            text_parts = []
            for col in text_columns:
                if col in df.columns:
                    val = row[col]
                    if isinstance(val, str) and val.strip():
                        text_parts.append(val.strip())
                    elif val is not None:
                        val_str = str(val).strip()
                        if val_str and val_str.lower() != "nan":
                            text_parts.append(val_str)
            
            text = "\n\n".join(text_parts).strip()
            
            if text:
                doc_id = f"{os.path.basename(self.file_path)}:row:{idx}"
                yield (doc_id, title, text)
    
    def _load_csv(self) -> Iterator[Tuple[str, str, str]]:
        """加载CSV文件"""
        try:
            csv.field_size_limit(min(sys.maxsize, 1_000_000_000))
        except Exception:
            pass
        
        with open(self.file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            
            # 自动检测文本列
            text_columns = self.text_columns
            if text_columns is None:
                text_candidates = ["text", "content", "article", "body", "paragraph", "wiki_text"]
                text_col = None
                for candidate in text_candidates:
                    if candidate in header:
                        text_col = candidate
                        break
                if not text_col and header:
                    text_col = header[0]
                text_columns = [text_col] if text_col else []
            
            # 检测标题列
            title_col = None
            for candidate in ["title", "heading", "subject", "name"]:
                if candidate in header:
                    title_col = candidate
                    break
            
            print(f"[DataLoader] CSV文件列: {header}")
            print(f"[DataLoader] 使用文本列: {text_columns}")
            print(f"[DataLoader] 使用标题列: {title_col}")
            
            for i, row in enumerate(reader):
                # 提取文本
                text_parts = []
                for col in text_columns:
                    val = row.get(col, "")
                    if isinstance(val, str) and val.strip():
                        text_parts.append(val.strip())
                
                text = "\n\n".join(text_parts).strip()
                
                # 提取标题
                title = row.get(title_col, "") if title_col else ""
                
                if text:
                    doc_id = f"{os.path.basename(self.file_path)}:row:{i}"
                    yield (doc_id, title, text)


def load_files(file_paths: List[str], text_columns: Optional[List[str]] = None) -> Iterator[Tuple[str, str, str]]:
    """加载多个文件
    
    Args:
        file_paths: 文件路径列表
        text_columns: 文本列名（可选）
    
    Yields:
        (doc_id, title, text) tuples
    """
    for file_path in file_paths:
        print(f"\n正在加载: {file_path}")
        loader = DataLoader(file_path, text_columns)
        yield from loader.load()
