import os
from typing import List
from llama_cpp import Llama

class JinaEmbeddingGGUF:
    def __init__(self, repo_id: str, filename: str, n_ctx: int = 8192, n_threads: int = 8, n_batch: int = 32, n_seq_max: int = 128):
        # 可从环境读取 HF token（不要在代码中硬编码）
        hf_token = os.environ.get("huggingface_token", None)
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            n_seq_max=n_seq_max,
            embedding=True,
            # 如果要走私有/带token的下载，可设置环境变量 HUGGINGFACEHUB_API_TOKEN
        )
        self._embedding_dim = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        # 输入校验，避免传入空列表导致底层报错
        if not texts:
            raise ValueError("Embedding texts list is empty. Make sure documents and chunks are loaded correctly.")
        # llama.cpp 的 create_embedding 接口兼容 OpenAI 风格，支持 list[str]
        try:
            res = self.llm.create_embedding(texts)
            raw_vecs = [item.get("embedding") for item in res.get("data", [])]
        except Exception:
            # 降级：逐条嵌入，避免批量 seq_id 限制导致的失败
            raw_vecs = []
            for t in texts:
                r = self.llm.create_embedding(t)
                raw_vecs.append(r["data"][0]["embedding"])

        # 规范化与维度校验，过滤不合法向量
        vecs: List[List[float]] = []
        for v in raw_vecs:
            if not isinstance(v, (list, tuple)) or not v:
                continue
            v_list = list(v)
            if self._embedding_dim is None:
                self._embedding_dim = len(v_list)
            if len(v_list) != self._embedding_dim:
                # 跳过维度不匹配的向量，避免 FAISS 构建报错
                continue
            vecs.append(v_list)

        if not vecs:
            raise RuntimeError("No valid embeddings produced by Jina GGUF backend.")
        return vecs