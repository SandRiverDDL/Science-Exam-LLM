from typing import List, Tuple
from llama_cpp import Llama

class JinaRerankerGGUF:
    def __init__(self, repo_id: str, filename: str, n_ctx: int = 4096):
        self.llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, n_ctx=n_ctx)

    def score(self, query: str, docs: List[str]) -> List[Tuple[int, float]]:
        results = []
        for i, doc in enumerate(docs):
            prompt = (
                "You are a reranker. Score relevance between 0 and 1.\n"
                f"Query: {query}\n"
                f"Document: {doc}\n"
                "Score:"
            )
            out = self.llm(prompt, max_tokens=8, temperature=0.0)
            text = out["choices"][0]["text"].strip()
            try:
                score = float(text.split()[0])
            except Exception:
                score = 0.0
            results.append((i, max(0.0, min(1.0, score))))
        # 返回按分数排序的 (index_in_docs, score)
        results.sort(key=lambda x: x[1], reverse=True)
        return results