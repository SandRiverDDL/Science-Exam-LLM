from typing import List, Tuple
from rank_bm25 import BM25Okapi

class BM25Retrieval:
    def __init__(self, corpus_tokens: List[List[str]]):
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query_tokens: List[str], top_k: int) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked