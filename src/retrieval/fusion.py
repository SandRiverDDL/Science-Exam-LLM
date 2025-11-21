from typing import Dict, List, Tuple

def rrf_fusion(rank_lists: Dict[str, List[Tuple[int, float]]], k: int = 60, top_k: int = 50):
    # rank_lists: {"embedding": [(doc_id, score), ...], "bm25": [...]} 等
    # 仅用名次进行 RRF；分值可用作后续 tie-break
    rrf_scores = {}
    for name, lst in rank_lists.items():
        for rank, (doc_id, _) in enumerate(lst, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return merged  # [(doc_id, rrf_score)]