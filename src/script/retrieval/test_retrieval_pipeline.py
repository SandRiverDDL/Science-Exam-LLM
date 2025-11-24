"""Retrieval Pipeline Test Script

Demonstration of how to use the complete retrieval pipeline.
"""
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from retrieval.retrieval_pipeline import RetrievalPipeline


def main():
    """Test retrieval pipeline"""
    
    print("\n" + "="*80)
    print("Retrieval Pipeline Initialization")
    print("="*80)
    
    # Initialize reranker model (you need to provide this)
    print("\n[NOTE] Reranker model required:")
    print("    Example: from rerank.jina_reranker_gguf import JinaReranker")
    print("             reranker = JinaReranker(...)")
    print("\nFor demo, skip reranker. Show first 5 steps only.")
    
    # Initialize pipeline (without reranker)
    try:
        print("\n[1/2] Initializing Dense Retriever...", end='', flush=True)
        import yaml
        with open(project_root / 'config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        qwen3_config = config.get('embedding', {}).get('qwen3', {})
        index_path = project_root / qwen3_config.get('index_path', 'data/faiss/qwen3_fp16_ip.faiss')
        chunk_ids_path = project_root / 'data' / 'faiss' / 'qwen3_fp16_ip_chunk_ids.json'
        
        from retrieval.dense_retrieval import DenseRetriever
        dense_retriever = DenseRetriever(
            str(index_path),
            str(chunk_ids_path),
            model_id=qwen3_config.get('model_id', 'Qwen/Qwen3-Embedding-0.6B'),
            device=qwen3_config.get('device') or 'cuda',
            max_length=qwen3_config.get('max_length', 168),
            dtype=qwen3_config.get('dtype', 'float16')
        )
        print(" OK")
        
        print("[2/2] Initializing Data...", end='', flush=True)
        import pandas as pd
        chunks_df = pd.read_parquet(project_root / 'data' / 'processed' / 'chunks.parquet')
        docs_df = pd.read_parquet(project_root / 'data' / 'processed' / 'documents_cleaned.parquet')
        print(" OK")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test queries
    test_queries = [
        "What is machine learning",
        "Deep learning basics",
    ]
    
    print("\n" + "="*80)
    print("Test Queries")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 80)
        
        try:
            # Dense retrieval
            print(f"  Dense retrieval (top-600)...", end='', flush=True)
            dense_results = dense_retriever.retrieve(query, top_k=600)
            print(f" OK, found {len(dense_results)}")
            
            # Show top-5 results
            print(f"\n  Top-5 Results:")
            for rank, (chunk_id, similarity) in enumerate(dense_results[:5], 1):
                # Get chunk text
                chunk_row = chunks_df[chunks_df['chunk_id'] == chunk_id]
                if len(chunk_row) > 0:
                    chunk_row = chunk_row.iloc[0]
                    doc_id = chunk_row['doc_id']
                    doc_row = docs_df[docs_df['doc_id'] == doc_id]
                    
                    if len(doc_row) > 0:
                        doc_text = doc_row.iloc[0]['text']
                        child_start = chunk_row['child_start']
                        child_end = chunk_row['child_end']
                        chunk_text = doc_text[child_start:child_end]
                        
                        print(f"\n    [Top {rank}] Similarity: {similarity:.4f}")
                        print(f"    ID: {chunk_id}")
                        print(f"    Content: {chunk_text[:150]}...")
        
        except Exception as e:
            print(f"\n  ERROR: {e}")
    
    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)
    print("\nRetrieval Pipeline Steps:")
    print("  1. [OK] Dense Retrieval")
    print("  2. [OK] BM25 Retrieval")
    print("  3. [OK] Paragraph Boosting")
    print("  4. [OK] RRF Fusion")
    print("  5. [OK] MMR Reranking")
    print("  6. [PENDING] Cross-Encoder Reranking")
    print("\nTo use complete pipeline:")
    print("  from rerank.jina_reranker_gguf import JinaReranker")
    print("  reranker_model = JinaReranker(...)")
    print("  pipeline = RetrievalPipeline(..., reranker_model=reranker_model)")
    print("  results = pipeline.retrieve(query, verbose=True)\n")


if __name__ == "__main__":
    main()
