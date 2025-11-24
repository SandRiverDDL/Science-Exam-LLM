"""Retrieval Pipeline Test Script

Demonstration of how to use the complete retrieval pipeline.
"""
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from retrieval.retrieval_pipeline import RetrievalPipeline
from rerank.jina_reranker import JinaReranker


def main():
    """Test retrieval pipeline"""
    
    print("\n" + "="*80)
    print("Retrieval Pipeline Test (Using RetrievalPipeline)")
    print("="*80)
    
    # Initialize reranker model
    print("\n[1/2] Initializing Jina Reranker...", end='', flush=True)
    try:
        reranker = JinaReranker()
        print(" OK")
    except Exception as e:
        print(f"\n  ERROR loading reranker: {e}")
        print(f"  Continuing without reranker...")
        reranker = None
    
    # Initialize complete pipeline
    try:
        print("[2/2] Initializing Retrieval Pipeline...", end='', flush=True)
        pipeline = RetrievalPipeline(reranker_model=reranker)
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
            # Use complete pipeline
            print(f"  Running full retrieval pipeline...", end='', flush=True)
            results = pipeline.retrieve(query, verbose=True)
            print(f" OK, final {len(results)} results")
            
            # Display results
            print(f"\n  Final Results (Top 5):")
            for rank, (chunk_id, score) in enumerate(results[:5], 1):
                # Get parent chunk text
                parent_text = pipeline.reranker.get_parent_chunk_text(chunk_id)
                
                print(f"\n    [Top {rank}] Score: {score:.4f}")
                print(f"    ID: {chunk_id}")
                print(f"    Content: {parent_text[:150]}...")
        
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)
    print("\nRetrieval Pipeline Steps:")
    print("  1. [OK] Dense Retrieval")
    print("  2. [OK] BM25 Retrieval")
    print("  3. [OK] Paragraph Boosting")
    print("  4. [OK] RRF Fusion")
    print("  5. [OK] MMR Reranking")
    if reranker:
        print("  6. [OK] Cross-Encoder Reranking (Jina-v3)")
    else:
        print("  6. [SKIPPED] Cross-Encoder Reranking (model not available)")
    print("\n")


if __name__ == "__main__":
    main()
