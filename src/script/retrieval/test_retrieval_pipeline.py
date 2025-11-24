"""Retrieval Pipeline Test Script

Demonstration of how to use the complete retrieval pipeline.
Supports batch query testing to verify batch functionality and measure pipeline performance.
"""
import sys
import time
import argparse
from pathlib import Path
from typing import List

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from retrieval.retrieval_pipeline import RetrievalPipeline


def main():
    """Test retrieval pipeline with configurable batch query testing"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Retrieval Pipeline with batch query support')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of queries to send as a batch (default: 1 for single query)'
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='Repeat each query N times to test caching/performance (default: 1)'
    )
    parser.add_argument(
        '--no-reranker',
        action='store_true',
        help='Skip loading reranker model'
    )
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Custom query string (can be overridden with --batch-size)'
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Retrieval Pipeline Test (Using RetrievalPipeline)")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"  Batch Size: {args.batch_size} queries")
    print(f"  Repeat per Query: {args.repeat} times")
    print(f"  Total Queries: {args.batch_size * args.repeat}")
    print(f"  Reranker: {'Disabled' if args.no_reranker else 'Enabled'}")
    
    # Initialize reranker model
    reranker = None
    if not args.no_reranker:
        print("\n[1/2] Initializing Jina Reranker...", end='', flush=True)
        try:
            from rerank.jina_reranker import JinaReranker
            reranker = JinaReranker()
            print(" ✅")
        except Exception as e:
            print(f"\n  ⚠️  ERROR loading reranker: {e}")
            print(f"  Continuing without reranker...")
            reranker = None
    else:
        print("\n[1/2] Skipping Reranker (disabled by --no-reranker)")
    
    # Initialize complete pipeline
    try:
        print("[2/2] Initializing Retrieval Pipeline...", end='', flush=True)
        pipeline = RetrievalPipeline(reranker_model=reranker)
        print(" ✅")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Prepare test queries
    if args.query:
        # Custom query from command line
        test_queries = [args.query] * args.batch_size
    else:
        # Default queries
        default_queries = [
            "What is machine learning",
            "Deep learning basics",
            "Neural network architecture",
            "Transformer models",
            "Natural language processing",
        ]
        # Repeat default queries to reach batch_size
        test_queries = []
        for _ in range(args.batch_size):
            test_queries.append(default_queries[len(test_queries) % len(default_queries)])
    
    print("\n" + "="*80)
    print(f"Test Queries (Batch Size: {args.batch_size})")
    print("="*80)
    for i, query in enumerate(test_queries, 1):
        print(f"  [{i}] {query}")
    
    # ===== Test 1: Single query (baseline) =====
    if args.batch_size == 1:
        print("\n" + "="*80)
        print("Test Mode: Single Query Baseline")
        print("="*80)
        query = test_queries[0]
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        try:
            time_start = time.time()
            print("  Running retrieval pipeline...", end='', flush=True)
            results = pipeline.retrieve(query, verbose=True)
            time_elapsed = time.time() - time_start
            print(f" ✅")
            print(f"\n  Total Time: {time_elapsed:.2f}s")
            print(f"  Final Results: {len(results)} items")
            
            # Display top results
            print(f"\n  Top 5 Results:")
            for rank, (chunk_id, score) in enumerate(results[:5], 1):
                try:
                    parent_text = pipeline.reranker.get_parent_chunk_text(chunk_id) if pipeline.reranker else "[No reranker]"
                except:
                    parent_text = "[Cannot retrieve text]"
                
                print(f"\n    [Top {rank}] Score: {score:.4f}")
                print(f"    ID: {chunk_id}")
                print(f"    Content: {parent_text[:100]}...")
        
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== Test 2: Batch query with repeat =====
    else:
        print("\n" + "="*80)
        print(f"Test Mode: Batch Query (Size={args.batch_size}, Repeat={args.repeat})")
        print("="*80)
        
        # Run repeated batches to test performance consistency
        for repeat_idx in range(args.repeat):
            print(f"\n[Repeat {repeat_idx + 1}/{args.repeat}]")
            print("-" * 80)
            
            try:
                time_start = time.time()
                print("  Running batch retrieval...", end='', flush=True)
                batch_results = pipeline.retrieve(test_queries, verbose=True)
                time_elapsed = time.time() - time_start
                print(f" ✅")
                
                print(f"\n  Total Time: {time_elapsed:.2f}s")
                print(f"  Per Query: {time_elapsed / args.batch_size:.2f}s")
                print(f"\n  Batch Results Summary:")
                
                # Show summary for each query in batch
                for query_idx, (query, results) in enumerate(zip(test_queries, batch_results), 1):
                    print(f"\n    Query {query_idx}: {query[:50]}...")
                    print(f"      Results: {len(results)} items")
                    if len(results) > 0:
                        top_score = results[0][1]
                        print(f"      Top Score: {top_score:.4f}")
                        # Show top 3 for this query
                        print(f"      Top 3:")
                        for rank, (chunk_id, score) in enumerate(results[:3], 1):
                            print(f"        [{rank}] {chunk_id} (score: {score:.4f})")
            
            except Exception as e:
                print(f"\n  ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*80)
    print("Test Complete ✅")
    print("="*80)
    print("\nRetrieval Pipeline Steps:")
    print("  1. [✅] Dense Retrieval (FAISS on-disk)")
    print("  2. [✅] BM25 Retrieval (倒排索引)")
    print("  3. [✅] Paragraph Boosting")
    print("  4. [✅] RRF Fusion")
    print("  5. [✅] MMR Reranking")
    if reranker:
        print("  6. [✅] Cross-Encoder Reranking (Jina-v3)")
    else:
        print("  6. [⏭️  SKIPPED] Cross-Encoder Reranking (model not loaded)")
    print("\nBatch Testing:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Repeat Count: {args.repeat}")
    print(f"  Mode: {'Batch' if args.batch_size > 1 else 'Single'} Query")
    print("\n")


if __name__ == "__main__":
    main()


# # 批量3条查询，执行1次
# python src/script/retrieval/test_retrieval_pipeline.py --batch-size 3

# # 批量10条查询，重复执行2次（测试缓存影响）
# python src/script/retrieval/test_retrieval_pipeline.py --batch-size 10 --repeat 2

# # 批量5条自定义查询
# python src/script/retrieval/test_retrieval_pipeline.py --batch-size 5 --query "What is deep learning"

# # 跳过 reranker，只测试前几个步骤（节省显存）
# python src/script/retrieval/test_retrieval_pipeline.py --batch-size 5 --no-reranker