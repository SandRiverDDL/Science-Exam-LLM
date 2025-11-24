"""Example: Using Jina Reranker v3

Demonstrates how to use the new Jina Reranker for document reranking.
"""
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from rerank.jina_reranker import JinaReranker


def main():
    """Demonstrate Jina Reranker usage"""
    
    print("\n" + "="*80)
    print("Jina Reranker v3 Example")
    print("="*80)
    
    # Initialize reranker
    print("\n[1] Initializing Jina Reranker...", end='', flush=True)
    try:
        reranker = JinaReranker()
        print(" OK")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return
    
    # Example: Rerank documents
    print("\n[2] Reranking Documents")
    print("-" * 80)
    
    query = "What are the health benefits of green tea?"
    
    documents = [
        "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
        "El precio del cafe ha aumentado un 20% este ano debido a problemas en la cadena de suministro.",
        "Coffee is a popular beverage consumed worldwide for its caffeine content and rich flavor.",
        "Tea culture has been an important part of Asian traditions for centuries.",
        "Water is essential for human health and should be consumed daily.",
    ]
    
    print(f"\nQuery: {query}\n")
    print("Documents to rerank:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc[:80]}...")
    
    # Rerank
    print("\nReranking...", end='', flush=True)
    try:
        results = reranker.rerank(query, documents, top_k=len(documents))
        print(" OK")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results
    print("\n" + "="*80)
    print("Reranking Results (sorted by relevance)")
    print("="*80)
    
    for rank, result in enumerate(results, 1):
        print(f"\n[{rank}] Relevance Score: {result['relevance_score']:.4f}")
        print(f"    Document: {result['document'][:100]}...")
    
    # Example 2: Direct pair ranking
    print("\n" + "="*80)
    print("Direct Pair Ranking Example")
    print("="*80)
    
    pairs = [
        ["What is machine learning?", "Machine learning is a subset of AI that learns from data."],
        ["What is machine learning?", "Python is a programming language."],
        ["What is machine learning?", "Neural networks are computing systems inspired by biological neurons."],
    ]
    
    print(f"\nRanking {len(pairs)} query-document pairs...", end='', flush=True)
    try:
        scores = reranker.rank(pairs)
        print(" OK")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return
    
    print("\nPair Ranking Results:")
    for i, (pair, score) in enumerate(zip(pairs, scores), 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     Document: {pair[1][:70]}...")
    
    print("\n" + "="*80)
    print("Demo Complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
