"""
Embedding Model Testing
========================
This script loads pre-trained sentence embedding models and generates
vector representations for sample English sentences.

Models being tested:
1. all-MiniLM-L6-v2 (Fast, Lightweight)
2. all-mpnet-base-v2 (High Quality, Standard)
3. SciBERT (Specialized for Scientific/Academic text)

Library: sentence-transformers
Prerequisites: See README.md for environment setup.

Usage:
    python embedding_test.py
"""

from sentence_transformers import SentenceTransformer
import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────────
# List of models to test
MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "allenai/scibert_scivocab_uncased"  # SciBERT for sentence-transformers
]

# Sample sentences to encode
SAMPLE_SENTENCES = [
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing helps computers understand human language.",
    "Deep learning uses neural networks with many layers.",
    "Python is a popular programming language for data science.",
    "The weather is sunny and warm today.",
]


def test_model(model_name, sentences):
    """
    Load an embedding model and encode sample sentences.
    """
    print(f"\n[INFO] Testing Model: {model_name}")
    print("-" * 50)
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Encode sentences
    embeddings = model.encode(sentences)
    
    # Display details
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Demonstrate similarity between first two sentences (related topics)
    from numpy import dot
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))
    
    sim = cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity (Related sentences): {sim:.4f}")
    
    return embeddings


def main():
    """Main function to demonstrate multiple embedding models."""

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          EMBEDDING MODELS EVALUATION                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    summary_results = []

    for model_name in MODELS:
        try:
            embeddings = test_model(model_name, SAMPLE_SENTENCES)
            
            # Calculate similarity for summary
            from numpy import dot
            from numpy.linalg import norm
            sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
            
            summary_results.append({
                "model": model_name,
                "dimension": embeddings.shape[1],
                "similarity": sim
            })
        except Exception as e:
            print(f"[ERROR] Failed to test {model_name}: {e}")

    # ── Final Summary Table ──
    print("\n" + "=" * 70)
    print("║                   FINAL EVALUATION SUMMARY                         ║")
    print("=" * 70)
    print(f"{'Model Name':<30} | {'Dimension':<10} | {'Semantic Similarity':<15}")
    print("-" * 70)
    
    for res in summary_results:
        # Simplify model name for the table if it's the long SciBERT path
        display_name = res['model'].split('/')[-1]
        print(f"{display_name:<30} | {res['dimension']:<10} | {res['similarity']:.4f}")
    
    print("=" * 70)
    print("[DONE] All models evaluated.")
    print()


if __name__ == "__main__":
    main()
