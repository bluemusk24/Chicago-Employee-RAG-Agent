import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
import string
import numpy as np
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
from sentence_transformers import CrossEncoder

# Initialize the reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def bm25_tokenizer(doc):
    """Tokenize text for BM25 search"""
    tokenized_doc = []
    for token in doc.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)

    return tokenized_doc


def extract_docs(documents):
    """Extract text content and build BM25 index from Document objects"""
    print("Extracting docs from documents...\n")
    docs = [doc.page_content for doc in documents]

    # Tokenize the corpus
    print("Tokenizing corpus for BM25...\n")
    tokenized_corpus = []
    for doc in tqdm(docs, desc="Building BM25 index"):
        tokenized_corpus.append(bm25_tokenizer(doc))

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index ready with {len(tokenized_corpus)} documents... \n")

    return bm25, docs


def keyword_search_with_reranking(query, bm25, docs, documents, top_k=3, num_candidates=5):
    """
    BM25 keyword search with CrossEncoder reranking.

    Args:
        query: Search query string
        bm25: BM25Okapi index
        docs: List of raw text strings
        documents: List of Document objects
        top_k: Number of final results to return after reranking
        num_candidates: Number of candidates to retrieve with BM25 before reranking

    Returns:
        List of reranked results with scores
    """
    print(f"Searching for: '{query}'\n")

    # Step 1: BM25 retrieval
    print(f"Step 1: BM25 retrieval (getting {num_candidates} candidates)\n")

    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    print(f"Top-3 BM25 results (before reranking):\n")

    for i, hit in enumerate(bm25_hits[:3]):
        print(f"\n{i+1}. BM25 Score: {hit['score']:.3f}")
        print(f"   {docs[hit['corpus_id']][:300]}")

    # Step 2: Rerank
    print(f"Step 2: Reranking with CrossEncoder \n")

    pairs = [(query, docs[hit['corpus_id']]) for hit in bm25_hits]
    rerank_scores = reranker.predict(pairs)

    # Create results with Document objects
    reranked_results = []
    for hit, rerank_score in zip(bm25_hits, rerank_scores):
        reranked_results.append({
            'document': documents[hit['corpus_id']],
            'corpus_id': hit['corpus_id'],
            'bm25_score': hit['score'],
            'rerank_score': float(rerank_score)
        })

    # Sort by rerank score
    reranked_results = sorted(reranked_results, key=lambda x: x['rerank_score'], reverse=True)
    final_results = reranked_results[:top_k]

    # Display results
    print(f"\n Top-{top_k} results after reranking:\n")

    for i, result in enumerate(final_results):
        doc = result['document']
        print(f"{i+1}. Rerank Score: {result['rerank_score']:.4f} (BM25: {result['bm25_score']:.3f})")
        print(f"   Document ID: {result['corpus_id']}")
        print(f"   Metadata: {doc.metadata}")
        print(f"   Content:\n   {doc.page_content[:300]}")
        print()

    return final_results


if __name__ == "__main__":
    file_path = "clean_data/employee_data.csv"
    documents = load_data(file_path=file_path)  
    bm25, docs = extract_docs(documents)

    query = input('\nEnter query here: ')   # INVESTIGATOR INSPECTOR GENERAL as query example
    results = keyword_search_with_reranking(query, bm25, docs, documents, top_k=3, num_candidates=5)