import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import vector_store
from sentence_transformers import CrossEncoder

from langchain_ollama import ChatOllama
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

# Initialize the reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def dense_search_with_compression_and_reranking(query, top_k=3, num_candidates=10):
    """
    Dense retrieval → LLM Compression → CrossEncoder Reranking
    Three-stage approach for highest quality results.
    """
    print(f" Dense Search with Compression & Reranking: '{query}'")
    print("-"*50)
    
    # Step 1: Dense retrieval
    print(f"\n Step 1: Dense retrieval (getting {num_candidates} candidates)")
    print("-"*50)
    
    candidates = vector_store.similarity_search_with_score(query, k=num_candidates)
    print(f"Retrieved {len(candidates)} candidates")
    
    # Store original documents
    original_docs = {doc.metadata['row']: doc for doc, score in candidates}
    
    # Step 2: LLM Compression
    print(f"\n Step 2: LLM Compression")
    print("-"*50)
    
    # Setup LLM compressor
    llm = ChatOllama(temperature=0, model="gpt-oss:20b-cloud")
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": num_candidates})
    )
    
    # Get compressed docs
    compressed_docs = compression_retriever.invoke(query)
    print(f"Compressed to {len(compressed_docs)} documents")
    
    # Step 3: CrossEncoder Reranking
    print(f"\n Step 3: Reranking compressed documents")
    print("-"*50)
    
    # Rerank compressed docs and get metadata
    pairs = [(query, doc.page_content) for doc in compressed_docs]
    rerank_scores = reranker.predict(pairs)

    
    reranked_results = []
    for doc, rerank_score in zip(compressed_docs, rerank_scores):
        row_id = doc.metadata['row']
        # Get the original full document
        original_doc = original_docs.get(row_id, doc)
        
        reranked_results.append({
            'document': original_doc,
            'metadata': doc.metadata,
            'compressed_content': doc.page_content,  # What LLM extracted
            'page_content': original_doc.page_content,  # FULL original content
            'rerank_score': float(rerank_score)
        })
    
    # Sort and get top-k
    reranked_results = sorted(reranked_results, key=lambda x: x['rerank_score'], reverse=True)
    final_results = reranked_results[:top_k]
    
    # Display results
    print(f"\n Top-{top_k} final results:\n")
    for i, result in enumerate(final_results):
        print(f"{i+1}. Rerank Score: {result['rerank_score']:.4f}")
        print(f"   Metadata: {result['metadata']}")
        print(f"   Content:\n{result['page_content']}")
        print()
    
    return final_results


if __name__ == "__main__":
    query = input('\nEnter query here: ')   # People in the Chicago Police Department
    print(f"\nSearching for: '{query}'")
    results = dense_search_with_compression_and_reranking(query, top_k=3, num_candidates=5)