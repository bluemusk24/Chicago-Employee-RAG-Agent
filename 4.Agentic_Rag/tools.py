import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.lexical_retrieval import bm25_tokenizer, extract_docs, keyword_search_with_reranking
from src.dense_retrieval import dense_search_with_compression_and_reranking
from src.graph_retrieval import extract_metadata_from_content, build_graph_retriever, pretty_print_retrieval
from src.ingestion import vector_store
from src.data_loader import load_data


class RetrieverTools:
    def __init__(self, file_path: str = None):
        print("Initializing RetrieverTools...")
        if file_path is None:
            file_path = os.path.join(PROJECT_ROOT, "clean_data", "employee_data.csv")
        
        self.documents = load_data(file_path=file_path)

        if self.documents is None:
            raise ValueError(f"Failed to load documents from '{file_path}'.")

        # Setup lexical search
        print("\nSetting up BM25 index...")
        self.bm25, self.docs = extract_docs(self.documents)

        # Setup graph retrieval
        print("\nSetting up Graph Retriever...")
        all_docs = vector_store.similarity_search("", k=1000)
        enriched_docs = extract_metadata_from_content(all_docs)
        self.traversal_retriever = build_graph_retriever(enriched_docs)

        print("\nAll retrievers ready!")

    def lexical_search(self, query: str, top_k: int = 3, num_candidates: int = 5):
        """BM25 keyword search with CrossEncoder reranking."""
        print(f"\n[Lexical Search] Query: '{query}'")
        return keyword_search_with_reranking(
            query, self.bm25, self.docs, self.documents,
            top_k=top_k, num_candidates=num_candidates
        )

    def dense_search(self, query: str, top_k: int = 3, num_candidates: int = 10):
        """Dense vector search with LLM compression and reranking."""
        print(f"\n[Dense Search] Query: '{query}'")
        return dense_search_with_compression_and_reranking(
            query, top_k=top_k, num_candidates=num_candidates
        )

    def graph_search(self, query: str):
        """Graph traversal retrieval using metadata relationships."""
        print(f"\n[Graph Search] Query: '{query}'")
        results = self.traversal_retriever.invoke(query)
        pretty_print_retrieval(results)
        return results

    def search_all(self, query: str, top_k: int = 3):
        """Run all three retrievers and combine results."""
        print(f"\n{'-'*50}")
        print(f"Running all retrievers for: '{query}'")
        print(f"{'-'*50}")

        lexical_results = self.lexical_search(query, top_k=top_k)
        dense_results = self.dense_search(query, top_k=top_k)
        graph_results = self.graph_search(query)

        return {
            "lexical": lexical_results,
            "dense": dense_results,
            "graph": graph_results
        }


if __name__ == "__main__":
    retriever = RetrieverTools()
    query = input("\nEnter query here: ")
    results = retriever.search_all(query)