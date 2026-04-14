import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from src.ingestion import vector_store, embedding_model

from langchain_core.vectorstores import InMemoryVectorStore
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever


def extract_metadata_from_content(documents):
    """Extract structured fields from page_content and add to metadata."""
    enriched_docs = []
    
    for doc in documents:
        content = doc.page_content
        
        patterns = {
            'name': r'name:\s*(.+)',
            'job_titles': r'job_titles:\s*(.+)',
            'department': r'department:\s*(.+)',
            'full_or_part_time': r'full_or_part_time:\s*(.+)',
            'salary_or_hourly': r'salary_or_hourly:\s*(.+)',
            'annual_salary': r'annual_salary:\s*(.+)',
            'typical_hours': r'typical_hours:\s*(.+)',
            'hourly_rate': r'hourly_rate:\s*(.+)',
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1).strip()
                doc.metadata[field] = value
        
        job_title = doc.metadata.get('job_titles', '')
        if job_title:
            doc.metadata['job_category'] = job_title.split()[0]
        
        enriched_docs.append(doc)
    
    return enriched_docs


def build_graph_retriever(enriched_docs):
    """Create in-memory store and setup graph retriever."""
    edges = [
        ("department", "department"),
        ("job_titles", "job_titles"),
        ("job_category", "job_category"),
        ("full_or_part_time", "full_or_part_time"),
    ]

    in_memory_store = InMemoryVectorStore(embedding=embedding_model)
    in_memory_store.add_documents(enriched_docs)
    print(f"In-memory store created with {len(enriched_docs)} enriched documents")

    traversal_retriever = GraphRetriever(
        store=in_memory_store,
        edges=edges,
        strategy=Eager(k=5, start_k=1, max_depth=2),
    )

    return traversal_retriever


def pretty_print_retrieval(results):
    print("\n" + "-"*80)
    print(f"Total Results Found: {len(results)}")
    print("="*80)

    depths, categories, departments = {}, {}, {}

    for doc in results:
        depth = doc.metadata.get('_depth', 0)
        depths[depth] = depths.get(depth, 0) + 1

        category = doc.metadata.get('job_category', 'Unknown')
        categories[category] = categories.get(category, 0) + 1

        dept = doc.metadata.get('department', 'Unknown')
        departments[dept] = departments.get(dept, 0) + 1

    print(f"\nResults by Depth:")
    for depth in sorted(depths.keys()):
        print(f"   Depth {depth}: {depths[depth]} documents")

    print(f"\nResults by Job Category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {cat}: {count} documents")

    print(f"\nResults by Department:")
    for dept, count in sorted(departments.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {dept}: {count} documents")

    print("\n" + "-"*80)

    for i, doc in enumerate(results):
        depth = doc.metadata.get('_depth', 'N/A')
        similarity = doc.metadata.get('_similarity_score', 0)
        similarity_str = f"{similarity:.4f}" if isinstance(similarity, (int, float)) else str(similarity)

        print(f"\n Document {i+1}")
        print(f"   Depth: {depth} | Similarity: {similarity_str}")
        print(f"   Name: {doc.metadata.get('name', 'N/A')}")
        print(f"   Job Title: {doc.metadata.get('job_titles', 'N/A')}")
        print(f"   Job Category: {doc.metadata.get('job_category', 'N/A')}")
        print(f"   Department: {doc.metadata.get('department', 'N/A')}")
        print(f"   Full/Part Time: {doc.metadata.get('full_or_part_time', 'N/A')}")


if __name__ == "__main__":
    print("Loading documents from PGVector...")
    all_docs = vector_store.similarity_search("", k=1000)
    print(f"Loaded {len(all_docs)} documents")

    print("\nExtracting metadata from page_content...")
    enriched_docs = extract_metadata_from_content(all_docs)
    print(f"Enriched {len(enriched_docs)} documents")

    traversal_retriever = build_graph_retriever(enriched_docs)

    query = input('\nEnter query here: ')   # COORDINATING ENGINEER
    print(f"\nSearching for: '{query}'")

    graph_docs = traversal_retriever.invoke(query)
    pretty_print_retrieval(graph_docs)