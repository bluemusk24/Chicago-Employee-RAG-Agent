import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from tqdm import tqdm
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings


connection = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = PGVector(
   embeddings=embedding_model,
   collection_name="chicago_employee_docs",
   connection=connection,
   use_jsonb=True
)

def batch_insert(documents: list, batch_size: int = 100):
    total_batches = range(0, len(documents), batch_size)
    
    for i in tqdm(total_batches, desc="Inserting batches"):
        batch = documents[i:i + batch_size]
        
        vector_store.add_documents(
            batch,
            ids=[str(doc.metadata["row"]) for doc in batch]
        )

    return vector_store
    

if __name__ == "__main__":
    try:
        file_path = "clean_data/employee_data.csv"
        documents = load_data(file_path=file_path)
        vector_db = batch_insert(documents, batch_size=100)  # will process all 1000 docs in 10 batches
        print(f"Collection name: {vector_db.collection_name} has total {len(documents)} documents in vector store.")
    except Exception as e:
        print(f'Error during ingestion: {e}')
    