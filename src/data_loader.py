from langchain_community.document_loaders.csv_loader import CSVLoader

def load_data(file_path: str):
    try:
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from '{file_path}' \n")
        return documents
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    file_path = "clean_data/employee_data.csv"
    documents = load_data(file_path=file_path)
    print(documents[999].page_content, '\n',documents[999].metadata)  # preview last doc