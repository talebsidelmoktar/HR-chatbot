from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

class CustomEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model using SentenceTransformer.
        :param model_name: The name of the model to use.
        """
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        """
        Generate embeddings for a list of documents.
        :param texts: List of document strings.
        :return: NumPy array of embeddings.
        """
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    def embed_query(self, text):
        """
        Generate an embedding for a single query.
        :param text: A single query string.
        :return: A NumPy array of the query embedding.
        """
        return self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]

def upload_htmls():
    """
    Upload HTML documents into a FAISS vector database using CustomEmbeddings.
    """
    # Load documents from the specified directory
    loader = DirectoryLoader(path="hr-policies")
    documents = loader.load()
    print(f"{len(documents)} Pages Loaded")
    
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50, 
        separators=["\n\n", "\n", " ", ""]
    )
    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_documents)} Documents...")
    
    # Create embeddings and store in FAISS
    embeddings = CustomEmbeddings()
    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local("faiss_index")
    print("Vector database created and saved successfully.")

def faiss_query(query_text="Explain the Candidate Onboarding process."):
    """
    Query the FAISS vector database for semantic similarity.
    :param query_text: The query string to search for.
    """
    # Load the saved database
    embeddings = CustomEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings)
    
    # Perform similarity search
    docs = new_db.similarity_search(query_text)
    
    # Print results
    for doc in docs:
        print("\n##---- Page ---##")
        print(doc.metadata['source'])
        print("##---- Content ---##")
        print(doc.page_content)

if __name__ == "__main__":
    # First, upload documents and create the vector database
    print("Starting document upload and database creation...")
    upload_htmls()
    
    # Then, run a sample query
    print("\nRunning sample query...")
    faiss_query()