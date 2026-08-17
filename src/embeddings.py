from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
texts = [
    "RAG combines retrieval with generation.",
    "FAISS is used for similarity search.",
    "Embeddings represent text as numerical vectors."
]

# Generate embeddings
embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)
print("Number of documents:", len(texts))
print("Embedding dimension:", embeddings.shape[1])