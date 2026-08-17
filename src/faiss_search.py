import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Our documents
documents = [
    "RAG combines retrieval with generation.",
    "FAISS is used for similarity search.",
    "Embeddings represent text as numerical vectors."
]


# 3. Convert documents into embeddings
embeddings = model.encode(documents)

# FAISS expects float32
embeddings = np.array(embeddings).astype("float32")


# 4. Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)


# 5. Add document embeddings
index.add(embeddings)

print("Total vectors in index:", index.ntotal)


# 6. User query
query = "What is FAISS used for?"

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")


# 7. Search for top 2 results
k = 2

distances, indices = index.search(query_embedding, k)


# 8. Display results
print("\nQuery:", query)

for i in range(k):
    document_index = indices[0][i]

    print(f"\nRank {i + 1}")
    print("Document:", documents[document_index])
    print("Distance:", distances[0][i])