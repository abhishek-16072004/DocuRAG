import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(
        self,
        index_path="vector_store/index.faiss",
        metadata_path="vector_store/chunks.pkl"
    ):
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as file:
            self.chunks = pickle.load(file)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("FAISS vectors:", self.index.ntotal)
        print("Total chunks:", len(self.chunks))

    def retrieve(self, query, k=3, distance_threshold=None):

        # Convert query to embedding
        query_embedding = self.model.encode([query])

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        # Search FAISS
        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for rank in range(k):

            chunk_index = indices[0][rank]
            distance = float(distances[0][rank])

            # Apply optional threshold
            if (
                distance_threshold is not None
                and distance > distance_threshold
            ):
                continue

            result = {
                "rank": len(results) + 1,
                "text": self.chunks[chunk_index]["text"],
                "page_number": self.chunks[chunk_index]["page_number"],
                "distance": distance
            }

            results.append(result)

        return results


# Test
if __name__ == "__main__":

    retriever = Retriever()

    query = "What is a perceptron?"

    results = retriever.retrieve(
        query,
        k=3
    )

    print("\nQuery:", query)

    for result in results:

        print("\n-----------------------------")
        print("Rank:", result["rank"])
        print("Page:", result["page_number"])
        print("Distance:", result["distance"])
        print("Text:")
        print(result["text"])