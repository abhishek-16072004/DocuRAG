import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    Handles:
    1. Embedding generation
    2. FAISS index creation
    3. Persistent storage
    4. Loading saved indexes
    5. Similarity search
    """

    def __init__(
        self,
        index_path,
        metadata_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.model = SentenceTransformer(model_name)

        self.index = None
        self.chunks = []

    # ========================================================
    # BUILD INDEX
    # ========================================================

    def build(self, chunks):
        """
        Create embeddings and build FAISS index.
        """

        self.chunks = chunks

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        embeddings = embeddings.astype(
            np.float32
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            embeddings
        )

        return self.index

    # ========================================================
    # SAVE
    # ========================================================

    def save(self):
        """
        Save FAISS index and chunk metadata.
        """

        if self.index is None:
            raise ValueError(
                "Index has not been built."
            )

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        with open(
            self.metadata_path,
            "wb",
        ) as file:

            pickle.dump(
                self.chunks,
                file,
            )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self):
        """
        Load FAISS index and chunk metadata.
        """

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: "
                f"{self.metadata_path}"
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "rb",
        ) as file:

            self.chunks = pickle.load(
                file
            )

        return self.index, self.chunks

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        k=3,
        distance_threshold=None,
    ):
        """
        Convert query into an embedding
        and search FAISS.
        """

        if self.index is None:
            raise ValueError(
                "Index is not loaded or built."
            )

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        query_embedding = query_embedding.astype(
            np.float32
        )

        distances, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for distance, index_number in zip(
            distances[0],
            indices[0],
        ):

            if index_number == -1:
                continue

            if (
                distance_threshold is not None
                and distance > distance_threshold
            ):
                continue

            chunk = self.chunks[
                index_number
            ]

            results.append(
                {
                    "text": chunk["text"],
                    "page_number": chunk[
                        "page_number"
                    ],
                    "distance": float(
                        distance
                    ),
                }
            )

        return results