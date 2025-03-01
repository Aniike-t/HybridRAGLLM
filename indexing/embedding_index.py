# indexing/embedding_index.py
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    raise  # Re-raise

try:
    import faiss
except ImportError:
    print("Please install faiss: conda install -c conda-forge faiss-cpu (or faiss-gpu)")
    raise  # Re-raise

import config
from data_processing.chunking import create_overlapping_chunks

class EmbeddingIndex:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME,
                index_type: str = config.FAISS_INDEX_TYPE):

        if SentenceTransformer is None:
            raise ImportError("Sentence Transformers not installed.")
        if faiss is None:
            raise ImportError("Faiss not installed")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Select FAISS index type based on config
        if index_type == "IndexFlatIP":  # Inner Product (for cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "IndexFlatL2":  # L2 distance
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IndexIVFFlat": # Inverted File with Flat
            nlist = 100  # Number of clusters (adjust as needed)
            quantizer = faiss.IndexFlatL2(self.dimension) #Define quantizer
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
        # Add other index types (IndexHNSW, etc.) as needed
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")
        self.chunk_id_mapping: Dict[int, str] = {}  # faiss_index -> chunk_id
        self.next_index = 0

    def build(self, chunks: List[Tuple[str, str]]):
        """Builds the embedding index."""
        embeddings = self.model.encode([chunk_text for _, chunk_text in chunks], convert_to_tensor=False)
        embeddings = np.array(embeddings)

        if config.FAISS_INDEX_TYPE == "IndexFlatIP":
            faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        elif config.FAISS_INDEX_TYPE == "IndexIVFFlat":
            self.index.train(embeddings) #IVF needs training


        self.index.add(embeddings)

        # Keep track of chunk IDs
        for i, (chunk_id, _) in enumerate(chunks):
            self.chunk_id_mapping[self.next_index] = chunk_id
            self.next_index += 1

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Retrieves the top_k most similar chunks."""
        query_embedding = self.model.encode([query], convert_to_tensor=False)[0]
        query_embedding = query_embedding.reshape(1, -1)

        if config.FAISS_INDEX_TYPE == "IndexFlatIP":
            faiss.normalize_L2(query_embedding) #Normalize if using IP

        distances, indices = self.index.search(query_embedding, top_k)
        scores = distances[0]

        # Adjust score based on index type (if needed)
        #   FAISS returns distances: lower is better for L2, higher for IP
        #   For cosine similarity, after normalization, inner product IS cosine sim.
        if config.FAISS_INDEX_TYPE == "IndexFlatL2":
            scores = -scores  # Negate L2 distances to make higher scores better

        results = [(self.chunk_id_mapping[indices[0][i]], scores[i]) for i in range(len(indices[0]))]
        return results


    def save(self, filepath: str):
        """Saves the FAISS index and chunk ID mapping."""
        faiss.write_index(self.index, filepath)
        mapping_path = filepath + ".mapping"
        with open(mapping_path, "wb") as f:
            pickle.dump(self.chunk_id_mapping, f)

    def load(self, filepath: str):
        """Loads the FAISS index and chunk ID mapping."""
        mapping_path = filepath + ".mapping"
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

        self.index = faiss.read_index(filepath)
        with open(mapping_path, "rb") as f:
            self.chunk_id_mapping = pickle.load(f)
        self.next_index = len(self.chunk_id_mapping)

    # --- Debugging Method ---
    def test_retrieval(self, query: str, top_k: int = 5):
        """Tests retrieval with a given query (for debugging)."""
        print(f"Testing embedding retrieval with query: '{query}'")
        results = self.retrieve(query, top_k)
        if results:
            print("  Results:")
            for chunk_id, score in results:
                print(f"    - {chunk_id}: {score:.4f}")
        else:
            print("  No results found.")

def build_embedding_index(text:str) -> EmbeddingIndex:

    chunks = create_overlapping_chunks(text, config.CHUNK_SIZE, config.OVERLAP_SIZE)
    embedding_index = EmbeddingIndex()
    embedding_index.build(chunks)
    return embedding_index