# retrieval/hybrid_retriever.py (with added print statement)
from typing import List, Tuple, Dict, Optional

import config
from indexing.inverted_index import InvertedIndex
from indexing.signature_index import SignatureIndex
from indexing.embedding_index import EmbeddingIndex
from retrieval.query_processor import QueryProcessor
from retrieval.prompt_builder import PromptBuilder
import faiss
import numpy as np
import google.generativeai as genai


class HybridRetriever:
    def __init__(self, inverted_index: Optional[InvertedIndex] = None,
                 signature_index: Optional[SignatureIndex] = None,
                 embedding_index: Optional[EmbeddingIndex] = None):

        self.inverted_index = inverted_index
        self.signature_index = signature_index
        self.embedding_index = embedding_index
        self.chunk_store: Dict[str, str] = {}  # chunk_id -> chunk_text
        self.prompt_builder = PromptBuilder()
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        self.query_processor = QueryProcessor()

    def add_chunks_to_store(self, chunks: List[Tuple[str, str]]):
        for chunk_id, chunk_text in chunks:
            self.chunk_store[chunk_id] = chunk_text

    def retrieve(self, query: str, top_k_initial: int = config.TOP_K_INITIAL,
                 top_k_final: int = config.TOP_K_FINAL) -> List[Tuple[str, float]]:
        """Performs hybrid retrieval."""

        processed_query = self.query_processor.preprocess_query(query)

        # --- ADD THIS PRINT STATEMENT FOR DEBUGGING ---
        print(f"Processed query: '{processed_query}'")
        # ----------------------------------

        # 1. Initial Filtering (Inverted Index or Signature File)
        if self.inverted_index:
            candidate_chunk_ids = self.inverted_index.retrieve(processed_query)
        elif self.signature_index:
            candidate_chunk_ids = self.signature_index.retrieve(processed_query)
        else:
            raise ValueError("No initial retrieval method provided.")

        print(f"Initial retrieval (inverted index) returned: {candidate_chunk_ids}")

        candidate_chunk_ids = candidate_chunk_ids[:top_k_initial]

        if not candidate_chunk_ids:
            return []

        # --- Embedding Retrieval and Re-ranking (CORRECTED) ---
        assert self.embedding_index is not None, "Embedding index not provided."

        # 2. Re-rank with Embeddings (within the candidates)
        if not candidate_chunk_ids:
            return []

        # Retrieve pre-calculated embeddings *directly* from the main embedding index.
        candidate_indices = [self.embedding_index.next_index - len(self.embedding_index.chunk_id_mapping) + i  for i in range(len(self.embedding_index.chunk_id_mapping)) if self.embedding_index.chunk_id_mapping[i] in candidate_chunk_ids]

        candidate_embeddings = self.embedding_index.index.reconstruct_n(candidate_indices[0], len(candidate_indices))


        # Create a temporary FAISS index.
        temp_embedding_index = faiss.IndexFlatIP(self.embedding_index.dimension) # Use same dimension
        temp_embedding_index.add(candidate_embeddings) # Add the embeddings


        # Get query embedding
        query_embedding = self.embedding_index.model.encode([query], convert_to_tensor=False)[0]
        query_embedding = query_embedding.reshape(1, -1)
        if config.FAISS_INDEX_TYPE == "IndexFlatIP":
            faiss.normalize_L2(query_embedding)

        # Search using temporary index
        distances, indices = temp_embedding_index.search(query_embedding, top_k_final)

        # Convert indices back to doc_ids, and include scores
        scores = distances[0]
        results = [(candidate_chunk_ids[indices[0][i]], scores[i]) for i in range(len(indices[0]))] #correct indices
        return results

    def generate_response(self, query: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
        """Generates a response using the Gemini API and retrieved chunks."""
        if not retrieved_chunks:
            return "No relevant information found."

        context = "\n".join([self.chunk_store[chunk_id] for chunk_id, _ in retrieved_chunks])
        prompt = self.prompt_builder.build_prompt(context, query)

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            return f"An error occurred while generating the response: {e}"