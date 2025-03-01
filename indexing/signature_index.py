# indexing/signature_index.py
import pickle
from typing import List, Dict, Tuple
import hashlib

import config
from utils.helpers import calculate_signature
from data_processing.chunking import create_overlapping_chunks

class SignatureIndex:
    def __init__(self, signature_size: int = config.SIGNATURE_SIZE):
        self.index: Dict[str, int] = {}  # chunk_id -> signature
        self.signature_size = signature_size

    def build(self, chunks: List[Tuple[str, str]]):
        """Builds the signature index."""
        for chunk_id, chunk_text in chunks:
            signature = calculate_signature(chunk_text, self.signature_size)
            self.index[chunk_id] = signature

    def retrieve(self, query: str) -> List[str]:
        """Retrieves candidate chunk IDs (bitwise AND check)."""
        query_signature = calculate_signature(query, self.signature_size)
        candidate_chunk_ids = []
        for chunk_id, chunk_signature in self.index.items():
            # Correct bitwise AND check:
            if (query_signature & chunk_signature) == query_signature:
                candidate_chunk_ids.append(chunk_id)
        return candidate_chunk_ids

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.index, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.index = pickle.load(f)

    # --- Debugging Methods (NOW INCLUDED) ---
    def print_signature_stats(self):
        print(f"Number of signatures: {len(self.index)}")

    def check_signature(self, chunk_id: str):
        if chunk_id in self.index:
            print(f"Signature for {chunk_id}: {self.index[chunk_id]}")
        else:
            print("Chunk ID not found")

def build_signature_index(text:str) -> SignatureIndex:

    chunks = create_overlapping_chunks(text, config.CHUNK_SIZE, config.OVERLAP_SIZE)
    signature_index = SignatureIndex()
    signature_index.build(chunks)
    return signature_index