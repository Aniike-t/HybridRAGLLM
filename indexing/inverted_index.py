# indexing/inverted_index.py (with OR query)
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import config
from data_processing.chunking import create_overlapping_chunks

class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # term -> [(chunk_id, tf), ...]
        self.stemmer = SnowballStemmer("english") if config.STEMMING else None #Changed to snowball
        self.stop_words = set(stopwords.words('english')) if config.STOP_WORDS else set()

    def _preprocess_text(self, text: str) -> List[str]:
        """Tokenizes, stems, and removes stop words."""

        if config.TOKENIZER == "nltk":
            tokens = word_tokenize(text)
        else:
            raise ValueError("Invalid Tokenizer in Config")


        processed_tokens = []
        for token in tokens:
            token = token.lower()
            if config.STOP_WORDS and token in self.stop_words:
                continue
            if config.STEMMING and self.stemmer:
                token = self.stemmer.stem(token)
            processed_tokens.append(token)
        return processed_tokens


    def build(self, chunks: List[Tuple[str, str]]):
        """Builds the inverted index."""
        for chunk_id, chunk_text in chunks:
            tokens = self._preprocess_text(chunk_text)
            term_counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1
            for term, tf in term_counts.items():
                self.index[term].append((chunk_id, tf))
    def retrieve(self, query: str) -> List[str]:
        """Retrieves chunk IDs (disjunctive OR query)."""
        query_terms = self._preprocess_text(query)
        if not query_terms:
            return []

        all_results = set()  # Use a set to avoid duplicates
        for term in query_terms:
            if term in self.index:
                chunk_ids = {chunk_id for chunk_id, _ in self.index[term]}
                all_results.update(chunk_ids)  # Add all matching chunks

        return list(all_results)


    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.index, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.index = pickle.load(f)

    # --- Debugging Methods (NOW INCLUDED) ---
    def print_index_stats(self):
        """Prints statistics about the index (for debugging)."""
        print(f"Number of unique terms: {len(self.index)}")
        total_postings = sum(len(postings) for postings in self.index.values())
        print(f"Total number of postings: {total_postings}")

    def print_term_postings(self, term: str):
        """Prints the postings list for a given term (for debugging)."""
        processed_term = self._preprocess_text(term)[0] if self._preprocess_text(term) else "" #Handle empty
        if processed_term in self.index:
            print(f"Postings for term '{processed_term}': {self.index[processed_term]}")
        else:
            print(f"Term '{processed_term}' not found in index.")

def build_inverted_index(text: str) -> InvertedIndex:
    chunks = create_overlapping_chunks(text, config.CHUNK_SIZE, config.OVERLAP_SIZE)
    inverted_index = InvertedIndex()
    inverted_index.build(chunks)
    return inverted_index