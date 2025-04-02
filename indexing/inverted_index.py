# indexing/inverted_index.py
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math 

import config
from data_processing.chunking import create_overlapping_chunks

class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  
        self.stemmer = SnowballStemmer("english") if config.STEMMING else None
        self.stop_words = set(stopwords.words('english')) if config.STOP_WORDS else set()
        self.doc_lengths: Dict[str, int] = {}  # chunk_id -> length (in words)
        self.avg_doc_length: float = 0  # Average document (chunk) length
        self.k1 = 1.2  # k1 parameter for BM25
        self.b = 0.75   # b parameter for BM25


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
        """Builds the inverted index and calculates document lengths."""
        total_length = 0
        for chunk_id, chunk_text in chunks:
            tokens = self._preprocess_text(chunk_text)
            term_counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1
            for term, tf in term_counts.items():
                self.index[term].append((chunk_id, tf))

            self.doc_lengths[chunk_id] = len(tokens)
            total_length += len(tokens)


        if chunks:  
            self.avg_doc_length = total_length / len(chunks)


    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """Retrieves chunk IDs and scores using BM25."""
        query_terms = self._preprocess_text(query)
        if not query_terms:
            return []

        scores: Dict[str, float] = defaultdict(float)  # chunk_id -> score
        num_chunks = len(self.doc_lengths)

        for term in query_terms:
            if term in self.index:
                idf = math.log(1 + (num_chunks - len(self.index[term]) + 0.5) / (len(self.index[term]) + 0.5))
                for chunk_id, tf in self.index[term]:
                    # BM25 formula
                    numerator = idf * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (self.doc_lengths[chunk_id] / self.avg_doc_length))
                    scores[chunk_id] += numerator / denominator

        ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_results

    def save(self, filepath: str):
        """Saves the inverted index, document lengths, and average length."""
        with open(filepath, 'wb') as f:
            pickle.dump((self.index, self.doc_lengths, self.avg_doc_length), f) 

    def load(self, filepath: str):
        """Loads the inverted index, document lengths, and average length."""
        with open(filepath, 'rb') as f:
            self.index, self.doc_lengths, self.avg_doc_length = pickle.load(f) 

    def get_all_chunk_ids(self):
        """Gets all the stored chunk_ids"""
        all_chunks = []
        for term_list in self.index.values():
            for chunk_id, _ in term_list:
                all_chunks.append(chunk_id)

        return list(set(all_chunks)) #Return unique

    def print_index_stats(self):
        print(f"Number of unique terms: {len(self.index)}")
        total_postings = sum(len(postings) for postings in self.index.values())
        print(f"Total number of postings: {total_postings}")
        print(f"Average document length: {self.avg_doc_length}")


    def print_term_postings(self, term: str):
        processed_term = self._preprocess_text(term)[0] if self._preprocess_text(term) else ""
        if processed_term in self.index:
            print(f"Postings for term '{processed_term}': {self.index[processed_term]}")
        else:
            print(f"Term '{processed_term}' not found in index.")

def build_inverted_index(text: str) -> InvertedIndex:
    chunks = create_overlapping_chunks(text, config.CHUNK_SIZE, config.OVERLAP_SIZE)
    inverted_index = InvertedIndex()
    inverted_index.build(chunks)
    return inverted_index