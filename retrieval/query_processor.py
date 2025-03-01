# retrieval/query_processor.py
from typing import List, Dict
import google.generativeai as genai
import config

class QueryProcessor:
    def __init__(self):
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)

    def preprocess_query(self, query: str) -> str:
        """Preprocesses the query using Gemini for reformulation/expansion."""

        prompt = f"""
        You are a helpful query processing assistant.  Your task is to reformulate and expand the user's search query to improve retrieval results.

        Original Query:
        {query}

        Instructions:
        1.  Rephrase the query to be more precise and clear.
        2.  Provide a list of related terms, synonyms, and alternative phrasings.
        3.  If the query is ambiguous, provide possible interpretations.
        4.  Output the processed query for an inverted index search, space separated.
        Output:
        """
        try:
            response = self.model.generate_content(prompt)
            processed_query = response.text
            return processed_query

        except Exception as e:
            print(f"Error during Gemini query preprocessing: {e}")
            # Fallback to basic preprocessing if Gemini fails
            return self._basic_preprocess(query)

    def _basic_preprocess(self, query: str) -> str:
        """Basic preprocessing (fallback)."""
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer
        from nltk.corpus import stopwords
        tokens = word_tokenize(query.lower())
        stemmer = PorterStemmer() if config.STEMMING else None
        stop_words = set(stopwords.words('english')) if config.STOP_WORDS else set()
        processed_tokens = []
        for token in tokens:
            if config.STOP_WORDS and token in stop_words:
                continue
            if config.STEMMING and stemmer:
                token = stemmer.stem(token)
            processed_tokens.append(token)
        return " ".join(processed_tokens)