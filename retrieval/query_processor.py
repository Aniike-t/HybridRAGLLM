# retrieval/query_processor.py
from typing import List

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
import config
import google.generativeai as genai

class QueryProcessor:
    def __init__(self):
        self.stemmer = SnowballStemmer("english") if config.STEMMING else None # Use Snowball
        self.stop_words = set(stopwords.words('english')) if config.STOP_WORDS else set()
        if config.USE_GEMINI_QUERY_PROCESSING: # Add to config
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)

    def preprocess_query(self, query: str) -> str:
        """Preprocesses the query, with option to use Gemini."""
        if config.USE_GEMINI_QUERY_PROCESSING:
            return self._gemini_preprocess(query)
        else:
            return self._basic_preprocess(query)

    def _basic_preprocess(self, query: str) -> str:
        """Basic preprocessing (stemming, stop words).  Matches inverted index."""
        if config.TOKENIZER == "nltk":
            tokens = word_tokenize(query.lower())
        else:
            raise ValueError("Invalid Tokenizer selected")

        processed_tokens = []
        for token in tokens:
            if config.STOP_WORDS and token in self.stop_words:
                continue
            if config.STEMMING and self.stemmer:
                token = self.stemmer.stem(token)
            processed_tokens.append(token)

        return " ".join(processed_tokens)

    def _gemini_preprocess(self, query: str) -> str:
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
            response = self.gemini_model.generate_content(prompt)
             # --- Basic Preprocessing on Gemini Output ---
            processed_query = self._basic_preprocess(response.text) #Apply Basic
            return processed_query

        except Exception as e:
            print(f"Error during Gemini query preprocessing: {e}")
            return self._basic_preprocess(query) # Fallback