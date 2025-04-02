# --- PDF Processing ---
PDF_EXTRACTOR = "PyMuPDF"

# --- Chunking ---
CHUNK_SIZE = 200  
OVERLAP_SIZE = 50 

# --- Inverted Index ---
STEMMING = True  
STOP_WORDS = True
TOKENIZER = "nltk"

# --- Signature File ---
SIGNATURE_SIZE = 64

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_TYPE = "IndexFlatIP"

#--- Retrieval ---
TOP_K_INITIAL = 100
TOP_K_FINAL = 10

#--- File Paths ---
INDEX_DIR = "indices"
INVERTED_INDEX_PATH = f"{INDEX_DIR}/inverted_index.pkl"
SIGNATURE_INDEX_PATH = f"{INDEX_DIR}/signature_index.pkl"
EMBEDDING_INDEX_PATH = f"{INDEX_DIR}/embedding_index.faiss"
METADATA_PATH = f"{INDEX_DIR}/metadata.json"

# --- Data ---
DATA_DIR = "data"

# --- Gemini API ---
GEMINI_API_KEY = "KEY"  
GEMINI_MODEL_NAME = "models/gemini-1.5-pro-latest"  # Model
QUERY_EXPANSION = True
USE_GEMINI_QUERY_PROCESSING = True # True/False to enable

# --- Ngrams ---
NGRAMS = 0 