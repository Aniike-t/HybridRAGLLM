# config.py

# --- PDF Processing ---
PDF_EXTRACTOR = "PyMuPDF"

# --- Chunking ---
CHUNK_SIZE = 200  # Reduced for debugging
OVERLAP_SIZE = 50 # Reduced

# --- Inverted Index ---
STEMMING = True  # Try True/False
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
GEMINI_API_KEY = "AIzaSyCT6Nbne2XwYtuHX9htjcGqGDxhTYJaD28"  # Replace with your actual API key
GEMINI_MODEL_NAME = "models/gemini-1.5-pro-latest"  # Or "gemini-pro-vision" if you need image support
QUERY_EXPANSION = False #Disable temporarily

# --- Ngrams ---
NGRAMS = 0 #Disable temporarily