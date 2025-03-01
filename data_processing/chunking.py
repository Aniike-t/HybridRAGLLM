# data_processing/chunking.py
from typing import List, Tuple, Dict
import nltk
from nltk.tokenize import word_tokenize
import config

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def create_overlapping_chunks(text: str, chunk_size: int, overlap_size: int) -> List[Tuple[str, str]]:
  """
  Creates overlapping text chunks, handling edge cases gracefully.
  chunk_size and overlap_size are now in *words*, not tokens.
  """
  if config.TOKENIZER == "nltk":
      words = word_tokenize(text)  # Tokenize into words
  else:
      raise ValueError("Invalid Tokenizer in Config")

  chunks = []
  start = 0
  chunk_count = 0

  # Handle cases where the text is shorter than the chunk size
  if len(words) <= chunk_size:
      chunk_id = f"chunk_{chunk_count}"
      chunks.append((chunk_id, " ".join(words)))
      return chunks

  while start < len(words):
      end = min(start + chunk_size, len(words))
      chunk_text = " ".join(words[start:end])
      chunk_id = f"chunk_{chunk_count}"
      chunks.append((chunk_id, chunk_text))

      # Prevent infinite loop if overlap_size >= chunk_size
      if overlap_size >= chunk_size:
          start += 1  # Move by at least one word
      else:
          start += (chunk_size - overlap_size)

      chunk_count += 1

  return chunks