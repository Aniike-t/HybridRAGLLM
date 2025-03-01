# main.py
import os
import json
from typing import Dict, Tuple, List
import time
import logging

import config
from data_processing.pdf_extractor import process_pdf
from data_processing.chunking import create_overlapping_chunks
from indexing.inverted_index import build_inverted_index, InvertedIndex
from indexing.signature_index import build_signature_index, SignatureIndex
from indexing.embedding_index import build_embedding_index, EmbeddingIndex
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_processor import QueryProcessor

# --- Configure Logging ---
logging.basicConfig(
    filename="retrieval_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)

def build_and_save_indices(pdf_path: str) -> Tuple[InvertedIndex, SignatureIndex, EmbeddingIndex, List[Tuple[str,str]]]:
    """Builds and saves all indices."""
    logging.info("Building indices...")
    start_time = time.time()

    pdf_text, metadata = process_pdf(pdf_path)
    print(f"Extracted Text (First 500 characters):\n{pdf_text[:500]}...")
    logging.info(f"Extracted text length: {len(pdf_text)}")

    chunks = create_overlapping_chunks(pdf_text, config.CHUNK_SIZE, config.OVERLAP_SIZE)
    logging.info(f"Number of Chunks created: {len(chunks)}")

    inverted_index = build_inverted_index(pdf_text)
    signature_index = build_signature_index(pdf_text)
    embedding_index = build_embedding_index(pdf_text)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    inverted_index.save(config.INVERTED_INDEX_PATH)
    signature_index.save(config.SIGNATURE_INDEX_PATH)
    embedding_index.save(config.EMBEDDING_INDEX_PATH)

    metadata["pdf_text"] = pdf_text
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    end_time = time.time()
    logging.info(f"Indices built and saved in {end_time - start_time:.2f} seconds.")
    return inverted_index, signature_index, embedding_index, chunks

def load_indices() -> Tuple[InvertedIndex, SignatureIndex, EmbeddingIndex, List[Tuple[str,str]]]:
    """Loads indices from disk."""
    logging.info("Loading indices from disk...")
    start_time = time.time()

    if not (os.path.exists(config.INVERTED_INDEX_PATH) and
            os.path.exists(config.SIGNATURE_INDEX_PATH) and
            os.path.exists(config.EMBEDDING_INDEX_PATH) and
            os.path.exists(config.EMBEDDING_INDEX_PATH + ".mapping") and
            os.path.exists(config.METADATA_PATH)):
        raise FileNotFoundError("One or more index files are missing. Rebuilding indices.")

    inverted_index = InvertedIndex()
    inverted_index.load(config.INVERTED_INDEX_PATH)

    signature_index = SignatureIndex()
    signature_index.load(config.SIGNATURE_INDEX_PATH)

    embedding_index = EmbeddingIndex()
    embedding_index.load(config.EMBEDDING_INDEX_PATH)

    with open(config.METADATA_PATH, "r") as f:
        metadata = json.load(f)
    chunks = create_overlapping_chunks(metadata["pdf_text"], config.CHUNK_SIZE, config.OVERLAP_SIZE)

    end_time = time.time()
    logging.info(f"Indices loaded in {end_time - start_time:.2f} seconds.")
    return inverted_index, signature_index, embedding_index, chunks


def main():
    """Main function."""
    pdf_filename = "example.pdf"
    pdf_path = os.path.join(config.DATA_DIR, pdf_filename)

    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        print(f"Error: PDF file not found: {pdf_path}")
        return

    use_inverted_index = True

    try:
        inverted_index, signature_index, embedding_index, chunks = load_indices()
    except FileNotFoundError:
        inverted_index, signature_index, embedding_index, chunks = build_and_save_indices(pdf_path)

    if inverted_index:
        inverted_index.print_index_stats()
        inverted_index.print_term_postings("sabotage")
        inverted_index.print_term_postings("author")
        inverted_index.print_term_postings("published")
    if signature_index:
        signature_index.print_signature_stats()

    # --- Test Embedding Retrieval ---
    if embedding_index: #added if
        embedding_index.test_retrieval("sabotage principles")

    if use_inverted_index:
        retriever = HybridRetriever(inverted_index=inverted_index, embedding_index=embedding_index)
    else:
        retriever = HybridRetriever(signature_index=signature_index, embedding_index=embedding_index)

    retriever.add_chunks_to_store(chunks)

    print("--- Sample Chunks ---")
    for chunk_id, chunk_text in chunks[:5]:
        print(f"{chunk_id}: {chunk_text[:100]}...")
    print("-" * 20)

    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        logging.info(f"Original Query: {query}")

        start_time = time.time()
        retrieved_chunks = retriever.retrieve(query)  # Use the retriever
        answer = retriever.generate_response(query, retrieved_chunks)
        end_time = time.time()

        print("\n--- Hybrid Retrieval Results ---")
        if retrieved_chunks:
            for chunk_id, score in retrieved_chunks:
                print(f"  - {chunk_id}: {score:.4f} - {retriever.chunk_store[chunk_id][:100]}...")
                logging.info(f"Retrieved Chunk: {chunk_id}, Score: {score:.4f}")
        else:
            print("No results found.")
            logging.info("No results found.")
        print(f"Retrieval time: {end_time - start_time:.4f} seconds")
        print(f"\n--- LLM Response ---\n{answer}")
        logging.info(f"LLM Response: {answer}")

if __name__ == "__main__":
    main()