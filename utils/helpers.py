# utils/helpers.py
import hashlib

def calculate_signature(text: str, signature_size: int = 64) -> int:
    """Calculates a simple signature for a text using SHA-256 hashing."""
    if signature_size > 256:
        raise ValueError("Signature size cannot be greater than 256 bits")

    sha256_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    signature = int(sha256_hash, 16) % (2**signature_size)
    return signature