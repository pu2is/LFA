"""64-bit character n-gram SimHash for Rescan fuzzy recovery (ADR-0001b D3/D5).

Pure text -> signature computation and comparison, no I/O. The signature is a
coarse locality-sensitive fingerprint of already-normalized text (see
processing/cleaning.py) -- it is a candidate-narrowing signal, never identity
proof; a human still confirms every match (D5).
"""

import hashlib

SIGNATURE_BITS = 64
NGRAM_SIZE = 4

# Application constants (ADR-0001b D3): tunable, not derived from any proof.
# THRESHOLD ~ Hamming distance <= 6/64 bits; MARGIN ~ 3/64 bits of separation
# from the runner-up candidate before a metadata-ambiguous match is trusted.
SIMILARITY_THRESHOLD = 0.90
UNIQUENESS_MARGIN = 0.05


def compute_text_signature(text: str) -> str | None:
    """64-bit SimHash of `text`'s character n-grams, as lowercase hex.

    None when `text` (lowercased) has fewer than NGRAM_SIZE characters --
    too short to yield a single voting n-gram, so no signature is possible.
    """
    normalized = text.lower()
    ngrams = [normalized[i : i + NGRAM_SIZE] for i in range(len(normalized) - NGRAM_SIZE + 1)]
    if not ngrams:
        return None

    bit_votes = [0] * SIGNATURE_BITS
    for ngram in ngrams:
        ngram_hash = int.from_bytes(hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(SIGNATURE_BITS):
            if ngram_hash & (1 << bit):
                bit_votes[bit] += 1
            else:
                bit_votes[bit] -= 1

    signature = 0
    for bit in range(SIGNATURE_BITS):
        if bit_votes[bit] > 0:
            signature |= 1 << bit
    return f"{signature:0{SIGNATURE_BITS // 4}x}"


def similarity(signature_a: str, signature_b: str) -> float:
    """Hamming similarity between two hex-encoded 64-bit signatures, in [0, 1]."""
    hamming_distance = bin(int(signature_a, 16) ^ int(signature_b, 16)).count("1")
    return 1 - (hamming_distance / SIGNATURE_BITS)
