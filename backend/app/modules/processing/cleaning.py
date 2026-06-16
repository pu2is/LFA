"""Conservative text cleaning before chunking and labeling.

No database access -- a pure function so it can be unit-tested without any
infrastructure, and called from both the sync (issue #6) and async (#9) paths.
"""

import re

_WHITESPACE_RUN = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Lines where fewer than this fraction of non-whitespace characters are
# alphabetic letters are treated as noise (table borders, page numbers,
# barcode rows, etc.) and dropped. 0.3 is intentionally conservative: a
# sentence like "Invoice #2024-001, due 30.06.2024" still has enough letters
# to survive. Raise the threshold only if real noise slips through in testing.
MIN_LETTER_RATIO = 0.3


def clean(text: str) -> str:
    """Normalize extracted document text before chunking and labeling.

    Two conservative transforms only:
    1. Collapse repeated horizontal whitespace on each line.
    2. Drop lines where < 30 % of non-whitespace characters are letters
       (table borders, sequences of digits, symbol-heavy separators).

    Paragraph breaks (blank lines) are preserved but capped at one blank line
    between paragraphs so the splitter sees clean boundaries.
    """
    kept: list[str] = []
    for line in text.splitlines():
        normalized = _WHITESPACE_RUN.sub(" ", line).strip()
        if normalized and _is_noise_line(normalized):
            continue
        kept.append(normalized)

    joined = "\n".join(kept)
    return _EXCESS_BLANK_LINES.sub("\n\n", joined).strip()


def _is_noise_line(line: str) -> bool:
    non_space = [ch for ch in line if not ch.isspace()]
    if not non_space:
        return False
    letter_count = sum(1 for ch in non_space if ch.isalpha())
    return (letter_count / len(non_space)) < MIN_LETTER_RATIO
