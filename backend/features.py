"""
features.py
Turns a question into numbers that the router can learn from.

Two kinds of features are used:
  1. Hand-made features: length, how many digits, maths symbols, keywords
     such as "prove" or "calculate", whether it is a multiple-choice question...
  2. A sentence embedding: a vector of 384 numbers that captures the meaning
     of the question (from the all-MiniLM-L6-v2 model).

The same embedding is reused by the semantic cache in Phase 5, so it is
computed once per query and shared.
"""

import math
import re

import numpy as np

from backend import config

# ---------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------
# ml/build_dataset.py adds one of these instructions to every training prompt.
# Real users do not type them, so we cut them off before computing features.
_INSTRUCTION_MARKERS = [
    "\n\nSolve this maths problem.",
    "\n\nAnswer this multiple-choice question.",
]


def clean_query(text):
    """Remove the grading instructions from a training prompt. Normal queries are unchanged."""
    text = text or ""
    for marker in _INSTRUCTION_MARKERS:
        position = text.find(marker)
        if position != -1:
            text = text[:position]
    return text.strip()


# ---------------------------------------------------------------
# Hand-made features
# ---------------------------------------------------------------
KEYWORD_GROUPS = {
    "kw_reasoning": [
        "prove", "derive", "show that", "step by step", "why", "suppose", "assume",
        "therefore", "implies", "contradiction", "if and only if", "logical", "valid",
        "argument", "conclude",
    ],
    "kw_calculation": [
        "calculate", "compute", "how many", "how much", "total", "average", "percent",
        "each", "sum", "product", "remaining", "left", "times", "ratio", "probability",
        "integral", "derivative", "equation", "solve", "per", "cost", "twice", "half",
    ],
    "kw_code": ["code", "function", "python", "algorithm", "program", "javascript", "sql", "bug"],
    "kw_explain": [
        "explain", "describe", "discuss", "compare", "difference", "summarize",
        "summarise", "analyze", "analyse",
    ],
    "kw_lookup": ["what is", "who", "when", "where", "which", "capital", "define", "name"],
    "kw_negation": ["not", "except", "least", "never", "false", "incorrect"],
}

# One compiled pattern per keyword group, matching whole words or phrases only.
_KEYWORD_PATTERNS = {
    name: re.compile(r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\b")
    for name, words in KEYWORD_GROUPS.items()
}

# Maths symbols: + * / = ^ % < > and the characters for root, integral, sum, pi, times, divide
_MATH_SYMBOLS = set("+*/=^%<>\u221a\u222b\u2211\u03c0\u00d7\u00f7")
_FORMULA_RE = re.compile(r"\\frac|\\sqrt|\\\(|\$|\^|_\{")
_OPTION_RE = re.compile(r"(?m)^[A-D]\. ")


def handcrafted_features(text):
    """Return a dictionary: feature name -> number."""
    text = clean_query(text)
    lower = text.lower()
    words = re.findall(r"[A-Za-z']+", text)
    letters = [c for c in text if c.isalpha()]
    digit_count = sum(1 for c in text if c.isdigit())
    option_lines = _OPTION_RE.findall(text)
    word_count = len(words)

    values = {
        "char_count_log": math.log1p(len(text)),
        "word_count_log": math.log1p(word_count),
        "line_count": text.count("\n") + 1,
        "sentence_count": max(len(re.findall(r"[.!?]+(?:\s|$)", text)), 1),
        "avg_word_length": (sum(len(w) for w in words) / word_count) if word_count else 0.0,
        "long_word_ratio": (sum(1 for w in words if len(w) >= 9) / word_count) if word_count else 0.0,
        "digit_ratio": digit_count / max(len(text), 1),
        "number_count": len(re.findall(r"\d+(?:\.\d+)?", text)),
        "math_symbol_count": sum(1 for c in text if c in _MATH_SYMBOLS),
        "has_formula_markup": 1.0 if _FORMULA_RE.search(text) else 0.0,
        "question_mark_count": text.count("?"),
        "uppercase_ratio": (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0,
        "comma_count": text.count(","),
        "has_options": 1.0 if option_lines else 0.0,
        "option_count": len(option_lines),
    }
    for name, pattern in _KEYWORD_PATTERNS.items():
        values[name] = len(pattern.findall(lower))
    return values


# The fixed order of the hand-made features (the trained model depends on this order).
FEATURE_NAMES = list(handcrafted_features("example").keys())


def handcrafted_matrix(texts):
    """Hand-made features for many questions: one row per question."""
    rows = []
    for text in texts:
        values = handcrafted_features(text)
        rows.append([values[name] for name in FEATURE_NAMES])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------
# Sentence embeddings
# ---------------------------------------------------------------
_embedder = None


def get_embedder():
    """Load the embedding model once, the first time it is needed, then reuse it."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def embed_texts(texts):
    """
    Embeddings for many questions: one row of 384 numbers per question.
    The vectors are normalised to length 1, so similarity between two questions
    is just a dot product (that is what the cache will use in Phase 5).
    """
    cleaned = [clean_query(text) for text in texts]
    vectors = get_embedder().encode(
        cleaned,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_text(text):
    """Embedding for a single question."""
    return embed_texts([text])[0]


# ---------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------
def build_features(texts, embeddings=None):
    """
    The full feature matrix: hand-made features first, then the embedding.
    Pass embeddings if they were already computed (so we do not compute them twice).
    """
    hand = handcrafted_matrix(texts)
    if embeddings is None:
        embeddings = embed_texts(texts)
    return np.hstack([hand, embeddings])