"""
cache.py
The semantic cache.

A normal cache only recognises identical text. A semantic cache recognises
questions that MEAN the same thing, using the sentence embeddings from features.py:

    similarity = embedding of the new question . embedding of a cached question

(The embeddings are normalised, so a simple dot product gives the cosine
similarity: 1.0 means identical meaning, values near 0 mean unrelated.)

A cached answer is reused only if BOTH are true:
  1. similarity >= CACHE_THRESHOLD, and
  2. three safety guards pass (numbers, negation, multiple choice).
The guards exist because two questions can look almost identical to an embedding
model and still need different answers ("15% of 200" vs "15% of 300").

Entries are stored in the SQLite file logs/requests.db, so the cache survives
server restarts, and are also kept in memory for fast lookup.
"""

import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

from backend import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT    NOT NULL,
    embedding_model  TEXT    NOT NULL,
    query            TEXT    NOT NULL,
    embedding        BLOB    NOT NULL,
    answer           TEXT    NOT NULL,
    tier             TEXT    NOT NULL,
    model_name       TEXT    NOT NULL,
    input_tokens     INTEGER NOT NULL,
    output_tokens    INTEGER NOT NULL,
    hit_count        INTEGER NOT NULL DEFAULT 0
)
"""

# FastAPI can serve several requests at once, so changes to the cache are
# protected by a lock.
_lock = threading.Lock()

# The cache in memory: a list of entries and one matrix with one embedding per row.
_entries = []
_matrix = None


@contextmanager
def _connect():
    """Open a connection, save changes if everything worked, always close it."""
    connection = sqlite3.connect(config.DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


# ---------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------
def normalize(text):
    """Lower-case, single spaces, plain apostrophes: for comparing texts."""
    return " ".join((text or "").lower().replace("\u2019", "'").split())


# Whole numbers, decimals and numbers with thousands separators: 15, 3.14, 1,200
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def extract_numbers(text):
    """All numbers in a text, written in one standard form and sorted."""
    found = []
    for token in _NUMBER_RE.findall(text or ""):
        value = float(token.replace(",", ""))
        found.append(str(int(value)) if value.is_integer() else str(value))
    return sorted(found)


_NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor",
    "cannot", "without", "except", "false", "incorrect",
}


def has_negation(text):
    """True if the text contains a common negation word (not, never, isn't, ...)."""
    for word in re.findall(r"[a-z']+", normalize(text)):
        if word in _NEGATION_WORDS or word.endswith("n't"):
            return True
    return False


# A line that starts like a multiple-choice option:  "A. something"  or  "(B) something"
_OPTION_LINE_RE = re.compile(r"(?m)^\s*\(?[A-D][.)]\s+\S")


def is_multiple_choice(text):
    return len(_OPTION_LINE_RE.findall(text or "")) >= 2


def guard_failure(query_a, query_b):
    """
    Checks whether two similar questions are SAFE to treat as the same.
    Returns None if they are safe, or a short reason why they are not.
    """
    if normalize(query_a) == normalize(query_b):
        return None  # the same question: always safe
    if is_multiple_choice(query_a) or is_multiple_choice(query_b):
        return "multiple-choice question: only an exact repeat may match"
    if extract_numbers(query_a) != extract_numbers(query_b):
        return "the two questions contain different numbers"
    if has_negation(query_a) != has_negation(query_b):
        return "one question is negated and the other is not"
    return None


# ---------------------------------------------------------------
# Loading and storing
# ---------------------------------------------------------------
def init():
    """
    Create the table if needed and load saved entries into memory.
    Called once when the server starts. Entries made with a different embedding
    model are ignored, because their vectors would not be comparable.
    """
    global _entries, _matrix

    loaded = []
    vectors = []
    skipped = 0

    with _connect() as connection:
        connection.execute(SCHEMA)
        rows = connection.execute("SELECT * FROM cache_entries ORDER BY id").fetchall()

    for row in rows:
        if row["embedding_model"] != config.EMBEDDING_MODEL:
            skipped += 1
            continue
        vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
        loaded.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "query": row["query"],
                "answer": row["answer"],
                "tier": row["tier"],
                "model_name": row["model_name"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "hit_count": row["hit_count"],
            }
        )

    with _lock:
        _entries = loaded
        _matrix = np.vstack(vectors) if vectors else None

    message = f"Cache: loaded {len(loaded)} entries from the database"
    if skipped:
        message += f" (ignored {skipped} made with a different embedding model)"
    print(message)


def add(query, embedding, answer, tier, model_name, input_tokens, output_tokens):
    """Store an answered question. Returns the new entry's id."""
    global _matrix

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _lock:
        with _connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO cache_entries (
                    created_at, embedding_model, query, embedding, answer,
                    tier, model_name, input_tokens, output_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at, config.EMBEDDING_MODEL, query, vector.tobytes(), answer,
                    tier, model_name, int(input_tokens), int(output_tokens),
                ),
            )
            entry_id = cursor.lastrowid

        _entries.append(
            {
                "id": entry_id,
                "created_at": created_at,
                "query": query,
                "answer": answer,
                "tier": tier,
                "model_name": model_name,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "hit_count": 0,
            }
        )
        if _matrix is None:
            _matrix = vector.reshape(1, -1).copy()
        else:
            _matrix = np.vstack([_matrix, vector])
    return entry_id


def _bump_hit_count(entry_id):
    with _connect() as connection:
        connection.execute(
            "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?", (entry_id,)
        )


# ---------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------
def lookup(query, embedding, threshold=None, use_guards=True):
    """
    Look for a cached answer to a question that means the same thing.

    Returns a dictionary:
        hit         True if a cached answer can be reused
        entry       the cached entry (question, answer, tokens...) if hit, else None
        similarity  on a hit: similarity to the matched question
                    on a miss: similarity to the closest cached question (None if the cache is empty)
        blocked_by  on a miss: why a similar-enough question was NOT reused (a guard), else None
    """
    if threshold is None:
        threshold = config.CACHE_THRESHOLD
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

    with _lock:
        if _matrix is None or len(_entries) == 0:
            return {"hit": False, "entry": None, "similarity": None, "blocked_by": None}

        similarities = _matrix @ vector
        order = np.argsort(-similarities)  # most similar first
        best_similarity = float(similarities[order[0]])
        blocked_by = None

        for index in order:
            similarity = float(similarities[index])
            if similarity < threshold:
                break  # everything after this is even less similar

            entry = _entries[index]
            reason = guard_failure(query, entry["query"]) if use_guards else None
            if reason is None:
                entry["hit_count"] += 1
                _bump_hit_count(entry["id"])
                return {"hit": True, "entry": dict(entry), "similarity": similarity, "blocked_by": None}
            if blocked_by is None:
                blocked_by = reason

        return {"hit": False, "entry": None, "similarity": best_similarity, "blocked_by": blocked_by}


# ---------------------------------------------------------------
# Information and clearing
# ---------------------------------------------------------------
def info():
    """Short summary, for the /health endpoint."""
    with _lock:
        count = len(_entries)
        total_hits = sum(entry["hit_count"] for entry in _entries)
    return {"entries": count, "total_hits": total_hits, "threshold": config.CACHE_THRESHOLD}


def recent(limit=20):
    """The most recently stored entries, newest first."""
    with _lock:
        newest = list(reversed(_entries[-limit:]))
    return [dict(entry) for entry in newest]


def clear():
    """Delete every cached entry. Returns how many were removed."""
    global _entries, _matrix
    with _lock:
        with _connect() as connection:
            cursor = connection.execute("DELETE FROM cache_entries")
            deleted = cursor.rowcount
            connection.execute("DELETE FROM sqlite_sequence WHERE name = 'cache_entries'")
        _entries = []
        _matrix = None
    return deleted