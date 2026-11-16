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
  2. the safety guards pass.
The guards exist because two questions can look almost identical to an embedding
model and still need different answers ("15% of 200" vs "15% of 300").

The guards are simple rules, not understanding. They block a match when they see
a sign that the answers should differ: different numbers (digits, number words,
ordinals, in order), negation, a different arithmetic operation, opposing words,
different names, the same words in a different order, non-English text, or a
multiple-choice question. When a guard is unsure it errs on the side of NOT
reusing the answer: a lost hit costs a little money, a wrong reuse costs a wrong answer.

is_time_sensitive() recognises questions whose answer changes over time
("what is today's date", "latest iPhone"). The server never caches those.

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
# Safety guards: helpers
# ---------------------------------------------------------------
def normalize(text):
    """Lower-case, single spaces, plain apostrophes: for comparing texts."""
    return " ".join((text or "").lower().replace("\u2019", "'").split())


def _word_list(text):
    """
    The words of a text, in order, lower-case. A trailing 's is removed
    ("Japan's" -> "japan"). Hyphens split words ("non-leap" -> "non", "leap").
    """
    words = []
    for token in re.findall(r"[a-z]+(?:'[a-z]+)?", normalize(text)):
        if token.endswith("'s"):
            token = token[:-2]
        words.append(token)
    return words


# ---------------------------------------------------------------
# Time-sensitive questions
# ---------------------------------------------------------------
_TIME_SENSITIVE_RE = re.compile(
    r"\b(?:"
    r"today|tonight|tomorrow|yesterday|now|currently|current|latest|newest|recent|recently|"
    r"right now|at the moment|nowadays|up to date|"
    r"this (?:morning|afternoon|evening|week|weekend|month|year)|"
    r"last (?:night|week|weekend|month|year)|"
    r"next (?:week|weekend|month|year)|"
    r"weather|forecast|news|trending|score|scores|"
    r"stock price|share price|price of|exchange rate|"
    r"what time is it|what(?:'s| is) the time(?! complexity)|what day is it|"
    r"what(?:'s| is) the date(?! of)|"
    r"who is the (?:president|prime minister|ceo|chief minister|governor)"
    r")\b",
    re.IGNORECASE,
)


def is_time_sensitive(text):
    """
    True if the answer to this question probably changes over time, so a stored
    answer would go stale. Errs on the side of True: a question wrongly flagged
    is simply not cached, which costs a little money and nothing else.
    """
    return bool(_TIME_SENSITIVE_RE.search((text or "").replace("\u2019", "'")))


# ---------------------------------------------------------------
# Guard 1: numbers (digits, number words and ordinals), in order
# ---------------------------------------------------------------
# Whole numbers, decimals and numbers with thousands separators: 15, 3.14, 1,200
_DIGIT_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_SMALL_NUMBERS = {**_UNITS, **_ORDINALS}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}


def _spoken_numbers(words):
    """
    Numbers written as words, as whole numbers.
    "twenty five" -> 25, "two hundred" -> 200, "first" -> 1, "third" -> 3.
    Longer phrases such as "two hundred and fifty" come out as separate numbers
    (200 and 50), which will not match "250": the guard then blocks the match,
    which is the safe direction.
    """
    values = []
    i = 0
    while i < len(words):
        word = words[i]
        if word in _TENS:
            value = _TENS[word]
            if i + 1 < len(words) and 1 <= _SMALL_NUMBERS.get(words[i + 1], 0) <= 9:
                value += _SMALL_NUMBERS[words[i + 1]]
                i += 1
            i += 1
        elif word in _SMALL_NUMBERS:
            value = _SMALL_NUMBERS[word]
            i += 1
        elif word in _SCALES:
            value = _SCALES[word]
            i += 1
        else:
            i += 1
            continue
        while i < len(words) and words[i] in _SCALES:
            value *= _SCALES[words[i]]
            i += 1
        values.append(value)
    return values


def extract_numbers(text):
    """
    The numbers in a text, each in one standard form: first the digits, then the
    number words, each in the order they appear. The ORDER matters:
    "1000 > 999" is not "999 > 1000", and 03/04 is not 04/03.
    """
    found = []
    for token in _DIGIT_NUMBER_RE.findall(text or ""):
        value = float(token.replace(",", ""))
        found.append(str(int(value)) if value.is_integer() else str(value))
    for value in _spoken_numbers(_word_list(text)):
        found.append(str(value))
    return found


# ---------------------------------------------------------------
# Guard 2: negation (including the prefix "non-")
# ---------------------------------------------------------------
_NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "non",
    "cannot", "without", "except", "false", "incorrect",
    # common negations from a few other languages, written with English letters
    "pas", "jamais", "nicht", "kein", "keine", "nie", "niet", "nunca", "nao",
}


def has_negation(text):
    """True if the text contains a negation word (not, never, non-, isn't, n'est, ...)."""
    for word in _word_list(text):
        if word in _NEGATION_WORDS or word.endswith("n't") or word.startswith("n'"):
            return True
    return False


# ---------------------------------------------------------------
# Guard 3: arithmetic operations
# ---------------------------------------------------------------
_OPERATION_WORDS = {
    "add": {"add", "adds", "added", "adding", "plus", "sum", "addition"},
    "subtract": {"subtract", "subtracts", "subtracted", "subtracting", "minus", "subtraction"},
    "multiply": {"multiply", "multiplied", "multiplying", "multiplication", "times"},
    "divide": {"divide", "divided", "dividing", "division", "quotient"},
}
_OPERATION_SYMBOLS = {
    "+": "add",
    "\u2212": "subtract",   # the true minus sign (an ordinary hyphen is NOT used)
    "\u00d7": "multiply",
    "*": "multiply",
    "\u00f7": "divide",
}


def _operations_in(text):
    """Which arithmetic operations does a text ask for? A set such as {"add"}."""
    found = set()
    words = set(_word_list(text))
    for name, vocabulary in _OPERATION_WORDS.items():
        if words & vocabulary:
            found.add(name)
    for symbol, name in _OPERATION_SYMBOLS.items():
        if symbol in (text or ""):
            found.add(name)
    return found


# ---------------------------------------------------------------
# Guard 4: opposing words
# ---------------------------------------------------------------
# If both questions use words from the same group, but different ones,
# they are probably asking opposite things. Synonyms (big and large) are
# deliberately NOT in the same group.
_OPPOSITE_GROUPS = [
    {"ascending", "descending"}, {"increasing", "decreasing"}, {"increase", "decrease"},
    {"maximum", "minimum"}, {"largest", "smallest"}, {"biggest", "smallest"},
    {"tallest", "shortest"}, {"longest", "shortest"}, {"highest", "lowest"},
    {"fastest", "slowest"}, {"oldest", "youngest"}, {"greatest", "least"},
    {"before", "after"}, {"above", "below"}, {"north", "south"}, {"east", "west"},
    {"left", "right"}, {"clockwise", "counterclockwise"}, {"hot", "cold"},
    {"boiling", "freezing"}, {"melting", "freezing"}, {"true", "false"},
    {"positive", "negative"}, {"odd", "even"}, {"prime", "composite"},
    {"encrypt", "decrypt"}, {"compress", "decompress"}, {"import", "export"},
    {"push", "pop"}, {"enqueue", "dequeue"}, {"upper", "lower"}, {"input", "output"},
    {"open", "close"}, {"start", "stop"}, {"win", "lose"}, {"won", "lost"},
    {"buy", "sell"}, {"accept", "reject"}, {"allow", "deny"}, {"include", "exclude"},
    {"public", "private"}, {"derivative", "integral"}, {"differentiate", "integrate"},
    {"synchronous", "asynchronous"}, {"mean", "median", "mode"},
    {"sine", "cosine", "tangent"},
    # added after the flaw tests of Phase 8
    {"easy", "hard"}, {"easy", "difficult"}, {"easier", "harder"}, {"easiest", "hardest"},
    {"bigger", "smaller"}, {"larger", "smaller"},
    {"small", "large", "medium"}, {"small", "big"}, {"little", "big"}, {"tiny", "huge"},
    {"radius", "diameter", "circumference"}, {"raw", "cooked"},
    {"more", "less", "fewer"}, {"older", "younger"},
    {"today", "tomorrow", "yesterday", "tonight"},
]


def _opposite_word_conflict(words_a, words_b):
    """A reason string if the two word sets use different words of one opposing group."""
    for group in _OPPOSITE_GROUPS:
        in_a = group & words_a
        in_b = group & words_b
        if in_a and in_b and in_a != in_b:
            return (
                "the questions use opposing words ("
                + "/".join(sorted(in_a)) + " vs " + "/".join(sorted(in_b)) + ")"
            )
    return None


# ---------------------------------------------------------------
# Guard 5: names (capitalised words that are not at the start of a sentence)
# ---------------------------------------------------------------
def _proper_nouns(text):
    """
    Capitalised words that are not the first word of a sentence: names of places,
    people, languages, technologies ("Australia", "Spanish", "Python", "CPU").
    Returned in lower case.
    """
    names = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        tokens = re.findall(r"[A-Za-z][A-Za-z'\u2019\-]*", sentence)
        for position, token in enumerate(tokens):
            if position == 0:
                continue
            token = token.strip("'\u2019-")
            if token.lower().endswith(("'s", "\u2019s")):
                token = token[:-2]
            if len(token) < 2 or not token[0].isupper():
                continue
            if token.startswith(("I'", "I\u2019")):
                continue
            names.add(token.lower())
    return names


# ---------------------------------------------------------------
# Guard 6: word order
# ---------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "as", "and", "or", "but", "if", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "you", "your", "we", "our", "they", "their", "he", "she",
    "his", "her", "can", "could", "would", "should", "will", "shall", "may",
    "might", "must", "what", "which", "who", "whom", "whose", "how", "why",
    "when", "where", "there", "here", "about", "into", "than", "then", "so",
    "such", "any", "some", "also", "please", "tell", "give", "between",
}


def _content_words(words):
    """The meaningful words of a question, in order (small filler words removed)."""
    return [word for word in words if word not in _STOPWORDS]


# ---------------------------------------------------------------
# Non-English text and multiple choice
# ---------------------------------------------------------------
def _has_non_english_letters(text):
    """True if the text contains letters outside plain English (accents, Hindi, Chinese, ...)."""
    return any(ch.isalpha() and ord(ch) > 127 for ch in (text or ""))


# A line that starts like a multiple-choice option:  "A. something"  or  "(B) something"
_OPTION_LINE_RE = re.compile(r"(?m)^\s*\(?[A-D][.)]\s+\S")


def is_multiple_choice(text):
    return len(_OPTION_LINE_RE.findall(text or "")) >= 2


# ---------------------------------------------------------------
# The combined check
# ---------------------------------------------------------------
def guard_failure(query_a, query_b):
    """
    Checks whether two similar questions are SAFE to treat as the same.
    Returns None if they are safe, or a short reason why they are not.
    """
    if normalize(query_a) == normalize(query_b):
        return None  # the same question: always safe

    if _has_non_english_letters(query_a) or _has_non_english_letters(query_b):
        return "non-English text: the guards cannot read it, so only an exact repeat may match"

    if is_multiple_choice(query_a) or is_multiple_choice(query_b):
        return "multiple-choice question: only an exact repeat may match"

    if extract_numbers(query_a) != extract_numbers(query_b):
        return "the two questions contain different numbers"

    if has_negation(query_a) != has_negation(query_b):
        return "one question is negated and the other is not"

    operations_a = _operations_in(query_a)
    operations_b = _operations_in(query_b)
    if operations_a and operations_b and operations_a != operations_b:
        return "the two questions ask for different arithmetic operations"

    words_a = _word_list(query_a)
    words_b = _word_list(query_b)

    conflict = _opposite_word_conflict(set(words_a), set(words_b))
    if conflict:
        return conflict

    # Block only if EACH question names something that the other does not contain.
    # (If just one side has an extra name, the questions may still be paraphrases.)
    names_a = _proper_nouns(query_a)
    names_b = _proper_nouns(query_b)
    if names_a and names_b and not names_a <= set(words_b) and not names_b <= set(words_a):
        return "each question names something (a person, place or term) that the other does not"

    content_a = _content_words(words_a)
    content_b = _content_words(words_b)
    if len(content_a) >= 2 and content_a != content_b and sorted(content_a) == sorted(content_b):
        return "the same words appear in a different order"

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