"""
config.py
Central place for every setting in the project.

Other files do:  from backend import config
and then use:    config.SMALL_MODEL, config.DB_PATH, ...

Values come from the .env file, with safe defaults if a value is missing.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# The project root is two levels up from this file:
# llm-router-project/backend/config.py  ->  llm-router-project/
BASE_DIR = Path(__file__).resolve().parent.parent

# Read the .env file in the project root and load its values
# into environment variables.
load_dotenv(BASE_DIR / ".env")


def _float_env(name, default):
    """Read a decimal number from the environment, falling back to a default."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _int_env(name, default):
    """Read a whole number from the environment, falling back to a default."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


# ---------------------------------------------------------------
# Groq API and model names
# ---------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SMALL_MODEL = os.getenv("SMALL_MODEL", "openai/gpt-oss-20b")
LARGE_MODEL = os.getenv("LARGE_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------
# Generation settings (the same for both models, so comparisons are fair)
# ---------------------------------------------------------------
# "low", "medium" or "high". An empty value disables the setting.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low").strip().lower()
# Maximum tokens a model may generate (thinking tokens count toward this)
MAX_OUTPUT_TOKENS = _int_env("MAX_OUTPUT_TOKENS", 1024)

# ---------------------------------------------------------------
# Embedding model (used by the semantic cache and the router)
# ---------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------
# Tunable thresholds: the two main "dials" of the research project
# ---------------------------------------------------------------
# Cache hit if cosine similarity >= this value
CACHE_THRESHOLD = _float_env("CACHE_THRESHOLD", 0.90)
# Use the small model if P(small model is good enough) >= this value
ROUTER_THRESHOLD = _float_env("ROUTER_THRESHOLD", 0.75)

# ---------------------------------------------------------------
# Prices in USD per 1 million tokens (used to compute cost)
# ---------------------------------------------------------------
SMALL_INPUT_PRICE_PER_M = _float_env("SMALL_INPUT_PRICE_PER_M", 0.075)
SMALL_OUTPUT_PRICE_PER_M = _float_env("SMALL_OUTPUT_PRICE_PER_M", 0.30)
LARGE_INPUT_PRICE_PER_M = _float_env("LARGE_INPUT_PRICE_PER_M", 0.15)
LARGE_OUTPUT_PRICE_PER_M = _float_env("LARGE_OUTPUT_PRICE_PER_M", 0.60)

# ---------------------------------------------------------------
# Lookup tables used by the rest of the code.
# A "tier" is simply the word "small" or "large".
# ---------------------------------------------------------------
MODEL_BY_TIER = {
    "small": SMALL_MODEL,
    "large": LARGE_MODEL,
}

# tier -> (input price, output price), each per 1 million tokens
PRICES_PER_M = {
    "small": (SMALL_INPUT_PRICE_PER_M, SMALL_OUTPUT_PRICE_PER_M),
    "large": (LARGE_INPUT_PRICE_PER_M, LARGE_OUTPUT_PRICE_PER_M),
}

# ---------------------------------------------------------------
# Folder and file locations
# ---------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RESPONSES_DIR = DATA_DIR / "responses"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "ml" / "models"
RESULTS_DIR = BASE_DIR / "experiments" / "results"
FRONTEND_DIR = BASE_DIR / "frontend"
LOGS_DIR = BASE_DIR / "logs"

DB_PATH = LOGS_DIR / "requests.db"


def ensure_directories():
    """
    Create any project folder that is missing.
    Git does not store empty folders, so on a fresh download or on a
    hosting service some folders would not exist without this.
    """
    for folder in [
        RAW_DIR,
        RESPONSES_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FRONTEND_DIR,
        LOGS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


# Run once, as soon as this file is imported.
ensure_directories()