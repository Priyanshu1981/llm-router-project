"""
check_setup.py
Run this from the project root to verify that everything set up in Phase 1 works:

    python check_setup.py

It checks, in order:
  1. Python version
  2. Installed packages
  3. Settings loaded from .env
  4. The embedding model downloads and runs
  5. Both Groq models answer a test prompt
"""

import importlib
import sys
import time


def header(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_python():
    header("1. Python version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if (version.major, version.minor) < (3, 10):
        print("FAIL: Python 3.10 or newer is required.")
        return False
    print("OK")
    return True


def check_packages():
    header("2. Installed packages")
    packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("python-dotenv", "dotenv"),
        ("groq", "groq"),
        ("sentence-transformers", "sentence_transformers"),
        ("scikit-learn", "sklearn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("joblib", "joblib"),
        ("datasets", "datasets"),
    ]
    all_ok = True
    for pip_name, import_name in packages:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "version unknown")
            print(f"  OK    {pip_name} ({version})")
        except Exception as error:
            all_ok = False
            print(f"  FAIL  {pip_name}: {error}")
    if not all_ok:
        print("Some packages are missing. Make sure (.venv) is active, then run:")
        print("    pip install -r requirements.txt")
    return all_ok


def check_config():
    header("3. Settings from .env")
    try:
        from backend import config
    except Exception as error:
        print(f"FAIL: could not load backend/config.py: {error}")
        return None

    key = config.GROQ_API_KEY
    if not key or key == "paste_your_key_here":
        print("FAIL: GROQ_API_KEY is missing or still the placeholder.")
        print("Open the .env file and paste your real Groq key after GROQ_API_KEY=")
        return None

    masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "(too short)"
    print(f"  GROQ_API_KEY      : {masked}")
    print(f"  SMALL_MODEL       : {config.SMALL_MODEL}")
    print(f"  LARGE_MODEL       : {config.LARGE_MODEL}")
    print(f"  EMBEDDING_MODEL   : {config.EMBEDDING_MODEL}")
    print(f"  CACHE_THRESHOLD   : {config.CACHE_THRESHOLD}")
    print(f"  ROUTER_THRESHOLD  : {config.ROUTER_THRESHOLD}")
    print("OK")
    return config


def check_embedding(config):
    header("4. Embedding model")
    print("Loading the model (first run downloads about 90 MB, please wait)...")
    try:
        from sentence_transformers import SentenceTransformer

        start = time.time()
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        vector = model.encode("Hello world")
        elapsed = time.time() - start
        print(f"  Loaded and ran in {elapsed:.1f}s")
        print(f"  Embedding vector length: {len(vector)} (expected 384 for all-MiniLM-L6-v2)")
        print("OK")
        return True
    except Exception as error:
        print(f"FAIL: {error}")
        print("Check your internet connection. The model downloads from Hugging Face.")
        return False


def check_groq(config):
    header("5. Groq models")
    try:
        from groq import Groq

        client = Groq(api_key=config.GROQ_API_KEY)
    except Exception as error:
        print(f"FAIL: could not create the Groq client: {error}")
        return False

    all_ok = True
    for label, model_name in [("small", config.SMALL_MODEL), ("large", config.LARGE_MODEL)]:
        print(f"Testing {label} model: {model_name}")
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Reply with exactly one word: ready"}],
            )
            elapsed = time.time() - start
            text = (response.choices[0].message.content or "").strip()
            usage = response.usage
            print(f"  Reply  : {text!r}")
            print(
                f"  Time   : {elapsed:.2f}s | input tokens: {usage.prompt_tokens} "
                f"| output tokens: {usage.completion_tokens}"
            )
            print("  OK")
        except Exception as error:
            all_ok = False
            print(f"  FAIL: {error}")
            print("  Things to check:")
            print("   - Is the API key in .env correct (no quotes, no extra spaces)?")
            print("   - Does the model name exist on Groq? Models get retired over time.")
            print("   - If you see a rate limit error, wait a minute and run again.")
    return all_ok


def main():
    results = {}

    results["Python version"] = check_python()
    results["Packages"] = check_packages()

    config = check_config()
    results["Settings (.env)"] = config is not None

    if config is not None:
        results["Embedding model"] = check_embedding(config)
        results["Groq models"] = check_groq(config)
    else:
        results["Embedding model"] = False
        results["Groq models"] = False
        print()
        print("Skipping steps 4 and 5 because the settings could not be loaded.")

    header("SUMMARY")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    if all(results.values()):
        print()
        print("All checks passed. You are ready for Phase 2.")
    else:
        print()
        print("Some checks failed. Fix the items marked FAIL above and run this script again.")
        sys.exit(1)


if __name__ == "__main__":
    main()