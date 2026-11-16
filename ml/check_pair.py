"""
check_pair.py
Shows what the semantic cache would do with two questions.

    python ml/check_pair.py "How do I turn on dark mode?" "How do I turn off dark mode?"

Uses no Groq tokens. Run from the project root with (.venv) active.
"""

import sys
from pathlib import Path

# Make "from backend import ..." work when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles can fail on non-English text when output is redirected.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from backend import cache, config, features  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print('Usage: python ml/check_pair.py "first question" "second question"')
        sys.exit(1)

    first, second = sys.argv[1], sys.argv[2]
    vectors = features.embed_texts([first, second])
    similarity = float(vectors[0] @ vectors[1])
    reason = cache.guard_failure(first, second)
    threshold = config.CACHE_THRESHOLD

    print(f"similarity : {similarity:.3f}   (threshold {threshold:.2f})")
    print(f"guards     : {'blocked: ' + reason if reason else 'pass'}")
    if similarity >= threshold and reason is None:
        print("result     : the cache WOULD reuse the answer to the first question")
    elif reason is not None:
        print("result     : not reused (a guard blocked it)")
    else:
        print("result     : not reused (similarity below the threshold)")


if __name__ == "__main__":
    main()