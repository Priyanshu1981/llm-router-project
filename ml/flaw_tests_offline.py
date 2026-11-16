"""
flaw_tests_offline.py
Hunts for weaknesses in the semantic cache WITHOUT using any Groq tokens.

  1. Trap pairs: questions that look alike but need different answers. A trap is only
     reused if the similarity is at or above CACHE_THRESHOLD AND the guards let it through.
     Paraphrase pairs: same meaning, so reuse is good.
  2. Time-sensitive questions ("what is today's date?"): a cached answer goes stale.
  3. The rate limiter and the daily budget.

Run from the project root:
    python ml/flaw_tests_offline.py

These pairs were written knowing how the guards work, so passing proves little.
Every failure is real. Add your own pairs to the PAIRS list below.
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

import pandas as pd  # noqa: E402

from backend import cache, config, features, security  # noqa: E402

THRESHOLD = config.CACHE_THRESHOLD
LOWER_THRESHOLD = 0.85

# (kind, group, question A, question B)
#   "trap"       : the answers differ, so the cache must NOT reuse an answer
#   "paraphrase" : same meaning, reuse is good
PAIRS = [
    ("trap", "unit direction", "Convert 5 kilometers to miles.", "Convert 5 miles to kilometers."),
    ("trap", "unit name", "What is the boiling point of water in Celsius?", "What is the boiling point of water in Fahrenheit?"),
    ("trap", "ordinal word", "Who was the third president of the United States?", "Who was the fourth president of the United States?"),
    ("trap", "date year", "What happened on 4 July 1776?", "What happened on 4 July 1777?"),
    ("trap", "one-word swap", "Is it safe to eat raw eggs?", "Is it safe to eat cooked eggs?"),
    ("trap", "antonym not listed", "What is the easiest sorting algorithm to learn?", "What is the hardest sorting algorithm to learn?"),
    ("trap", "lowercase name", "what is the capital of australia?", "what is the capital of austria?"),
    ("trap", "language name", "Translate hello into German.", "Translate hello into Dutch."),
    ("trap", "programming language", "Does Python use curly braces?", "Does Java use curly braces?"),
    ("trap", "comparison direction", "Is Python faster than C?", "Is C faster than Python?"),
    ("trap", "word order", "Does the dog chase the cat?", "Does the cat chase the dog?"),
    ("trap", "arithmetic symbol", "What is 6+3?", "What is 6*3?"),
    ("trap", "currency direction", "Convert 100 USD to INR.", "Convert 100 INR to USD."),
    ("trap", "letter swap", "How many r's are in strawberry?", "How many s's are in strawberry?"),
    ("trap", "size word", "How long should I bake a small cake?", "How long should I bake a large cake?"),
    ("trap", "radius vs diameter", "What is the area of a circle with radius 5?", "What is the area of a circle with diameter 5?"),
    ("trap", "number order", "Is 1000 greater than 999?", "Is 999 greater than 1000?"),
    ("trap", "date format", "What day of the week is 03/04/2026?", "What day of the week is 04/03/2026?"),
    ("trap", "Spanish negation", "\u00bfLa Tierra es plana?", "\u00bfLa Tierra no es plana?"),
    ("trap", "Hindi negation", "\u0915\u094d\u092f\u093e \u092a\u0943\u0925\u094d\u0935\u0940 \u091a\u092a\u091f\u0940 \u0939\u0948?", "\u0915\u094d\u092f\u093e \u092a\u0943\u0925\u094d\u0935\u0940 \u091a\u092a\u091f\u0940 \u0928\u0939\u0940\u0902 \u0939\u0948?"),
    ("trap", "French negation", "Est-ce que le ciel est bleu ?", "Est-ce que le ciel n'est pas bleu ?"),
    ("trap", "Spanish number words", "\u00bfQu\u00e9 es el cinco por ciento de doscientos?", "\u00bfQu\u00e9 es el diez por ciento de doscientos?"),
    ("trap", "antonym not listed (2)", "Which is bigger, a mouse or an elephant?", "Which is smaller, a mouse or an elephant?"),
    ("trap", "time word", "Will it rain today?", "Will it rain tomorrow?"),
    ("paraphrase", "contraction", "What's the capital of France?", "What is the capital of France?"),
    ("paraphrase", "number word vs article", "How many seconds are in an hour?", "How many seconds are in one hour?"),
    ("paraphrase", "how do I / how can I", "How do I install Python on Windows 11?", "How can I install Python on Windows 11?"),
    ("paraphrase", "words vs digits", "What is two plus two?", "What is 2 plus 2?"),
    ("paraphrase", "rephrased", "How many days are in a week?", "How many days does a week have?"),
    ("paraphrase", "define / mean", "Define photosynthesis.", "What does photosynthesis mean?"),
    ("paraphrase", "article dropped", "Is the Earth round?", "Is Earth round?"),
    ("paraphrase", "question reordered", "What is the largest ocean?", "Which ocean is the largest?"),
]

# Questions whose correct answer changes over time. A cached answer goes stale.
TIME_SENSITIVE = [
    "What is today's date?",
    "What is the weather like right now?",
    "Who is the current prime minister of India?",
    "What is the latest iPhone model?",
    "What day is it tomorrow?",
    "What is the price of Bitcoin?",
    "Who won the match last night?",
]
NOT_TIME_SENSITIVE = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "What is 15% of 200?",
]


def short(text, limit=40):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def section(title):
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def check(description, passed):
    print(f"  {'PASS' if passed else 'FAIL'}  {description}")
    return passed


def main():
    # -----------------------------------------------------------
    # 1. Trap and paraphrase pairs
    # -----------------------------------------------------------
    print("Computing embeddings (loads the embedding model, so the first run can take a minute) ...")
    texts = []
    for _, _, a, b in PAIRS:
        texts.extend([a, b])
    vectors = features.embed_texts(texts)

    rows = []
    for index, (kind, group, a, b) in enumerate(PAIRS):
        similarity = float(vectors[2 * index] @ vectors[2 * index + 1])
        reason = cache.guard_failure(a, b)
        reused = similarity >= THRESHOLD and reason is None
        reused_low = similarity >= LOWER_THRESHOLD and reason is None

        if kind == "trap":
            if reused:
                outcome = "REUSED  <- FLAW"
            elif reason is not None:
                outcome = "blocked by a guard"
            elif reused_low:
                outcome = f"fragile: safe only because similarity < {THRESHOLD:.2f}"
            else:
                outcome = "safe: low similarity"
        else:
            if reused:
                outcome = "reused (good)"
            elif reason is not None:
                outcome = "lost: blocked by a guard"
            else:
                outcome = "lost: below threshold"

        rows.append({
            "kind": kind, "group": group, "question_a": a, "question_b": b,
            "similarity": round(similarity, 4), "guard_reason": reason or "", "outcome": outcome,
        })

    df = pd.DataFrame(rows)

    section(f"1. TRAP pairs (the cache must NOT reuse them), threshold {THRESHOLD:.2f}")
    print(f"  {'similarity':>10}  {'what differs':<24} {'question A':<42} outcome")
    for _, row in df[df["kind"] == "trap"].iterrows():
        print(f"  {row['similarity']:>10.3f}  {row['group']:<24} {short(row['question_a']):<42} {row['outcome']}")

    section("2. PARAPHRASE pairs (the cache SHOULD reuse them)")
    print(f"  {'similarity':>10}  {'type':<24} {'question A':<42} outcome")
    for _, row in df[df["kind"] == "paraphrase"].iterrows():
        print(f"  {row['similarity']:>10.3f}  {row['group']:<24} {short(row['question_a']):<42} {row['outcome']}")

    traps = df[df["kind"] == "trap"]
    paraphrases = df[df["kind"] == "paraphrase"]
    flaws = traps[traps["outcome"].str.contains("FLAW")]
    fragile = traps[traps["outcome"].str.startswith("fragile")]
    blocked = traps[traps["outcome"] == "blocked by a guard"]
    print()
    print(f"  Trap pairs wrongly reused (flaws)      : {len(flaws)} of {len(traps)}")
    print(f"  Trap pairs blocked by a guard          : {len(blocked)} of {len(traps)}")
    print(f"  Trap pairs safe only by the threshold  : {len(fragile)} of {len(traps)}")
    print(f"  Paraphrases reused                     : {int((paraphrases['outcome'] == 'reused (good)').sum())} of {len(paraphrases)}")
    if len(flaws):
        print()
        print("  Flaws found:")
        for _, row in flaws.iterrows():
            print(f"    - {row['group']}: {short(row['question_a'], 30)} | {short(row['question_b'], 30)}")

    # -----------------------------------------------------------
    # 3. Time-sensitive questions
    # -----------------------------------------------------------
    section("3. Time-sensitive questions (a cached answer goes stale)")
    detector = getattr(cache, "is_time_sensitive", None)
    if detector is None:
        print("  There is no time-sensitivity check in the cache, so ALL of these would be cached")
        print("  and served unchanged tomorrow, next week and next year:")
        for question in TIME_SENSITIVE:
            print(f"    FLAW  {question}")
        time_flaws = len(TIME_SENSITIVE)
    else:
        time_flaws = 0
        for question in TIME_SENSITIVE:
            caught = bool(detector(question))
            time_flaws += 0 if caught else 1
            print(f"  {'PASS' if caught else 'FAIL'}  detected as time-sensitive: {question}")
        for question in NOT_TIME_SENSITIVE:
            wrongly = bool(detector(question))
            time_flaws += 1 if wrongly else 0
            print(f"  {'FAIL' if wrongly else 'PASS'}  ordinary question left cacheable: {question}")

    # -----------------------------------------------------------
    # 4. Rate limiter and daily budget
    # -----------------------------------------------------------
    section("4. Rate limiter and daily budget (the protection used in public mode)")
    limiter = security.SlidingWindowLimiter(3, 60)
    first_three = [limiter.check("visitor-a") for _ in range(3)]
    fourth = limiter.check("visitor-a")
    results = [
        check("a visitor may make 3 requests per window", first_three == [0, 0, 0]),
        check("the 4th request is refused, with a wait time", fourth > 0),
        check("a different visitor is not affected", limiter.check("visitor-b") == 0),
        check("a limit of 0 means unlimited", all(security.SlidingWindowLimiter(0, 60).check("x") == 0 for _ in range(50))),
    ]
    budget = security.DailyBudget(2)
    takes = [budget.take(), budget.take(), budget.take()]
    results.append(check("a daily budget of 2 allows two calls, then refuses", takes == [True, True, False]))
    results.append(check("the budget reports its usage", budget.used() == 2))
    results.append(check("a budget of 0 means unlimited", all(security.DailyBudget(0).take() for _ in range(100))))

    # -----------------------------------------------------------
    # Summary and save
    # -----------------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "flaw_pairs_offline.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    section("Summary")
    print(f"  Cache traps wrongly reused         : {len(flaws)} of {len(traps)}")
    print(f"  Traps protected only by threshold  : {len(fragile)}")
    print(f"  Time-sensitive problems            : {time_flaws}")
    print(f"  Rate limit / budget checks passed  : {sum(results)} of {len(results)}")
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()