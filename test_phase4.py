"""
test_phase4.py
Shows what the trained router thinks of some sample questions.
No server and no Groq tokens needed: everything runs on your computer.

    python test_phase4.py
"""

import sys

from backend import config, router

SAMPLES = [
    "What is the capital of Japan?",
    "Who wrote the play Romeo and Juliet?",
    "Which planet is known as the Red Planet?",
    "Maria buys 3 notebooks at $4 each and 2 pens at $1.50 each. She pays with a $20 bill. How much change does she get?",
    "A train travels 120 km in 1.5 hours and then 180 km in 2.5 hours. What is its average speed for the whole trip?",
    "Prove that the square root of 2 is irrational.",
    "Let G be a group of order 15. Show that G is cyclic.",
    "A contract was signed under duress and ratified a year later without any threats. Is the original contract voidable? Explain with reference to common-law doctrine.",
    "Explain the difference between a Type I and a Type II error in hypothesis testing.",
    "Write a Python function that checks whether a string is a palindrome.",
]


def main():
    router.load()
    if not router.is_ready():
        print("The router is not loaded. Run  python ml/train_router.py  first.")
        sys.exit(1)

    threshold = config.ROUTER_THRESHOLD
    print()
    print(f"Threshold from .env: {threshold:.2f}  (P(easy) at or above it -> small model)")
    print()
    print(f"  {'P(easy)':>7}   {'goes to':<8} question")
    print("  " + "-" * 76)

    small_count = 0
    for question in SAMPLES:
        p_easy = router.predict_p_easy(question)
        tier = "small" if p_easy >= threshold else "large"
        small_count += tier == "small"
        shortened = question if len(question) <= 62 else question[:62] + "..."
        print(f"  {p_easy:>7.2f}   {tier:<8} {shortened}")

    print()
    print(f"{small_count} of {len(SAMPLES)} sample questions would go to the small model.")
    print("The router was trained on benchmark-style questions (school maths and exam multiple choice),")
    print("so its choices on casual or open-ended questions are only a rough guide.")


if __name__ == "__main__":
    main()