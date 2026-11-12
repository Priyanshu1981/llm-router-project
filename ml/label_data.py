"""
label_data.py
Grades every saved response and decides, for each question, whether the
small model was "good enough". These labels are what the router learns from.

Run from the project root, after collect_responses.py:
    python ml/label_data.py

Output:
    data/processed/labeled.csv
"""

import json
import math
import re
import sys
from pathlib import Path

# Make "from backend import ..." work when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from backend import config, cost  # noqa: E402

TEST_FRACTION = 0.30   # 30% of the questions are kept aside for testing
SPLIT_SEED = 42        # fixed, so the split is the same every time

# ---------------------------------------------------------------
# Finding the answer inside a model's response
# ---------------------------------------------------------------
NUMBER = r"-?\d[\d,]*\.?\d*"

# "Final answer: 42"  (also copes with **bold**, $ signs and commas)
FINAL_ANSWER_RE = re.compile(r"(?i:final\s+answer)\s*[:\-]?\s*[\*\s]*\$?\s*(" + NUMBER + ")")
# \boxed{42}
BOXED_RE = re.compile(r"\\boxed\{\s*\$?\s*(" + NUMBER + ")")
ANY_NUMBER_RE = re.compile(NUMBER)

# "Answer: B"  /  "The answer is (B)"  (the letter must be a capital A-D)
LETTER_RE = re.compile(r"(?i:answer)\s*(?:is)?\s*[:\-]?\s*[\*\s]*[\(\[]?\s*([ABCD])\b")
# A response that is only a letter, such as "B" or "(B)" or "B."
SINGLE_LETTER_RE = re.compile(r"[\(\[]?([ABCD])[\)\]]?[.)]?", re.IGNORECASE)


def to_number(text):
    """Turn text such as '1,200.50' or '$18.' into a float. Returns None if impossible."""
    if text is None:
        return None
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def format_number(value):
    """Show 18.0 as '18' so the CSV is easy to read."""
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return str(round(value, 6))


def extract_number_answer(text):
    """Find the final numeric answer in a response, or None."""
    matches = FINAL_ANSWER_RE.findall(text)
    if matches:
        return to_number(matches[-1])
    matches = BOXED_RE.findall(text)
    if matches:
        return to_number(matches[-1])
    matches = ANY_NUMBER_RE.findall(text)
    if matches:
        return to_number(matches[-1])
    return None


def extract_letter_answer(text):
    """Find the chosen option letter (A-D) in a response, or None."""
    matches = LETTER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    stripped = text.strip().strip("*").strip()
    single = SINGLE_LETTER_RE.fullmatch(stripped)
    if single:
        return single.group(1).upper()
    return None


def grade_response(source, response, reference):
    """
    Returns (prediction, is_correct, was_truncated).
    A response that was cut off (ran out of output tokens) counts as wrong.
    """
    if response.get("finish_reason") == "length":
        return "", False, True

    text = response.get("text", "")

    if source == "gsm8k":
        predicted = extract_number_answer(text)
        expected = to_number(reference)
        is_correct = (
            predicted is not None
            and expected is not None
            and math.isclose(predicted, expected, rel_tol=1e-6, abs_tol=1e-6)
        )
        return format_number(predicted), is_correct, False

    letter = extract_letter_answer(text)
    return (letter or ""), (letter == reference), False


def categorize(small_correct, large_correct):
    """The four possible outcomes for a question."""
    if small_correct and large_correct:
        return "both_correct"
    if large_correct:
        return "only_large_correct"   # the only "hard" case: the large model is needed
    if small_correct:
        return "only_small_correct"
    return "both_wrong"


# ---------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------
def load_responses(tier):
    """Read data/responses/<tier>.jsonl into a dictionary: question id -> response."""
    path = config.RESPONSES_DIR / f"{tier}.jsonl"
    responses = {}
    if not path.exists():
        return responses
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            responses[record["id"]] = record
    return responses


def accuracy_table(df, column):
    """Accuracy of both models for each value of a column (source, group or subject)."""
    table = df.groupby(column).agg(
        questions=("id", "count"),
        small_acc=("small_correct", "mean"),
        large_acc=("large_correct", "mean"),
        needs_large=("small_ok", lambda values: int((values == 0).sum())),
    )
    table["small_acc"] = (table["small_acc"] * 100).round(1)
    table["large_acc"] = (table["large_acc"] * 100).round(1)
    return table


def section(title):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    queries_path = config.RAW_DIR / "queries.json"
    if not queries_path.exists():
        print(f"Could not find {queries_path}. Run  python ml/build_dataset.py  first.")
        sys.exit(1)
    with open(queries_path, "r", encoding="utf-8") as file:
        queries = json.load(file)

    small_responses = load_responses("small")
    large_responses = load_responses("large")
    print(
        f"Loaded {len(queries)} questions, {len(small_responses)} small-model responses, "
        f"{len(large_responses)} large-model responses."
    )

    # Grade every question that has a response from BOTH models.
    rows = []
    for item in queries:
        question_id = item["id"]
        if question_id not in small_responses or question_id not in large_responses:
            continue
        small = small_responses[question_id]
        large = large_responses[question_id]

        small_pred, small_correct, small_truncated = grade_response(item["source"], small, item["reference"])
        large_pred, large_correct, large_truncated = grade_response(item["source"], large, item["reference"])
        category = categorize(small_correct, large_correct)

        rows.append(
            {
                "id": question_id,
                "source": item["source"],
                "subject": item["subject"],
                "group": item["group"],
                "query": item["query"],
                "reference": item["reference"],
                "small_pred": small_pred,
                "small_correct": int(small_correct),
                "small_truncated": int(small_truncated),
                "large_pred": large_pred,
                "large_correct": int(large_correct),
                "large_truncated": int(large_truncated),
                "category": category,
                # 1 = the small model was good enough, 0 = the large model was needed
                "small_ok": 0 if category == "only_large_correct" else 1,
                "small_input_tokens": small["input_tokens"],
                "small_output_tokens": small["output_tokens"],
                "small_latency_ms": small["latency_ms"],
                "large_input_tokens": large["input_tokens"],
                "large_output_tokens": large["output_tokens"],
                "large_latency_ms": large["latency_ms"],
            }
        )

    if not rows:
        print("No question has responses from both models yet. Run  python ml/collect_responses.py  first.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    missing = len(queries) - len(df)
    print(f"Questions with responses from both models: {len(df)}" + (f"  ({missing} still missing)" if missing else ""))

    # Cost of each answer, from the token counts and the prices in .env
    df["small_cost_usd"] = df.apply(
        lambda r: cost.calculate_cost("small", r["small_input_tokens"], r["small_output_tokens"]), axis=1
    )
    df["large_cost_usd"] = df.apply(
        lambda r: cost.calculate_cost("large", r["large_input_tokens"], r["large_output_tokens"]), axis=1
    )

    # Train/test split, balanced by label so both parts contain hard questions.
    label_counts = df["small_ok"].value_counts()
    can_stratify = len(label_counts) == 2 and label_counts.min() >= 2
    train_ids, test_ids = train_test_split(
        df["id"],
        test_size=TEST_FRACTION,
        random_state=SPLIT_SEED,
        stratify=df["small_ok"] if can_stratify else None,
    )
    test_set = set(test_ids)
    df["split"] = df["id"].apply(lambda question_id: "test" if question_id in test_set else "train")

    # -----------------------------------------------------------
    # Report
    # -----------------------------------------------------------
    n = len(df)

    section("Accuracy of each model")
    print(f"  small ({config.SMALL_MODEL}): {df['small_correct'].mean() * 100:.1f}%")
    print(f"  large ({config.LARGE_MODEL}): {df['large_correct'].mean() * 100:.1f}%")
    print(f"  Answers with no readable answer (small / large): "
          f"{int((df['small_pred'] == '').sum())} / {int((df['large_pred'] == '').sum())}")
    print(f"  Answers cut off by the token limit (small / large): "
          f"{int(df['small_truncated'].sum())} / {int(df['large_truncated'].sum())}")

    section("Accuracy by group (percent correct)")
    print(accuracy_table(df, "group").to_string())

    section("Accuracy by subject (percent correct)")
    print(accuracy_table(df, "subject").to_string())

    section("Outcome categories")
    category_counts = df["category"].value_counts()
    notes = {
        "only_large_correct": "<- hard: only the large model got it right",
        "both_wrong": "<- neither model got it right",
    }
    for name in ["both_correct", "only_large_correct", "only_small_correct", "both_wrong"]:
        count = int(category_counts.get(name, 0))
        print(f"  {name:<20} {count:>4}  ({count / n * 100:5.1f}%)  {notes.get(name, '')}")

    section("Router labels and split")
    easy = int((df["small_ok"] == 1).sum())
    hard = int((df["small_ok"] == 0).sum())
    print(f"  small_ok = 1 (easy): {easy}")
    print(f"  small_ok = 0 (hard): {hard}")
    print(f"  train questions: {int((df['split'] == 'train').sum())}   "
          f"test questions: {int((df['split'] == 'test').sum())}")
    print(f"  hard in train / test: "
          f"{int(((df['split'] == 'train') & (df['small_ok'] == 0)).sum())} / "
          f"{int(((df['split'] == 'test') & (df['small_ok'] == 0)).sum())}")
    if hard < 15:
        print()
        print("  WARNING: very few hard questions. The router will have little to learn from.")
        print("  Tell me the numbers above and we will add more questions from harder subjects.")

    section("What a perfect router could achieve on this data")
    always_small_cost = df["small_cost_usd"].sum()
    always_large_cost = df["large_cost_usd"].sum()
    perfect_cost = df["small_cost_usd"].where(df["small_ok"] == 1, df["large_cost_usd"]).sum()
    perfect_accuracy = df["small_correct"].where(df["small_ok"] == 1, df["large_correct"]).mean()

    def line(name, accuracy, total_cost):
        relative = (total_cost / always_large_cost * 100) if always_large_cost > 0 else 0.0
        print(f"  {name:<22} accuracy {accuracy * 100:5.1f}%   cost ${total_cost:.6f}   ({relative:5.1f}% of always-large)")

    line("Always small", df["small_correct"].mean(), always_small_cost)
    line("Always large", df["large_correct"].mean(), always_large_cost)
    line("Perfect router", perfect_accuracy, perfect_cost)
    print()
    print(f"  Average latency: small {df['small_latency_ms'].mean():.0f} ms, "
          f"large {df['large_latency_ms'].mean():.0f} ms")

    output_path = config.PROCESSED_DIR / "labeled.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print()
    print(f"Saved {len(df)} labeled questions to {output_path}")
    print("When reading this file later, use:  pd.read_csv(path, encoding='utf-8-sig')")


if __name__ == "__main__":
    main()