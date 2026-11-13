"""
expand_dataset.py
Adds more questions from hard subjects to data/raw/queries.json.

Why: the first dataset had only about a dozen "hard" questions (questions that
only the large model answered correctly), which is too few to train and test a
router reliably. This script adds ~200 more questions from subjects that
language models often find difficult.

Existing question ids and their saved responses are NOT changed. New questions
get new ids (q221, q222, ...) and are marked with a "batch" tag.

Run from the project root:
    python ml/expand_dataset.py                 add the first extra batch (seed 1001)
    python ml/expand_dataset.py --seed 2002     add a second, different batch later

Afterwards:
    python ml/collect_responses.py              collects ONLY the new questions
    python ml/label_data.py
    python ml/train_router.py
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

# Make "from backend import config" work when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from backend import config  # noqa: E402

DEFAULT_SEED = 1001
MAX_PROMPT_CHARS = 3500   # the API accepts at most 4000 characters per query
LETTERS = "ABCD"

# This text must match ml/build_dataset.py exactly, so old and new prompts look alike.
MMLU_INSTRUCTION = (
    "Answer this multiple-choice question. Finish with a final line in exactly this form:\n"
    "Answer: <letter>"
)

# subject -> (group label, how many NEW questions to add)
EXPANSION_PLAN = {
    # Subjects already in the dataset: take more questions that have not been used yet
    "professional_law": ("knowledge-hard", 15),
    "moral_scenarios": ("knowledge-hard", 15),
    "formal_logic": ("reasoning-hard", 15),
    "abstract_algebra": ("reasoning-hard", 15),
    "college_mathematics": ("reasoning-hard", 15),
    "college_physics": ("reasoning-hard", 15),
    # New subjects
    "college_chemistry": ("reasoning-hard", 10),
    "econometrics": ("reasoning-hard", 10),
    "electrical_engineering": ("reasoning-hard", 10),
    "professional_accounting": ("reasoning-hard", 10),
    "high_school_statistics": ("reasoning-hard", 10),
    "global_facts": ("knowledge-hard", 10),
    "professional_medicine": ("knowledge-hard", 10),
    "international_law": ("knowledge-hard", 10),
    "jurisprudence": ("knowledge-hard", 10),
    "medical_genetics": ("knowledge-hard", 10),
    "anatomy": ("knowledge-hard", 10),
}


def load_existing():
    path = config.RAW_DIR / "queries.json"
    if not path.exists():
        print(f"Could not find {path}. Run  python ml/build_dataset.py  first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as file:
        return path, json.load(file)


def average_tokens_per_question():
    """Average tokens (input + output) per question in the responses saved so far."""
    averages = {}
    for tier in ("small", "large"):
        path = config.RESPONSES_DIR / f"{tier}.jsonl"
        totals = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    totals.append(record.get("input_tokens", 0) + record.get("output_tokens", 0))
        averages[tier] = (sum(totals) / len(totals)) if totals else None
    return averages


def pick_questions(subject, group, count, rng, seen_queries):
    """Pick `count` unused questions from one MMLU subject."""
    dataset = load_dataset("cais/mmlu", subject, split="test")
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    picked = []
    for index in indices:
        row = dataset[index]
        options = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(row["choices"]))
        prompt = f"{row['question'].strip()}\n\n{options}\n\n{MMLU_INSTRUCTION}"
        if len(prompt) > MAX_PROMPT_CHARS or prompt in seen_queries:
            continue
        seen_queries.add(prompt)
        picked.append(
            {
                "source": "mmlu",
                "subject": subject,
                "group": group,
                "query": prompt,
                "reference": LETTERS[row["answer"]],
            }
        )
        if len(picked) == count:
            break
    return picked


def main():
    parser = argparse.ArgumentParser(description="Add more hard questions to the dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed; use a new one for a second batch (default: {DEFAULT_SEED})")
    parser.add_argument("--force", action="store_true",
                        help="add this batch even if a batch with the same seed was already added")
    args = parser.parse_args()

    path, existing = load_existing()
    batch_tag = f"expansion-{args.seed}"

    if any(item.get("batch") == batch_tag for item in existing) and not args.force:
        print(f"The batch '{batch_tag}' is already in queries.json, so nothing was changed.")
        print("To add a different batch, use a new seed, for example:  python ml/expand_dataset.py --seed 2002")
        sys.exit(1)

    # Keep a copy of the original questions (only the first time).
    backup = config.RAW_DIR / "queries_before_expansion.json"
    if not backup.exists():
        shutil.copyfile(path, backup)
        print(f"Backup of the original questions saved to {backup}")
        print()

    seen_queries = {item["query"] for item in existing}
    rng = random.Random(args.seed)

    new_records = []
    for subject, (group, count) in EXPANSION_PLAN.items():
        print(f"Loading MMLU subject: {subject} (want {count}) ...")
        picked = pick_questions(subject, group, count, rng, seen_queries)
        if len(picked) < count:
            print(f"  Only found {len(picked)} unused questions for {subject}.")
        new_records.extend(picked)

    if not new_records:
        print("No new questions were found. Nothing was changed.")
        sys.exit(1)

    rng.shuffle(new_records)  # mix subjects together

    # New ids continue after the highest existing id (q220 -> q221, q222, ...)
    highest = max(int(item["id"][1:]) for item in existing)
    added = []
    for offset, record in enumerate(new_records, start=1):
        added.append({"id": f"q{highest + offset:03d}", **record, "batch": batch_tag})

    merged = existing + added
    with open(path, "w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=False, indent=2)

    print()
    print(f"Added {len(added)} new questions ({added[0]['id']} to {added[-1]['id']}).")
    print(f"queries.json now has {len(merged)} questions.")
    print()
    print("New questions by group:")
    for name, count in Counter(item["group"] for item in added).items():
        print(f"  {name:<18} {count}")
    print("New questions by subject:")
    for name, count in sorted(Counter(item["subject"] for item in added).items()):
        print(f"  {name:<26} {count}")

    averages = average_tokens_per_question()
    print()
    print("Rough token estimate for collecting the new questions (based on your first responses):")
    for tier in ("small", "large"):
        if averages[tier]:
            print(f"  {tier:<6} about {int(averages[tier] * len(added)):,} tokens")
    print("Harder subjects often need more thinking, so the real number may be higher.")
    print("If Groq's daily limit stops the run, nothing is lost: run the same command again later.")
    print()
    print("Next step:  python ml/collect_responses.py --limit 225")


if __name__ == "__main__":
    main()