"""
build_dataset.py
Builds the question set that will train and test the router.

It mixes two public benchmarks so that some questions are easy for a small
model and some are hard:
  - GSM8K: grade-school maths word problems (the answer is a number)
  - MMLU : multiple-choice questions from many school and university subjects
           (the answer is a letter A to D)

Both have a known correct answer, so we can grade the models automatically.

Run from the project root:
    python ml/build_dataset.py

Output:
    data/raw/queries.json
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

# Make "from backend import config" work when this file is run as a script.
# Python only looks in the script's own folder (ml/) by default, so we add the
# project root to its search path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from backend import config  # noqa: E402

# ---------------------------------------------------------------
# Settings you could change (but then delete data/responses/*.jsonl first)
# ---------------------------------------------------------------
SEED = 42                    # same seed = same questions every time
NUM_GSM8K = 100              # number of maths word problems
QUESTIONS_PER_SUBJECT = 10   # questions taken from each MMLU subject
MAX_PROMPT_CHARS = 3500      # the API accepts at most 4000 characters per query

MMLU_SUBJECTS = {
    # General knowledge: usually easy for a small model
    "high_school_geography": "knowledge-easy",
    "marketing": "knowledge-easy",
    "sociology": "knowledge-easy",
    "world_religions": "knowledge-easy",
    # Need step-by-step reasoning
    "abstract_algebra": "reasoning-hard",
    "college_mathematics": "reasoning-hard",
    "formal_logic": "reasoning-hard",
    "college_physics": "reasoning-hard",
    # Need many specialised facts: bigger models usually know more
    "professional_law": "knowledge-hard",
    "virology": "knowledge-hard",
    "moral_scenarios": "knowledge-hard",
    "college_medicine": "knowledge-hard",
}

# These instructions end every prompt so the final answer is easy to find and grade.
GSM8K_INSTRUCTION = (
    "Solve this maths problem. Finish with a final line in exactly this form:\n"
    "Final answer: <number>"
)
MMLU_INSTRUCTION = (
    "Answer this multiple-choice question. Finish with a final line in exactly this form:\n"
    "Answer: <letter>"
)

LETTERS = "ABCD"


def build_gsm8k(rng):
    """Pick maths word problems. The reference answer is a number stored as text."""
    print("Loading GSM8K (maths word problems) ...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    records = []
    for index in indices:
        row = dataset[index]
        prompt = f"{row['question'].strip()}\n\n{GSM8K_INSTRUCTION}"
        if len(prompt) > MAX_PROMPT_CHARS:
            continue
        # In GSM8K the correct answer comes after "####" at the end of the solution.
        reference = row["answer"].split("####")[-1].strip().replace(",", "")
        records.append(
            {
                "source": "gsm8k",
                "subject": "gsm8k",
                "group": "maths-word-problems",
                "query": prompt,
                "reference": reference,
            }
        )
        if len(records) == NUM_GSM8K:
            break
    return records


def build_mmlu(rng):
    """Pick multiple-choice questions from each chosen subject. The reference is a letter."""
    records = []
    for subject, group in MMLU_SUBJECTS.items():
        print(f"Loading MMLU subject: {subject} ...")
        dataset = load_dataset("cais/mmlu", subject, split="test")

        indices = list(range(len(dataset)))
        rng.shuffle(indices)

        taken = 0
        for index in indices:
            row = dataset[index]
            options = "\n".join(
                f"{LETTERS[i]}. {choice}" for i, choice in enumerate(row["choices"])
            )
            prompt = f"{row['question'].strip()}\n\n{options}\n\n{MMLU_INSTRUCTION}"
            if len(prompt) > MAX_PROMPT_CHARS:
                continue
            records.append(
                {
                    "source": "mmlu",
                    "subject": subject,
                    "group": group,
                    "query": prompt,
                    "reference": LETTERS[row["answer"]],
                }
            )
            taken += 1
            if taken == QUESTIONS_PER_SUBJECT:
                break
    return records


def responses_already_exist():
    """True if any saved model responses exist (they belong to the current questions)."""
    for tier in ("small", "large"):
        path = config.RESPONSES_DIR / f"{tier}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            return True
    return False


def main():
    output_path = config.RAW_DIR / "queries.json"

    # Rebuilding with different settings would give the questions new ids and
    # silently mismatch any responses already collected. Refuse unless forced.
    if responses_already_exist() and "--force" not in sys.argv:
        print("Saved model responses already exist in data/responses/.")
        print("Rebuilding the questions could make them not match those responses.")
        print("If you really want to start over, delete data/responses/small.jsonl and")
        print("data/responses/large.jsonl, or run this script again with --force.")
        sys.exit(1)

    rng = random.Random(SEED)
    records = build_gsm8k(rng) + build_mmlu(rng)
    rng.shuffle(records)  # mix the sources together

    # Give each question a short id: q001, q002, ...
    output = [{"id": f"q{number:03d}", **record} for number, record in enumerate(records, start=1)]

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    lengths = [len(item["query"]) for item in output]
    print()
    print(f"Saved {len(output)} questions to {output_path}")
    print()
    print("By source:")
    for name, count in Counter(item["source"] for item in output).items():
        print(f"  {name:<8} {count}")
    print("By group:")
    for name, count in Counter(item["group"] for item in output).items():
        print(f"  {name:<22} {count}")
    print()
    print(
        f"Prompt length (characters): shortest {min(lengths)}, "
        f"average {sum(lengths) // len(lengths)}, longest {max(lengths)}"
    )

    example = output[0]
    print()
    print(f"Example ({example['id']}, {example['source']}, correct answer: {example['reference']}):")
    print("-" * 60)
    print(example["query"][:500])
    print("-" * 60)


if __name__ == "__main__":
    main()