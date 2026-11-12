"""
collect_responses.py
Sends every question in data/raw/queries.json to the small and the large model
and saves each response to disk, one line per response:

    data/responses/small.jsonl
    data/responses/large.jsonl

Safe to stop and restart: questions that already have a saved response are
skipped, so nothing is ever collected (or paid for) twice.

Usage (from the project root):
    python ml/collect_responses.py --limit 5     quick test with the first 5 questions
    python ml/collect_responses.py               everything, small model then large model
    python ml/collect_responses.py --tier small  only the small model
    python ml/collect_responses.py --tier large  only the large model
    python ml/collect_responses.py --tpm 4000    slower pace, if you keep hitting rate limits
"""

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# Make "from backend import ..." work when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402
from backend.llm_client import LLMError, generate  # noqa: E402


# ---------------------------------------------------------------
# Pacing: stay under the tokens-per-minute limit
# ---------------------------------------------------------------
class Pacer:
    """
    Remembers how many tokens were used in the last 60 seconds and makes us
    wait before the next call if that would go over the budget.
    """

    def __init__(self, tokens_per_minute, min_gap_seconds=2.0):
        self.budget = tokens_per_minute
        self.min_gap = min_gap_seconds
        self.window = deque()  # (time of call, tokens used)
        self.last_call_time = 0.0

    def _forget_old_calls(self, now):
        while self.window and now - self.window[0][0] > 60:
            self.window.popleft()

    def wait_turn(self):
        """Block until it is safe to make the next call."""
        while True:
            now = time.time()
            self._forget_old_calls(now)

            used = sum(tokens for _, tokens in self.window)
            if self.window:
                expected_next = used / len(self.window)  # average size of recent calls
            else:
                expected_next = 600  # a first guess

            gap_is_fine = (now - self.last_call_time) >= self.min_gap
            fits_budget = (used + expected_next) <= self.budget or not self.window
            if gap_is_fine and fits_budget:
                return
            time.sleep(0.5)

    def record(self, tokens):
        """Tell the pacer a call just finished and how many tokens it used."""
        now = time.time()
        self.window.append((now, tokens))
        self.last_call_time = now


# ---------------------------------------------------------------
# Reading and writing files
# ---------------------------------------------------------------
def load_queries():
    path = config.RAW_DIR / "queries.json"
    if not path.exists():
        print(f"Could not find {path}")
        print("Run this first:  python ml/build_dataset.py")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def response_path(tier):
    return config.RESPONSES_DIR / f"{tier}.jsonl"


def load_done_ids(path):
    """Ids of the questions that already have a saved response."""
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue  # ignore a half-written last line
    return done


# ---------------------------------------------------------------
# Collecting
# ---------------------------------------------------------------
def run_queue(tier, todo, path, tokens_per_minute, stats):
    """
    Go through the questions one by one. Returns "done", "limit" or "error".
    Progress numbers are written into the stats dictionary as we go.
    """
    pacer = Pacer(tokens_per_minute)
    rate_limit_streak = 0   # rate-limit errors in a row
    failed_streak = 0       # questions in a row that could not be answered

    with open(path, "a", encoding="utf-8") as file:
        for position, item in enumerate(todo, start=1):
            result = None
            attempts = 0

            while result is None:
                pacer.wait_turn()
                try:
                    result = generate(tier, item["query"])
                except LLMError as error:
                    if error.status_code == 429:
                        rate_limit_streak += 1
                        if rate_limit_streak >= 3:
                            print()
                            print("  Groq is still refusing requests after waiting.")
                            print("  This is most likely the free daily token limit for this model.")
                            print("  Everything so far is saved. Run the same command again later")
                            print("  and the script will carry on where it stopped.")
                            return "limit"
                        print(f"  Rate limit hit. Waiting 65 seconds ({rate_limit_streak} of 2 waits)...")
                        time.sleep(65)
                        continue

                    if error.status_code in (400, 500):
                        # Missing key or invalid request: retrying will not help.
                        print(f"  Stopping: {error.message}")
                        return "error"

                    attempts += 1
                    print(f"  Problem on {item['id']} (attempt {attempts} of 3): {error.message}")
                    if attempts >= 3:
                        break
                    time.sleep(5)

            if result is None:
                failed_streak += 1
                if failed_streak >= 3:
                    print("  Stopping: 3 questions in a row failed. Check the messages above.")
                    return "error"
                continue  # skip this one for now; the next run will retry it

            rate_limit_streak = 0
            failed_streak = 0

            record = {
                "id": item["id"],
                "tier": tier,
                "model_name": result["model_name"],
                "text": result["text"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "latency_ms": round(result["latency_ms"], 1),
                "finish_reason": result["finish_reason"],
                "reasoning_effort": config.REASONING_EFFORT,
                "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()  # write to disk right now, so nothing is lost if we stop

            tokens = result["input_tokens"] + result["output_tokens"]
            pacer.record(tokens)
            stats["collected"] += 1
            stats["tokens"] += tokens
            stats["latency_ms"] += result["latency_ms"]
            if result["finish_reason"] == "length":
                stats["truncated"] += 1

            print(
                f"  [{position:>3}/{len(todo)}] {item['id']} {item['source']:<6} "
                f"{result['latency_ms'] / 1000:5.1f}s {tokens:>6} tokens   "
                f"(run total: {stats['tokens']:,})"
            )

    return "done"


def print_run_summary(tier, path, stats):
    saved_in_file = len(load_done_ids(path))
    print()
    print(f"  --- {tier} model summary ---")
    print(f"  Collected in this run : {stats['collected']}")
    print(f"  Saved in total        : {saved_in_file}")
    print(f"  Tokens used this run  : {stats['tokens']:,}")
    if stats["collected"]:
        print(f"  Average latency       : {stats['latency_ms'] / stats['collected']:.0f} ms")
        print(f"  Average tokens/query  : {stats['tokens'] // stats['collected']}")
    if stats["truncated"]:
        print(
            f"  WARNING: {stats['truncated']} answer(s) were cut off because they hit "
            f"MAX_OUTPUT_TOKENS ({config.MAX_OUTPUT_TOKENS})."
        )


def collect_tier(tier, queries, limit, tokens_per_minute):
    path = response_path(tier)
    done_ids = load_done_ids(path)
    wanted = queries[:limit] if limit else queries
    todo = [item for item in wanted if item["id"] not in done_ids]

    print()
    print("=" * 66)
    print(f"{tier.upper()} model: {config.MODEL_BY_TIER[tier]}")
    print(f"  already saved: {len(done_ids)}   to collect now: {len(todo)}")
    print("=" * 66)

    if not todo:
        print("  Nothing to do for this model.")
        return "done"

    stats = {"collected": 0, "tokens": 0, "truncated": 0, "latency_ms": 0.0}
    try:
        status = run_queue(tier, todo, path, tokens_per_minute, stats)
    finally:
        print_run_summary(tier, path, stats)
    return status


def main():
    parser = argparse.ArgumentParser(description="Collect model responses for every question.")
    parser.add_argument("--tier", choices=["small", "large", "both"], default="both",
                        help="which model to collect for (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N questions (good for a quick test)")
    parser.add_argument("--tpm", type=int, default=6000,
                        help="tokens-per-minute budget for pacing (default: 6000)")
    args = parser.parse_args()

    queries = load_queries()
    tiers = ["small", "large"] if args.tier == "both" else [args.tier]

    outcomes = {}
    try:
        for tier in tiers:
            outcomes[tier] = collect_tier(tier, queries, args.limit, args.tpm)
            if outcomes[tier] == "error":
                break
    except KeyboardInterrupt:
        print()
        print("Stopped by you. Everything collected so far is saved.")
        print("Run the same command again to carry on.")
        return

    print()
    print("=" * 66)
    for tier, outcome in outcomes.items():
        label = {"done": "finished", "limit": "stopped at the rate limit", "error": "stopped because of an error"}[outcome]
        print(f"  {tier:<6} {label}")
    print("=" * 66)

    if all(outcome == "done" for outcome in outcomes.values()) and len(outcomes) == len(tiers):
        print("Next step:  python ml/label_data.py")
    else:
        print("Some models are not finished. Fix the problem or wait, then run the same command again.")


if __name__ == "__main__":
    main()