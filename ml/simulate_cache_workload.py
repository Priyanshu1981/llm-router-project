"""
simulate_cache_workload.py
Estimates how much the semantic cache would save on workloads that contain
repeated and paraphrased questions, and how often it would return a wrong answer.

It replays workloads through the SAME guard code the server uses
(backend/cache.guard_failure) and the same lookup rule (similarity >= threshold
and the guards pass). It never touches the real cache or the request log, and it
uses no Groq tokens.

Each question belongs to a "family": the same question, or paraphrases of it.
A cache hit is CORRECT if the matched cached question is in the same family as
the request, and WRONG if it is in a different family.

The workload mixes three kinds of questions:
  benchmark  : your distinct benchmark questions (they never repeat)
  paraphrase : the paraphrase pairs from data/raw/cache_pairs*.json (two wordings, one meaning)
  trap       : the "different" pairs (similar wording, different meaning)

The share of REPEAT requests is not measured from real traffic: it is a number we
choose and vary, so the results show how savings scale, not what you will save.

Run from the project root:
    python ml/simulate_cache_workload.py
"""

import argparse
import json
import sys
from pathlib import Path

# Make "from backend import ..." work when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw charts into files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend import cache, config, features  # noqa: E402

SEED = 7
FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # share of requests that repeat an earlier question
PRINTED_FRACTIONS = [0.1, 0.3, 0.5]


def section(title):
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


# ---------------------------------------------------------------
# Building the question families
# ---------------------------------------------------------------
def build_families(df):
    """
    Returns a list of families, each {"kind": ..., "variants": [question texts]}.
    A question text is never allowed to belong to two families, because the same
    text asked twice is the same question.
    """
    families = []
    text_to_family = {}

    def add_family(kind, variants):
        keys = [cache.normalize(text) for text in variants]
        if any(key in text_to_family for key in keys):
            return False
        index = len(families)
        families.append({"kind": kind, "variants": list(variants)})
        for key in keys:
            text_to_family[key] = index
        return True

    for query in df["query"]:
        add_family("benchmark", [features.clean_query(query)])

    skipped = 0
    pair_files = [config.RAW_DIR / "cache_pairs.json", config.RAW_DIR / "cache_pairs_fresh.json"]
    loaded = []
    for path in pair_files:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                loaded.append(json.load(file))
            for pair in loaded[-1]["paraphrase_pairs"]:
                if not add_family("paraphrase", [pair["a"], pair["b"]]):
                    skipped += 1

    for data in loaded:
        for pair in data["different_pairs"]:
            for text in (pair["a"], pair["b"]):
                # If this text already belongs to a family (for example it is also one
                # of the paraphrase questions), it is simply reused.
                add_family("trap", [text])

    if not loaded:
        print("Note: no cache_pairs files found, so the workload has only benchmark questions.")
    if skipped:
        print(f"Note: skipped {skipped} paraphrase pair(s) whose wording was already used elsewhere.")
    return families


# ---------------------------------------------------------------
# A copy of the cache's lookup rule, kept in memory only
# ---------------------------------------------------------------
class SimCache:
    def __init__(self, threshold, use_guards, dim, capacity):
        self.threshold = threshold
        self.use_guards = use_guards
        self.matrix = np.zeros((capacity, dim), dtype=np.float32)
        self.count = 0
        self.texts = []
        self.families = []

    def lookup(self, text, vector):
        """Index of a cached question whose answer may be reused, or None."""
        if self.count == 0:
            return None
        similarities = self.matrix[: self.count] @ vector
        for index in np.argsort(-similarities):
            if similarities[index] < self.threshold:
                break
            if self.use_guards and cache.guard_failure(text, self.texts[index]) is not None:
                continue
            return int(index)
        return None

    def add(self, text, family, vector):
        self.matrix[self.count] = vector
        self.count += 1
        self.texts.append(text)
        self.families.append(family)


# ---------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------
def make_workload(families, n_requests, repeat_fraction, rng):
    """
    A list of (family index, question text, is_repeat).
    A repeat re-asks a family that was already asked, using a randomly chosen wording of it.
    Everything else is a family asked for the first time.
    """
    order = rng.permutation(len(families))
    next_new = 0
    seen = []
    requests = []
    for _ in range(n_requests):
        want_repeat = bool(seen) and (rng.random() < repeat_fraction or next_new >= len(order))
        if want_repeat:
            family = seen[int(rng.integers(len(seen)))]
            is_repeat = True
        else:
            family = int(order[next_new])
            next_new += 1
            seen.append(family)
            is_repeat = False
        variants = families[family]["variants"]
        text = variants[int(rng.integers(len(variants)))]
        requests.append((family, text, is_repeat))
    return requests


def run_workload(requests, vector_of, threshold, use_guards):
    """Replay a workload through the cache. A miss stores the question, as the real server does."""
    dim = len(next(iter(vector_of.values())))
    sim = SimCache(threshold, use_guards, dim, capacity=len(requests) + 1)
    hits = correct_hits = wrong_hits = 0
    repeats = repeats_served = 0

    for family, text, is_repeat in requests:
        vector = vector_of[text]
        match = sim.lookup(text, vector)
        if is_repeat:
            repeats += 1
        if match is None:
            sim.add(text, family, vector)
            continue
        hits += 1
        if sim.families[match] == family:
            correct_hits += 1
            if is_repeat:
                repeats_served += 1
        else:
            wrong_hits += 1

    return {
        "requests": len(requests), "hits": hits, "correct_hits": correct_hits,
        "wrong_hits": wrong_hits, "repeats": repeats, "repeats_served": repeats_served,
    }


def summarize(runs, miss_ms, hit_ms):
    n = np.array([r["requests"] for r in runs], dtype=float)
    hits = np.array([r["hits"] for r in runs], dtype=float)
    wrong = np.array([r["wrong_hits"] for r in runs], dtype=float)
    repeats = sum(r["repeats"] for r in runs)
    served = sum(r["repeats_served"] for r in runs)

    avoided = hits / n * 100
    latency = (1 - hits / n) * miss_ms + (hits / n) * hit_ms
    return {
        "calls_avoided_mean": float(avoided.mean()),
        "calls_avoided_min": float(avoided.min()),
        "calls_avoided_max": float(avoided.max()),
        "repeat_recall": (served / repeats * 100) if repeats else float("nan"),
        "wrong_hits_mean": float(wrong.mean()),
        "wrong_hits_max": int(wrong.max()),
        "wrong_share_of_hits": float(wrong.sum() / hits.sum() * 100) if hits.sum() > 0 else 0.0,
        "est_latency_ms": float(latency.mean()),
    }


def show_pct(value):
    return "n/a" if value != value else f"{value:.0f}%"


def main():
    parser = argparse.ArgumentParser(description="Simulate cache workloads.")
    parser.add_argument("--requests", type=int, default=400, help="requests per simulated workload (default 400)")
    parser.add_argument("--seeds", type=int, default=20, help="random workloads per setting (default 20)")
    parser.add_argument("--hit-ms", type=float, default=20.0,
                        help="assumed time of a cache hit in milliseconds (default 20, from the live test)")
    args = parser.parse_args()

    labeled_path = config.PROCESSED_DIR / "labeled.csv"
    if not labeled_path.exists():
        print(f"Could not find {labeled_path}. Run  python ml/label_data.py  first.")
        sys.exit(1)
    df = pd.read_csv(labeled_path, encoding="utf-8-sig")
    miss_ms = float(df["large_latency_ms"].mean())

    families = build_families(df)
    kinds = pd.Series([family["kind"] for family in families]).value_counts()

    section("Setup")
    print(f"  Question families: {len(families)} "
          f"({int(kinds.get('benchmark', 0))} benchmark, {int(kinds.get('paraphrase', 0))} paraphrase, "
          f"{int(kinds.get('trap', 0))} trap)")
    print(f"  Workloads: {args.requests} requests, {args.seeds} random workloads per setting")
    print(f"  Assumed time per request: a miss takes {miss_ms / 1000:.2f} s (average saved large-model answer),")
    print(f"  a hit takes {args.hit_ms:.0f} ms. These are assumptions used only for the latency estimate.")

    texts = list(dict.fromkeys(text for family in families for text in family["variants"]))
    print()
    print(f"Computing embeddings for {len(texts)} questions (this loads the embedding model) ...")
    vectors = features.embed_texts(texts)
    vector_of = {text: vectors[i] for i, text in enumerate(texts)}

    thresholds = sorted(set([0.85, round(config.CACHE_THRESHOLD, 2), 0.95]))

    print("Running workloads ...")
    runs = {}
    for fraction in FRACTIONS:
        for seed in range(args.seeds):
            rng = np.random.default_rng(SEED + 1000 * seed + int(round(fraction * 100)))
            requests = make_workload(families, args.requests, fraction, rng)
            for threshold in thresholds:
                for guards in (True, False):
                    runs.setdefault((fraction, threshold, guards), []).append(
                        run_workload(requests, vector_of, threshold, guards)
                    )

    summaries = {key: summarize(value, miss_ms, args.hit_ms) for key, value in runs.items()}

    # -----------------------------------------------------------
    # Tables
    # -----------------------------------------------------------
    def print_table(guards):
        print(f"  {'repeat share':>12} {'threshold':>9} | {'calls avoided':>24} {'repeat recall':>13} "
              f"{'wrong hits per workload':>24} {'est. avg latency':>17}")
        print("  " + "-" * 108)
        for fraction in PRINTED_FRACTIONS:
            for threshold in thresholds:
                s = summaries[(fraction, threshold, guards)]
                marker = "  <- .env" if abs(threshold - config.CACHE_THRESHOLD) < 1e-9 else ""
                print(f"  {fraction * 100:>11.0f}% {threshold:>9.2f} | "
                      f"{s['calls_avoided_mean']:>6.1f}% [{s['calls_avoided_min']:>4.1f}-{s['calls_avoided_max']:>4.1f}]     "
                      f"{show_pct(s['repeat_recall']):>13} "
                      f"{s['wrong_hits_mean']:>12.2f} (max {s['wrong_hits_max']:>2})      "
                      f"{s['est_latency_ms'] / 1000:>10.2f} s{marker}")
            print()

    section("Semantic cache with the guards ON (what the server does)")
    print("  'repeat share' = share of requests that re-ask an earlier question (assumed, not measured).")
    print("  'calls avoided' = share of requests answered from the cache, with the lowest and highest")
    print("  value over the random workloads. 'repeat recall' = share of repeats that were served correctly.")
    print()
    print_table(True)

    section("The same workloads with the guards OFF (similarity threshold only)")
    print_table(False)

    # -----------------------------------------------------------
    # Distinct benchmark questions: every hit here is a false hit
    # -----------------------------------------------------------
    section("False hits on distinct benchmark questions (no question repeats, so every hit is wrong)")
    benchmark_ids = [i for i, family in enumerate(families) if family["kind"] == "benchmark"]
    print(f"  Each test streams all {len(benchmark_ids)} distinct benchmark questions through an empty cache.")
    print(f"  {'threshold':>9} | {'wrong hits, guards ON (mean / max)':>36} | {'wrong hits, guards OFF (mean / max)':>36}")
    print("  " + "-" * 88)
    false_hit_rows = []
    for threshold in thresholds:
        cells = {}
        for guards in (True, False):
            counts = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(SEED + 500000 + seed)
                order = rng.permutation(benchmark_ids)
                requests = [(int(i), families[int(i)]["variants"][0], False) for i in order]
                counts.append(run_workload(requests, vector_of, threshold, guards)["hits"])
            cells[guards] = (float(np.mean(counts)), int(np.max(counts)))
            false_hit_rows.append({"threshold": threshold, "guards": guards,
                                   "wrong_hits_mean": cells[guards][0], "wrong_hits_max": cells[guards][1]})
        print(f"  {threshold:>9.2f} | {cells[True][0]:>26.2f} / {cells[True][1]:<7} | {cells[False][0]:>26.2f} / {cells[False][1]:<7}")
    print()
    print("  Multiple-choice questions share long template text, so the similarity threshold alone")
    print("  can match different questions. This shows what the guards prevent on realistic questions.")

    # -----------------------------------------------------------
    # Save
    # -----------------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for (fraction, threshold, guards), s in sorted(summaries.items()):
        rows.append({"repeat_share": fraction, "threshold": threshold, "guards": guards, **{k: round(v, 3) for k, v in s.items()}})
    results_path = config.RESULTS_DIR / "cache_workload_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    false_path = config.RESULTS_DIR / "cache_false_hits_benchmark.csv"
    pd.DataFrame(false_hit_rows).to_csv(false_path, index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = [f * 100 for f in FRACTIONS]
    for threshold in thresholds:
        ys = [summaries[(f, threshold, True)]["calls_avoided_mean"] for f in FRACTIONS]
        ax.plot(xs, ys, marker="o", label=f"Cache, threshold {threshold:.2f}, guards on")
    ax.plot(xs, xs, linestyle="--", color="grey", label="Upper bound: every repeat served")
    ax.set_xlabel("Share of requests that repeat an earlier question (%)")
    ax.set_ylabel("Requests answered from the cache (%)")
    ax.set_title("Cache savings grow with repetition (simulated workloads)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    chart_path = config.RESULTS_DIR / "cache_workload_curve.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print()
    print("Saved:")
    for path in (results_path, false_path, chart_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()