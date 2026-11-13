"""
eval_cache_pairs.py
Measures how the semantic cache behaves on hand-written question pairs.

For each threshold it reports:
  - hit rate on PARAPHRASE pairs  (good: the cache reuses an answer correctly)
  - wrong-hit rate on DIFFERENT pairs  (bad: the cache reuses a wrong answer)
with the safety guards on and off.

Uses no Groq tokens. Run from the project root:
    python ml/eval_cache_pairs.py
"""

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

THRESHOLDS = np.round(np.arange(0.50, 1.0001, 0.01), 2)
SHOWN_THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def short(text, limit=44):
    text = text.replace("\n", " / ")
    return text if len(text) <= limit else text[:limit] + "..."


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    path = config.RAW_DIR / "cache_pairs.json"
    if not path.exists():
        print(f"Could not find {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    paraphrase = data["paraphrase_pairs"]
    different = data["different_pairs"]
    pairs = [("paraphrase", p) for p in paraphrase] + [("different", p) for p in different]

    print("Computing embeddings (loads the embedding model, so the first run can take a minute) ...")
    texts = []
    for _, pair in pairs:
        texts.extend([pair["a"], pair["b"]])
    vectors = features.embed_texts(texts)

    rows = []
    for index, (kind, pair) in enumerate(pairs):
        similarity = float(vectors[2 * index] @ vectors[2 * index + 1])
        reason = cache.guard_failure(pair["a"], pair["b"])
        rows.append(
            {
                "kind": kind,
                "question_a": pair["a"],
                "question_b": pair["b"],
                "similarity": round(similarity, 4),
                "guard_blocks": reason is not None,
                "guard_reason": reason or "",
                "why_different": pair.get("why", ""),
            }
        )
    df = pd.DataFrame(rows)

    # -----------------------------------------------------------
    # Every pair with its similarity
    # -----------------------------------------------------------
    section("PARAPHRASE pairs: the cache SHOULD reuse the answer (highest similarity first)")
    print(f"  {'similarity':>10}  {'guards':<8}  question A")
    for _, row in df[df["kind"] == "paraphrase"].sort_values("similarity", ascending=False).iterrows():
        print(f"  {row['similarity']:>10.3f}  {'BLOCKED' if row['guard_blocks'] else 'passes':<8}  {short(row['question_a'])}")

    section("DIFFERENT pairs: the cache must NOT reuse the answer (highest similarity first)")
    print(f"  {'similarity':>10}  {'guards':<8}  question A   |   what differs")
    for _, row in df[df["kind"] == "different"].sort_values("similarity", ascending=False).iterrows():
        print(f"  {row['similarity']:>10.3f}  {'BLOCKED' if row['guard_blocks'] else 'passes':<8}  "
              f"{short(row['question_a'], 34)}   |   {row['why_different']}")

    # -----------------------------------------------------------
    # Threshold sweep
    # -----------------------------------------------------------
    similarities = df["similarity"].to_numpy()
    guard_ok = ~df["guard_blocks"].to_numpy()
    is_paraphrase = (df["kind"] == "paraphrase").to_numpy()

    def rates(threshold, use_guards):
        allowed = guard_ok if use_guards else np.ones_like(guard_ok)
        hit = (similarities >= threshold) & allowed
        return float(hit[is_paraphrase].mean()), float(hit[~is_paraphrase].mean())

    sweep_rows = []
    for threshold in THRESHOLDS:
        hit_g, wrong_g = rates(threshold, True)
        hit_n, wrong_n = rates(threshold, False)
        sweep_rows.append(
            {
                "threshold": float(threshold),
                "hit_rate_guards": hit_g,
                "wrong_hit_guards": wrong_g,
                "hit_rate_no_guards": hit_n,
                "wrong_hit_no_guards": wrong_n,
            }
        )
    sweep = pd.DataFrame(sweep_rows)

    section("Threshold sweep (percent of pairs)")
    print(f"  {'threshold':>9} | {'paraphrases reused':>18} {'wrong reuse':>12} | "
          f"{'paraphrases reused':>18} {'wrong reuse':>12}")
    print(f"  {'':>9} | {'(guards ON)':>18} {'(guards ON)':>12} | {'(guards OFF)':>18} {'(guards OFF)':>12}")
    print("  " + "-" * 76)
    shown = sorted(set(SHOWN_THRESHOLDS + [round(config.CACHE_THRESHOLD, 2)]))
    for threshold in shown:
        row = sweep[np.isclose(sweep["threshold"], threshold)].iloc[0]
        marker = "  <- your .env threshold" if abs(threshold - config.CACHE_THRESHOLD) < 1e-9 else ""
        print(f"  {threshold:>9.2f} | {row['hit_rate_guards'] * 100:>17.0f}% {row['wrong_hit_guards'] * 100:>11.0f}% | "
              f"{row['hit_rate_no_guards'] * 100:>17.0f}% {row['wrong_hit_no_guards'] * 100:>11.0f}%{marker}")

    # -----------------------------------------------------------
    # Overlap summary
    # -----------------------------------------------------------
    section("Where the two groups overlap")
    para_sims = df[df["kind"] == "paraphrase"]["similarity"]
    diff_all = df[df["kind"] == "different"]
    diff_unblocked = diff_all[~diff_all["guard_blocks"]]
    print(f"  Lowest similarity among paraphrases      : {para_sims.min():.3f}")
    print(f"  Median similarity among paraphrases      : {para_sims.median():.3f}")
    print(f"  Highest similarity among 'different' pairs, guards off : {diff_all['similarity'].max():.3f}")
    if len(diff_unblocked):
        print(f"  Highest similarity among 'different' pairs the guards let through : "
              f"{diff_unblocked['similarity'].max():.3f}")
    else:
        print("  The guards blocked every 'different' pair.")
    blocked_paraphrases = int(df[(df["kind"] == "paraphrase") & df["guard_blocks"]].shape[0])
    print(f"  Paraphrases wrongly blocked by a guard   : {blocked_paraphrases} of {len(paraphrase)}")
    print()
    print("  Reading this: if the highest 'different' similarity is above the lowest paraphrase")
    print("  similarity, no threshold separates them perfectly, and the guards matter.")
    print(f"  With only {len(paraphrase)} pairs per group, each pair moves a rate by {100 / len(paraphrase):.0f} points.")
    print("  Treat these as a development set. Do not quote them as a final accuracy.")

    # -----------------------------------------------------------
    # Save
    # -----------------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_path = config.RESULTS_DIR / "cache_threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False)
    pairs_path = config.RESULTS_DIR / "cache_pairs_scored.csv"
    df.to_csv(pairs_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(sweep["threshold"], sweep["hit_rate_guards"] * 100, color="tab:green",
            label="Paraphrases reused (guards on)")
    ax.plot(sweep["threshold"], sweep["wrong_hit_guards"] * 100, color="tab:red",
            label="Wrong reuse (guards on)")
    ax.plot(sweep["threshold"], sweep["wrong_hit_no_guards"] * 100, color="tab:red",
            linestyle="--", label="Wrong reuse (guards off)")
    ax.axvline(config.CACHE_THRESHOLD, color="grey", linestyle=":",
               label=f"Threshold in .env ({config.CACHE_THRESHOLD:.2f})")
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Share of pairs (%)")
    ax.set_title("Semantic cache: reuse vs wrong reuse")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout()
    chart_path = config.RESULTS_DIR / "cache_threshold_curve.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print()
    print(f"Saved: {sweep_path}")
    print(f"       {pairs_path}")
    print(f"       {chart_path}")


if __name__ == "__main__":
    main()