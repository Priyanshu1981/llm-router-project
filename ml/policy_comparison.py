"""
policy_comparison.py
Combines two earlier results into one table:
  - experiments/results/strategy_comparison.csv      (cost and accuracy of each routing policy)
  - experiments/results/cache_workload_results.csv   (how many calls the cache avoids)

For a given repeat share, the cache answers some requests for free, so a policy's
cost falls to:   policy cost  x  (1 - share of calls the cache avoids)

Assumptions (state them in your report):
  - a cached answer has the same quality as the policy that produced it
  - a cache hit takes about --hit-ms milliseconds
  - wrong cache hits are not counted here (they are reported in the cache experiments)
  - the repeat share is an assumption, not a measurement of real traffic

Uses no Groq tokens. Run from the project root:
    python ml/policy_comparison.py
"""

import argparse
import sys
from pathlib import Path

# Make "from backend import config" work when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backend import config  # noqa: E402

POLICIES = [
    ("Always large", "Always large"),
    ("Router (nested out-of-fold)", "Router (as trained)"),
    ("Always small", "Always small"),
]
SHARES = [0.0, 0.1, 0.3, 0.5]


def main():
    parser = argparse.ArgumentParser(description="Combine policy and cache results.")
    parser.add_argument("--hit-ms", type=float, default=20.0, help="assumed time of a cache hit (default 20 ms)")
    args = parser.parse_args()

    strategy_path = config.RESULTS_DIR / "strategy_comparison.csv"
    workload_path = config.RESULTS_DIR / "cache_workload_results.csv"
    for path, script in [(strategy_path, "evaluate_strategies.py"), (workload_path, "simulate_cache_workload.py")]:
        if not path.exists():
            print(f"Could not find {path}. Run  python ml/{script}  first.")
            sys.exit(1)

    strategy = pd.read_csv(strategy_path)
    workload = pd.read_csv(workload_path)

    guards_on = workload["guards"].astype(str).str.lower() == "true"
    same_threshold = (workload["threshold"] - config.CACHE_THRESHOLD).abs() < 1e-6
    workload = workload[guards_on & same_threshold]
    if workload.empty:
        print(f"No cache results found for threshold {config.CACHE_THRESHOLD:.2f} with the guards on.")
        sys.exit(1)

    by_policy = {row["policy"]: row for _, row in strategy.iterrows()}
    for key, _ in POLICIES:
        if key not in by_policy:
            print(f"The policy '{key}' is missing from {strategy_path.name}. Rerun evaluate_strategies.py.")
            sys.exit(1)

    print()
    print("Accuracy of each policy on the benchmark (95% interval):")
    for key, label in POLICIES:
        p = by_policy[key]
        print(f"  {label:<22} {p['accuracy_pct']:.1f}%  [{p['accuracy_ci_low']:.1f}, {p['accuracy_ci_high']:.1f}]")
    print("  (the cache does not change these, apart from rare wrong hits)")

    print()
    print(f"Cost (% of always-large without a cache) and average latency, cache threshold {config.CACHE_THRESHOLD:.2f}, hit = {args.hit_ms:.0f} ms")
    print()
    head = f"  {'repeat share':>12} {'cache avoids':>13} |"
    for _, label in POLICIES:
        head += f" {label:^19} |"
    print(head)
    print("  " + "-" * (len(head) - 2))

    rows = []
    for share in SHARES:
        match = workload[(workload["repeat_share"] - share).abs() < 1e-9]
        if match.empty:
            continue
        avoided = float(match.iloc[0]["calls_avoided_mean"]) / 100.0
        line = f"  {share * 100:>11.0f}% {avoided * 100:>12.1f}% |"
        for key, label in POLICIES:
            p = by_policy[key]
            cost_pct = float(p["cost_pct_of_large"]) * (1 - avoided)
            latency_ms = (1 - avoided) * float(p["avg_latency_ms"]) + avoided * args.hit_ms
            line += f" {cost_pct:>7.1f}%  {latency_ms / 1000:>5.2f} s   |"
            rows.append({
                "repeat_share": share, "cache_avoids_pct": round(avoided * 100, 2), "policy": label,
                "accuracy_pct": round(float(p["accuracy_pct"]), 2),
                "cost_pct_of_large_with_cache": round(cost_pct, 2),
                "avg_latency_ms_with_cache": round(latency_ms, 1),
            })
        print(line)

    result = pd.DataFrame(rows)
    out_path = config.RESULTS_DIR / "policy_with_cache.csv"
    result.to_csv(out_path, index=False)

    print()
    print("Reading this table:")
    print("  - At 0% repeats the cache changes nothing, so the policy alone decides the cost.")
    print("  - Always small is the cheapest policy, and its accuracy gap to always large is in the")
    print("    accuracy list above. Judge that gap against its interval, not just the point value.")
    print("  - The router costs more than always small without winning clearly on accuracy: the")
    print("    evaluation found no evidence that it picks the right model better than chance.")
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()