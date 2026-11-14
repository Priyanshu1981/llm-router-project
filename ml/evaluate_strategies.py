"""
evaluate_strategies.py
Compares routing strategies on ALL labeled questions, using the answers that are
already saved (data/processed/labeled.csv), so it uses no Groq tokens.

Why this exists: in Phase 4 the router was tested on one split with very few
hard questions, which is too little to say anything reliable. Here every hard
question is used for testing, but never by a router that was trained on it:

  nested cross-validation
    outer loop : split all questions into folds and hold one fold out
    inner loop : using ONLY the remaining questions, choose the routing threshold
                 (the cheapest threshold whose accuracy stays within the tolerance
                 of always-large accuracy)
    then       : train on the remaining questions and route the held-out fold

Every question is therefore routed by a model and a threshold that never saw it.
This is repeated with different random splits and the results are averaged.

Confidence intervals come from a bootstrap: the questions are resampled many
times and each policy is re-scored on every resample.

Run from the project root:
    python ml/evaluate_strategies.py
    python ml/evaluate_strategies.py --repeats 1      quicker test run
"""

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

# Make "from backend import ..." work when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw charts into files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from backend import config, features  # noqa: E402
from train_router import choose_threshold, make_pipeline, to_arrays  # noqa: E402

warnings.filterwarnings("ignore", module="sklearn")

SEED = 42
VALID_VARIANTS = ("handcrafted_lr", "embedding_lr", "combined_lr", "combined_rf")

# A simple rule: requests that ask for a proof or a derivation go straight to the
# large model, whatever the router says.
GUARDRAIL_RE = re.compile(r"\b(?:prove|derive|show that|demonstrate that)\b", re.IGNORECASE)

ALWAYS_SMALL = "Always small"
ALWAYS_LARGE = "Always large"
RANDOM = "Random (same share as router)"
ROUTER = "Router (nested out-of-fold)"
GUARDED = "Router + keyword guardrail"
PERFECT = "Perfect router (oracle)"
POLICY_ORDER = [ALWAYS_SMALL, ALWAYS_LARGE, RANDOM, ROUTER, GUARDED, PERFECT]


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_labeled():
    path = config.PROCESSED_DIR / "labeled.csv"
    if not path.exists():
        print(f"Could not find {path}. Run  python ml/label_data.py  first.")
        sys.exit(1)
    df = pd.read_csv(path, encoding="utf-8-sig")
    needed = ["id", "source", "query", "small_ok", "small_correct", "large_correct",
              "small_cost_usd", "large_cost_usd", "small_latency_ms", "large_latency_ms"]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        print(f"labeled.csv is missing these columns: {missing}")
        print("Run  python ml/label_data.py  again.")
        sys.exit(1)
    return df


# ---------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------
def nested_out_of_fold(X, d, variant, n_hand, n_emb, tolerance, repeats, outer_folds):
    """
    Returns two arrays of shape (repeats, number of questions):
      probabilities : P(small model is good enough), from a model that never saw the question
      thresholds    : the threshold chosen without ever seeing the question
    """
    y = d["y"]
    n = len(y)
    probabilities = np.zeros((repeats, n))
    thresholds = np.zeros((repeats, n))

    for repeat in range(repeats):
        outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=SEED + repeat)
        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
            d_train = {key: value[train_idx] for key, value in d.items()}
            y_train = y[train_idx]

            # Inner loop: honest predictions on the training part, used only to pick the threshold.
            inner_folds = min(5, int((y_train == 0).sum()), int((y_train == 1).sum()))
            inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED + 100 + repeat)
            p_inner = cross_val_predict(
                make_pipeline(variant, n_hand, n_emb), X[train_idx], y_train,
                cv=inner, method="predict_proba",
            )[:, 1]
            threshold, _found = choose_threshold(p_inner, d_train, tolerance)

            # Train on the whole training part and route the held-out questions.
            model = make_pipeline(variant, n_hand, n_emb).fit(X[train_idx], y_train)
            classes = list(model.classes_)
            probabilities[repeat, test_idx] = model.predict_proba(X[test_idx])[:, classes.index(1)]
            thresholds[repeat, test_idx] = threshold

        print(f"  repeat {repeat + 1} of {repeats} done")
    return probabilities, thresholds


# ---------------------------------------------------------------
# Policies and statistics
# ---------------------------------------------------------------
def policy_vectors(share_to_small, small_c, large_c, small_cost, large_cost, small_lat, large_lat):
    """
    Per-question expected correctness, cost and latency of a policy.
    share_to_small is, for each question, the probability that the policy sends it to the small model
    (0 or 1 for a fixed policy, in between for something averaged over repeats).
    """
    share = np.asarray(share_to_small, dtype=float)
    return {
        "share": share,
        "correct": share * small_c + (1 - share) * large_c,
        "cost": share * small_cost + (1 - share) * large_cost,
        "latency": share * small_lat + (1 - share) * large_lat,
    }


def point_estimates(policy, large):
    return {
        "share": float(policy["share"].mean()),
        "accuracy": float(policy["correct"].mean() * 100),
        "accuracy_vs_large": float((policy["correct"].mean() - large["correct"].mean()) * 100),
        "cost_usd": float(policy["cost"].sum()),
        "cost_pct": float(policy["cost"].sum() / large["cost"].sum() * 100),
        "latency_ms": float(policy["latency"].mean()),
    }


def bootstrap_policies(policies, n_boot, rng):
    """Re-score every policy on the same resampled questions. Returns draws per policy."""
    n = len(policies[ALWAYS_LARGE]["correct"])
    idx = rng.integers(0, n, size=(n_boot, n))
    large_accuracy = policies[ALWAYS_LARGE]["correct"][idx].mean(axis=1)
    large_cost = policies[ALWAYS_LARGE]["cost"][idx].sum(axis=1)

    draws = {}
    for name, policy in policies.items():
        accuracy = policy["correct"][idx].mean(axis=1)
        cost = policy["cost"][idx].sum(axis=1)
        draws[name] = {
            "accuracy": accuracy * 100,
            "accuracy_vs_large": (accuracy - large_accuracy) * 100,
            "cost_pct": cost / large_cost * 100,
        }
    return draws


def interval(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def safe_auc(y_true, scores):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def bootstrap_auc(y_true, scores, n_boot, rng):
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(roc_auc_score(y_true[idx], scores[idx]))
    if not values:
        return float("nan"), float("nan")
    return interval(values)


def verdict(low, high):
    return "(interval includes 0: no clear difference)" if low <= 0 <= high else "(interval excludes 0)"


def show_auc(value):
    return "n/a" if value != value else f"{value:.2f}"


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare routing strategies on all labeled questions.")
    parser.add_argument("--repeats", type=int, default=3, help="how many times to repeat the nested cross-validation (default 3)")
    parser.add_argument("--folds", type=int, default=5, help="outer cross-validation folds (default 5)")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="accuracy the router may lose versus always-large when choosing a threshold (default 0.02)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples (default 2000)")
    parser.add_argument("--variant", default=None, help="router model type (default: the one chosen in training)")
    args = parser.parse_args()

    df = load_labeled()
    n = len(df)
    d = to_arrays(df)
    y = d["y"]
    hard = y == 0
    n_hard = int(hard.sum())

    if n_hard < args.folds or int((y == 1).sum()) < args.folds:
        print(f"Not enough hard questions ({n_hard}) for {args.folds} folds. Use fewer folds with --folds.")
        sys.exit(1)

    small_c, large_c = d["small_correct"], d["large_correct"]
    small_cost, large_cost = d["small_cost"], d["large_cost"]
    small_lat = df["small_latency_ms"].to_numpy(dtype=float)
    large_lat = df["large_latency_ms"].to_numpy(dtype=float)

    variant = args.variant
    if variant is None:
        variant = "combined_rf"
        report_path = config.MODELS_DIR / "router_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as file:
                variant = json.load(file).get("chosen_variant", variant)
    if variant not in VALID_VARIANTS:
        print(f"Unknown router variant '{variant}'. Choose one of: {', '.join(VALID_VARIANTS)}")
        sys.exit(1)

    section("Setup")
    print(f"  Questions          : {n} ({n_hard} hard, {n - n_hard} easy)")
    print(f"  Router model type  : {variant}")
    print(f"  Cross-validation   : {args.folds} outer folds x {args.repeats} repeats, tolerance {args.tolerance * 100:.0f} points")
    print(f"  Bootstrap          : {args.bootstrap} resamples for the 95% intervals")

    print()
    print("Computing features (this loads the embedding model) ...")
    X = features.build_features(df["query"].tolist())
    n_hand = len(features.FEATURE_NAMES)
    n_emb = X.shape[1] - n_hand

    print("Running nested cross-validation (a few minutes) ...")
    started = time.time()
    probabilities, thresholds = nested_out_of_fold(
        X, d, variant, n_hand, n_emb, args.tolerance, args.repeats, args.folds
    )
    print(f"  finished in {time.time() - started:.0f} seconds")

    use_small = probabilities >= thresholds                       # (repeats, questions)
    router_share = float(use_small.mean())
    guard_hit = np.array([bool(GUARDRAIL_RE.search(features.clean_query(q))) for q in df["query"]])
    use_small_guarded = use_small & ~guard_hit[None, :]

    def make(share):
        return policy_vectors(share, small_c, large_c, small_cost, large_cost, small_lat, large_lat)

    policies = {
        ALWAYS_SMALL: make(np.ones(n)),
        ALWAYS_LARGE: make(np.zeros(n)),
        RANDOM: make(np.full(n, router_share)),
        ROUTER: make(use_small.mean(axis=0)),
        GUARDED: make(use_small_guarded.mean(axis=0)),
        PERFECT: make((y == 1).astype(float)),
    }
    points = {name: point_estimates(policies[name], policies[ALWAYS_LARGE]) for name in POLICY_ORDER}

    rng = np.random.default_rng(SEED)
    draws = bootstrap_policies(policies, args.bootstrap, rng)

    # -----------------------------------------------------------
    # Strategy table
    # -----------------------------------------------------------
    section(f"Strategy comparison on all {n} questions ({n_hard} hard)")
    print(f"  {'policy':<32} {'to small':>9}   {'accuracy % [95% CI]':<22} {'cost % of always-large [95% CI]':<33} {'avg latency':>11}")
    print("  " + "-" * 112)
    for name in POLICY_ORDER:
        p = points[name]
        a_lo, a_hi = interval(draws[name]["accuracy"])
        c_lo, c_hi = interval(draws[name]["cost_pct"])
        print(f"  {name:<32} {p['share'] * 100:>8.0f}%   "
              f"{p['accuracy']:>5.1f} [{a_lo:>5.1f}, {a_hi:>5.1f}]    "
              f"{p['cost_pct']:>6.1f} [{c_lo:>6.1f}, {c_hi:>6.1f}]      "
              f"{p['latency_ms'] / 1000:>7.2f} s")
    print()
    print("  Latency is the average time of the saved model answers. It leaves out routing and")
    print("  embedding time (a few milliseconds) and network variation.")

    section("Head-to-head comparisons (paired bootstrap, same resampled questions)")
    for a, b in [(ROUTER, RANDOM), (ROUTER, ALWAYS_SMALL), (GUARDED, ROUTER)]:
        acc_draws = draws[a]["accuracy"] - draws[b]["accuracy"]
        cost_draws = draws[a]["cost_pct"] - draws[b]["cost_pct"]
        acc_point = points[a]["accuracy"] - points[b]["accuracy"]
        cost_point = points[a]["cost_pct"] - points[b]["cost_pct"]
        acc_lo, acc_hi = interval(acc_draws)
        cost_lo, cost_hi = interval(cost_draws)
        print(f"  {a}  minus  {b}")
        print(f"      accuracy: {acc_point:+.1f} points   [{acc_lo:+.1f}, {acc_hi:+.1f}]   {verdict(acc_lo, acc_hi)}")
        print(f"      cost    : {cost_point:+.1f} points of always-large cost   [{cost_lo:+.1f}, {cost_hi:+.1f}]   {verdict(cost_lo, cost_hi)}")
        print()

    # -----------------------------------------------------------
    # Does the router have any signal?
    # -----------------------------------------------------------
    section("Does the router have any signal?")
    mean_scores = probabilities.mean(axis=0)
    auc = safe_auc(y, mean_scores)
    auc_lo, auc_hi = bootstrap_auc(y, mean_scores, args.bootstrap, rng)
    per_repeat = [safe_auc(y, probabilities[r]) for r in range(args.repeats)]
    print(f"  AUC over all questions : {show_auc(auc)}   95% interval [{show_auc(auc_lo)}, {show_auc(auc_hi)}]")
    print(f"  AUC in each repeat     : {', '.join(show_auc(v) for v in per_repeat)}")
    print("  (0.50 = no better than guessing, 1.00 = perfect. An interval that includes 0.50")
    print("   means the data cannot rule out that the router has no skill.)")

    print()
    print("  AUC inside each question source (is it only recognising the format?):")
    sources = df["source"].to_numpy()
    source_auc = {}
    for source in sorted(set(sources)):
        mask = sources == source
        hard_here = int((y[mask] == 0).sum())
        value = safe_auc(y[mask], mean_scores[mask])
        source_auc[source] = {"questions": int(mask.sum()), "hard": hard_here, "auc": None if value != value else value}
        print(f"    {source:<8} {int(mask.sum()):>4} questions, {hard_here:>3} hard   AUC {show_auc(value)}")
    print("    A source with only a few hard questions gives an AUC too noisy to rely on.")

    print()
    print("  Hard questions the router sent to the large model, compared with chance:")
    rng_perm = np.random.default_rng(SEED + 1)
    permutation_rows = []
    for repeat in range(args.repeats):
        to_large = ~use_small[repeat]
        k_large = int(to_large.sum())
        observed = int((hard & to_large).sum())
        expected = k_large * n_hard / n
        if k_large == 0:
            p_value = 1.0
        else:
            simulated = rng_perm.hypergeometric(n_hard, n - n_hard, k_large, size=10000)
            p_value = (int((simulated >= observed).sum()) + 1) / 10001
        permutation_rows.append({"repeat": repeat + 1, "sent_to_large": k_large, "hard_caught": observed,
                                 "expected_by_chance": expected, "p_value": p_value})
        print(f"    repeat {repeat + 1}: sent {k_large} questions to large; caught {observed} of {n_hard} hard; "
              f"chance would catch about {expected:.1f}; p = {p_value:.3f}")
    print("    p is the chance of catching at least this many hard questions by routing at random")
    print("    with the same share. Small p (say under 0.05) suggests real signal; large p does not.")

    hard_to_large = float(np.mean([(hard & ~use_small[r]).sum() for r in range(args.repeats)]))
    easy_to_small = float(np.mean([(~hard & use_small[r]).sum() for r in range(args.repeats)]))
    print()
    print(f"  Router decisions on all {n} questions (average over repeats):")
    print(f"    hard -> large model (right choice)   : {hard_to_large:.1f}")
    print(f"    hard -> small model (quality risk)   : {n_hard - hard_to_large:.1f}")
    print(f"    easy -> small model (right choice)   : {easy_to_small:.1f}")
    print(f"    easy -> large model (wasted money)   : {(n - n_hard) - easy_to_small:.1f}")

    unique_thresholds = sorted(set(np.round(thresholds.ravel(), 2).tolist()))
    print()
    print(f"  Thresholds chosen across folds: {unique_thresholds}")
    print("  Very different values from fold to fold mean the threshold is unstable.")

    # -----------------------------------------------------------
    # Guardrail
    # -----------------------------------------------------------
    section("Keyword guardrail")
    guard_count = int(guard_hit.sum())
    print(f"  Questions matching prove / derive / show that / demonstrate that: {guard_count} of {n}")
    if guard_count:
        print(f"  Small-model accuracy on them: {small_c[guard_hit].mean() * 100:.0f}%   "
              f"(hard among them: {int((hard & guard_hit).sum())})")
    print("  The benchmark is multiple choice and word problems, so few questions ask for proofs.")
    print("  The guardrail targets open-ended requests that this benchmark barely contains,")
    print("  so this table cannot show its real value. It only shows it does not hurt here.")

    # -----------------------------------------------------------
    # Save
    # -----------------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in POLICY_ORDER:
        p = points[name]
        a_lo, a_hi = interval(draws[name]["accuracy"])
        c_lo, c_hi = interval(draws[name]["cost_pct"])
        v_lo, v_hi = interval(draws[name]["accuracy_vs_large"])
        rows.append({
            "policy": name,
            "share_sent_to_small": round(p["share"], 4),
            "accuracy_pct": round(p["accuracy"], 2), "accuracy_ci_low": round(a_lo, 2), "accuracy_ci_high": round(a_hi, 2),
            "accuracy_vs_large_pts": round(p["accuracy_vs_large"], 2),
            "accuracy_vs_large_ci_low": round(v_lo, 2), "accuracy_vs_large_ci_high": round(v_hi, 2),
            "cost_pct_of_large": round(p["cost_pct"], 2), "cost_ci_low": round(c_lo, 2), "cost_ci_high": round(c_hi, 2),
            "total_cost_usd": round(p["cost_usd"], 8),
            "avg_latency_ms": round(p["latency_ms"], 1),
        })
    comparison_path = config.RESULTS_DIR / "strategy_comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)

    scores_path = config.RESULTS_DIR / "router_oof_scores.csv"
    pd.DataFrame({
        "id": df["id"], "source": df["source"], "small_ok": y,
        "p_easy_mean": np.round(mean_scores, 4),
    }).to_csv(scores_path, index=False)

    summary = {
        "questions": n, "hard_questions": n_hard, "router_variant": variant,
        "repeats": args.repeats, "folds": args.folds, "tolerance": args.tolerance,
        "router_share_sent_to_small": router_share,
        "auc": None if auc != auc else auc,
        "auc_ci": [None if auc_lo != auc_lo else auc_lo, None if auc_hi != auc_hi else auc_hi],
        "auc_per_repeat": [None if v != v else v for v in per_repeat],
        "auc_by_source": source_auc,
        "hard_caught_vs_chance": permutation_rows,
        "thresholds_chosen": unique_thresholds,
        "guardrail_matches": guard_count,
    }
    summary_path = config.RESULTS_DIR / "strategy_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name in POLICY_ORDER:
        p = points[name]
        a_lo, a_hi = interval(draws[name]["accuracy"])
        c_lo, c_hi = interval(draws[name]["cost_pct"])
        ax.errorbar(
            p["cost_pct"], p["accuracy"],
            xerr=[[max(0.0, p["cost_pct"] - c_lo)], [max(0.0, c_hi - p["cost_pct"])]],
            yerr=[[max(0.0, p["accuracy"] - a_lo)], [max(0.0, a_hi - p["accuracy"])]],
            fmt="o", capsize=3, label=name,
        )
    ax.set_xlabel("Cost (% of always using the large model)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Routing strategies on {n} questions (bars: 95% bootstrap intervals)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    chart_path = config.RESULTS_DIR / "strategy_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print()
    print("Saved:")
    for path in (comparison_path, summary_path, scores_path, chart_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()