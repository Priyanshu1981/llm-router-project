"""
train_router.py
Trains the router: a classifier that predicts, for a question, the probability
that the SMALL model's answer is good enough (label small_ok = 1).

Run from the project root, after label_data.py:
    python ml/train_router.py
    python ml/train_router.py --tolerance 0.03     allow a 3-point accuracy drop

What it does:
  1. Loads data/processed/labeled.csv and its train/test split.
  2. Builds features for every question.
  3. Compares four candidate models with cross-validation on the TRAIN rows only.
  4. Picks the routing threshold on the train rows (cross-validated predictions).
  5. Evaluates on the TEST rows, which were never used for any choice.
  6. Saves the model, a JSON report, a threshold-sweep CSV and a chart.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Make "from backend import ..." work when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw charts into files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from backend import config, features  # noqa: E402

# Hide routine scikit-learn notices so the output stays readable.
warnings.filterwarnings("ignore", module="sklearn")

SEED = 42
THRESHOLD_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
MIN_EXAMPLES_PER_CLASS_IN_TRAIN = 8

VARIANT_DESCRIPTIONS = {
    "handcrafted_lr": "hand-made features only, logistic regression",
    "embedding_lr": "sentence embedding only, logistic regression",
    "combined_lr": "hand-made + embedding, logistic regression",
    "combined_rf": "hand-made + embedding, random forest",
}


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------
# Loading
# ---------------------------------------------------------------
def load_labeled():
    path = config.PROCESSED_DIR / "labeled.csv"
    if not path.exists():
        print(f"Could not find {path}. Run  python ml/label_data.py  first.")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig")
    needed = ["id", "query", "small_ok", "small_correct", "large_correct",
              "small_cost_usd", "large_cost_usd", "split"]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        print(f"labeled.csv is missing these columns: {missing}")
        print("Run  python ml/label_data.py  again.")
        sys.exit(1)
    return df


def to_arrays(df):
    """The columns we need for scoring a routing policy, as plain numpy arrays."""
    return {
        "y": df["small_ok"].to_numpy(dtype=int),
        "small_correct": df["small_correct"].to_numpy(dtype=float),
        "large_correct": df["large_correct"].to_numpy(dtype=float),
        "small_cost": df["small_cost_usd"].to_numpy(dtype=float),
        "large_cost": df["large_cost_usd"].to_numpy(dtype=float),
    }


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
def make_pipeline(variant, n_hand, n_emb):
    """Build one of the four candidate models (a preprocessing step plus a classifier)."""
    hand_columns = list(range(n_hand))
    embedding_columns = list(range(n_hand, n_hand + n_emb))
    all_columns = hand_columns + embedding_columns

    def logistic():
        # LogisticRegressionCV picks its own regularisation strength with an inner cross-validation.
        return LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
            cv=3,
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )

    if variant == "handcrafted_lr":
        prep = ColumnTransformer([("scale", StandardScaler(), hand_columns)], remainder="drop")
        return Pipeline([("prep", prep), ("clf", logistic())])
    if variant == "embedding_lr":
        prep = ColumnTransformer([("scale", StandardScaler(), embedding_columns)], remainder="drop")
        return Pipeline([("prep", prep), ("clf", logistic())])
    if variant == "combined_lr":
        prep = ColumnTransformer([("scale", StandardScaler(), all_columns)], remainder="drop")
        return Pipeline([("prep", prep), ("clf", logistic())])
    if variant == "combined_rf":
        prep = ColumnTransformer([("keep", "passthrough", all_columns)], remainder="drop")
        forest = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=1,
        )
        return Pipeline([("prep", prep), ("clf", forest)])
    raise ValueError(f"Unknown variant: {variant}")


def safe_auc(y_true, scores):
    """ROC AUC, or NaN if only one class is present."""
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def show(value):
    return "n/a" if value != value else f"{value:.2f}"


# ---------------------------------------------------------------
# Scoring routing policies
# ---------------------------------------------------------------
def evaluate_policy(use_small, d):
    """
    use_small is True/False for each question (True = send to the small model).
    Returns (accuracy, total cost) using the REAL saved answers and costs.
    """
    correct = np.where(use_small, d["small_correct"], d["large_correct"])
    total_cost = np.where(use_small, d["small_cost"], d["large_cost"])
    return float(correct.mean()), float(total_cost.sum())


def sweep(p_easy, d, thresholds):
    """Score the routing policy at every threshold."""
    large_total = float(d["large_cost"].sum())
    rows = []
    for threshold in thresholds:
        use_small = p_easy >= threshold
        accuracy, total_cost = evaluate_policy(use_small, d)
        rows.append(
            {
                "threshold": float(threshold),
                "small_share": float(use_small.mean()),
                "accuracy": accuracy,
                "cost": total_cost,
                "cost_pct_of_large": (total_cost / large_total * 100.0) if large_total > 0 else 0.0,
            }
        )
    return rows


def find_row(rows, threshold):
    return next(row for row in rows if abs(row["threshold"] - threshold) < 1e-9)


def choose_threshold(p_oof, d, tolerance):
    """
    Pick the cheapest threshold whose accuracy stays within `tolerance` of always
    using the large model. Uses cross-validated predictions on the TRAIN rows only.
    Returns (threshold, found_one).
    """
    large_accuracy = float(d["large_correct"].mean())
    rows = sweep(p_oof, d, THRESHOLD_GRID)

    acceptable = [row for row in rows if row["accuracy"] >= large_accuracy - tolerance]
    if acceptable:
        cheapest = min(row["cost"] for row in acceptable)
        ties = [row for row in acceptable if row["cost"] <= cheapest + 1e-12]
        # If several thresholds give the same cost, take the most cautious (highest) one.
        return max(row["threshold"] for row in ties), True

    best_accuracy = max(row["accuracy"] for row in rows)
    top = [row for row in rows if row["accuracy"] >= best_accuracy - 1e-12]
    cheapest_top = min(top, key=lambda row: row["cost"])
    return cheapest_top["threshold"], False


# ---------------------------------------------------------------
# Chart
# ---------------------------------------------------------------
def save_chart(test_rows, chosen_row, references, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))

    xs = [row["cost_pct_of_large"] for row in test_rows]
    ys = [row["accuracy"] * 100 for row in test_rows]
    ax.plot(xs, ys, "-", color="tab:blue", alpha=0.7, label="Router (threshold sweep)")
    ax.scatter(
        [chosen_row["cost_pct_of_large"]], [chosen_row["accuracy"] * 100],
        marker="*", s=220, color="tab:red", zorder=5,
        label=f"Chosen threshold ({chosen_row['threshold']:.2f})",
    )

    markers = {"Always small": "s", "Always large": "o", "Perfect router": "D"}
    for name, (cost_pct, accuracy) in references.items():
        ax.scatter([cost_pct], [accuracy * 100], marker=markers[name], s=70, label=name, zorder=4)

    ax.set_xlabel("Cost (% of always using the large model)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Router: accuracy vs cost on the test questions")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the router.")
    parser.add_argument(
        "--tolerance", type=float, default=0.02,
        help="how much accuracy (as a fraction, 0.02 = 2 points) the router may lose "
             "compared with always using the large model (default: 0.02)",
    )
    args = parser.parse_args()

    df = load_labeled()
    train_mask = (df["split"] == "train").to_numpy()
    test_mask = ~train_mask
    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    y_train = train_df["small_ok"].to_numpy(dtype=int)
    y_test = test_df["small_ok"].to_numpy(dtype=int)

    hard_train = int((y_train == 0).sum())
    easy_train = int((y_train == 1).sum())

    section("Data")
    print(f"  Train: {len(train_df)} questions ({easy_train} easy, {hard_train} hard)")
    print(f"  Test : {len(test_df)} questions ({int((y_test == 1).sum())} easy, {int((y_test == 0).sum())} hard)")

    if hard_train < MIN_EXAMPLES_PER_CLASS_IN_TRAIN or easy_train < MIN_EXAMPLES_PER_CLASS_IN_TRAIN:
        print()
        print(f"Need at least {MIN_EXAMPLES_PER_CLASS_IN_TRAIN} easy and {MIN_EXAMPLES_PER_CLASS_IN_TRAIN} hard "
              f"questions in the train split to train a meaningful router.")
        print("Tell me these numbers and we will add more questions first.")
        sys.exit(1)

    print()
    print("Computing features (this loads the embedding model, so the first run can take a minute) ...")
    X_all = features.build_features(df["query"].tolist())
    X_train = X_all[train_mask]
    X_test = X_all[test_mask]
    n_hand = len(features.FEATURE_NAMES)
    n_emb = X_all.shape[1] - n_hand
    print(f"  {n_hand} hand-made features + {n_emb} embedding numbers = {X_all.shape[1]} per question")

    d_train = to_arrays(train_df)
    d_test = to_arrays(test_df)

    # -----------------------------------------------------------
    # Compare the four candidate models
    # -----------------------------------------------------------
    section("Comparing candidate models")
    folds = min(5, hard_train, easy_train)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    print(f"  Cross-validation: {folds} folds on the train questions only.")
    print("  AUC = how well the model ranks easy questions above hard ones")
    print("        (1.00 perfect, 0.50 no better than guessing).")
    print()
    print(f"  {'variant':<16} {'CV AUC (train)':<18} {'Test AUC':<9} description")

    results = {}
    for variant, description in VARIANT_DESCRIPTIONS.items():
        pipeline = make_pipeline(variant, n_hand, n_emb)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        pipeline.fit(X_train, y_train)
        classes = list(pipeline.classes_)
        test_scores = pipeline.predict_proba(X_test)[:, classes.index(1)]
        test_auc = safe_auc(y_test, test_scores)
        results[variant] = {
            "pipeline": pipeline,
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "test_auc": test_auc,
            "test_scores": test_scores,
        }
        print(f"  {variant:<16} {scores.mean():.2f} +/- {scores.std():<9.2f} {show(test_auc):<9} {description}")

    best_variant = max(results, key=lambda name: results[name]["cv_mean"])
    best = results[best_variant]
    print()
    print(f"  Chosen model: {best_variant}  (highest cross-validated AUC; the test set was NOT used to choose)")

    # -----------------------------------------------------------
    # Choose the threshold (train rows only)
    # -----------------------------------------------------------
    section("Choosing the routing threshold (train questions, cross-validated)")
    oof_pipeline = make_pipeline(best_variant, n_hand, n_emb)
    p_oof = cross_val_predict(oof_pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    chosen_threshold, found = choose_threshold(p_oof, d_train, args.tolerance)
    oof_rows = sweep(p_oof, d_train, THRESHOLD_GRID)
    oof_row = find_row(oof_rows, chosen_threshold)
    train_large_accuracy = float(d_train["large_correct"].mean())
    print(f"  Rule: the cheapest threshold whose accuracy is within {args.tolerance * 100:.0f} points "
          f"of always-large ({train_large_accuracy * 100:.1f}% on train).")
    if found:
        print(f"  Chosen threshold: {chosen_threshold:.2f}")
    else:
        print(f"  No threshold met the rule, so the most accurate cheapest one was used: {chosen_threshold:.2f}")
    print(f"  On the train questions this gives accuracy {oof_row['accuracy'] * 100:.1f}% "
          f"at {oof_row['cost_pct_of_large']:.1f}% of the large model's cost, "
          f"sending {oof_row['small_share'] * 100:.0f}% of questions to the small model.")
    if chosen_threshold <= 0.01:
        print("  Note: a threshold of 0.00 means 'always small'. The small model is nearly as good as the large one here.")
    if chosen_threshold >= 0.99:
        print("  Note: a threshold near 1.00 means 'always large'. The router could not find a safe way to save cost.")

    # -----------------------------------------------------------
    # Evaluate on the test rows
    # -----------------------------------------------------------
    section("Final evaluation on the TEST questions (never used for any choice)")
    p_test = best["test_scores"]
    n_test = len(test_df)

    small_acc, small_cost = evaluate_policy(np.ones(n_test, dtype=bool), d_test)
    large_acc, large_cost = evaluate_policy(np.zeros(n_test, dtype=bool), d_test)
    perfect_acc, perfect_cost = evaluate_policy(d_test["y"] == 1, d_test)
    router_use_small = p_test >= chosen_threshold
    router_acc, router_cost = evaluate_policy(router_use_small, d_test)
    small_share = float(router_use_small.mean())

    def pct_of_large(total_cost):
        return (total_cost / large_cost * 100.0) if large_cost > 0 else 0.0

    print(f"  {'policy':<18} {'accuracy':>9}   {'cost':>12}   {'% of always-large':>18}")
    for name, accuracy, total_cost in [
        ("Always small", small_acc, small_cost),
        ("Always large", large_acc, large_cost),
        ("Perfect router", perfect_acc, perfect_cost),
        ("YOUR ROUTER", router_acc, router_cost),
    ]:
        print(f"  {name:<18} {accuracy * 100:>8.1f}%   ${total_cost:>11.6f}   {pct_of_large(total_cost):>17.1f}%")

    random_acc = small_share * small_acc + (1 - small_share) * large_acc
    random_cost = small_share * small_cost + (1 - small_share) * large_cost
    print(f"  {'Random router*':<18} {random_acc * 100:>8.1f}%   ${random_cost:>11.6f}   {pct_of_large(random_cost):>17.1f}%")
    print(f"  * expected result of choosing randomly with the same share ({small_share * 100:.0f}%) sent to small")

    is_hard = d_test["y"] == 0
    hard_to_large = int((is_hard & ~router_use_small).sum())
    hard_to_small = int((is_hard & router_use_small).sum())
    easy_to_small = int((~is_hard & router_use_small).sum())
    easy_to_large = int((~is_hard & ~router_use_small).sum())
    sent_small = int(router_use_small.sum())

    print()
    print(f"  Router decisions on the {n_test} test questions (threshold {chosen_threshold:.2f}):")
    print(f"    hard question -> large model (correct choice) : {hard_to_large}")
    print(f"    hard question -> small model (quality risk)   : {hard_to_small}")
    print(f"    easy question -> small model (correct choice) : {easy_to_small}")
    print(f"    easy question -> large model (wasted money)   : {easy_to_large}")
    if hard_to_large + hard_to_small:
        print(f"    share of hard questions caught: {hard_to_large / (hard_to_large + hard_to_small) * 100:.0f}%")
    if sent_small:
        print(f"    share of 'small' decisions that were truly easy: {easy_to_small / sent_small * 100:.0f}%")
    print(f"    Test AUC of the chosen model: {show(best['test_auc'])}")

    # Threshold sweep on the test rows
    test_rows = sweep(p_test, d_test, THRESHOLD_GRID)
    print()
    print("  Threshold sweep on the test questions:")
    print(f"  {'threshold':>9}  {'sent to small':>13}  {'accuracy':>9}  {'% of large cost':>15}")
    shown = sorted(set([0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, round(chosen_threshold, 2)]))
    for threshold in shown:
        row = find_row(test_rows, threshold)
        marker = "  <- chosen" if abs(threshold - chosen_threshold) < 1e-9 else ""
        print(f"  {threshold:>9.2f}  {row['small_share'] * 100:>12.0f}%  {row['accuracy'] * 100:>8.1f}%  "
              f"{row['cost_pct_of_large']:>14.1f}%{marker}")

    print()
    print(f"  Remember: the test split has only {int(is_hard.sum())} hard questions, so each one moves these "
          f"numbers a lot. Report them honestly, with that caveat.")

    # -----------------------------------------------------------
    # Save everything
    # -----------------------------------------------------------
    section("Saving")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": best["pipeline"],
        "variant": best_variant,
        "feature_names": features.FEATURE_NAMES,
        "n_handcrafted": n_hand,
        "embedding_model": config.EMBEDDING_MODEL,
        "recommended_threshold": float(chosen_threshold),
        "n_train": int(len(train_df)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    model_path = config.MODELS_DIR / "router.joblib"
    joblib.dump(bundle, model_path)
    print(f"  Model        : {model_path}")

    report = {
        "trained_at": bundle["trained_at"],
        "chosen_variant": best_variant,
        "tolerance": args.tolerance,
        "recommended_threshold": float(chosen_threshold),
        "n_train": int(len(train_df)),
        "n_test": int(n_test),
        "hard_in_train": hard_train,
        "hard_in_test": int(is_hard.sum()),
        "candidates": {
            name: {
                "cv_auc_mean": round(info["cv_mean"], 4),
                "cv_auc_std": round(info["cv_std"], 4),
                "test_auc": None if info["test_auc"] != info["test_auc"] else round(info["test_auc"], 4),
            }
            for name, info in results.items()
        },
        "test_results": {
            "always_small": {"accuracy": small_acc, "cost_usd": small_cost},
            "always_large": {"accuracy": large_acc, "cost_usd": large_cost},
            "perfect_router": {"accuracy": perfect_acc, "cost_usd": perfect_cost},
            "router": {
                "accuracy": router_acc,
                "cost_usd": router_cost,
                "share_sent_to_small": small_share,
                "hard_caught": hard_to_large,
                "hard_missed": hard_to_small,
            },
        },
    }
    report_path = config.MODELS_DIR / "router_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"  Report       : {report_path}")

    sweep_path = config.RESULTS_DIR / "router_threshold_sweep.csv"
    pd.DataFrame(test_rows).to_csv(sweep_path, index=False)
    print(f"  Sweep (CSV)  : {sweep_path}")

    chart_path = config.RESULTS_DIR / "router_threshold_curve.png"
    references = {
        "Always small": (pct_of_large(small_cost), small_acc),
        "Always large": (pct_of_large(large_cost), large_acc),
        "Perfect router": (pct_of_large(perfect_cost), perfect_acc),
    }
    save_chart(test_rows, find_row(test_rows, chosen_threshold), references, chart_path)
    print(f"  Chart        : {chart_path}")

    section("Next step")
    print("  1. Open the .env file and set this line (replace the old ROUTER_THRESHOLD value):")
    print()
    print(f"       ROUTER_THRESHOLD={chosen_threshold:.2f}")
    print()
    print("  2. Restart the server so it loads the trained router and the new threshold.")


if __name__ == "__main__":
    main()