"""
diagnose_router.py
Checks whether the router separates easy from hard questions WITHIN each source
(GSM8K, MMLU), or mostly just recognises the format of the question.
Uses only the held-out test questions. No Groq tokens needed.

Run from the project root:
    python ml/diagnose_router.py
"""

import sys
from pathlib import Path

# Make "from backend import ..." work when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from backend import config, features  # noqa: E402


def main():
    labeled_path = config.PROCESSED_DIR / "labeled.csv"
    model_path = config.MODELS_DIR / "router.joblib"
    if not labeled_path.exists() or not model_path.exists():
        print("Need data/processed/labeled.csv and ml/models/router.joblib.")
        print("Run label_data.py and train_router.py first.")
        sys.exit(1)

    df = pd.read_csv(labeled_path, encoding="utf-8-sig")
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]

    print("Computing router scores ...")
    X = features.build_features(df["query"].tolist())
    df["p_easy"] = pipeline.predict_proba(X)[:, list(pipeline.classes_).index(1)]

    test = df[df["split"] == "test"]
    threshold = config.ROUTER_THRESHOLD

    print()
    print(f"Test questions: {len(test)}   threshold in .env: {threshold:.2f}")
    print()
    print(f"{'source':<8} {'questions':>9} {'hard':>5} {'mean P(easy)':>13} {'mean P(easy)':>13} "
          f"{'within-source':>14} {'sent to':>8}")
    print(f"{'':<8} {'':>9} {'':>5} {'(easy ones)':>13} {'(hard ones)':>13} "
          f"{'AUC':>14} {'small':>8}")
    print("-" * 80)

    for source, part in test.groupby("source"):
        easy = part[part["small_ok"] == 1]
        hard = part[part["small_ok"] == 0]

        easy_mean = f"{easy['p_easy'].mean():.2f}" if len(easy) else "n/a"
        hard_mean = f"{hard['p_easy'].mean():.2f}" if len(hard) else "n/a"
        if len(easy) and len(hard):
            auc = f"{roc_auc_score(part['small_ok'], part['p_easy']):.2f}"
        else:
            auc = "n/a"
        share_small = f"{(part['p_easy'] >= threshold).mean() * 100:.0f}%"

        print(f"{source:<8} {len(part):>9} {len(hard):>5} {easy_mean:>13} {hard_mean:>13} {auc:>14} {share_small:>8}")

    print()
    print("How to read this:")
    print("  - Within-source AUC near 0.50 means that inside one format the router cannot")
    print("    tell hard from easy. It is mostly recognising the format.")
    print("  - Above about 0.70 means it does pick up some real difficulty signal.")
    print("  - Hard questions are rare per source on the test split, so these numbers are")
    print("    noisy. Report them with that caveat.")


if __name__ == "__main__":
    main()