"""
show_report.py
Prints a readable summary of the last router training run.
It only reads ml/models/router_report.json. It changes nothing and uses no Groq tokens.

    python show_report.py
"""

import json
import sys

from backend import config


def main():
    path = config.MODELS_DIR / "router_report.json"
    if not path.exists():
        print(f"Could not find {path}")
        print("Run  python ml/train_router.py  first.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as file:
        report = json.load(file)

    print()
    print(f"Trained at          : {report['trained_at']}")
    print(f"Chosen model        : {report['chosen_variant']}")
    print(f"Threshold chosen    : {report['recommended_threshold']}")
    print(f"Questions           : {report['n_train']} train, {report['n_test']} test")
    print(f"Hard questions      : {report['hard_in_train']} in train, {report['hard_in_test']} in test")

    print()
    print("Candidate models (AUC: 0.50 = guessing, 1.00 = perfect)")
    print(f"  {'name':<16} {'cross-validated AUC':<22} test AUC")
    for name, info in report["candidates"].items():
        test_auc = "n/a" if info["test_auc"] is None else f"{info['test_auc']:.2f}"
        cross_validated = f"{info['cv_auc_mean']:.2f} +/- {info['cv_auc_std']:.2f}"
        print(f"  {name:<16} {cross_validated:<22} {test_auc}")

    results = report["test_results"]
    large_cost = results["always_large"]["cost_usd"]
    small = results["always_small"]
    large = results["always_large"]
    router = results["router"]

    share = router["share_sent_to_small"]
    random_accuracy = share * small["accuracy"] + (1 - share) * large["accuracy"]
    random_cost = share * small["cost_usd"] + (1 - share) * large["cost_usd"]

    def line(label, accuracy, cost):
        print(f"  {label:<18} {accuracy * 100:>7.1f}%   ${cost:>10.6f}   {cost / large_cost * 100:>6.1f}%")

    print()
    print("Test-set results")
    print(f"  {'policy':<18} {'accuracy':>8}   {'cost':>11}   {'% of large cost'}")
    line("Always small", small["accuracy"], small["cost_usd"])
    line("Always large", large["accuracy"], large["cost_usd"])
    line("Perfect router", results["perfect_router"]["accuracy"], results["perfect_router"]["cost_usd"])
    line("YOUR ROUTER", router["accuracy"], router["cost_usd"])
    line("Random router*", random_accuracy, random_cost)
    print(f"  * expected result of choosing at random with the same share ({share * 100:.0f}%) sent to small")

    print()
    print(f"Hard questions caught by the router: {router['hard_caught']} of "
          f"{router['hard_caught'] + router['hard_missed']}")


if __name__ == "__main__":
    main()