"""
test_phase6.py
Checks that the server delivers the web page and the data the dashboard needs.
Uses no Groq tokens.

The server must already be running in another terminal:
    uvicorn backend.main:app --reload

    python test_phase6.py

This does not test the page's appearance. Open http://127.0.0.1:8000/ for that.
"""

import sys

import httpx

BASE_URL = "http://127.0.0.1:8000"

STATS_FIELDS = {
    "total_requests", "cache_hits", "cache_hit_rate", "small_calls", "large_calls",
    "total_input_tokens", "total_output_tokens", "total_cost_usd",
    "total_baseline_cost_usd", "total_saved_usd", "savings_percent", "avg_latency_ms",
}
HISTORY_FIELDS = {"id", "tier", "cost_usd", "baseline_cost_usd"}


def main():
    checks = []

    with httpx.Client(timeout=30.0) as client:
        try:
            client.get(f"{BASE_URL}/health").raise_for_status()
        except httpx.HTTPError:
            print("Could not reach the server at", BASE_URL)
            print("Start it in another terminal with:  uvicorn backend.main:app --reload")
            sys.exit(1)

        page = client.get(f"{BASE_URL}/")
        content_type = page.headers.get("content-type", "")
        checks.append(("home page returns HTML", page.status_code == 200 and "text/html" in content_type))
        checks.append(("home page contains the app title", "Cost-Aware LLM Router" in page.text))

        for path in ("/static/style.css", "/static/app.js"):
            response = client.get(f"{BASE_URL}{path}")
            checks.append((f"{path} is served", response.status_code == 200 and len(response.text) > 500))

        stats = client.get(f"{BASE_URL}/stats")
        stats_ok = stats.status_code == 200 and STATS_FIELDS <= set(stats.json())
        checks.append(("/stats has every field the dashboard reads", stats_ok))

        history = client.get(f"{BASE_URL}/history?limit=5")
        history_ok = history.status_code == 200 and isinstance(history.json(), list)
        if history_ok and history.json():
            history_ok = HISTORY_FIELDS <= set(history.json()[0])
        checks.append(("/history has the fields the cost chart reads", history_ok))

        health = client.get(f"{BASE_URL}/health").json()
        checks.append(("/health reports the router and cache", "router_loaded" in health and "cache" in health))

    print()
    print("Checks")
    print("=" * 60)
    failed = 0
    for description, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {description}")
        failed += 0 if passed else 1

    print()
    if failed == 0:
        print("All checks passed. Now open http://127.0.0.1:8000/ in your browser.")
    else:
        print(f"{failed} check(s) failed. Read the list above.")
        sys.exit(1)


if __name__ == "__main__":
    main()