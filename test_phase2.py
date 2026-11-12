"""
test_phase2.py
Sends three test queries to the running server and prints the results.

Usage (the server must already be running in another terminal):

    python test_phase2.py            run the tests and keep the logged rows
    python test_phase2.py --clean    run the tests, then delete the logged rows
"""

import sys

import httpx

BASE_URL = "http://127.0.0.1:8000"

# (mode, query) pairs. One for each mode.
TEST_QUERIES = [
    ("small", "What is the capital of France?"),
    ("large", "Explain in two sentences why the sky is blue."),
    ("auto", "What is 17 multiplied by 23?"),
]


def money(value):
    return f"${value:.8f}"


def print_result(data):
    answer = data["answer"].replace("\n", " ")
    if len(answer) > 160:
        answer = answer[:160] + "..."
    print(f"  Model used    : {data['model_name']} (tier: {data['tier']})")
    print(f"  Route reason  : {data['route_reason']}")
    print(f"  Latency       : {data['latency_ms']} ms")
    print(f"  Tokens        : {data['input_tokens']} in / {data['output_tokens']} out")
    print(f"  Cost          : {money(data['cost_usd'])}")
    print(f"  Baseline cost : {money(data['baseline_cost_usd'])} (if the large model answered)")
    print(f"  Saved         : {money(data['saved_usd'])}")
    print(f"  Answer        : {answer}")


def main():
    failures = 0

    with httpx.Client(timeout=120.0) as client:
        # 1. Is the server running?
        try:
            health = client.get(f"{BASE_URL}/health")
            health.raise_for_status()
        except httpx.HTTPError:
            print("Could not reach the server at", BASE_URL)
            print("Start it in another terminal with:  uvicorn backend.main:app --reload")
            sys.exit(1)

        if not health.json().get("groq_key_loaded"):
            print("The server says no Groq key is loaded. Check your .env file and restart the server.")
            sys.exit(1)

        # 2. Send each test query.
        for number, (mode, query) in enumerate(TEST_QUERIES, start=1):
            print()
            print(f"Test {number}: mode={mode}")
            print(f"  Query         : {query}")
            try:
                response = client.post(f"{BASE_URL}/query", json={"query": query, "mode": mode})
            except httpx.HTTPError as error:
                failures += 1
                print(f"  FAILED: {error}")
                continue

            if response.status_code != 200:
                failures += 1
                print(f"  FAILED (HTTP {response.status_code}): {response.text}")
                continue

            print_result(response.json())

        # 3. Show the running totals.
        stats = client.get(f"{BASE_URL}/stats").json()
        print()
        print("=" * 60)
        print("Totals from /stats")
        print("=" * 60)
        print(f"  Requests logged   : {stats['total_requests']}")
        print(f"  Small / large     : {stats['small_calls']} / {stats['large_calls']}")
        print(f"  Total cost        : {money(stats['total_cost_usd'])}")
        print(f"  Baseline cost     : {money(stats['total_baseline_cost_usd'])}")
        print(f"  Saved             : {money(stats['total_saved_usd'])} ({stats['savings_percent']}%)")
        print(f"  Average latency   : {stats['avg_latency_ms']} ms")

        # 4. Optionally delete the test rows.
        if "--clean" in sys.argv:
            deleted = client.delete(f"{BASE_URL}/logs").json()["deleted_rows"]
            print()
            print(f"Cleaned up: deleted {deleted} logged rows.")

    print()
    if failures == 0:
        print("All tests passed. Phase 2 is working.")
    else:
        print(f"{failures} test(s) failed. Read the messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()