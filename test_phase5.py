"""
test_phase5.py
Tests the semantic cache against the running server.

The server must already be running in another terminal:
    uvicorn backend.main:app --reload

Usage:
    python test_phase5.py            run the tests
    python test_phase5.py --clean    run the tests, then clear the cache and the request log

NOTE: this clears the cache at the start, so the results do not depend on earlier requests.
"""

import sys

import httpx

BASE_URL = "http://127.0.0.1:8000"


def money(value):
    return f"${value:.8f}"


def ask(client, query):
    """Send one auto-mode query. Returns the JSON response, or None if it failed."""
    response = client.post(f"{BASE_URL}/query", json={"query": query, "mode": "auto"})
    if response.status_code != 200:
        print(f"  FAILED (HTTP {response.status_code}): {response.text}")
        return None
    return response.json()


def show(label, query, data):
    similarity = "n/a" if data["similarity"] is None else f"{data['similarity']:.3f}"
    reason = data["route_reason"]
    if len(reason) > 150:
        reason = reason[:150] + "..."
    print()
    print(f"{label}")
    print(f"  Question      : {query}")
    print(f"  Cache hit     : {data['cache_hit']}   (tier: {data['tier']}, similarity: {similarity})")
    print(f"  Latency       : {data['latency_ms']} ms")
    print(f"  Tokens used   : {data['input_tokens']} in / {data['output_tokens']} out")
    print(f"  Cost          : {money(data['cost_usd'])}   saved: {money(data['saved_usd'])}")
    print(f"  Reason        : {reason}")
    answer = data["answer"].replace("\n", " ")
    print(f"  Answer        : {answer[:120]}{'...' if len(answer) > 120 else ''}")


def main():
    checks = []  # (description, passed)

    with httpx.Client(timeout=120.0) as client:
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
        if "cache" not in health.json():
            print("The server has no cache. Did you replace backend/main.py in Phase 5 and restart the server?")
            sys.exit(1)

        deleted = client.delete(f"{BASE_URL}/cache").json()["deleted_entries"]
        print(f"Cleared the cache first ({deleted} entries removed).")
        print(f"Cache threshold in use: {health.json()['cache_threshold']}")

        # Test 1: the same question twice
        q1 = "What is the capital of France?"
        first = ask(client, q1)
        second = ask(client, q1)
        if first and second:
            show("Test 1a: first time (expect a cache miss)", q1, first)
            show("Test 1b: exactly the same question again (expect a cache HIT)", q1, second)
            checks.append(("1a is a miss", first["cache_hit"] is False))
            checks.append(("1b is a hit", second["cache_hit"] is True))
            checks.append(("1b costs nothing", second["cost_usd"] == 0))

        # Test 2: a paraphrase (informational: depends on the threshold)
        q2 = "Which city is the capital of France?"
        paraphrase = ask(client, q2)
        if paraphrase:
            show("Test 2: a paraphrase (a hit or a miss both depend on your threshold)", q2, paraphrase)

        # Test 3: same question but a different number. Must NEVER be a hit.
        q3a = "What is 15% of 200?"
        q3b = "What is 15% of 300?"
        number_a = ask(client, q3a)
        number_b = ask(client, q3b)
        if number_a and number_b:
            show("Test 3a: first number question", q3a, number_a)
            show("Test 3b: same question with a different number (must be a MISS)", q3b, number_b)
            checks.append(("3b is a miss (number guard)", number_b["cache_hit"] is False))

        # Totals
        stats = client.get(f"{BASE_URL}/stats").json()
        info = client.get(f"{BASE_URL}/cache").json()["info"]
        print()
        print("=" * 60)
        print("Totals")
        print("=" * 60)
        print(f"  Requests logged  : {stats['total_requests']}")
        print(f"  Cache hits       : {stats['cache_hits']} (hit rate {stats['cache_hit_rate'] * 100:.0f}%)")
        print(f"  Entries in cache : {info['entries']}")
        print(f"  Total cost       : {money(stats['total_cost_usd'])}")
        print(f"  Saved vs large   : {money(stats['total_saved_usd'])} ({stats['savings_percent']}%)")
        print(f"  Average latency  : {stats['avg_latency_ms']} ms")

        if "--clean" in sys.argv:
            client.delete(f"{BASE_URL}/cache")
            client.delete(f"{BASE_URL}/logs")
            print()
            print("Cleaned up: cache and request log cleared.")

    print()
    print("=" * 60)
    print("Checks")
    print("=" * 60)
    failed = 0
    for description, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {description}")
        failed += 0 if passed else 1

    if not checks:
        print("  No checks ran. Read the messages above.")
        sys.exit(1)
    if failed == 0:
        print()
        print("All checks passed. The semantic cache is working.")
    else:
        print()
        print(f"{failed} check(s) failed. Read the messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()