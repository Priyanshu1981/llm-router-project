"""
test_deployed.py
Checks a running copy of the app (local or public) from the outside, the way a
visitor would: the gate must keep strangers out, the right code must let you in,
and the cache and admin protection must behave.

Usage:
    python test_deployed.py                                   test the local server
    python test_deployed.py --url https://YOUR-SPACE.hf.space test the public one
    python test_deployed.py --no-live                         skip the one live model call

The access code is asked for with hidden input (or pass --code).
The live check makes ONE model call (the second, identical question is a cache hit).
"""

import argparse
import getpass
import sys
import uuid

import httpx


def main():
    parser = argparse.ArgumentParser(description="Check a running copy of the app.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="address of the app")
    parser.add_argument("--code", default=None, help="the access code (asked for if missing)")
    parser.add_argument("--no-live", action="store_true", help="skip the live model call")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    code = args.code or getpass.getpass("Access code: ")
    checks = []

    def record(description, passed, detail=""):
        checks.append((description, passed))
        print(f"  {'PASS' if passed else 'FAIL'}  {description}" + (f"   ({detail})" if detail else ""))

    print(f"Testing {base}")
    print()

    with httpx.Client(timeout=180.0, follow_redirects=False) as client:
        # 1. The public health check works without signing in.
        try:
            response = client.get(f"{base}/healthz")
        except httpx.HTTPError as error:
            print(f"Could not reach {base}: {error}")
            print("If this is a public Space, it may still be waking up. Wait a minute and try again.")
            sys.exit(1)
        record("/healthz answers without signing in", response.status_code == 200)

        # 2. A stranger is kept out.
        response = client.get(f"{base}/")
        looks_like_login = response.status_code == 200 and "access code" in response.text.lower()
        record("the home page shows the sign-in page to strangers", looks_like_login)
        response = client.post(f"{base}/query", json={"query": "hello", "mode": "auto"})
        record("/query refuses strangers (401)", response.status_code == 401, f"got {response.status_code}")
        response = client.get(f"{base}/stats")
        record("/stats refuses strangers (401)", response.status_code == 401, f"got {response.status_code}")

        # 3. A wrong code does not work, the right one does.
        response = client.post(f"{base}/login", json={"code": "definitely-not-the-code"})
        record("a wrong access code is rejected (401)", response.status_code == 401, f"got {response.status_code}")
        response = client.post(f"{base}/login", json={"code": code})
        record("the right access code signs you in (200)", response.status_code == 200, f"got {response.status_code}")
        if response.status_code != 200:
            print()
            print("Cannot continue without signing in. Check the access code and try again.")
            sys.exit(1)

        # 4. Signed in: the app works.
        response = client.get(f"{base}/")
        record("the chat page loads after signing in", response.status_code == 200 and "Ask a question" in response.text)
        health = client.get(f"{base}/health")
        health_ok = health.status_code == 200
        info = health.json() if health_ok else {}
        record("/health works after signing in", health_ok)
        record("the router is loaded", bool(info.get("router_loaded")))
        record("the Groq key is loaded", bool(info.get("groq_key_loaded")))
        limits = info.get("limits", {})
        record("the gate reports itself as ON", bool(limits.get("gate")))
        print(f"        settings: rate limit {limits.get('rate_limit_per_minute')}/min, "
              f"daily model calls {limits.get('model_calls_per_day')}, "
              f"admin actions {'on' if limits.get('admin_actions') else 'off'}")

        # 5. Admin actions are refused when switched off. They are never attempted when switched on.
        if limits.get("admin_actions") is False:
            response = client.delete(f"{base}/cache")
            record("clearing the cache is refused (403)", response.status_code == 403, f"got {response.status_code}")
            response = client.delete(f"{base}/logs")
            record("clearing the log is refused (403)", response.status_code == 403, f"got {response.status_code}")
        else:
            print("  SKIP  admin-action checks (admin actions are ON here, so nothing was deleted)")

        # 6. One live question, asked twice: a miss, then a hit.
        if args.no_live:
            print("  SKIP  live model call (--no-live)")
        else:
            question = f"What is the capital of Portugal? (test {uuid.uuid4().hex[:6]})"
            first = client.post(f"{base}/query", json={"query": question, "mode": "auto"})
            if first.status_code != 200:
                record("live question is answered", False, f"HTTP {first.status_code}: {first.text[:120]}")
            else:
                data1 = first.json()
                record("live question is answered", True, f"{data1['tier']} model, {data1['latency_ms']} ms")
                second = client.post(f"{base}/query", json={"query": question, "mode": "auto"})
                data2 = second.json() if second.status_code == 200 else {}
                record("the same question again is a cache hit", bool(data2.get("cache_hit")))
                record("the cache hit used no tokens and cost nothing",
                       data2.get("input_tokens") == 0 and data2.get("output_tokens") == 0 and data2.get("cost_usd") == 0)
                if data2:
                    print(f"        miss: {data1['latency_ms']} ms   hit: {data2['latency_ms']} ms")

    failed = sum(1 for _, passed in checks if not passed)
    print()
    if failed == 0:
        print("All checks passed.")
    else:
        print(f"{failed} check(s) failed. Read the list above.")
        sys.exit(1)


if __name__ == "__main__":
    main()