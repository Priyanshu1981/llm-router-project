"""
security.py
Protection for a public deployment.

  1. Access code: visitors sign in once with a code. A signed session cookie then
     proves they did. (If DEMO_ACCESS_CODE is empty, there is no gate.)
  2. Rate limit: a limit on /query requests per minute from one visitor.
  3. Daily budget: a limit on calls to the Groq models per day across ALL visitors.
     Cache hits cost nothing, so they do not count.
  4. Admin actions: clearing the cache or the request log can be switched off.

All counters live in memory, so they reset when the server restarts.
This is enough for a demo. It is not a full security system.
"""

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import HTTPException, Request

from backend import config

COOKIE_NAME = "router_demo_session"
SESSION_SECONDS = 12 * 60 * 60  # a sign-in lasts 12 hours

# These paths work without signing in. "/healthz" lets the hosting platform
# check that the server is alive, and "/login" is where visitors sign in.
PUBLIC_PATHS = {"/healthz", "/login"}


# ---------------------------------------------------------------
# Access code
# ---------------------------------------------------------------
def gate_enabled():
    return bool(config.DEMO_ACCESS_CODE)


def _session_token():
    """A value derived from the access code. Whoever knows the code can sign in."""
    key = config.DEMO_ACCESS_CODE.encode("utf-8")
    return hmac.new(key, b"llm-router-demo-session-v1", hashlib.sha256).hexdigest()


def make_session_token():
    return _session_token()


def code_is_correct(candidate):
    """Compare codes in constant time, so response timing reveals nothing."""
    expected = config.DEMO_ACCESS_CODE.encode("utf-8")
    return hmac.compare_digest((candidate or "").encode("utf-8"), expected)


def is_authenticated(request):
    """True if there is no gate, or the visitor carries a valid session cookie."""
    if not gate_enabled():
        return True
    cookie = request.cookies.get(COOKIE_NAME, "")
    return hmac.compare_digest(cookie.encode("utf-8"), _session_token().encode("utf-8"))


def client_ip(request):
    return request.client.host if request.client else "unknown"


def is_https(request):
    return request.url.scheme == "https"


# ---------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------
class SlidingWindowLimiter:
    """Allows at most `limit` events per `window_seconds` for each key (for example an IP address)."""

    def __init__(self, limit, window_seconds):
        self.limit = limit
        self.window = window_seconds
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key):
        """Returns 0 if the event is allowed (and counts it), otherwise the seconds to wait."""
        if self.limit <= 0:
            return 0
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return int(self.window - (now - hits[0])) + 1
            hits.append(now)
            if len(self._hits) > 5000:  # forget visitors with no recent activity
                for stale in [k for k, v in self._hits.items() if not v]:
                    del self._hits[stale]
            return 0


class DailyBudget:
    """Counts model calls per UTC day and refuses new ones after the limit."""

    def __init__(self, limit):
        self.limit = limit
        self._day = None
        self._used = 0
        self._lock = threading.Lock()

    def _roll_over(self):
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._day = today
            self._used = 0

    def take(self):
        with self._lock:
            self._roll_over()
            if self.limit > 0 and self._used >= self.limit:
                return False
            self._used += 1
            return True

    def used(self):
        with self._lock:
            self._roll_over()
            return self._used


query_limiter = SlidingWindowLimiter(config.RATE_LIMIT_PER_MINUTE, 60)
login_limiter = SlidingWindowLimiter(10, 60)  # 10 sign-in attempts per minute per visitor
daily_budget = DailyBudget(config.MODEL_CALLS_PER_DAY)


def enforce_query_rate(request):
    wait = query_limiter.check(client_ip(request))
    if wait:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait {wait} seconds and try again.",
            headers={"Retry-After": str(wait)},
        )


def enforce_login_rate(request):
    wait = login_limiter.check(client_ip(request))
    if wait:
        raise HTTPException(
            status_code=429,
            detail=f"Too many sign-in attempts. Please wait {wait} seconds.",
            headers={"Retry-After": str(wait)},
        )


def reserve_model_call():
    """Called just before a model is called. Refuses once the daily budget is used up."""
    if not daily_budget.take():
        raise HTTPException(
            status_code=429,
            detail=(
                "This demo has reached its daily limit of model calls. "
                "Answers already in the cache still work. Please try again tomorrow."
            ),
        )


def require_admin():
    if not config.ALLOW_ADMIN_ACTIONS:
        raise HTTPException(status_code=403, detail="This action is disabled on this deployment.")


def status():
    """A summary of the protection settings, for the /health endpoint."""
    return {
        "gate": gate_enabled(),
        "admin_actions": config.ALLOW_ADMIN_ACTIONS,
        "rate_limit_per_minute": config.RATE_LIMIT_PER_MINUTE,
        "model_calls_per_day": config.MODEL_CALLS_PER_DAY,
        "model_calls_today": daily_budget.used(),
    }