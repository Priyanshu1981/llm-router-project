"""
main.py
The web server. It connects the pieces together.

Mode "auto":
    embed the question once
      -> look in the semantic cache; if a similar question was answered, reuse it (cost $0)
      -> otherwise the router picks the small or large model, the model answers,
         and the answer is stored in the cache for next time
    Questions whose answer changes over time ("what is today's date") skip the cache.
    Identical questions that arrive at the same moment take turns, so only the first
    one calls the model.
Mode "small" / "large":
    use exactly that model. The cache is neither read nor written, because the
    caller asked for a specific model.

The web page (frontend/) is served at "/" and its files at "/static".
The API documentation is at "/docs".

If DEMO_ACCESS_CODE is set, everything except /login and /healthz needs a sign-in.
"""

import time
from contextlib import ExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import cache, config, cost, database, features, llm_client, router, security, singleflight
from backend.schemas import QueryRequest, QueryResponse, StatsResponse

LOGIN_PAGE = config.FRONTEND_DIR / "login.html"

# Added to an answer that the model did not finish (it reached the output limit).
TRUNCATION_NOTE = "\n\n*(This answer was cut off because it reached the length limit.)*"

# Makes identical simultaneous questions take turns.
flights = singleflight.KeyedLock()


class LoginRequest(BaseModel):
    code: str = Field("", max_length=200)


@asynccontextmanager
async def lifespan(app):
    """Runs once when the server starts: prepare the database, cache and router."""
    database.init_db()
    cache.init()
    router.load()
    try:
        # Load the embedding model now, so the first request is not slow.
        features.embed_text("warm up")
    except Exception as error:
        print(f"Warning: could not load the embedding model ({error}). Cache and router will not work.")
    print(f"Access gate: {'ON' if security.gate_enabled() else 'OFF (no DEMO_ACCESS_CODE set)'}")
    yield


class NoCacheStaticFiles(StaticFiles):
    """
    Serves the frontend files and tells the browser not to keep its own copy,
    so edits to index.html, style.css and app.js show up when you reload the page.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(
    title="Cost-Aware LLM Router",
    description="Routes queries between a small and a large model and caches similar answers.",
    version="0.7.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def access_gate(request: Request, call_next):
    """
    Lets a request through if there is no gate, the path is public, or the visitor
    has signed in. Otherwise shows the login page (for the home page) or answers 401.
    """
    if (
        not security.gate_enabled()
        or request.url.path in security.PUBLIC_PATHS
        or security.is_authenticated(request)
    ):
        return await call_next(request)

    if request.method == "GET" and request.url.path == "/":
        return FileResponse(LOGIN_PAGE, headers={"Cache-Control": "no-store"})
    return JSONResponse(
        status_code=401,
        content={"detail": "Access code required. Open the home page and enter it."},
    )


# The files in the frontend folder are available at /static/...
app.mount("/static", NoCacheStaticFiles(directory=config.FRONTEND_DIR), name="static")


def _shorten(text, limit=80):
    """One-line, shortened version of a text, for messages."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


@app.get("/healthz", include_in_schema=False)
def healthz():
    """A bare 'alive' check for the hosting platform. It reveals nothing."""
    return {"status": "ok"}


@app.get("/login", include_in_schema=False)
def login_page(request: Request):
    if security.is_authenticated(request):
        return RedirectResponse("/", status_code=303)
    return FileResponse(LOGIN_PAGE, headers={"Cache-Control": "no-store"})


@app.post("/login", include_in_schema=False)
def login(body: LoginRequest, request: Request):
    if not security.gate_enabled():
        return {"ok": True, "gate": False}
    security.enforce_login_rate(request)
    if not security.code_is_correct(body.code):
        raise HTTPException(status_code=401, detail="That access code is not correct.")

    response = JSONResponse({"ok": True})
    response.set_cookie(
        security.COOKIE_NAME,
        security.make_session_token(),
        max_age=security.SESSION_SECONDS,
        httponly=True,
        samesite="lax",
        secure=security.is_https(request),
        path="/",
    )
    return response


@app.get("/", include_in_schema=False)
def home():
    """The chat page. If the frontend files are missing, show a plain message instead."""
    index_file = config.FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers={"Cache-Control": "no-cache"})
    return {"message": "Cost-Aware LLM Router is running, but frontend/index.html was not found. Visit /docs for the API docs."}


@app.get("/health")
def health():
    """
    Shows which settings the server loaded.
    It never reveals the API key or the access code, only whether they were found.
    """
    key_loaded = bool(config.GROQ_API_KEY) and config.GROQ_API_KEY != "paste_your_key_here"
    return {
        "status": "ok",
        "groq_key_loaded": key_loaded,
        "small_model": config.SMALL_MODEL,
        "large_model": config.LARGE_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "reasoning_effort": config.REASONING_EFFORT,
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "cache_threshold": config.CACHE_THRESHOLD,
        "router_threshold": config.ROUTER_THRESHOLD,
        "router_loaded": router.is_ready(),
        "router_info": router.info(),
        "cache": cache.info(),
        "limits": security.status(),
        "prices_per_million_tokens": {
            "small": {"input": config.SMALL_INPUT_PRICE_PER_M, "output": config.SMALL_OUTPUT_PRICE_PER_M},
            "large": {"input": config.LARGE_INPUT_PRICE_PER_M, "output": config.LARGE_OUTPUT_PRICE_PER_M},
        },
    }


# This is a normal "def" (not "async def") on purpose. Calling Groq waits on the
# network, and FastAPI runs normal "def" endpoints in a separate worker thread,
# so one slow request does not freeze the whole server.
@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest, http_request: Request):
    """Answer one query (from the cache if possible) and log it."""
    security.enforce_query_rate(http_request)
    started = time.perf_counter()

    # Questions whose answer changes over time are never read from or stored in the cache.
    time_sensitive = request.mode == "auto" and cache.is_time_sensitive(request.query)
    use_cache = request.mode == "auto" and not time_sensitive

    with ExitStack() as stack:
        if use_cache:
            # If identical questions arrive at the same moment, the first one calls the
            # model and stores its answer. The others wait here, then find it in the cache.
            stack.enter_context(flights.hold(cache.normalize(request.query)))

        # 1. Embed the question once and look in the cache.
        embedding = None
        lookup = None
        if use_cache:
            try:
                embedding = features.embed_text(request.query)
                lookup = cache.lookup(request.query, embedding)
            except Exception as error:
                # If the cache fails for any reason, carry on without it.
                print(f"Warning: cache lookup skipped ({error})")
                embedding = None
                lookup = None

        similarity = lookup["similarity"] if lookup else None
        cache_hit = bool(lookup and lookup["hit"])

        if cache_hit:
            # 2a. Cache hit: reuse the stored answer. No model is called, so no tokens are used.
            entry = lookup["entry"]
            tier = "cache"
            model_name = f"cache (answer from {entry['model_name']})"
            answer = entry["answer"]
            route_reason = (
                f"semantic cache hit: similarity {similarity:.3f} with an earlier question: "
                f"\"{_shorten(entry['query'])}\""
            )
            input_tokens = 0
            output_tokens = 0
            # For the "always large" baseline we use the token counts of the original answer.
            baseline_input_tokens = entry["input_tokens"]
            baseline_output_tokens = entry["output_tokens"]
        else:
            # 2b. Cache miss (or cache not used): decide which model to use.
            if request.mode == "small":
                tier = "small"
                route_reason = "forced by the caller (mode=small)"
            elif request.mode == "large":
                tier = "large"
                route_reason = "forced by the caller (mode=large)"
            else:
                tier, route_reason, _router_score = router.route(request.query, embedding)
                if time_sensitive:
                    route_reason += " | time-sensitive question: the cache was skipped"
                if lookup and lookup["blocked_by"]:
                    route_reason += f" | a similar cached question was not reused ({lookup['blocked_by']})"

            # 3. Call the model (after checking the daily budget).
            security.reserve_model_call()
            try:
                result = llm_client.generate(tier, request.query)
            except llm_client.LLMError as error:
                raise HTTPException(status_code=error.status_code, detail=error.message)

            answer = result["text"]
            if result["finish_reason"] == "length":
                answer += TRUNCATION_NOTE
            model_name = result["model_name"]
            input_tokens = result["input_tokens"]
            output_tokens = result["output_tokens"]
            baseline_input_tokens = input_tokens
            baseline_output_tokens = output_tokens

            # 4. Store the answer for next time (only complete answers, and only if the cache is used).
            if embedding is not None and result["finish_reason"] == "stop":
                try:
                    cache.add(request.query, embedding, answer, tier, model_name, input_tokens, output_tokens)
                except Exception as error:
                    print(f"Warning: could not store the answer in the cache ({error})")

    # 5. Time for the whole request (waiting, embedding, lookup, routing and model call).
    latency_ms = (time.perf_counter() - started) * 1000.0

    # 6. Cost, and what the "always large" baseline would have cost.
    cost_usd = cost.calculate_cost(tier, input_tokens, output_tokens)
    baseline_cost_usd = cost.calculate_baseline_cost(baseline_input_tokens, baseline_output_tokens)
    saved_usd = baseline_cost_usd - cost_usd

    # 7. Save everything to the request log.
    request_id = database.log_request(
        {
            "query": request.query,
            "mode_requested": request.mode,
            "tier": tier,
            "model_name": model_name,
            "route_reason": route_reason,
            "cache_hit": cache_hit,
            "similarity": similarity,
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "baseline_cost_usd": baseline_cost_usd,
            "saved_usd": saved_usd,
        }
    )

    # 8. Send the response back to the client.
    return QueryResponse(
        request_id=request_id,
        answer=answer,
        tier=tier,
        model_name=model_name,
        mode_requested=request.mode,
        route_reason=route_reason,
        cache_hit=cache_hit,
        similarity=round(similarity, 4) if similarity is not None else None,
        latency_ms=round(latency_ms, 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost_usd, 10),
        baseline_cost_usd=round(baseline_cost_usd, 10),
        saved_usd=round(saved_usd, 10),
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    """Running totals across all logged requests (used by the dashboard)."""
    return database.get_summary()


@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)):
    """The most recent requests, newest first."""
    return database.get_recent(limit)


@app.delete("/logs")
def clear_logs():
    """Delete all logged requests. Can be switched off with ALLOW_ADMIN_ACTIONS=false."""
    security.require_admin()
    deleted = database.reset_logs()
    return {"deleted_rows": deleted}


@app.get("/cache")
def cache_contents(limit: int = Query(20, ge=1, le=200)):
    """Cache status and the most recently stored entries."""
    return {"info": cache.info(), "recent_entries": cache.recent(limit)}


@app.delete("/cache")
def clear_cache():
    """Delete every cached answer. Can be switched off with ALLOW_ADMIN_ACTIONS=false."""
    security.require_admin()
    deleted = cache.clear()
    return {"deleted_entries": deleted}