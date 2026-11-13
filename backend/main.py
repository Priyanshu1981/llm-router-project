"""
main.py
The web server. It connects the pieces together:

    request -> choose a model -> call the model -> compute cost -> log it -> respond

Phase 4 version: the ML router chooses the model in "auto" mode.
Still to come: the semantic cache (Phase 5).
  mode "small" / "large"  -> uses exactly that model
  mode "auto"             -> the trained router decides
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from backend import config, cost, database, llm_client, router
from backend.schemas import QueryRequest, QueryResponse, StatsResponse


@asynccontextmanager
async def lifespan(app):
    """Runs once when the server starts: prepare the database and load the router."""
    database.init_db()
    router.load()
    yield


app = FastAPI(
    title="Cost-Aware LLM Router",
    description="Routes queries between a small and a large model and caches similar answers.",
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Simple welcome message to confirm the server is up."""
    return {"message": "Cost-Aware LLM Router is running. Visit /docs for the API docs."}


@app.get("/health")
def health():
    """
    Shows which settings the server loaded.
    It never reveals the API key itself, only whether one was found.
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
        "prices_per_million_tokens": {
            "small": {"input": config.SMALL_INPUT_PRICE_PER_M, "output": config.SMALL_OUTPUT_PRICE_PER_M},
            "large": {"input": config.LARGE_INPUT_PRICE_PER_M, "output": config.LARGE_OUTPUT_PRICE_PER_M},
        },
    }


# This is a normal "def" (not "async def") on purpose. Calling Groq waits on the
# network, and FastAPI runs normal "def" endpoints in a separate worker thread,
# so one slow request does not freeze the whole server.
@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """Answer one query and log it."""

    # 1. Decide which model to use.
    if request.mode == "small":
        tier = "small"
        route_reason = "forced by the caller (mode=small)"
    elif request.mode == "large":
        tier = "large"
        route_reason = "forced by the caller (mode=large)"
    else:
        # The trained router decides. If it is missing, it falls back to the large model.
        tier, route_reason, _router_score = router.route(request.query)

    # 2. Call the model.
    try:
        result = llm_client.generate(tier, request.query)
    except llm_client.LLMError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)

    # 3. Work out the cost, and what the "always large" baseline would have cost.
    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]
    cost_usd = cost.calculate_cost(tier, input_tokens, output_tokens)
    baseline_cost_usd = cost.calculate_baseline_cost(input_tokens, output_tokens)
    saved_usd = baseline_cost_usd - cost_usd

    # 4. Save everything to the database.
    request_id = database.log_request(
        {
            "query": request.query,
            "mode_requested": request.mode,
            "tier": tier,
            "model_name": result["model_name"],
            "route_reason": route_reason,
            "cache_hit": False,
            "similarity": None,
            "answer": result["text"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": result["latency_ms"],
            "cost_usd": cost_usd,
            "baseline_cost_usd": baseline_cost_usd,
            "saved_usd": saved_usd,
        }
    )

    # 5. Send the response back to the client.
    return QueryResponse(
        request_id=request_id,
        answer=result["text"],
        tier=tier,
        model_name=result["model_name"],
        mode_requested=request.mode,
        route_reason=route_reason,
        cache_hit=False,
        similarity=None,
        latency_ms=round(result["latency_ms"], 1),
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
    """Delete all logged requests. Useful for clearing test data before real experiments."""
    deleted = database.reset_logs()
    return {"deleted_rows": deleted}