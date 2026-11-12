"""
schemas.py
Defines the exact shape of the data the API receives and returns.

FastAPI uses these classes to:
  - reject bad input automatically (for example, an empty query),
  - convert Python objects to JSON,
  - generate the interactive documentation at /docs.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """What the client sends to POST /query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's question or prompt.",
    )
    mode: Literal["auto", "small", "large"] = Field(
        "auto",
        description=(
            "'auto' lets the router decide (Phase 4). "
            "'small' or 'large' force a specific model."
        ),
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value):
        """Reject queries that are only spaces or line breaks."""
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class QueryResponse(BaseModel):
    """What POST /query returns."""

    request_id: int
    answer: str
    # "small" or "large" = a model answered; "cache" = a stored answer was reused
    tier: Literal["small", "large", "cache"]
    model_name: str
    mode_requested: str
    route_reason: str
    cache_hit: bool
    similarity: Optional[float] = None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    baseline_cost_usd: float  # what the "always large model" setup would have cost
    saved_usd: float          # baseline_cost_usd - cost_usd


class StatsResponse(BaseModel):
    """What GET /stats returns: totals across all logged requests."""

    total_requests: int
    cache_hits: int
    cache_hit_rate: float
    small_calls: int
    large_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    total_baseline_cost_usd: float
    total_saved_usd: float
    savings_percent: float
    avg_latency_ms: float