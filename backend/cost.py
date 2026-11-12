"""
cost.py
Turns token counts into dollars.

    cost = (input_tokens x input_price + output_tokens x output_price) / 1,000,000

Prices are quoted per 1 million tokens, which is why we divide by 1,000,000.
"""

from backend import config

TOKENS_PER_PRICE_UNIT = 1_000_000


def calculate_cost(tier, input_tokens, output_tokens):
    """
    Cost in USD of one request.

    tier is "small", "large" or "cache". A cache hit calls no model,
    so we count it as free.
    """
    if tier == "cache":
        return 0.0

    if tier not in config.PRICES_PER_M:
        raise ValueError(f"Unknown tier '{tier}'. Use 'small', 'large' or 'cache'.")

    input_price, output_price = config.PRICES_PER_M[tier]
    total = input_tokens * input_price + output_tokens * output_price
    return total / TOKENS_PER_PRICE_UNIT


def calculate_baseline_cost(input_tokens, output_tokens):
    """
    What the same request would have cost if it had gone to the LARGE model.
    This is the "always large" baseline that we measure our savings against.

    Note: this is an approximation. We reuse the token counts we actually saw,
    but the large model might have produced a longer or shorter answer.
    Mention this honestly in your report.
    """
    return calculate_cost("large", input_tokens, output_tokens)