"""
llm_client.py
The only file that talks to the Groq API.

The main function is generate(tier, query). It sends the query to the small or
large model, and returns the answer together with the measurements we need:
latency and token counts.
"""

import time

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from backend import config

# The same instruction is sent to both models so the comparison is fair.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer accurately and concisely. "
    "For maths or logic questions, show the final answer clearly."
)


class LLMError(Exception):
    """
    Raised when something goes wrong while calling a model.
    It carries an HTTP status code so the web server can pass it on to the client.
    """

    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# The Groq client is created once, the first time it is needed, then reused.
_client = None


def _get_client():
    global _client
    if _client is None:
        key = config.GROQ_API_KEY
        if not key or key == "paste_your_key_here":
            raise LLMError(
                "GROQ_API_KEY is missing. Put your key in the .env file and restart the server.",
                status_code=500,
            )
        # The Groq library automatically retries briefly on temporary errors.
        _client = Groq(api_key=key, timeout=60.0)
    return _client


def generate(tier, query, max_output_tokens=None):
    """
    Send one query to the model for the given tier ("small" or "large").

    Returns a dictionary with:
        text           the model's answer
        model_name     the exact model that answered
        tier           "small" or "large"
        input_tokens   tokens in the prompt
        output_tokens  tokens generated (includes the model's hidden thinking)
        latency_ms     wall-clock time of the API call, in milliseconds
        finish_reason  "stop" normally, "length" if it ran out of tokens
    """
    if tier not in config.MODEL_BY_TIER:
        raise LLMError(f"Unknown model tier '{tier}'. Use 'small' or 'large'.", status_code=400)

    model_name = config.MODEL_BY_TIER[tier]
    client = _get_client()

    request_args = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "max_completion_tokens": max_output_tokens or config.MAX_OUTPUT_TOKENS,
    }
    # How much the gpt-oss models "think" before answering.
    # Sent through extra_body so it works on any version of the Groq library.
    if config.REASONING_EFFORT:
        request_args["extra_body"] = {"reasoning_effort": config.REASONING_EFFORT}

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**request_args)
    except RateLimitError as error:
        raise LLMError(
            "Groq rate limit reached. Wait a minute and try again. "
            "If it keeps happening, you may have used up the free daily token limit.",
            status_code=429,
        ) from error
    except APIConnectionError as error:
        raise LLMError(
            "Could not reach Groq. Check your internet connection and try again.",
            status_code=503,
        ) from error
    except APIStatusError as error:
        detail = getattr(error, "message", str(error))
        raise LLMError(
            f"Groq returned an error (HTTP {error.status_code}): {detail}",
            status_code=502,
        ) from error
    latency_ms = (time.perf_counter() - start) * 1000.0

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    finish_reason = choice.finish_reason

    if not text:
        if finish_reason == "length":
            text = (
                "(The model used all its output tokens before finishing its answer. "
                "Increase MAX_OUTPUT_TOKENS in .env.)"
            )
        else:
            text = "(The model returned an empty answer.)"

    usage = response.usage
    return {
        "text": text,
        "model_name": model_name,
        "tier": tier,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
    }