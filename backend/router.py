"""
router.py
Decides, for each query, whether the small or the large model should answer.

The trained model (saved by ml/train_router.py) predicts P(easy): the probability
that the small model's answer is good enough.

    P(easy) >= ROUTER_THRESHOLD   ->  small model
    otherwise                     ->  large model

If no trained model exists yet, or anything goes wrong, we choose the large
model: it costs more but is the safe choice for answer quality.
"""

import joblib
import numpy as np

from backend import config, features

BUNDLE_PATH = config.MODELS_DIR / "router.joblib"

# The loaded model lives here. None means "not loaded".
_bundle = None


def load():
    """Load the trained router. Called once when the server starts. Returns True on success."""
    global _bundle
    _bundle = None

    if not BUNDLE_PATH.exists():
        print("Router: no trained model found. Mode 'auto' will use the large model.")
        print("        Train one with:  python ml/train_router.py")
        return False

    try:
        bundle = joblib.load(BUNDLE_PATH)

        if bundle.get("embedding_model") != config.EMBEDDING_MODEL:
            print(
                f"Router: the model was trained with embeddings '{bundle.get('embedding_model')}' "
                f"but .env says '{config.EMBEDDING_MODEL}'. Retrain with: python ml/train_router.py"
            )
            return False

        # Load the embedding model now, so the first request is not slow.
        features.embed_text("warm up")
    except Exception as error:
        print(f"Router: could not load the trained model ({error}). Mode 'auto' will use the large model.")
        return False

    _bundle = bundle
    print(
        f"Router: loaded '{bundle['variant']}' trained on {bundle['n_train']} questions "
        f"(threshold in use: {config.ROUTER_THRESHOLD:.2f}, suggested by training: "
        f"{bundle['recommended_threshold']:.2f})"
    )
    return True


def is_ready():
    return _bundle is not None


def info():
    """Short description of the loaded router, for the /health endpoint."""
    if _bundle is None:
        return None
    return {
        "variant": _bundle["variant"],
        "trained_on_questions": _bundle["n_train"],
        "trained_at": _bundle["trained_at"],
        "suggested_threshold": _bundle["recommended_threshold"],
    }


def predict_p_easy(query, embedding=None):
    """
    Probability (0 to 1) that the small model's answer is good enough.
    Pass embedding if it was already computed, to avoid computing it twice.
    """
    if _bundle is None:
        raise RuntimeError("The router is not loaded.")

    embedding_row = None
    if embedding is not None:
        embedding_row = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

    X = features.build_features([query], embeddings=embedding_row)
    pipeline = _bundle["pipeline"]
    probabilities = pipeline.predict_proba(X)[0]
    return float(probabilities[list(pipeline.classes_).index(1)])


def route(query, embedding=None):
    """
    Choose a model for this query.
    Returns (tier, reason, p_easy), where tier is "small" or "large"
    and p_easy is None if no prediction could be made.
    """
    if _bundle is None:
        return "large", "router model not found (run: python ml/train_router.py), so the large model is used", None

    threshold = config.ROUTER_THRESHOLD
    try:
        p_easy = predict_p_easy(query, embedding)
    except Exception as error:
        return "large", f"router error ({error}), so the large model is used", None

    if p_easy >= threshold:
        reason = f"router: P(small model is enough) = {p_easy:.2f}, at or above threshold {threshold:.2f}"
        return "small", reason, p_easy

    reason = f"router: P(small model is enough) = {p_easy:.2f}, below threshold {threshold:.2f}"
    return "large", reason, p_easy