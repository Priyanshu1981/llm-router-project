# Cost-Aware Query Routing and Semantic Caching for LLM-Powered Web Applications

A web app that sends easy queries to a small LLM and hard queries to a large LLM,
and reuses stored answers for semantically similar questions. A live dashboard shows
cost, latency, and savings.

## Setup

1. Create and activate a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and paste your Groq API key.
4. Verify the setup: `python check_setup.py`
5. Start the server: `uvicorn backend.main:app --reload`

## Models

- Small: `openai/gpt-oss-20b` (Groq)
- Large: `openai/gpt-oss-120b` (Groq)
- Embeddings: `all-MiniLM-L6-v2` (sentence-transformers)

## Project structure

- `backend/`: FastAPI server, router, cache, cost tracking, logging
- `ml/`: scripts to build the dataset and train the router
- `data/`: datasets and saved model responses
- `experiments/`: scripts and results comparing routing strategies
- `frontend/`: chat page and live dashboard
