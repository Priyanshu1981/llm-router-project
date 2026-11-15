FROM python:3.12-slim

# Hugging Face runs the container as the user with id 1000, so we create it
# and give it a home folder that it owns.
RUN useradd -m -u 1000 user \
 && mkdir -p /home/user/app \
 && chown -R user:user /home/user

ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /home/user/app

# CPU-only PyTorch. The default Linux build includes GPU libraries and is
# roughly 2 GB larger, which a CPU-only server cannot use.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# The other libraries, pinned to the versions that worked on your computer
# (the file is written by deploy_to_space.py).
COPY --chown=user requirements-deploy.txt ./
RUN pip install -r requirements-deploy.txt

USER user

# The application: server code, web page and the trained router.
COPY --chown=user backend ./backend
COPY --chown=user frontend ./frontend
COPY --chown=user ml/models ./ml/models

# Download the embedding model while building, so the server starts quickly.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Hugging Face Spaces expects the app to listen on port 7860.
EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]