"""
deploy_to_space.py
Deploys the app to a Hugging Face Space (Docker).

What it does, in order:
  1. asks for your Hugging Face token (hidden input) and finds your username
  2. writes requirements-deploy.txt with the library versions installed on THIS computer
  3. creates the Space (or reuses it)
  4. stores secrets (your Groq key, the access code) and settings in the Space
  5. uploads only the files the server needs, in a single commit
  6. waits for the build and tells you the address

Usage (from the project root, with (.venv) active):
    python deploy_to_space.py
    python deploy_to_space.py --name my-router-demo
    python deploy_to_space.py --calls-per-day 150
"""

import argparse
import getpass
import os
import secrets
import sys
import time
from importlib import metadata
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from backend import config

BASE_DIR = config.BASE_DIR

# (what to write in requirements, installed package name)
PINNED_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn[standard]", "uvicorn"),
    ("python-dotenv", "python-dotenv"),
    ("groq", "groq"),
    ("sentence-transformers", "sentence-transformers"),
    ("scikit-learn", "scikit-learn"),
    ("numpy", "numpy"),
    ("joblib", "joblib"),
]

SPACE_README = """---
title: Cost-Aware LLM Router
emoji: 💸
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Cost-Aware Query Routing and Semantic Caching for LLM-Powered Web Applications

A web app that answers questions from a semantic cache when a similar question
was answered before, and otherwise lets a trained router choose between a small
and a large language model. A live dashboard shows latency, tokens, cost and
the saving compared with always using the large model.

The demo is protected by an access code to keep its API usage under control.

Honest summary of the experiments: on the benchmark used, the router did not
show a measurable advantage over random routing, while the semantic cache
(with safety guards) answered repeated and paraphrased questions at no cost.
"""

END_STAGES_BAD = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "DELETING"}


def write_requirements():
    """Writes requirements-deploy.txt using the versions installed on this computer."""
    lines = []
    for spec, distribution in PINNED_PACKAGES:
        try:
            lines.append(f"{spec}=={metadata.version(distribution)}")
        except metadata.PackageNotFoundError:
            print(f"Could not find the installed package '{distribution}'. Is (.venv) active?")
            sys.exit(1)
    path = BASE_DIR / "requirements-deploy.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, lines


def collect_files():
    """
    Returns a list of (path on this computer, path in the Space).
    Must run AFTER write_requirements(), because it checks that file too.
    """
    files = [
        (BASE_DIR / "Dockerfile", "Dockerfile"),
        (BASE_DIR / "requirements-deploy.txt", "requirements-deploy.txt"),
        (BASE_DIR / "ml" / "models" / "router.joblib", "ml/models/router.joblib"),
        (BASE_DIR / "ml" / "models" / "router_report.json", "ml/models/router_report.json"),
    ]
    for folder, pattern in [("backend", "*.py"), ("frontend", "*")]:
        for path in sorted((BASE_DIR / folder).glob(pattern)):
            if path.is_file():
                files.append((path, f"{folder}/{path.name}"))

    missing = [local for local, _ in files if not local.exists()]
    if missing:
        print("These required files are missing:")
        for local in missing:
            print(f"  {local}")
        print()
        print("What to do:")
        print("  - Dockerfile: create it as described in Step 7 (exact name, no extension).")
        print("  - router.joblib or router_report.json: run  python ml/train_router.py")
        print("  - login.html or other frontend files: check the frontend folder.")
        sys.exit(1)

    empty = [local for local, _ in files if local.stat().st_size == 0]
    if empty:
        print("These files are empty (a paste probably did not save):")
        for local in empty:
            print(f"  {local}")
        sys.exit(1)
    return files


def stage_name(api, repo_id):
    try:
        stage = api.get_space_runtime(repo_id).stage
        return str(getattr(stage, "value", stage))
    except Exception as error:  # noqa: BLE001
        return f"UNKNOWN ({error})"


def wait_for_running(api, repo_id, minutes):
    deadline = time.time() + minutes * 60
    last = None
    while time.time() < deadline:
        stage = stage_name(api, repo_id)
        if stage != last:
            print(f"  [{time.strftime('%H:%M:%S')}] status: {stage}")
            last = stage
        if stage == "RUNNING":
            return True
        if stage in END_STAGES_BAD:
            return False
        time.sleep(15)
    return None


def main():
    parser = argparse.ArgumentParser(description="Deploy the app to a Hugging Face Space.")
    parser.add_argument("--name", default="llm-router-demo", help="name of the Space (default: llm-router-demo)")
    parser.add_argument("--private", action="store_true", help="make the Space private (then it is NOT a public URL)")
    parser.add_argument("--access-code", default=None, help="the access code (a random one is made if you leave it out)")
    parser.add_argument("--calls-per-day", type=int, default=200,
                        help="daily limit of model calls across all visitors (default: 200)")
    parser.add_argument("--rate-per-minute", type=int, default=12,
                        help="requests per minute allowed from one visitor (default: 12)")
    parser.add_argument("--wait-minutes", type=int, default=25, help="how long to wait for the build (default: 25)")
    args = parser.parse_args()

    # --- Groq key ---
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "paste_your_key_here":
        print("No Groq key found in .env. Fix that first.")
        sys.exit(1)

    # --- Hugging Face token ---
    token = os.getenv("HF_TOKEN") or getpass.getpass("Hugging Face token (typing is hidden): ").strip()
    api = HfApi(token=token)
    try:
        username = api.whoami()["name"]
    except Exception as error:  # noqa: BLE001
        print(f"Could not sign in to Hugging Face with that token: {error}")
        print("Create a token of type 'Write' at huggingface.co/settings/tokens and try again.")
        sys.exit(1)
    repo_id = f"{username}/{args.name}"
    print(f"Signed in as {username}. The Space will be {repo_id}")

    # --- Access code ---
    access_code = args.access_code
    generated = False
    if not access_code:
        entered = getpass.getpass("Access code for visitors (press Enter to generate one): ").strip()
        access_code = entered or secrets.token_urlsafe(9)
        generated = not entered
    if len(access_code) < 6:
        print("The access code is very short. Use at least 6 characters.")
        sys.exit(1)

    # --- Requirements first, then check that every file exists ---
    path, lines = write_requirements()
    print(f"Wrote {path.name}:")
    for line in lines:
        print(f"  {line}")
    files = collect_files()

    # --- Create the Space ---
    print()
    print("Creating the Space ...")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)

    # --- Secrets and settings (never stored in files) ---
    print("Storing secrets and settings ...")
    api.add_space_secret(repo_id=repo_id, key="GROQ_API_KEY", value=config.GROQ_API_KEY)
    api.add_space_secret(repo_id=repo_id, key="DEMO_ACCESS_CODE", value=access_code)
    variables = {
        "SMALL_MODEL": config.SMALL_MODEL,
        "LARGE_MODEL": config.LARGE_MODEL,
        "REASONING_EFFORT": config.REASONING_EFFORT,
        "MAX_OUTPUT_TOKENS": "1024",
        "ROUTER_THRESHOLD": str(config.ROUTER_THRESHOLD),
        "CACHE_THRESHOLD": str(config.CACHE_THRESHOLD),
        "SMALL_INPUT_PRICE_PER_M": str(config.SMALL_INPUT_PRICE_PER_M),
        "SMALL_OUTPUT_PRICE_PER_M": str(config.SMALL_OUTPUT_PRICE_PER_M),
        "LARGE_INPUT_PRICE_PER_M": str(config.LARGE_INPUT_PRICE_PER_M),
        "LARGE_OUTPUT_PRICE_PER_M": str(config.LARGE_OUTPUT_PRICE_PER_M),
        "ALLOW_ADMIN_ACTIONS": "false",
        "RATE_LIMIT_PER_MINUTE": str(args.rate_per_minute),
        "MODEL_CALLS_PER_DAY": str(args.calls_per_day),
    }
    for key, value in variables.items():
        api.add_space_variable(repo_id=repo_id, key=key, value=value)

    # --- Upload everything in one commit ---
    print(f"Uploading {len(files) + 1} files (this can take a minute) ...")
    operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for local, remote in files]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=SPACE_README.encode("utf-8")))
    api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        operations=operations,
        commit_message="Deploy cost-aware LLM router",
    )

    # --- Addresses ---
    host = f"{username}-{args.name}".lower().replace("_", "-").replace(".", "-")
    direct_url = f"https://{host}.hf.space"
    page_url = f"https://huggingface.co/spaces/{repo_id}"

    print()
    print("=" * 70)
    print("Uploaded. Hugging Face is now building the app (about 5 to 12 minutes).")
    print(f"  Space page   : {page_url}")
    print(f"  Build logs   : {page_url}?logs=build")
    print(f"  Public URL   : {direct_url}")
    print("  (If that address does not open, use the Space page: menu with three dots,")
    print("   'Embed this Space', and copy the 'Direct URL'.)")
    if generated:
        print()
        print(f"  ACCESS CODE  : {access_code}")
        print("  Save it now. It is not stored anywhere on your computer.")
    print("=" * 70)
    print()
    print("Waiting for the build ...")

    outcome = wait_for_running(api, repo_id, args.wait_minutes)
    print()
    if outcome is True:
        print(f"The app is running. Open {direct_url}")
        print(f"Then check it with:  python test_deployed.py --url {direct_url}")
    elif outcome is False:
        print(f"The build or start failed. Open the logs: {page_url}?logs=build")
        print("Read the last red lines, and paste them to me.")
        sys.exit(1)
    else:
        print("Still building after the wait. That can be normal for a first build.")
        print(f"Watch it at {page_url} and open the logs if it shows an error.")


if __name__ == "__main__":
    main()