"""
run_public.py
Starts the server in PUBLIC MODE, meant to be exposed through a tunnel.

Public-mode settings live in .env.public (created on the first run, never committed
to git). They override .env, so your normal local use is not affected:
  - an access code is required
  - the "clear cache" and "clear request log" actions are switched off
  - requests per visitor and model calls per day are limited

It refuses to start without an access code, so the app cannot be exposed by accident.
The server listens only on this computer (127.0.0.1). Only the tunnel makes it public.

    python run_public.py
"""

import os
import secrets
import sys
from pathlib import Path

from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_ENV = BASE_DIR / ".env.public"

TEMPLATE = """# Settings used ONLY by run_public.py. This file is not committed to git.
# They override the values in .env while the server runs in public mode.
DEMO_ACCESS_CODE={code}
ALLOW_ADMIN_ACTIONS=false
RATE_LIMIT_PER_MINUTE=12
MODEL_CALLS_PER_DAY=200
MAX_OUTPUT_TOKENS=1024
"""


def main():
    if not PUBLIC_ENV.exists():
        code = secrets.token_urlsafe(9)
        PUBLIC_ENV.write_text(TEMPLATE.format(code=code), encoding="utf-8")
        print("Created .env.public with a new access code:")
        print()
        print(f"  Access code: {code}")
        print()
        print("Save it. It is also stored in .env.public (which git ignores).")
        print("Edit that file if you want other limits. Then run this again to start the server:")
        print("  python run_public.py")
        return

    # Load the public-mode values BEFORE the application reads its settings.
    for key, value in dotenv_values(PUBLIC_ENV).items():
        if value is not None:
            os.environ[key] = value

    code = os.environ.get("DEMO_ACCESS_CODE", "").strip()
    if len(code) < 6:
        print("Refusing to start: DEMO_ACCESS_CODE in .env.public is missing or shorter than 6 characters.")
        print("Fix it, or delete .env.public and run this again to get a new one.")
        sys.exit(1)

    import uvicorn  # imported late, on purpose

    print("=" * 66)
    print("PUBLIC MODE")
    print(f"  Access gate        : ON")
    print(f"  Admin actions      : {'ON' if os.environ.get('ALLOW_ADMIN_ACTIONS', '').lower() in ('1', 'true', 'yes', 'on') else 'off'}")
    print(f"  Rate limit         : {os.environ.get('RATE_LIMIT_PER_MINUTE', '0')} requests per minute per visitor")
    print(f"  Daily model calls  : {os.environ.get('MODEL_CALLS_PER_DAY', '0')}")
    print()
    print("Next: open a SECOND terminal and run")
    print("  cloudflared tunnel --url http://127.0.0.1:8000")
    print("Then give visitors the https://...trycloudflare.com address and the access code.")
    print("=" * 66)

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
    )


if __name__ == "__main__":
    main()