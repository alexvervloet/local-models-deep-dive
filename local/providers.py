"""
local/providers.py: the ONE idea of this whole repo, in code.

Open-weight models served locally speak the **OpenAI-compatible API**. So the
exact `openai` SDK from the sibling repos works against a model on your own
machine. You change `base_url` and nothing else:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

`base_url` points the SDK at the local server (Ollama's default port is 11434);
`api_key` is required by the SDK but ignored by the local server, so any
non-empty string works. Roles, streaming, token usage, structured output, tool
calling, all the shapes you already know, come along unchanged.

This file is the keystone, the local equivalent of the sibling repos'
`providers.py`. It does three things:

  client()          -> a singleton OpenAI client aimed at the local server
  chat() / stream() -> send messages, get a reply (with friendly errors)
  server_up()       -> a no-LLM probe: is a local server reachable at all?

Everything is read from `.env` so you can point at a different runtime (LM Studio
on :1234, llama.cpp on :8080, vLLM on :8000) without touching code.
"""

import os
import sys
import urllib.error
import urllib.request
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# The OpenAI-compatible endpoint of your local server. Ollama serves this on
# 11434 automatically once it's running. (.env.example lists the others.)
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

# The model tag to talk to. With Ollama this is whatever you `ollama pull`ed
# (e.g. "llama3.2", "qwen2.5", "phi3.5"). See README section 3 for picking one.
CHAT_MODEL = os.getenv("LOCAL_MODEL", "llama3.2")

# A small local embedding model, used by the embeddings example (section 7).
EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "nomic-embed-text")

# The SDK requires *some* key; the local server ignores it. Any string works.
API_KEY = os.getenv("OPENAI_API_KEY", "local-no-key-needed")


def describe() -> str:
    """One-line summary of where we're pointed, handy for examples to print."""
    return f"local  (base_url={BASE_URL}, chat={CHAT_MODEL})"


@lru_cache(maxsize=1)
def client():
    """A singleton OpenAI client pointed at the local server.

    Lazily imported and cached, so merely importing this module never forces the
    SDK to load or touches the network.
    """
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


# --- The "is anything even running?" probe --------------------------------
#
# Crucial for a repo with no hosted fallback: before we make an LLM call we want
# to tell the difference between "server down" (start Ollama) and "model missing"
# (pull it). This hits the server's root with the stdlib only: no SDK, no model,
# so it works even before `pip install`.


def server_up(timeout: float = 1.5) -> bool:
    """True if *something* answers at the local server's host:port.

    We strip the `/v1` suffix and probe the root; Ollama answers "Ollama is
    running" there. We only care that the TCP port is alive and speaks HTTP, not
    what it says, so any HTTP response (even a 404) counts as "up".
    """
    root = BASE_URL.removesuffix("/v1").rstrip("/") or BASE_URL
    try:
        urllib.request.urlopen(root, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # answered with an HTTP error -> the server is up
    except (urllib.error.URLError, OSError):
        return False  # nothing listening / connection refused / timeout


def list_models() -> list[str]:
    """Model tags the local server currently has available (empty on failure).

    Uses the OpenAI-standard `GET /v1/models` route, which Ollama and the others
    implement. Handy for telling "model not pulled" apart from "server down".
    """
    try:
        resp = client().models.list()
        return [m.id for m in resp.data]
    except Exception:
        return []


# --- Friendly guards ------------------------------------------------------
#
# This repo is FREE but not zero-effort: a local server has to be running. These
# helpers turn the two common failures into instructive messages instead of
# stack traces: the local analogue of the sibling repos' "missing API key" guard.

_START_HELP = (
    f"No local server is reachable at {BASE_URL}.\n"
    "  This repo runs models on YOUR machine, so a local runtime must be up.\n"
    "  Quickest path (Ollama):\n"
    "    1. Install Ollama:        https://ollama.com\n"
    "    2. Start it (the app, or `ollama serve`); it listens on :11434.\n"
    f"    3. Pull a small model:    ollama pull {CHAT_MODEL}\n"
    "    4. Re-run this script.\n"
    "  Already using LM Studio / llama.cpp / vLLM? Set OPENAI_BASE_URL in .env\n"
    "  (see .env.example) and re-run."
)


def ensure_server(exit_code: int = 0) -> None:
    """Exit cleanly with start-up instructions if no local server is reachable.

    Call this at the top of any script that makes an LLM call. We exit 0 by
    default so an absent server is treated as "skipped", not "failed". Nothing
    was charged and no key was used.
    """
    if not server_up():
        print(_START_HELP)
        print("\n(Nothing was charged and no key was used; local serving is free.)")
        sys.exit(exit_code)


def _model_help(model: str) -> str:
    have = list_models()
    have_line = f"  Models you have: {', '.join(have)}\n" if have else ""
    return (
        f"The server is up, but the call for model {model!r} failed.\n"
        f"{have_line}"
        f"  Most likely it isn't pulled yet. Run:  ollama pull {model}\n"
        "  (Tags must match exactly; `ollama list` shows what you have.)"
    )


def chat(messages: list[dict], model: str | None = None, **kwargs) -> str:
    """Send chat messages to the local model and return the reply text.

    On a missing model or a failed call it prints an instructive message and
    exits cleanly (code 0) rather than crashing, the same graceful-degradation rule
    as the rest of the repo. Extra kwargs (temperature, max_tokens, ...) pass
    straight through to the OpenAI-compatible endpoint.
    """
    model = model or CHAT_MODEL
    try:
        resp = client().chat.completions.create(model=model, messages=messages, **kwargs)  # type: ignore[arg-type]
    except Exception as exc:  # model not pulled, bad request, etc.
        from openai import APIConnectionError

        if isinstance(exc, APIConnectionError):
            print(_START_HELP)
            sys.exit(0)
        print(_model_help(model))
        print(f"\n  (raw error: {exc})")
        sys.exit(0)
    return resp.choices[0].message.content or ""


def stream(messages: list[dict], model: str | None = None, **kwargs):
    """Yield reply chunks as they arrive, for a live, typewriter feel.

    The caller is expected to have run `ensure_server()` already; this keeps the
    generator simple. The first chunk arriving slowly is normal: that delay is
    *prompt processing* (section 6), not the network.
    """
    model = model or CHAT_MODEL
    stream_resp = client().chat.completions.create(  # type: ignore[call-overload]
        model=model, messages=messages, stream=True, **kwargs  # type: ignore[arg-type]
    )
    for chunk in stream_resp:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
