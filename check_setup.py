"""
Setup check: run this first.

    python check_setup.py

This repo is different from its siblings: there's no API key to check, because the
model runs on YOUR machine. So instead of a key, this checks the thing that
matters here: **is a local model server actually running, and does it have a
model pulled?** It also checks your Python version and the (tiny) package list.
Makes no hosted API call and costs nothing.

The one offline section (the sizing calculator) needs none of this, so even with
no server running this check will still point you there.

Uses only the standard library for the core checks, so it runs before `pip install`.
"""

import importlib.util
import os
import sys
import urllib.error
import urllib.request

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def ok(msg):
    print(f"  {_c('✓', '32')} {msg}")


def warn(msg):
    print(f"  {_c('!', '33')} {msg}")


def fail(msg):
    print(f"  {_c('✗', '31')} {msg}")


HERE = os.path.dirname(os.path.abspath(__file__))


def _read_env_file():
    env_path = os.path.join(HERE, ".env")
    values = {}
    if not os.path.exists(env_path):
        return None
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _get(env, name, default=""):
    return os.getenv(name) or (env or {}).get(name, default)


ALWAYS = [
    ("openai", "openai", "the client for your local server"),
    ("dotenv", "python-dotenv", "loads settings from .env"),
    ("rich", "rich", "pretty output in the capstone"),
]


def check_python():
    print("Python version")
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        ok(f"Python {major}.{minor} (3.10+ required)")
        return True
    fail(f"Python {major}.{minor}: this repo needs Python 3.10 or newer.")
    print("    Install a newer Python from https://www.python.org/downloads/")
    return False


def check_dependencies():
    print("\nDependencies")
    missing = []
    for import_name, pip_name, purpose in ALWAYS:
        if importlib.util.find_spec(import_name) is not None:
            ok(f"{pip_name}: {purpose}")
        else:
            fail(f"{pip_name} MISSING: {purpose}")
            missing.append(pip_name)
    if missing:
        print("\n    Install everything with:")
        print("        pip install -r requirements.txt")
    return not missing


def _probe(url, timeout=1.5):
    """True if something HTTP answers at url (any status, even an error)."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError):
        return False


def _models_at(base_url):
    """Model tags the server reports via GET /v1/models (empty on failure)."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=2.0) as r:
            import json

            data = json.loads(r.read())
        return [m.get("id", "?") for m in data.get("data", [])]
    except Exception:
        return []


def check_server(env):
    print("\nLocal model server")
    base_url = _get(env, "OPENAI_BASE_URL", "http://localhost:11434/v1")
    model = _get(env, "LOCAL_MODEL", "llama3.2")
    root = base_url.removesuffix("/v1").rstrip("/") or base_url

    if not _probe(root):
        fail(f"No server is answering at {base_url}.")
        print("    This repo runs models locally, so a runtime must be up. Quickest path:")
        print("      1. Install Ollama:   https://ollama.com")
        print("      2. Start it (the app, or `ollama serve`); it listens on :11434.")
        print(f"      3. Pull a model:     ollama pull {model}")
        print("    (Using LM Studio / llama.cpp / vLLM instead? Set OPENAI_BASE_URL in .env.)")
        return False

    ok(f"a server is up at {base_url}")
    models = _models_at(base_url)
    if not models:
        warn("server is up but reports no models. Pull one, e.g.:")
        print(f"        ollama pull {model}")
        return True  # server's up; that's the hard part
    if model in models or any(m.startswith(model) for m in models):
        ok(f"your LOCAL_MODEL '{model}' is available")
    else:
        warn(f"LOCAL_MODEL '{model}' isn't pulled. Available: {', '.join(models)}")
        print(f"        ollama pull {model}   (or set LOCAL_MODEL to one above)")
    return True


def main():
    print(_c("Checking your setup for the Local Models deep dive...\n", "1"))
    env = _read_env_file()
    if env is None:
        warn(".env not found; using defaults. Create it with:  cp .env.example .env")
    py = check_python()
    deps = check_dependencies()
    server = check_server(env)

    print()
    if py and deps and server:
        print(_c("All set! 🎉", "1;32"))
        print("Start here:  python examples/01_quant_math.py   (offline, no server needed)")
        print("Then:        python examples/02_first_local_request.py")
        return 0
    if py and deps and not server:
        print(_c("Almost. Packages are fine, but no local server is running.", "1;33"))
        print("You can still run the OFFLINE section now:")
        print("    python examples/01_quant_math.py")
        print("Start a server (see above) to run the rest.")
        return 1
    print(_c("Not ready yet. Fix the ✗ items above, then run this again.", "1;31"))
    print("(The offline calculator `examples/01_quant_math.py` works regardless.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
