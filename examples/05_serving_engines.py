"""
Example 05: serving engines: Ollama vs llama.cpp vs vLLM.

The model is a file; a SERVING ENGINE loads it and answers requests. They all
expose the same OpenAI-compatible API (that's why your code doesn't change), but
they trade off ease vs. speed vs. scale. Pick by your situation:

  Ollama      Easiest. `ollama pull` + `ollama run`. Great for a laptop and for
              this repo. Handles model download, quantization, and GPU/CPU for you.
              Default endpoint :11434.

  llama.cpp   The engine UNDER a lot of others (incl. Ollama). Run `llama-server`
              directly for maximum control over GGUF files and flags. Endpoint
              :8080. Reach for it when you want to tune every knob.

  LM Studio   A GUI for the above, nice for browsing/trying models. Endpoint :1234.

  vLLM        Built for SERVING at scale: high throughput, continuous batching,
              many concurrent users, on a real GPU. Overkill for one laptop user;
              the right call for a shared internal endpoint. Endpoint :8000.

This script probes the common ports and tells you what's currently running, then
explains which one to choose. It makes no model call.

    python examples/05_serving_engines.py
"""

import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENGINES = [
    ("Ollama", "http://localhost:11434", "easiest; the repo default"),
    ("LM Studio", "http://localhost:1234", "GUI for browsing/running models"),
    ("llama.cpp", "http://localhost:8080", "max control over GGUF + flags"),
    ("vLLM", "http://localhost:8000", "high-throughput GPU serving"),
]


def _up(url, timeout=1.0):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError):
        return False


def main():
    print("Probing the usual local-serving ports...\n")
    any_up = False
    for name, url, note in ENGINES:
        running = _up(url)
        any_up = any_up or running
        mark = "● running" if running else "○ not found"
        print(f"  {mark:<12} {name:<10} {url:<26} {note}")
    print()

    if not any_up:
        print("Nothing is serving yet. For this repo, the simplest path is Ollama:")
        print("  1. install: https://ollama.com   2. `ollama pull llama3.2`   3. re-run.")
    else:
        print("Found a server. Point this repo at it with OPENAI_BASE_URL in .env")
        print("(append /v1 to the base, e.g. http://localhost:11434/v1).")

    print("\nHow to choose:")
    print("  • Just learning / one user on a laptop  -> Ollama (or LM Studio).")
    print("  • Want to hand-tune GGUF files & flags  -> llama.cpp (llama-server).")
    print("  • Serving many users on a GPU box       -> vLLM.")
    print("  • Either way your CODE is identical; only the base_url changes.")


if __name__ == "__main__":
    main()
