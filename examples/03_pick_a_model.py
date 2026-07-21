"""
Example 03: picking a model: params, families, and what fits.

"Which model should I run?" has two halves: capability (bigger ≈ smarter, to a
point) and whether it fits your machine (Example 01's arithmetic). This script
lists what your server actually has pulled, then maps the common small-model
families to their sizes and a fit estimate, so the choice is concrete.

Rules of thumb:
  * Parameter count is the first lever: 3B is fast and fine for simple tasks; 7-8B
    is the sweet spot for a laptop; 70B needs a serious GPU (or patience).
  * Prefer an INSTRUCT/chat-tuned tag over a base model for assistant work.
  * A higher-quality quant of a SMALLER model usually beats a crushed quant of a
    bigger one. When in doubt, smaller model + q4/q6.

Needs a server for the "what do I have?" part; the sizing guidance is offline.

    python examples/03_pick_a_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import providers, sizing

# A few popular small/instruct families and their rough parameter counts (B).
# Tags are Ollama-style; the point is the size, not the exact name.
COMMON = [
    ("llama3.2", 3, "Meta: small, fast, solid general chat"),
    ("phi3.5", 3.8, "Microsoft: strong reasoning for its size"),
    ("qwen2.5", 7, "Alibaba: excellent all-rounder, good at code/JSON"),
    ("mistral", 7, "Mistral: classic 7B workhorse"),
    ("llama3.1", 8, "Meta: the 8B everyone benchmarks against"),
    ("gemma2", 9, "Google: strong mid-size"),
    ("qwen2.5", 14, "Alibaba: bigger, noticeably smarter"),
    ("llama3.3", 70, "Meta: frontier-ish, needs lots of memory"),
]


def main():
    # 1) What's actually on this machine right now?
    if providers.server_up():
        have = providers.list_models()
        print(f"Models your server has pulled ({providers.describe()}):")
        if have:
            for m in have:
                print(f"  • {m}")
        else:
            print("  (none yet; `ollama pull llama3.2` to get a small one)")
    else:
        print("(No server running; showing the sizing guide only.)")
    print()

    # 2) The size/fit map, assuming a typical laptop with ~16 GB usable.
    free_gb = 16
    print(f"Common small models, size and whether they fit ~{free_gb} GB at q4:")
    print(f"  {'model':<12}{'params':>7}   {'q4 total':>9}   fit")
    for tag, params, note in COMMON:
        est = sizing.model_memory_gb(params, "q4", 4096)
        fits = sizing.fits_in(free_gb, params, "q4", 4096)
        mark = "✓ fits" if fits else "✗ too big"
        print(f"  {tag:<12}{params:>6}B   {est.total_gb:>7.1f} GB   {mark}   {note}")
    print()
    print("Change `free_gb` in this file to your real free memory, then re-read")
    print("the fit column. Best first pull for most laptops: `ollama pull llama3.2`")
    print("(tiny) or `ollama pull qwen2.5` (7B, a great default).")


if __name__ == "__main__":
    main()
