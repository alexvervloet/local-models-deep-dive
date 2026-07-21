"""
Example 04: quantization: the size / quality / speed tradeoff.

Quantization stores each weight in fewer bits. fp16 uses 2 bytes per weight; q8
about 1; q4 about half a byte. Fewer bits = a smaller file, less memory, and
(because there's less to move) often faster generation, at some cost to quality.
For most tasks q4–q6 is the sweet spot: most of the quality, a fraction of the
size.

This script shows the tradeoff two ways:
  1. OFFLINE: the size and memory of one model across every quant level.
  2. LIVE (if a server is up): it asks your model the same question and reports
     tokens/sec, so you can feel the speed. (To compare quality across quants for
     real, pull two tags, e.g. `qwen2.5:7b-instruct-q4_0` and `...-q8_0`, and run
     this against each.)

    python examples/04_quantization_tradeoff.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import providers, sizing


def main():
    # 1) The offline tradeoff table for a 7B model.
    print("A 7B model at each quantization (4k context):\n")
    print(f"  {'quant':<6}{'bytes/param':>12}{'weights':>10}{'total mem':>11}   quality")
    notes = {
        "fp16": "reference (full)",
        "q8": "near-lossless",
        "q6": "great, common default",
        "q5": "very good",
        "q4": "good, the laptop default",
        "q3": "noticeable degradation",
        "q2": "usually too lossy",
    }
    for quant in ("fp16", "q8", "q6", "q5", "q4", "q3", "q2"):
        bpp = sizing.BYTES_PER_PARAM[quant]
        w = sizing.weights_gb(7, quant)
        total = sizing.model_memory_gb(7, quant, 4096).total_gb
        print(f"  {quant:<6}{bpp:>12.2f}{w:>9.1f}G{total:>10.1f}G   {notes[quant]}")
    print("\n  Rule of thumb: start at q4. Go up to q6/q8 if you have memory to spare")
    print("  and want a little more quality; only go below q4 if you must.\n")

    # 2) Live speed check, if a server is available.
    if not providers.server_up():
        print("(Start a local server to measure real tokens/sec for your model.)")
        return

    print(f"Measuring generation speed for {providers.describe()}...")
    messages = [{"role": "user", "content": "Write three sentences about why laptops can run LLMs now."}]
    t0 = time.time()
    reply = providers.chat(messages, temperature=0.5)
    dt = time.time() - t0
    # Rough token count: ~1.3 tokens/word is a decent english approximation.
    approx_tokens = max(1, round(len(reply.split()) * 1.3))
    print(f"\n  generated ~{approx_tokens} tokens in {dt:.1f}s  ≈  {approx_tokens / dt:.1f} tok/s")
    print("  (Run this against a higher- and lower-bit tag of the same model to feel")
    print("   the size/speed tradeoff directly.)")


if __name__ == "__main__":
    main()
