"""
Example 06: performance: time-to-first-token, tokens/sec, and context.

Local inference has two speeds, and conflating them is the most common confusion:

  * TIME-TO-FIRST-TOKEN (TTFT): the pause before anything appears. This is the
    server *reading your prompt* (prompt processing / "prefill"). A long prompt or
    a cold model makes it longer. It is NOT the network.

  * GENERATION SPEED (tokens/sec): how fast text streams once it starts. This is
    the model producing tokens one at a time.

This script streams a reply and measures both, so the difference is concrete. Then
it shows that a longer prompt mostly grows TTFT, while generation speed stays
about the same. (The KV cache from Example 01 is why a longer context costs more
memory, and a bit more time, as you go.)

Needs a server. Streaming uses the same SDK shape as the API deep dives.

    python examples/06_performance.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import providers


def timed_stream(messages, label):
    """Stream a reply, printing TTFT and tokens/sec."""
    print(f"\n[{label}]")
    t0 = time.time()
    first = None
    chunks = 0
    for piece in providers.stream(messages, temperature=0.4):
        if first is None:
            first = time.time()
            print(f"  (first token after {first - t0:.2f}s)")
        chunks += 1
        print(piece, end="", flush=True)
    done = time.time()
    gen_time = max(1e-6, done - (first or done))
    print(f"\n  ~{chunks} chunks in {gen_time:.1f}s of generation  ≈  {chunks / gen_time:.1f} chunks/s")
    return (first or done) - t0


def main():
    providers.ensure_server()
    print(f"Measuring {providers.describe()}")
    print("Watch for the pause BEFORE text (TTFT = prompt processing), then the stream.")

    short = [{"role": "user", "content": "Say hello and tell me one fact about llamas."}]
    ttft_short = timed_stream(short, "short prompt")

    # A deliberately long prompt: same question, lots of preamble to chew on.
    filler = ("Consider the following context. " + "Local models run on consumer hardware. ") * 60
    long = [
        {"role": "system", "content": filler},
        {"role": "user", "content": "Given all that, say hello and tell me one fact about llamas."},
    ]
    ttft_long = timed_stream(long, "long prompt (more to read first)")

    print("\n---")
    print(f"TTFT short prompt: {ttft_short:.2f}s    TTFT long prompt: {ttft_long:.2f}s")
    print("The long prompt's first token took longer; that extra time is the model")
    print("READING the prompt, not generating. Generation speed barely changed.")
    print("Takeaways: keep prompts tight for snappy responses; a bigger model or a")
    print("higher quant lowers tokens/sec; the first call after load is always slowest.")


if __name__ == "__main__":
    main()
