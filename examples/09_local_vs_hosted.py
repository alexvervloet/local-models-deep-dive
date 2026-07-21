"""
Example 09: when local beats hosted (and when it doesn't).

Local isn't "better" or "cheaper" in the abstract. It's a set of tradeoffs. This
script lays out the decision so you can make it on purpose, and (if a server is up)
measures your local latency so one column isn't hypothetical.

The honest scorecard:

  Local WINS on:
    * Privacy / data control: nothing leaves your machine. Often the whole reason.
    * Cost at volume: after hardware, generation is free; no per-token bill.
    * Offline / air-gapped: works with no internet.
    * No rate limits, no vendor lock-in, full control of the model & version.

  Hosted WINS on:
    * Peak quality: the largest frontier models don't fit on your laptop.
    * Zero ops: no GPU, drivers, or model management to babysit.
    * Elastic scale: burst to thousands of requests without buying hardware.
    * Latest models on day one.

  Rule of thumb: prototype and reach for top quality on HOSTED; move high-volume,
  privacy-sensitive, or fixed tasks to LOCAL once a small model is good enough
  (prove "good enough" with the Evals deep dive).

    python examples/09_local_vs_hosted.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import providers

ROWS = [
    ("Privacy / data control", "LOCAL", "data never leaves your machine"),
    ("Cost at high volume", "LOCAL", "no per-token bill after hardware"),
    ("Works offline", "LOCAL", "no network needed"),
    ("Peak answer quality", "HOSTED", "biggest models don't fit locally"),
    ("Zero ops / no GPU", "HOSTED", "provider runs the hardware"),
    ("Elastic scale / bursts", "HOSTED", "scale without buying machines"),
    ("Latest models day one", "HOSTED", "providers ship first"),
    ("No rate limits", "LOCAL", "your hardware, your rules"),
]


def main():
    print("Local vs. hosted, picked per requirement rather than by vibes:\n")
    print(f"  {'requirement':<26}{'winner':<8}why")
    for req, winner, why in ROWS:
        print(f"  {req:<26}{winner:<8}{why}")
    print()

    if providers.server_up():
        print("Measuring YOUR local latency (so the 'speed' question isn't abstract)...")
        t0 = time.time()
        _ = providers.chat([{"role": "user", "content": "Reply with exactly: ok"}], temperature=0)
        print(f"  round-trip for a tiny request: {time.time() - t0:.2f}s")
        print("  No network hop, but speed is capped by YOUR hardware, not a datacenter GPU.")
    else:
        print("(Start a local server to measure your own latency here.)")

    print("\nThe best answer is often BOTH: a local small model for the 80% common,")
    print("private, high-volume path, falling back to a hosted frontier model for the")
    print("hard 20%. That routing is a production concern (see the Production dive).")


if __name__ == "__main__":
    main()
