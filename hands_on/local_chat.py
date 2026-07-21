"""
hands_on/local_chat.py: the capstone: a fully-local, zero-cost chat CLI.

Everything in the repo, wired into one tool you'd actually use: a streaming,
multi-turn chat assistant that runs entirely on your machine. No API key, no
per-token bill, no data leaving the laptop. It pulls together:

  - the OpenAI SDK pointed at localhost   (the whole "local" trick: providers.py)
  - streaming with a live tokens/sec readout   (the performance lesson, §6)
  - multi-turn memory                          (a growing message history)
  - an optional offline fit-check before you start  (sizing, §2)
  - graceful failure when no server is up          (start-up help, not a stack trace)

    # interactive chat (Ctrl-D or "quit" to exit):
    python hands_on/local_chat.py

    # one-shot question, then exit:
    python hands_on/local_chat.py "Explain quantization in one sentence."

    # use a specific model you've pulled:
    python hands_on/local_chat.py --model qwen2.5

    # print a memory fit-estimate for a model size first (offline):
    python hands_on/local_chat.py --fit 8
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import providers, sizing

SYSTEM = "You are a concise, friendly local assistant. Keep replies short unless asked for detail."


def stream_reply(history, model):
    """Stream one assistant turn, print a tokens/sec line, return the full text."""
    t0 = time.time()
    first = None
    pieces = []
    for piece in providers.stream(history, model=model, temperature=0.4):
        if first is None:
            first = time.time()
        pieces.append(piece)
        print(piece, end="", flush=True)
    done = time.time()
    text = "".join(pieces)
    gen = max(1e-6, done - (first or done))
    approx_tokens = max(1, round(len(text.split()) * 1.3))
    print(f"\n\033[2m  ({approx_tokens / gen:.0f} tok/s, first token {(first or done) - t0:.2f}s)\033[0m")
    return text


def main():
    parser = argparse.ArgumentParser(description="A fully-local chat CLI.")
    parser.add_argument("question", nargs="*", help="One-shot question; omit for interactive chat.")
    parser.add_argument("--model", default=None, help="Model tag to use (default: LOCAL_MODEL from .env).")
    parser.add_argument("--fit", type=float, metavar="PARAMS_B",
                        help="Offline: print a memory estimate for a model of this many billion params, then exit.")
    args = parser.parse_args()

    # Offline fit-check; needs no server.
    if args.fit is not None:
        print(sizing.format_estimate(sizing.model_memory_gb(args.fit, "q4", 4096)))
        print(f"\nFits in 16 GB at q4? {sizing.fits_in(16, args.fit, 'q4')}")
        return

    providers.ensure_server()  # graceful exit + instructions if nothing is listening
    model = args.model or providers.CHAT_MODEL
    print(f"Local chat: {providers.describe().replace(providers.CHAT_MODEL, model)}")

    history = [{"role": "system", "content": SYSTEM}]

    def turn(question):
        history.append({"role": "user", "content": question})
        print("\nassistant: ", end="", flush=True)
        reply = stream_reply(history, model)
        history.append({"role": "assistant", "content": reply})

    # One-shot mode.
    if args.question:
        turn(" ".join(args.question))
        return

    # Interactive mode.
    print('Type a message. Ctrl-D or "quit" to exit. Everything runs on your machine.\n')
    while True:
        try:
            question = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        turn(question)


if __name__ == "__main__":
    main()
