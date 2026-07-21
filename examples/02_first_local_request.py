"""
Example 02: your first local request: the same SDK, one changed URL.

This is the whole idea of the repo in one script. The code below is the *same*
OpenAI chat call you'd make against OpenAI's servers: same `client.chat.completions
.create`, same messages, same response shape. The only difference is in
`local/providers.py`: the client's `base_url` points at your machine
(http://localhost:11434/v1) instead of api.openai.com. No OpenAI key, no account,
no cost.

Needs a local server running (Ollama by default). If none is up, the script prints
how to start one and exits cleanly. Nothing was charged, because nothing can be.

    python examples/02_first_local_request.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import providers


def main():
    providers.ensure_server()  # exits with start-up help if nothing is listening
    print(f"Talking to: {providers.describe()}\n")

    messages = [
        {"role": "system", "content": "You are a concise assistant. One short paragraph, no preamble."},
        {"role": "user", "content": "In two sentences, what's the difference between RAM and VRAM for running LLMs?"},
    ]

    print("Asking the local model...\n")
    reply = providers.chat(messages, temperature=0.3)
    print(reply)

    print("\n---")
    print("That was a normal OpenAI SDK call; the only change was base_url.")
    print("Every shape you learned in the API deep dives (roles, temperature,")
    print("streaming, usage) works here unchanged. Local is an ops choice, not a")
    print("new API.")


if __name__ == "__main__":
    main()
