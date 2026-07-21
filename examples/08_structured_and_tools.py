"""
Example 08: structured output & tool calling on local models (the caveats).

Two features you rely on with hosted models, "give me JSON" and "call this
function", also work locally, but with rougher edges. Smaller models follow a
schema less reliably and sometimes wrap JSON in prose or markdown fences. The fix
is the same defensive habit from the API/prompt deep dives: ask clearly, and
parse forgivingly.

This script shows both:
  1. Structured output: request JSON (using the OpenAI-compatible `response_format`
     when the server supports it), then parse it defensively (strip ``` fences).
  2. Tool calling: describe one tool and let the model choose to call it. Support
     varies by model; if your model doesn't emit a tool call, the script says so
     rather than pretending.

Pick a capable small model for this (qwen2.5 and llama3.1 are good at JSON/tools).

    python examples/08_structured_and_tools.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import providers


def strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences a model often adds around JSON."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def structured_output():
    print("1) Structured output: asking for JSON\n")
    messages = [
        {"role": "system", "content": "Extract fields and reply with ONLY a JSON object, no prose, no fences."},
        {"role": "user", "content": "Order: 3 large coffees and 1 muffin for Dana, table 7."},
    ]
    # Many local servers accept response_format={"type":"json_object"}; some ignore
    # it. We pass it AND keep the defensive parse, so it works either way.
    try:
        raw = providers.chat(messages, temperature=0, response_format={"type": "json_object"})
    except Exception:
        raw = providers.chat(messages, temperature=0)  # server didn't accept the flag

    print("  raw reply:", raw.replace("\n", " ")[:160])
    try:
        data = json.loads(strip_fences(raw))
        print("  parsed OK:", data)
    except json.JSONDecodeError:
        print("  !! couldn't parse as JSON; small models sometimes drift. Tighten the")
        print("     prompt, lower temperature, or try a stronger model (qwen2.5/llama3.1).")


def tool_calling():
    print("\n2) Tool calling: letting the model choose a function\n")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    try:
        resp = providers.client().chat.completions.create(
            model=providers.CHAT_MODEL,
            messages=[{"role": "user", "content": "What's the weather in Oslo right now?"}],
            tools=tools,  # type: ignore[arg-type]
            temperature=0,
        )
    except Exception as exc:
        print(f"  this model/server didn't accept tools: {exc}")
        return

    msg = resp.choices[0].message
    calls = getattr(msg, "tool_calls", None)
    if calls:
        for c in calls:
            print(f"  model requested: {c.function.name}({c.function.arguments})")
        print("  -> you'd run the function and feed the result back (the agent loop).")
    else:
        print("  model answered without a tool call:")
        print("   ", (msg.content or "").strip()[:160])
        print("  Tool support varies across local models; capable ones (qwen2.5,")
        print("  llama3.1) emit the call. Weaker ones just answer in text.")


def main():
    providers.ensure_server()
    print(f"Using {providers.describe()}\n")
    structured_output()
    tool_calling()
    print("\nTakeaway: the FEATURES exist locally, but reliability tracks model size.")
    print("Keep the defensive parsing you learned for hosted models; you need it more.")


if __name__ == "__main__":
    main()
