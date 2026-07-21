# Local Models: A Guided Deep Dive

A hands-on playground for running **open-weight LLMs on your own machine**, and
discovering that "local" is mostly an *operations* choice, not a new API. You'll
serve a model with Ollama, talk to it with the exact same OpenAI SDK from the API
deep dives (one changed URL), and understand every moving part: model sizing,
quantization, serving engines, performance, local embeddings, structured output,
and the real local-vs-hosted tradeoff. No framework magic, just enough code to
*see* how it works.

The hook that makes this repo click: **open-weight models speak the
OpenAI-compatible API.** So the client you already know works against a model on
your laptop by changing `base_url` and nothing else, and there's no API key and no
per-token bill anywhere in this repo. The one offline section (the sizing
calculator) runs with nothing installed at all; the rest need only a local runtime.

This repo is **standalone**: it teaches everything it needs on its own. It's the
full version of the "use a local model" footnote in the
[OpenAI](https://github.com/alexvervloet/openai-api-deep-dive) and
[Prompt Engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive) dives,
and it's the place the [Fine-tuning dive](https://github.com/alexvervloet/fine-tuning-deep-dive)
points to for *running* open weights, but its code depends on none of them.

Like its siblings, it's meant to be *walked through*. Each section ends with
something to run; the first runs **offline and free**, with no server even needed.
[EXERCISES.md](EXERCISES.md) has a predict-then-run prompt for each section.

---

## 0. The one big idea

> **An open-weight model on your machine speaks the same OpenAI API, so "local"
> is mostly an *ops* choice: you trade hosted convenience for privacy, cost at
> volume, offline use, and control.**

That's the whole repo. The first time you point the OpenAI SDK at
`http://localhost:11434/v1` and get an answer back with no key, the trick is
revealed: the *code* doesn't change, the *operations* do. Everything below 
sizing a model to your RAM, picking a quantization, choosing a serving engine,
reading tokens/sec, is about running that model *well* on hardware you own. And
because there's no provider doing the work, "will it even fit?" becomes a question
you answer with arithmetic before you download a thing. Hold onto that and none of
this feels complicated.

---

## 1. Setup (5 minutes)

```bash
# 1. Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies (just the OpenAI SDK + helpers: no key needed)
pip install -r requirements.txt

# 3. Copy the env file (it holds settings, not secrets)
cp .env.example .env

# 4. Install a local runtime and pull a small model
#    Ollama is the easiest (https://ollama.com):
ollama pull llama3.2

# 5. Confirm everything is wired up (no hosted call, costs nothing)
python check_setup.py
```

Unlike its siblings, this repo has **no API key**; the model runs on your machine.
What it needs instead is a **local server**. Ollama is the default; any
OpenAI-compatible runtime works by setting `OPENAI_BASE_URL`:

| Runtime | `OPENAI_BASE_URL` | Notes |
|---------|-------------------|-------|
| **Ollama** (default) | `http://localhost:11434/v1` | Easiest; `ollama pull` + run. Best for this repo. |
| LM Studio | `http://localhost:1234/v1` | GUI for browsing/running models. |
| llama.cpp | `http://localhost:8080/v1` | Maximum control over GGUF files & flags. |
| vLLM | `http://localhost:8000/v1` | High-throughput GPU serving for many users. |

> **The whole repo is free, and Section 2 needs nothing at all.** The sizing
> calculator is pure arithmetic, so run it before you install a runtime. Everything
> else needs a local server up, but never a key and never a cent.

---

## 2. Will it fit?: sizing & quantization math

```bash
python examples/01_quant_math.py     # offline, no server needed
```

The most useful local-models skill is arithmetic you do *before* downloading
anything: **how much memory will this model need?** One formula answers it 
`memory ≈ parameters × bytes-per-parameter (+ KV cache + overhead)`, where
bytes-per-parameter is set by the **quantization** (fp16 = 2.0, q8 ≈ 1.0, q4 ≈
0.5). The example prints the size of a model at every quant level, shows how the KV
cache grows with context length (a quiet memory eater), and answers the practical
question: *given my RAM, the best version of this model I can run.* It needs no
server, no model, no key.

---

## 3. Your first local request

```bash
python examples/02_first_local_request.py
```

The repo's whole idea in one script. The chat call below is the *same*
`client.chat.completions.create` you'd make against OpenAI, with the same messages and
same response shape, and the only difference lives in [local/providers.py](local/providers.py):
`base_url` points at your machine instead of `api.openai.com`. No key, no cost. If
no server is running, the script tells you how to start one and exits cleanly 
it can't charge you, because there's nothing to charge.

---

## 4. Picking a model

```bash
python examples/03_pick_a_model.py
```

"Which model?" has two halves: capability (bigger ≈ smarter, to a point) and
whether it fits (Section 2's math). The example lists what your server has pulled,
then maps the popular small families (Llama 3.x, Qwen2.5, Phi, Mistral, Gemma) to
their sizes and a fit estimate. Rules of thumb: **3B** is fast and fine for simple
work; **7–8B** is the laptop sweet spot; pick an **instruct** tag for assistant
tasks; and a higher-quality quant of a *smaller* model usually beats a crushed
quant of a bigger one.

---

## 5. Quantization in practice: size vs. quality vs. speed

```bash
python examples/04_quantization_tradeoff.py
```

Section 2 sized quantization; this one is about its *cost*. Fewer bits means a
smaller file, less memory, and often faster generation, at some quality cost. For
most tasks **q4–q6 is the sweet spot**: most of the quality, a fraction of the
size. The example shows the tradeoff table offline, then (if a server is up)
measures real tokens/sec so you can feel it. Start at q4; go up to q6/q8 if you
have memory to spare.

---

## 6. Serving engines: Ollama vs. llama.cpp vs. vLLM

```bash
python examples/05_serving_engines.py
```

The model is a file; a **serving engine** loads it and answers requests. They all
expose the same OpenAI-compatible API (that's why your code never changes), but
trade off ease vs. control vs. scale. The example probes the common ports and tells
you what's running, then: **Ollama** for easy/laptop, **llama.cpp** for hand-tuning
GGUF files, **LM Studio** for a GUI, **vLLM** for high-throughput GPU serving.

---

## 7. Performance: time-to-first-token & tokens/sec

```bash
python examples/06_performance.py
```

Local inference has *two* speeds, and conflating them causes most confusion:
**time-to-first-token** (the pause while the model *reads your prompt*, the prompt
processing, not the network) and **generation speed** (tokens/sec once it starts).
The example streams a reply, measures both, then shows that a longer prompt mostly
grows TTFT while generation speed barely moves. Takeaway: keep prompts tight for
snappy responses; the first call after load is always the slowest.

---

## 8. Embeddings, locally

```bash
python examples/07_embeddings.py
```

Embeddings (text → vectors, so similar meanings sit close) power search and RAG 
and they run locally through the same endpoint, just a different model
(`ollama pull nomic-embed-text`). No per-token embedding bill, ever. The example
embeds a handful of sentences, ranks them against a query by cosine similarity
(five lines of standard library, no magic), and retrieves the right one by
*meaning*: the exact core of the RAG deep dive, for $0.

---

## 9. Structured output & tool calling

```bash
python examples/08_structured_and_tools.py
```

"Give me JSON" and "call this function" work locally too, but with rougher edges.
Smaller models follow a schema less reliably and sometimes wrap JSON in prose or
fences. The fix is the same defensive habit from the API/prompt dives: ask clearly,
parse forgivingly. The example requests JSON (with `response_format` when the server
supports it) and parses it defensively, then describes one tool and lets the model
choose to call it, reporting honestly when a weaker model just answers in text.
Reliability tracks model size; capable small models (qwen2.5, llama3.1) are good at
both.

---

## 10. When local beats hosted (and when it doesn't)

```bash
python examples/09_local_vs_hosted.py
```

Local isn't better in the abstract. It's a set of tradeoffs, and this is the
decision laid out so you make it on purpose. **Local wins** on privacy/data
control, cost at volume, offline use, and no rate limits. **Hosted wins** on peak
quality, zero ops, elastic scale, and day-one access to the newest models. The
example prints the scorecard and (if a server is up) measures *your* local latency
so the speed column isn't hypothetical. The best answer is often **both**: a local
small model for the common, private, high-volume path, falling back to a hosted
frontier model for the hard cases.

---

## 11. Run the rest of the series locally, for $0

```bash
python examples/10_run_the_series_locally.py
```

The payoff of "local speaks the OpenAI API": the sibling dives (prompt engineering,
RAG, agents, evals) were built on the OpenAI SDK, so they run against your local
model with only an env change, with no code edits. The example runs a tiny RAG loop
(retrieve → ground → answer) entirely on the local server. To point any sibling
repo here, set this in *its* `.env`:

```bash
OPENAI_API_KEY=local                          # any non-empty string; ignored locally
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL=llama3.2
```

That's the whole change: your whole learning series, now running at zero cost.

---

## The capstone: `local_chat.py`

Everything assembled into a tool you'd actually use: a streaming, multi-turn chat
assistant running entirely on your machine: no key, no bill, nothing leaving the
laptop. It shows a live tokens/sec readout (Section 7), remembers the conversation,
and can print an offline fit-estimate before you start (Section 2).

```bash
# interactive chat (Ctrl-D or "quit" to exit):
python hands_on/local_chat.py

# one-shot question, then exit:
python hands_on/local_chat.py "Explain quantization in one sentence."

# use a specific model you've pulled:
python hands_on/local_chat.py --model qwen2.5

# offline: estimate memory for an 8B model before running anything:
python hands_on/local_chat.py --fit 8
```

Read [hands_on/local_chat.py](hands_on/local_chat.py); it's just the library
(`providers.stream` + `sizing`) wired to a CLI. **Suggested exercise:** pull a
second model (`ollama pull qwen2.5`) and chat with both via `--model`; you'll feel
the size/speed/quality tradeoff from Sections 4–5 in your own hands.

---

## Where to go next

You've run a model end to end on your own hardware. The frontier is more control
and more scale:

- **Fine-tune the open weights you're running**: LoRA/PEFT with
  `transformers`/`peft`/`trl`; the [Fine-tuning dive](https://github.com/alexvervloet/fine-tuning-deep-dive)
  explains the concepts, this repo runs the result.
- **GPU serving with vLLM**: continuous batching and high throughput for many
  concurrent users.
- **Bigger models via more memory**: quantized 70B on a 64 GB machine, or
  multi-GPU.
- **Speculative decoding & draft models**: a small model drafts, a big one
  verifies, for faster generation.
- **Embeddings & reranking models locally**: a fully-local RAG stack with no
  hosted calls at all.

---

## From teaching code to production

The teaching shortcuts here are what you'd harden once a local model serves real
traffic:

| This repo's teaching shortcut | In production |
|-------------------------------|---------------|
| One Ollama instance on your laptop | A **GPU serving stack** (vLLM) with batching, behind a load balancer |
| `ensure_server()` exits if it's down | **Health checks, retries, and a hosted fallback** so a down model doesn't drop requests |
| Model tag hard-coded in `.env` | A **model registry** with pinned versions and a rollout process |
| Speed is whatever your laptop does | **Latency budgets** and capacity planning per GPU |
| Trust the local JSON/tool output | **Schema validation + guardrails**: small models drift more, so check more |
| Local = implicitly private | **Real data governance**: who can reach the endpoint, what's logged, retention |

The general ops machinery (observability, cost, reliability, caching, guardrails,
prompt versioning, eval gates) is built from scratch and wired into one running app
in **[Production](https://github.com/alexvervloet/ai-in-production-deep-dive)** (#8 in the
series), which runs offline on a mock provider.

---

## File map

```
check_setup.py              ← run first: Python, packages, and is a server up?
README.md                   ← this guide
EXERCISES.md                ← predict-then-run prompts, one per section
local/                      ← the tiny from-scratch library (read it!)
  providers.py              ← THE trick: the OpenAI SDK pointed at localhost
  sizing.py                 ← offline calculator: params × bits → memory, will it fit?
hands_on/
  local_chat.py             ← capstone: a streaming, fully-local chat CLI
examples/
  01_quant_math.py          ← will it fit? sizing & quantization (offline, no server)
  02_first_local_request.py ← the same SDK, one changed base_url
  03_pick_a_model.py        ← params, families, and what fits your machine
  04_quantization_tradeoff.py ← size vs. quality vs. speed
  05_serving_engines.py     ← Ollama vs. llama.cpp vs. vLLM (probes what's up)
  06_performance.py         ← time-to-first-token vs. tokens/sec
  07_embeddings.py          ← local embeddings + cosine similarity
  08_structured_and_tools.py ← JSON output & tool calling (and their caveats)
  09_local_vs_hosted.py     ← the decision, with your own latency measured
  10_run_the_series_locally.py ← point the other repos here; a local RAG loop
```

(`out/` is created by some examples and is git-ignored.)

---

## Troubleshooting

Run `python check_setup.py` first; it catches most problems. Then, by symptom:

| What you see | What it means / the fix |
|--------------|-------------------------|
| `No server is answering at http://localhost:11434/v1` | No runtime is up. Install/start Ollama and `ollama pull llama3.2`. Or set `OPENAI_BASE_URL` to your engine's port. |
| The call fails with "model not found" | The tag isn't pulled. `ollama list` shows what you have; `ollama pull <tag>` to get it. Tags must match exactly. |
| First response is very slow, later ones fast | Normal: the first call **loads the model into memory** and processes the prompt. That's TTFT (Section 7), not a bug. |
| Generation is painfully slow / the machine swaps | The model spilled out of RAM/VRAM. Use a **smaller model or lower quant**, and check the fit with Section 2 first. |
| Embeddings example errors | The embed model isn't pulled: `ollama pull nomic-embed-text`. |
| Tool calling just returns text | Your model is weak at tools. Try `qwen2.5` or `llama3.1`; reliability tracks model size (Section 9). |
| `ModuleNotFoundError` (openai / rich) | Dependencies aren't installed or the venv isn't active. `source .venv/bin/activate` then `pip install -r requirements.txt`. |
| `SyntaxError` / odd type errors on startup | You're likely on Python 3.9 or older; this repo needs 3.10+. `check_setup.py` confirms your version. |

Still stuck? Every file is small and self-contained. Open it, read the docstring
at the top, and run it. [local/providers.py](local/providers.py) is the whole story:
the OpenAI SDK, one changed URL.

---

## The series

This is one of sixteen standalone, hands-on deep dives into building with LLM APIs: eight core, plus eight bonus dives.
Each one stands on its own, with its own setup, examples, and capstone, and they
all share the same house style: provider-agnostic where it makes sense, built from
scratch (no frameworks), offline-first examples, and a real capstone. Do them in
any order; this sequence builds naturally:

1. [OpenAI API](https://github.com/alexvervloet/openai-api-deep-dive): the API from zero
2. [Claude API](https://github.com/alexvervloet/claude-api-deep-dive): the same ideas, the Anthropic way
3. [Prompt Engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive): shape model behavior with better prompts
4. [RAG](https://github.com/alexvervloet/rag-deep-dive): answer questions over your own documents
5. [Evals](https://github.com/alexvervloet/evals-deep-dive): measure whether a change actually helps
6. [Agents](https://github.com/alexvervloet/agents-deep-dive): give a model tools and a loop so it can act
7. [Prompt Injection & Guardrails](https://github.com/alexvervloet/prompt-injection-deep-dive): attack and defend all of the above
8. [Production](https://github.com/alexvervloet/ai-in-production-deep-dive): operate one app end to end

**Bonus dives**, standalone and slotting in where they're most useful:

- [Context Engineering](https://github.com/alexvervloet/context-engineering-deep-dive): manage what's in the window: memory, compaction, assembly
- [Multimodal](https://github.com/alexvervloet/multimodal-deep-dive): images & audio, not just text
- [Fine-tuning](https://github.com/alexvervloet/fine-tuning-deep-dive): teach a model new behavior by example
- [MCP](https://github.com/alexvervloet/mcp-deep-dive): serve tools, data & prompts to any LLM over a standard protocol
- [Local Models](https://github.com/alexvervloet/local-models-deep-dive): run open-weight models on your own machine
- [Agent Harnesses](https://github.com/alexvervloet/agent-harness-deep-dive): build on the loop: hooks, permissions, sandboxing, subagents
- [Realtime Voice](https://github.com/alexvervloet/realtime-voice-deep-dive): low-latency speech-to-speech agents
- [Observability](https://github.com/alexvervloet/observability-deep-dive): watch a running app over time: drift, quality, alerting, the flywheel

**Local Models is a bonus dive in the series.** It pairs naturally with the two API
dives (#1-2), since it's their code with one changed `base_url`, and with
[Fine-tuning](https://github.com/alexvervloet/fine-tuning-deep-dive), which explains
training the open weights you run here. Section 11 lets you run the *whole* series
locally at zero cost.
