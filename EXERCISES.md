# Exercises: make the learning stick

Reading code teaches you less than *predicting* what it will do and then checking.
This file turns each section of the [README](README.md) into a few quick
active-recall prompts.

How to use it: work the section first, then come back. **Commit to an answer
before you run or reveal.** The prediction is where the learning happens. Answers
are hidden behind ▸ toggles.

> The whole repo is **free**: there's no API key and no per-token bill anywhere.
> Section 2 needs no server at all; the rest just need a local runtime up.

---

## Section 2: Will it fit? **(offline)**

**Recall.** Write the memory formula, and give bytes-per-parameter for fp16, q8,
and q4.

<details><summary>▸ Answer</summary>

`memory ≈ parameters × bytes-per-parameter` (plus KV cache + overhead). fp16 ≈
**2.0**, q8 ≈ **1.0**, q4 ≈ **0.5** bytes per parameter. So a 7B model is ~14 GB at
fp16 but ~3.5 GB of weights at q4, which is why q4 is the laptop default.
</details>

**Predict, then run.** Before running `examples/01_quant_math.py`: you bump an 8B
model's context from 4k to 32k tokens. Roughly what happens to memory, and why?

<details><summary>▸ Answer</summary>

It **grows a lot**. The **KV cache** scales linearly with context length, and at
32k it can rival or exceed the weights themselves. The weights are fixed; the
context is the lever. The example prints this curve.
</details>

---

## Section 3: Your first local request

**Recall.** Compared to a hosted OpenAI call, what *exactly* is different in the
code for a local call?

<details><summary>▸ Answer</summary>

Only the client's **`base_url`** (pointed at `http://localhost:11434/v1`), and the
`api_key`, which is required by the SDK but **ignored** by the local server, so any
string works. The messages, parameters, and response shape are identical. That's
the entire repo in one sentence.
</details>

**Predict.** You run `examples/02_first_local_request.py` with no server running.
What happens: a crash, a hang, or something else?

<details><summary>▸ Answer</summary>

Something else: `providers.ensure_server()` probes the port first, prints
**start-up instructions**, and exits cleanly (code 0). No stack trace, no charge 
because there's nothing to charge. Graceful degradation is the local analogue of
the sibling repos' "missing API key" guard.
</details>

---

## Section 4: Picking a model

**Recall.** Two models: a 13B at q3 and a 7B at q6, and they use about the same
memory. Which is usually the better pick, and why?

<details><summary>▸ Answer</summary>

Usually the **7B at q6**. A higher-quality quant of a *smaller* model tends to beat
a heavily-crushed quant of a bigger one. Below ~q4 quality drops off, and q3 of a
13B may be noticeably degraded. Rule of thumb: smaller model + q4/q6 over bigger
model + q2/q3.
</details>

---

## Section 5: Quantization tradeoff

**Predict, then run.** Going from q8 to q4 on the same model: what happens to file
size, memory, generation speed, and quality?

<details><summary>▸ Answer</summary>

Size and memory roughly **halve**; generation often gets **a bit faster** (less data
to move); quality drops **a little**. For most tasks q4 keeps most of the quality at
a quarter of fp16's size, which is why it's the default. `examples/04_...` shows
the table and (with a server) measures tokens/sec.
</details>

---

## Section 6: Serving engines

**Recall.** You need a high-throughput endpoint serving many concurrent users on a
GPU box. Which engine, and why not Ollama?

<details><summary>▸ Answer</summary>

**vLLM**, which is built for serving at scale (continuous batching, high throughput).
Ollama is fantastic for one user on a laptop but isn't aimed at many-user GPU
serving. And **your code is identical either way**. Only the `base_url`
changes.
</details>

---

## Section 7: Performance

**Recall.** Name the two speeds of local inference and which one a longer *prompt*
mainly affects.

<details><summary>▸ Answer</summary>

**Time-to-first-token** (TTFT, the model *reading/prefilling* your prompt) and
**generation speed** (tokens/sec once it starts). A longer prompt mainly grows
**TTFT**; generation speed barely changes. And the very first call after the model
loads is always slowest.
</details>

**Predict, then run.** In `examples/06_performance.py`, why does the long-prompt run
show a bigger pause *before* text, but stream at about the same speed once it
starts?

<details><summary>▸ Answer</summary>

Because the extra time is **prompt processing** (reading all those tokens), which is
separate from **generation**. More input → longer TTFT; the per-token generation
rate is set by the model and hardware, not the prompt length.
</details>

---

## Section 8: Embeddings **(local)**

**Recall.** What changes about your code to get embeddings from a local model
instead of a hosted one, and what's the recurring cost?

<details><summary>▸ Answer</summary>

Just the **model name** (an embedding model like `nomic-embed-text`) on the same
local endpoint. The recurring cost is **zero**: no per-token embedding bill. This
is the core of local RAG.
</details>

---

## Section 9: Structured output & tools

**Predict.** You ask a small local model for "ONLY JSON" and it returns the JSON
wrapped in ```` ```json ```` fences. Is that a failure? What does the example do?

<details><summary>▸ Answer</summary>

Not a failure. Small models drift more than hosted ones. The example **strips the
fences before `json.loads`**, the same defensive parsing you learned for hosted
models. Lower temperature, a tighter prompt, or a stronger model (qwen2.5,
llama3.1) all help reliability.
</details>

---

## Section 10: Local vs. hosted

**Recall.** Give two things local wins on and two things hosted wins on.

<details><summary>▸ Answer</summary>

**Local wins:** privacy/data control, cost at volume, offline use, no rate limits.
**Hosted wins:** peak answer quality (the biggest models don't fit locally), zero
ops, elastic scale, day-one access to new models. The best design is often **both**
using local for the common/private/high-volume path and hosted for the hard cases.
</details>

---

## Section 11: Run the series locally

**Recall.** What three lines in a sibling repo's `.env` make it run against your
local model, and which one is a no-op locally?

<details><summary>▸ Answer</summary>

`OPENAI_BASE_URL=http://localhost:11434/v1`, `MODEL=<your local tag>`, and
`OPENAI_API_KEY=local`. The **API key is the no-op**: the local server ignores it,
but the SDK insists on a non-empty string. No code changes; the prompt-engineering,
RAG, agents, and evals repos all run locally for $0.
</details>

---

## Capstone: `local_chat.py`

**Do.** Run `python hands_on/local_chat.py --fit 8` (offline), then start a chat.
What did `--fit` tell you, and what's the tokens/sec line after each reply
measuring?

<details><summary>▸ Answer</summary>

`--fit 8` printed an **offline memory estimate** for an 8B model (weights + KV +
overhead) and whether it fits 16 GB, which is Section 2 without a server. The tokens/sec
line is the **generation speed** (Section 7) of your machine on each reply, plus the
time-to-first-token.
</details>

**Stretch.** Pull a second model (`ollama pull qwen2.5`) and chat with both via
`--model`. Do you feel the size/speed/quality tradeoff from Sections 4–5? Then point
a *sibling* repo at your local server (Section 11) and run its capstone for free 
the first time the RAG or agents dive runs with no key and no bill, "local is an ops
choice, not a new API" has fully landed.

---

### Where to take it next

Build something you'd actually keep: a fully-local note summarizer, a private RAG
over your own files, or a coding helper, all $0 to run and with nothing leaving
your machine. The moment you stop thinking about per-token cost while iterating,
the point of local has clicked.
