"""
Example 10: run the rest of the series locally, for $0.

The payoff of "local speaks the OpenAI API": the sibling deep dives (prompt
engineering, RAG, agents, evals) were built on the OpenAI SDK, so they run against
your local model with only an environment change, with no code edits. This script
proves it by running a tiny, self-contained RAG loop (retrieve → ground → answer)
entirely on the local server, exactly like the RAG deep dive's pipeline, but free.

How to point the other repos here (in each repo's .env):

    OPENAI_API_KEY=local            # any non-empty string; ignored locally
    OPENAI_BASE_URL=http://localhost:11434/v1
    MODEL=llama3.2                  # or whatever you pulled

That's the whole change. The prompt-engineering and OpenAI repos already document
this "local model" path; this is the same switch, applied to all of them.

    python examples/10_run_the_series_locally.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import providers

# A tiny "knowledge base": stand-in for the RAG deep dive's document corpus.
CORPUS = [
    "Acme Cloud's Plus plan costs $20 per month, billed monthly.",
    "The Free plan includes 5 GB of storage and community support.",
    "Annual billing on any paid plan gives two months free.",
    "Support response time on the Plus plan is within one business day.",
]
QUESTION = "How much is the Plus plan for a whole year if I pay annually?"


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed(texts):
    resp = providers.client().embeddings.create(model=providers.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    providers.ensure_server()
    print(f"Running a local RAG loop on {providers.describe()}\n")

    # 1) RETRIEVE: embed the corpus + question, pick the most similar chunks.
    try:
        doc_vecs = embed(CORPUS)
        (q_vec,) = embed([QUESTION])
    except Exception as exc:
        print(f"Embedding failed: {exc}\n  Pull the embed model: ollama pull {providers.EMBED_MODEL}")
        return
    ranked = sorted(zip((cosine(q_vec, d) for d in doc_vecs), CORPUS), reverse=True)
    top = [doc for _, doc in ranked[:2]]

    print("retrieved context:")
    for c in top:
        print(f"  • {c}")

    # 2) GROUND + ANSWER: feed only the retrieved context to the local model.
    context = "\n".join(f"- {c}" for c in top)
    messages = [
        {"role": "system", "content": "Answer ONLY from the context. If it's not there, say you don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {QUESTION}"},
    ]
    print("\nlocal model's grounded answer:")
    print(" ", providers.chat(messages, temperature=0.2))

    print("\n---")
    print("That was retrieve -> ground -> answer, the RAG deep dive's whole pipeline,")
    print("running on your laptop at zero cost. Set OPENAI_BASE_URL in the other repos")
    print("and they run locally too: prompt engineering, agents, evals, all of it.")


if __name__ == "__main__":
    main()
