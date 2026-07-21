"""
Example 07: embeddings, locally: turn text into vectors, for free.

Embeddings (text → a vector of numbers, so similar meanings sit close together)
power search, RAG, clustering, and classification. They also run locally, through
the same OpenAI-compatible endpoint, just a different model. With Ollama:
`ollama pull nomic-embed-text`. No per-token embedding bill, ever.

This script embeds a few sentences with your LOCAL embedding model and ranks them
by cosine similarity to a query: the exact core of the RAG deep dive, but $0 and
offline. (The cosine function is five lines of standard library, so you can see
there's no magic.)

    python examples/07_embeddings.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import providers

DOCS = [
    "Quantization stores model weights in fewer bits to save memory.",
    "Llamas are domesticated South American camelids.",
    "The KV cache grows with context length and uses extra RAM.",
    "Espresso is brewed by forcing hot water through ground coffee.",
    "Ollama serves local models over an OpenAI-compatible API.",
]
QUERY = "How do I make a model use less memory on my laptop?"


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed(texts):
    """Embed a list of strings via the local server's embeddings endpoint."""
    resp = providers.client().embeddings.create(model=providers.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    providers.ensure_server()
    print(f"Embedding with: {providers.EMBED_MODEL}  (via {providers.describe()})\n")

    try:
        doc_vecs = embed(DOCS)
        (query_vec,) = embed([QUERY])
    except Exception as exc:
        print(f"Embedding call failed: {exc}")
        print(f"Make sure the embed model is pulled:  ollama pull {providers.EMBED_MODEL}")
        return

    scored = sorted(
        ((cosine(query_vec, dv), doc) for dv, doc in zip(doc_vecs, DOCS)),
        reverse=True,
    )

    print(f"query: {QUERY}\n")
    print("documents ranked by similarity:")
    for score, doc in scored:
        print(f"  {score:.3f}  {doc}")
    print("\nThe top hit is about memory/quantization, retrieved by MEANING, not")
    print("keywords. Wire this into a chunk store and you have local RAG (Example 10).")


if __name__ == "__main__":
    main()
