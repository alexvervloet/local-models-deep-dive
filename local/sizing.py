"""
local/sizing.py: will this model fit on my machine? (pure offline, no server).

The single most useful number before you `ollama pull` anything: **how much
memory will this model need?** It comes from one back-of-the-envelope formula,
and getting it wrong is the #1 reason a local model is painfully slow (it spilled
out of RAM/VRAM) or won't load at all.

The formula:

    memory (bytes) ≈ parameters × bytes_per_parameter

`bytes_per_parameter` is set by the **quantization** (Section 4): fp16 is 2 bytes,
8-bit is ~1 byte, 4-bit is ~0.5 bytes. So a 7-billion-parameter model is roughly:

    fp16:  7e9 × 2.0  ≈ 14 GB
    q8:    7e9 × 1.0  ≈  7 GB
    q4:    7e9 × 0.5  ≈  3.5 GB     ← why 4-bit is what fits on a laptop

On top of the weights you need headroom for the **KV cache** (grows with context
length) and the runtime itself, and this module estimates that too. Everything here
is arithmetic on the standard library: no model, no server, no key, no network.
"""

from dataclasses import dataclass

# Bytes per parameter for the common quantization levels. These are the
# *effective* averages a real GGUF file lands near (quantized formats keep a few
# tensors at higher precision, so q4 is ~0.5-0.65 in practice, not exactly 0.5).
BYTES_PER_PARAM = {
    "fp16": 2.0,   # half precision, what most models are released in
    "bf16": 2.0,   # same size, different layout
    "q8": 1.0,     # 8-bit, near-lossless, half the size of fp16
    "q6": 0.75,    # 6-bit, a common "high quality, smaller" sweet spot
    "q5": 0.65,    # 5-bit
    "q4": 0.5,     # 4-bit, the laptop default; big size win, small quality cost
    "q3": 0.4,     # 3-bit, getting risky on quality
    "q2": 0.3,     # 2-bit, usually too lossy; here for the curve
}

# A rough overhead the runtime adds on top of weights + KV (activations, the
# program itself, fragmentation). A small flat fudge factor keeps estimates from
# being optimistic.
_RUNTIME_OVERHEAD_GB = 0.8


@dataclass
class Estimate:
    """The memory an estimate breaks down into, all in gigabytes."""

    params_b: float          # parameters, in billions
    quant: str               # quantization key, e.g. "q4"
    weights_gb: float        # the weights themselves
    kv_cache_gb: float       # the KV cache at the chosen context length
    overhead_gb: float       # runtime overhead
    context_tokens: int      # context length the KV estimate assumes

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.kv_cache_gb + self.overhead_gb


def _bytes_per_param(quant: str) -> float:
    q = quant.lower()
    if q not in BYTES_PER_PARAM:
        raise ValueError(
            f"unknown quant {quant!r}; pick one of {', '.join(BYTES_PER_PARAM)}"
        )
    return BYTES_PER_PARAM[q]


def weights_gb(params_b: float, quant: str = "q4") -> float:
    """GB just for the model weights: params × bytes-per-param.

    `params_b` is in BILLIONS (7 for a 7B model). This is the dominant term and
    often the only one people remember, but the KV cache (below) bites at long
    context.
    """
    return params_b * 1e9 * _bytes_per_param(quant) / 1e9  # = params_b * bytes_per_param


def kv_cache_gb(params_b: float, context_tokens: int = 4096) -> float:
    """A rough GB estimate for the KV cache at a given context length.

    The KV cache stores keys+values for every token in context, across every
    layer. The exact size needs the architecture (layers, heads, head_dim); we
    approximate from parameter count, which tracks model dimensions well enough
    for a "will it fit?" check. It scales LINEARLY with context length, which is
    why a long context can double your memory before you notice.
    """
    # Bytes of fp16 KV per token, per billion params. Calibrated to land near real
    # modern small models (Llama 3.x / Qwen2.5), which use grouped-query attention
    # (GQA) and so have a much smaller KV cache than older multi-head models. This
    # is a "will it fit?" estimate, not an architecture-exact figure.
    per_token_per_b_bytes = 35_000
    return params_b * context_tokens * per_token_per_b_bytes / 1e9


def model_memory_gb(params_b: float, quant: str = "q4", context_tokens: int = 4096) -> Estimate:
    """Full memory estimate: weights + KV cache + runtime overhead.

    This is the number to compare against your machine's RAM (CPU inference) or
    VRAM (GPU inference). Returns an `Estimate` so callers can show the breakdown.
    """
    w = weights_gb(params_b, quant)
    kv = kv_cache_gb(params_b, context_tokens)
    return Estimate(
        params_b=params_b,
        quant=quant.lower(),
        weights_gb=w,
        kv_cache_gb=kv,
        overhead_gb=_RUNTIME_OVERHEAD_GB,
        context_tokens=context_tokens,
    )


def fits_in(available_gb: float, params_b: float, quant: str = "q4", context_tokens: int = 4096) -> bool:
    """True if a model of this size/quant/context fits in `available_gb`.

    Use your free RAM for CPU inference, or your GPU's VRAM for GPU inference.
    Leave some slack; an estimate that *just* fits will likely swap and crawl.
    """
    return model_memory_gb(params_b, quant, context_tokens).total_gb <= available_gb


def largest_quant_that_fits(available_gb: float, params_b: float, context_tokens: int = 4096) -> str | None:
    """The highest-quality quant of a given model that fits in `available_gb`.

    Walks fp16 → q2 (best quality first) and returns the first that fits, or None
    if even the most aggressive quant is too big (you need a smaller model). This
    is the practical question: 'given my RAM, how good a version of this model can
    I run?'
    """
    for quant in ("fp16", "q8", "q6", "q5", "q4", "q3", "q2"):
        if fits_in(available_gb, params_b, quant, context_tokens):
            return quant
    return None


def format_estimate(est: Estimate) -> str:
    """A human-readable one-block summary of an estimate."""
    return (
        f"{est.params_b:g}B @ {est.quant} (ctx {est.context_tokens}):\n"
        f"  weights      {est.weights_gb:6.1f} GB\n"
        f"  KV cache     {est.kv_cache_gb:6.1f} GB\n"
        f"  overhead     {est.overhead_gb:6.1f} GB\n"
        f"  ------------------------\n"
        f"  TOTAL        {est.total_gb:6.1f} GB"
    )
