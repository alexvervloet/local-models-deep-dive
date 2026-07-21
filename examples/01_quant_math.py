"""
Example 01: will this model fit on my machine? OFFLINE, FREE, no server.

The first and most useful skill in running models locally is arithmetic, not
code: before you download anything, work out how much memory it needs. Get this
wrong and the model either won't load or spills out of RAM and crawls. This script
needs no server and no model; it's pure calculation on `local/sizing.py`.

The one formula:

    memory ≈ parameters × bytes-per-parameter   (+ KV cache + a little overhead)

`bytes-per-parameter` is set by the QUANTIZATION: fp16 = 2.0, 8-bit ≈ 1.0,
4-bit ≈ 0.5. That's why a 4-bit ("q4") build is what fits on a laptop: same model,
a quarter the size of fp16, for a small quality cost.

    python examples/01_quant_math.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import sizing


def main():
    print("How model memory works: parameters × bytes-per-parameter.\n")

    # 1) The same 8B model at every quantization. Watch the weights shrink.
    print("An 8B model, at each quantization level (weights only):")
    for quant in ("fp16", "q8", "q6", "q5", "q4", "q3"):
        gb = sizing.weights_gb(8, quant)
        bar = "█" * round(gb)
        print(f"  {quant:>4}  {gb:5.1f} GB  {bar}")
    print("  -> 4-bit is ~1/4 the size of fp16, for a small quality cost.\n")

    # 2) Full estimates (weights + KV cache + overhead) for common sizes at q4.
    print("Full memory estimate at q4, 4k context (this is what must fit in RAM/VRAM):")
    for params in (3, 7, 8, 13, 70):
        est = sizing.model_memory_gb(params, "q4", 4096)
        print(f"  {params:>2}B   {est.total_gb:5.1f} GB total"
              f"   (weights {est.weights_gb:.1f} + KV {est.kv_cache_gb:.1f} + overhead {est.overhead_gb:.1f})")
    print()

    # 3) The KV cache grows with context length: a quiet memory eater.
    print("The KV cache scales with context length (8B model):")
    for ctx in (2048, 8192, 32768, 131072):
        est = sizing.model_memory_gb(8, "q4", ctx)
        print(f"  ctx {ctx:>6}  ->  {est.total_gb:5.1f} GB total  (KV alone: {est.kv_cache_gb:.1f} GB)")
    print("  -> a long context can cost more memory than the weights.\n")

    # 4) The practical question: given MY RAM, how good a model can I run?
    print("Given your available memory, the best quant of an 8B model that fits:")
    for ram in (8, 16, 32, 64):
        quant = sizing.largest_quant_that_fits(ram, 8, 4096)
        verdict = f"run it at {quant}" if quant else "too big; pick a smaller model"
        print(f"  {ram:>2} GB free  ->  8B: {verdict}")
    print()

    print("Try it for your machine: edit the numbers above, or import local.sizing")
    print("in a REPL and call sizing.fits_in(your_free_gb, params_b, quant).")
    print("\n(Offline and free: no server, no model, no key was needed.)")


if __name__ == "__main__":
    main()
