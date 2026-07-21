"""
local/ : the tiny from-scratch library for running open-weight models locally.

Three small modules, each readable in one sitting:

  providers.py  the keystone: the OpenAI SDK pointed at http://localhost:11434/v1.
                  This is the whole "local" trick, the same code as the API dive,
                  one URL changed. Also a `server_up()` probe and friendly guards.
  sizing.py     a pure-offline calculator: params x bits -> memory, plus a
                  "does this fit my RAM/VRAM?" check. No model, no server, no key.

Import what you need:

    from local import providers
    from local.sizing import model_memory_gb, fits_in
"""

from . import providers, sizing

__all__ = ["providers", "sizing"]
