# Modified for RLinf: lazy dataset imports and PPO-compatible MoE replay.
# Licensed under Apache-2.0; see LICENSE.

"""VLA transforms remain usable without loading offline dataset dependencies."""

from importlib import import_module


__all__ = ["VLADataset", "MultiVLADataset"]
_EXPORTS = {"VLADataset": ".base_dataset", "MultiVLADataset": ".multi_vla_dataset"}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
