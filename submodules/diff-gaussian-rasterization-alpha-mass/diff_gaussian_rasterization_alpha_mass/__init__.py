"""Dedicated fused all-contributor mass backend.

This package deliberately has a different module name from the standard and
max-contributor rasterizers so a stale binary cannot be imported by accident.
"""

from . import _C

KERNEL_VERSION = "alpha-mass-fused-v5-warp-reduced-float"


def accumulate_alpha_mass(*args):
    return _C.accumulate_alpha_mass(*args)


__all__ = ["KERNEL_VERSION", "accumulate_alpha_mass", "_C"]
