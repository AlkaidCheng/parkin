"""Compiled core of parkin."""

from collections.abc import Sequence
from typing import Any

from numpy.typing import NDArray

def compiler() -> str:
    """Return the id and version of the compiler that built the core."""

def detected_cpu_features() -> list[str]:
    """Return the SIMD instruction sets the running CPU supports."""

def run_map(
    kernel: str,
    inputs: Sequence[NDArray[Any]],
    out: NDArray[Any],
    thread_count: int,
    block_size: int,
    instruction_set: str | None,
) -> None:
    """Apply a map kernel to one-dimensional float columns and write the result to out."""

def supported_instruction_sets() -> list[str]:
    """Return the instruction sets kernels can run with, narrowest first."""

def xsimd_version() -> str:
    """Return the version of the xsimd headers the core was compiled with."""
