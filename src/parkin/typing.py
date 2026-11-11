"""Types for annotations, in parkin and in code that uses it.

FloatArray
    A NumPy array of floating-point numbers of any precision.
ParticlesLike
    What `parkin.as_particles` reads: a `parkin.Particles`, a structured NumPy
    array, or a mapping of names to NumPy arrays.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np
import numpy.typing as npt

# Every module of parkin may import this one, so it imports a class of parkin
# only for the type checker and names it in quotes.
if TYPE_CHECKING:
    from parkin.particles import Particles

__all__ = ["FloatArray", "ParticlesLike"]

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]

ParticlesLike: TypeAlias = Union[
    "Particles", npt.NDArray[np.void], Mapping[str, npt.NDArray[Any]]
]
