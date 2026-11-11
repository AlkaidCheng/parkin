"""Tests of the types for annotations."""

import typing
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

import parkin


def test_module_lists_its_types() -> None:
    assert parkin.typing.__all__ == ["FloatArray", "ParticlesLike"]
    for name in parkin.typing.__all__:
        assert hasattr(parkin.typing, name)


def test_particles_like_names_the_particles_class() -> None:
    hints = typing.get_type_hints(parkin.as_particles)
    assert hints["return"] is parkin.Particles
    assert typing.get_args(hints["data"]) == (
        parkin.Particles,
        npt.NDArray[np.void],
        Mapping[str, npt.NDArray[Any]],
    )
