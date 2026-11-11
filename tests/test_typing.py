"""Tests of the types for annotations."""

import typing
from collections.abc import Mapping

import numpy as np

import parkin


def test_module_lists_its_types() -> None:
    assert parkin.typing.__all__ == ["FloatArray", "ParticlesLike"]
    for name in parkin.typing.__all__:
        assert hasattr(parkin.typing, name)


def test_particles_like_names_the_particles_class() -> None:
    hints = typing.get_type_hints(parkin.as_particles)
    assert hints["return"] is parkin.Particles
    particles, records, mapping = typing.get_args(hints["data"])
    assert particles is parkin.Particles
    assert typing.get_origin(records) is np.ndarray
    assert typing.get_origin(mapping) is Mapping
