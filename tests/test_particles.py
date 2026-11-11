"""parkin.Particles and parkin.as_particles: the standard format of input."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import parkin

COMPONENT_NAMES = ("pt", "eta", "phi", "mass")
FLOAT_DTYPES = (np.float32, np.float64)


def columns_of(count: int, dtype: Any = np.float64) -> dict[str, npt.NDArray[Any]]:
    """Four separate arrays of distinct values, by component name."""
    values = np.arange(4 * count, dtype=dtype).reshape(4, count)
    return {name: values[row].copy() for row, name in enumerate(COMPONENT_NAMES)}


def records_of(
    columns: dict[str, npt.NDArray[Any]], extra_fields: list[tuple[str, str]]
) -> npt.NDArray[Any]:
    """The columns as an array of records, with further fields in front."""
    fields = extra_fields + [(name, column.dtype) for name, column in columns.items()]
    records = np.zeros(len(next(iter(columns.values()))), dtype=fields)
    for name, column in columns.items():
        records[name] = column
    return records


# --- Particles --------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_particles_hold_a_view_of_each_array(dtype: Any) -> None:
    columns = columns_of(5, dtype)
    particles = parkin.Particles(**columns)
    assert len(particles) == 5
    assert particles.dtype == dtype
    for name in COMPONENT_NAMES:
        component = getattr(particles, name)
        assert type(component) is np.ndarray
        assert component.base is columns[name]
        assert np.array_equal(component, columns[name])

    columns["pt"][0] = -1.0
    assert particles.pt[0] == -1.0


def test_components_are_given_by_keyword() -> None:
    columns = columns_of(3)
    with pytest.raises(TypeError, match="positional"):
        parkin.Particles(*columns.values())
    with pytest.raises(TypeError, match="mass"):
        parkin.Particles(pt=columns["pt"], eta=columns["eta"], phi=columns["phi"])


def test_particles_cannot_be_given_other_components() -> None:
    particles = parkin.Particles(**columns_of(3))
    with pytest.raises(AttributeError):
        particles.pt = np.zeros(3)
    with pytest.raises(AttributeError):
        particles.charge = np.zeros(3)


def test_repr_states_components_length_and_dtype() -> None:
    particles = parkin.Particles(**columns_of(7, np.float32))
    assert repr(particles) == "<Particles (pt, eta, phi, mass), length 7, float32>"


def test_strides_of_the_arrays_are_kept() -> None:
    columns = columns_of(12)
    every_third = parkin.Particles(**{name: c[::3] for name, c in columns.items()})
    assert every_third.pt.strides == (24,)
    backwards = parkin.Particles(**{name: c[::-1] for name, c in columns.items()})
    assert backwards.eta.strides == (-8,)
    repeated = np.broadcast_to(np.float64(0.0), (12,))
    massless = parkin.Particles(**{**columns, "mass": repeated})
    assert massless.mass.strides == (0,)


def test_memory_map_is_held_as_a_plain_array(tmp_path: Path) -> None:
    columns = columns_of(6)
    paths = {name: tmp_path / f"{name}.bin" for name in COMPONENT_NAMES}
    for name, column in columns.items():
        column.tofile(paths[name])
    mapped = {
        name: np.memmap(path, dtype=np.float64, mode="r")
        for name, path in paths.items()
    }
    particles = parkin.Particles(**mapped)
    assert type(particles.phi) is np.ndarray
    assert not particles.phi.flags.writeable
    is_equal = np.array_equal(particles.phi, columns["phi"])
    # Windows cannot remove a file while it is mapped.
    del particles, mapped
    assert is_equal


@pytest.mark.parametrize("name", COMPONENT_NAMES)
def test_component_that_is_no_float_array(name: str) -> None:
    columns = columns_of(4)
    with pytest.raises(TypeError, match=f"{name} must be a NumPy array, got list"):
        parkin.Particles(**{**columns, name: [0.0, 1.0, 2.0, 3.0]})
    for dtype in ("i8", "f2", ">f8", "c16"):
        with pytest.raises(TypeError, match=f"{name} must be float32 or float64"):
            parkin.Particles(**{**columns, name: np.zeros(4, dtype=dtype)})


def test_masked_component_is_refused() -> None:
    columns = columns_of(4)
    masked = np.ma.masked_array(columns["eta"], mask=[False, True, False, False])
    with pytest.raises(TypeError, match="eta is a masked array"):
        parkin.Particles(**{**columns, "eta": masked})


def test_components_that_do_not_match() -> None:
    columns = columns_of(4)
    with pytest.raises(ValueError, match="phi must be one-dimensional"):
        parkin.Particles(**{**columns, "phi": np.zeros((4, 1))})
    with pytest.raises(
        ValueError,
        match="one dtype and one length: pt has float64 and 4 elements, "
        "phi has float64 and 5 elements",
    ):
        parkin.Particles(**{**columns, "phi": np.zeros(5)})
    with pytest.raises(ValueError, match="mass has float32 and 4 elements"):
        parkin.Particles(**{**columns, "mass": np.zeros(4, dtype="f4")})


# --- as_particles -----------------------------------------------------------


def test_particles_are_returned_as_they_are() -> None:
    particles = parkin.Particles(**columns_of(3))
    assert parkin.as_particles(particles) is particles


def test_mapping_is_read_in_place() -> None:
    columns = columns_of(5)
    particles = parkin.as_particles({**columns, "charge": np.ones(2, dtype="i1")})
    for name in COMPONENT_NAMES:
        assert getattr(particles, name).base is columns[name]


@pytest.mark.parametrize("array_type", [np.ndarray, np.recarray])
def test_fields_of_records_are_read_in_place(array_type: Any) -> None:
    columns = columns_of(5)
    records = records_of(columns, [("charge", "i1")]).view(array_type)
    particles = parkin.as_particles(records)
    for name in COMPONENT_NAMES:
        component = getattr(particles, name)
        assert type(component) is np.ndarray
        assert np.shares_memory(component, records)
        # The fields of a record are interleaved, so a component is strided.
        assert component.strides == (records.dtype.itemsize,)
        assert not component.flags.c_contiguous
        assert np.array_equal(component, columns[name])


@pytest.mark.parametrize(
    "names",
    [
        ("rho", "eta", "phi", "tau"),
        ("pt", "eta", "phi", "m"),
        ("pt", "eta", "phi", "M"),
    ],
)
def test_field_aliases(names: tuple[str, ...]) -> None:
    columns = columns_of(4)
    renamed = dict(zip(names, columns.values(), strict=True))
    for data in (renamed, records_of(renamed, [])):
        particles = parkin.as_particles(data)
        for name in COMPONENT_NAMES:
            assert np.array_equal(getattr(particles, name), columns[name])


def test_data_in_no_layout_of_particles() -> None:
    with pytest.raises(TypeError, match="data must be a Particles, a structured"):
        parkin.as_particles([1.0, 2.0])
    with pytest.raises(TypeError, match="data must be .* without fields"):
        parkin.as_particles(np.zeros((4, 4)))


def test_missing_and_ambiguous_fields() -> None:
    columns = columns_of(4)
    cartesian = {name: np.zeros(4) for name in ("px", "py", "pz", "E")}
    with pytest.raises(
        ValueError, match="data has no field for the transverse momentum"
    ):
        parkin.as_particles(cartesian)
    with pytest.raises(ValueError, match="data has several fields for the mass"):
        parkin.as_particles({**columns, "m": columns["mass"]})


def test_refused_field_is_named_with_its_source() -> None:
    columns = columns_of(4)
    renamed = {"pt": columns["pt"], "eta": columns["eta"], "phi": columns["phi"]}
    with pytest.raises(TypeError, match="field 'm' of data must be float32 or float64"):
        parkin.as_particles({**renamed, "m": np.zeros(4, dtype="i8")})
    with pytest.raises(
        ValueError, match="field 'pt' of data has float64 and 4 elements, field 'm' of"
    ):
        parkin.as_particles({**renamed, "m": np.zeros(3)})


def test_messages_name_the_argument() -> None:
    with pytest.raises(TypeError, match="^jets must be a Particles"):
        parkin.as_particles([1.0, 2.0], argument_name="jets")
    columns = columns_of(4)
    with pytest.raises(TypeError, match="field 'mass' of jets must be float32"):
        parkin.as_particles(
            {**columns, "mass": np.zeros(4, dtype="i8")}, argument_name="jets"
        )


# --- The module and subclasses ----------------------------------------------


class ChargedParticles(parkin.Particles):
    """Particles that carry a charge besides their components."""

    def __init__(self, *, charge: npt.NDArray[Any], **components: Any) -> None:
        super().__init__(**components)
        self.charge = charge


def test_module_is_public() -> None:
    assert parkin.particles.Particles is parkin.Particles
    assert parkin.particles.as_particles is parkin.as_particles
    assert parkin.particles.__all__ == ["Particles", "as_particles"]
    for name in parkin.particles.__all__:
        assert hasattr(parkin.particles, name)


def test_components_are_listed_by_name() -> None:
    particles = parkin.Particles(**columns_of(3))
    components = particles.components
    assert list(components) == list(COMPONENT_NAMES)
    for name in COMPONENT_NAMES:
        assert components[name] is getattr(particles, name)

    del components["pt"]
    assert list(particles.components) == list(COMPONENT_NAMES)


def test_subclass_is_taken_as_particles() -> None:
    columns = columns_of(3)
    charge = np.array([-1, 1, -1], dtype="i1")
    charged = ChargedParticles(charge=charge, **columns)
    assert isinstance(charged, parkin.Particles)
    assert charged.charge is charge
    assert charged.pt.base is columns["pt"]
    assert parkin.as_particles(charged) is charged
    assert repr(charged) == "<ChargedParticles (pt, eta, phi, mass), length 3, float64>"


def test_subclass_is_built_from_the_components_of_other_particles() -> None:
    particles = parkin.as_particles(records_of(columns_of(3), [("charge", "i1")]))
    charged = ChargedParticles(charge=np.zeros(3, dtype="i1"), **particles.components)
    for name in COMPONENT_NAMES:
        assert np.shares_memory(getattr(charged, name), getattr(particles, name))


def test_subclass_is_checked_like_particles() -> None:
    columns = columns_of(3)
    with pytest.raises(ValueError, match="one dtype and one length"):
        ChargedParticles(charge=np.zeros(3), **{**columns, "phi": np.zeros(4)})
