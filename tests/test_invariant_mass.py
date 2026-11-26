"""parkin.invariant_mass: values, accuracy, input layouts and execution settings."""

from typing import Any

import mpmath
import numpy as np
import numpy.typing as npt
import pytest
import vector

import parkin
from parkin import _core

FIELD_NAMES = ("pt", "eta", "phi", "mass")
FLOAT_DTYPES = (np.float32, np.float64)
INSTRUCTION_SETS = tuple(_core.supported_instruction_sets())

# The accuracy stated on the documentation page of the pair mass, in units in
# the last place of the result.
ERROR_BOUND_ULP = 8.0

# Decimal digits of the reference computation, far beyond the 16 of float64
# so that a reference loses nothing to cancellation.
REFERENCE_DIGITS = 60

Columns = tuple[npt.NDArray[Any], ...]


def mapping_of(columns: Columns) -> dict[str, npt.NDArray[Any]]:
    """Four columns as a mapping from field name to array."""
    return dict(zip(FIELD_NAMES, columns, strict=True))


def particles_of(columns: Columns) -> parkin.Particles:
    """Four columns as particles, one array per component."""
    return parkin.Particles(**mapping_of(columns))


def mass_of(columns: Columns) -> npt.NDArray[Any]:
    """The pair mass of eight columns, four per particle."""
    return parkin.invariant_mass(particles_of(columns[:4]), particles_of(columns[4:]))


def exact_mass(row: tuple[float, ...]) -> mpmath.mpf:
    """The mass of one pair, from the four-momentum sum at high precision."""
    with mpmath.workdps(REFERENCE_DIGITS):
        pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2 = (
            mpmath.mpf(float(value)) for value in row
        )
        energy = mpmath.mpf(0)
        px = mpmath.mpf(0)
        py = mpmath.mpf(0)
        pz = mpmath.mpf(0)
        for pt, eta, phi, mass in ((pt1, eta1, phi1, mass1), (pt2, eta2, phi2, mass2)):
            momentum = pt * mpmath.cosh(eta)
            energy += mpmath.sqrt(mass * mass + momentum * momentum)
            px += pt * mpmath.cos(phi)
            py += pt * mpmath.sin(phi)
            pz += pt * mpmath.sinh(eta)
        return mpmath.sqrt(energy * energy - px * px - py * py - pz * pz)


def relative_errors(masses: npt.NDArray[Any], columns: Columns) -> npt.NDArray[Any]:
    """The error of each mass relative to the exact mass of its inputs."""
    errors = []
    with mpmath.workdps(REFERENCE_DIGITS):
        for mass, row in zip(masses, zip(*columns, strict=True), strict=True):
            exact = exact_mass(row)
            errors.append(float(abs(mpmath.mpf(float(mass)) - exact) / exact))
    return np.array(errors)


def errors_in_ulp(masses: npt.NDArray[Any], columns: Columns) -> npt.NDArray[Any]:
    """The error of each mass in units in the last place of the exact mass."""
    errors = []
    with mpmath.workdps(REFERENCE_DIGITS):
        for mass, row in zip(masses, zip(*columns, strict=True), strict=True):
            exact = exact_mass(row)
            ulp = float(np.spacing(masses.dtype.type(float(exact))))
            errors.append(float(abs(mpmath.mpf(float(mass)) - exact)) / ulp)
    return np.array(errors)


def sample_pairs(kind: str, count: int, dtype: Any, seed: int) -> Columns:
    """Random pairs of one kind, within the domain the accuracy is stated for.

    The masses are non-negative, |eta| <= 5 and |phi1 - phi2| <= pi.
    """
    rng = np.random.default_rng(seed)
    pt1 = 10 ** rng.uniform(-2, 4, count)
    eta1 = rng.uniform(-5, 5, count)
    phi1 = rng.uniform(-np.pi, np.pi, count)
    mass1 = np.where(rng.random(count) < 0.3, 0.0, 10 ** rng.uniform(-4, 3, count))
    if kind == "generic":
        pt2 = 10 ** rng.uniform(-2, 4, count)
        eta2 = rng.uniform(-5, 5, count)
        phi2 = phi1 + rng.uniform(-3.0, 3.0, count)
        mass2 = np.where(rng.random(count) < 0.3, 0.0, 10 ** rng.uniform(-4, 3, count))
    elif kind == "collinear":
        # Angular separations down to the resolution of the element type.
        smallest = -7 if dtype == np.float64 else -4
        separation = 10 ** rng.uniform(smallest, -1, count)
        pt2 = pt1 * 10 ** rng.uniform(-1, 1, count)
        eta2 = np.clip(eta1 + separation * rng.normal(size=count), -5, 5)
        phi2 = phi1 + separation * rng.normal(size=count)
        mass1 = np.where(rng.random(count) < 0.5, 0.0, mass1 * 1e-3)
        mass2 = np.where(rng.random(count) < 0.5, 0.0, 10 ** rng.uniform(-4, 0, count))
    elif kind == "soft":
        pt2 = pt1 * 10 ** rng.uniform(-8, -3, count)
        eta2 = rng.uniform(-5, 5, count)
        phi2 = phi1 + rng.uniform(-3.0, 3.0, count)
        mass2 = np.zeros(count)
    elif kind == "heavy":
        pt1 = 10 ** rng.uniform(-3, 0, count)
        pt2 = 10 ** rng.uniform(-3, 0, count)
        eta2 = rng.uniform(-5, 5, count)
        phi2 = phi1 + rng.uniform(-3.0, 3.0, count)
        mass1 = 10 ** rng.uniform(1, 3, count)
        mass2 = 10 ** rng.uniform(1, 3, count)
    else:
        raise ValueError(kind)
    columns = (pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2)
    return tuple(np.ascontiguousarray(column, dtype=dtype) for column in columns)


@pytest.fixture
def pairs() -> Columns:
    """Pairs that span several chunks of the map loop and end in a partial one."""
    return sample_pairs("generic", 2_003, np.float64, seed=1)


@pytest.fixture
def expected(pairs: Columns) -> npt.NDArray[Any]:
    with parkin.execution(threads=1):
        return mass_of(pairs)


# --- Values -----------------------------------------------------------------


@pytest.mark.parametrize("simd", INSTRUCTION_SETS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_closed_form_values(simd: str, dtype: Any) -> None:
    # Rows: back to back and massless (m = 2 pT); at rest (m = m1 + m2); a
    # particle paired with itself (m = 2 m); both null (m = 0).
    p1 = np.array(
        [(45.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 3.0), (7.0, 1.5, 0.5, 2.0), (0.0,) * 4],
        dtype=[(name, dtype) for name in FIELD_NAMES],
    )
    p2 = np.array(
        [
            (45.0, 0.0, np.pi, 0.0),
            (0.0, 0.0, 0.0, 4.0),
            (7.0, 1.5, 0.5, 2.0),
            (0.0,) * 4,
        ],
        dtype=[(name, dtype) for name in FIELD_NAMES],
    )
    with parkin.execution(simd=simd):
        masses = parkin.invariant_mass(p1, p2)
    assert masses.dtype == dtype
    np.testing.assert_allclose(
        masses, [90.0, 7.0, 4.0, 0.0], rtol=4 * np.finfo(dtype).eps
    )


@pytest.mark.parametrize("simd", INSTRUCTION_SETS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("kind", ["generic", "collinear", "soft", "heavy"])
def test_error_is_within_the_stated_bound(kind: str, dtype: Any, simd: str) -> None:
    columns = sample_pairs(kind, 1_500, dtype, seed=20261110)
    with parkin.execution(simd=simd):
        masses = mass_of(columns)
    assert errors_in_ulp(masses, columns).max() < ERROR_BOUND_ULP


@pytest.mark.parametrize("simd", INSTRUCTION_SETS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pair_across_the_phi_seam_is_as_accurate_as_its_angles(
    simd: str, dtype: Any
) -> None:
    # Massless particles at equal eta, a small angle apart, on either side of
    # phi = +-pi. phi1 - phi2 is rounded at the size of 2 pi, which bounds the
    # relative error of the mass by 2 eps / separation on top of the usual one.
    rng = np.random.default_rng(3)
    count = 500
    eps = float(np.finfo(dtype).eps)
    smallest = -6 if dtype == np.float64 else -3
    below = 10 ** rng.uniform(smallest, -1, count)
    above = 10 ** rng.uniform(smallest, -1, count)
    columns = tuple(
        np.ascontiguousarray(column, dtype=dtype)
        for column in (
            10 ** rng.uniform(0, 2, count),
            np.zeros(count),
            np.pi - below,
            np.zeros(count),
            10 ** rng.uniform(0, 2, count),
            np.zeros(count),
            -np.pi + above,
            np.zeros(count),
        )
    )
    separation = 2 * np.pi - (columns[2].astype(np.float64) - columns[6])
    with parkin.execution(simd=simd):
        masses = mass_of(columns)
    allowed = (ERROR_BOUND_ULP + 4 / separation) * eps
    assert np.all(relative_errors(masses, columns) < allowed)


def test_agrees_with_vector(pairs: Columns, expected: npt.NDArray[Any]) -> None:
    first = vector.array(mapping_of(pairs[:4]))
    second = vector.array(mapping_of(pairs[4:]))
    total = first + second
    # vector takes m^2 = E^2 - |p|^2 of the Cartesian sum, whose rounding
    # error scales with E^2, so the squared masses are compared on that scale.
    tolerance = 64 * np.finfo(np.float64).eps * total.t**2
    assert np.all(np.abs(expected**2 - total.mass**2) < tolerance)


@pytest.mark.parametrize("simd", INSTRUCTION_SETS)
def test_spacelike_momenta_have_negative_mass(simd: str) -> None:
    # A negative mass is a spacelike input, and these two nearly collinear
    # ones sum to a spacelike pair. vector gives the reference.
    p1 = {"pt": np.array([5.0]), "eta": np.array([0.3]), "phi": np.array([0.1])}
    p2 = {"pt": np.array([4.0]), "eta": np.array([0.2]), "phi": np.array([0.3])}
    p1["mass"] = np.array([-4.0])
    p2["mass"] = np.array([-3.0])
    with parkin.execution(simd=simd):
        mass = parkin.invariant_mass(p1, p2)
    reference = (vector.array(p1) + vector.array(p2)).mass
    assert reference[0] < 0
    np.testing.assert_allclose(mass, reference, rtol=1e-12)


@pytest.mark.parametrize("simd", INSTRUCTION_SETS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("column", range(8))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_gives_non_finite_mass(
    value: float, column: int, dtype: Any, simd: str
) -> None:
    columns = list(sample_pairs("generic", 5, dtype, seed=2))
    columns[column] = np.full(5, value, dtype=dtype)
    with parkin.execution(simd=simd):
        masses = mass_of(tuple(columns))
    if np.isnan(value):
        assert np.all(np.isnan(masses))
    else:
        assert not np.any(np.isfinite(masses))


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_one_set_of_particles_has_the_masses_it_is_given(dtype: Any) -> None:
    columns = sample_pairs("generic", 300, dtype, seed=17)[:4]
    masses = parkin.invariant_mass(particles_of(columns))
    assert masses.dtype == dtype
    assert np.array_equal(masses, columns[3])
    # The result is a new array, not a view of the argument.
    assert not np.shares_memory(masses, columns[3])
    assert np.array_equal(parkin.invariant_mass(mapping_of(columns)), columns[3])


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_empty_input_gives_empty_result(dtype: Any) -> None:
    masses = mass_of(tuple(np.empty(0, dtype=dtype) for _ in range(8)))
    assert masses.shape == (0,)
    assert masses.dtype == dtype


# --- Input layouts ----------------------------------------------------------


def structured(
    columns: Columns, extra_fields: list[tuple[str, str]]
) -> npt.NDArray[Any]:
    """Four columns as an array of records, with further fields in front."""
    dtype = np.dtype(extra_fields + [(name, columns[0].dtype) for name in FIELD_NAMES])
    records = np.zeros(columns[0].shape[0], dtype=dtype)
    for name, column in zip(FIELD_NAMES, columns, strict=True):
        records[name] = column
    return records


def test_array_of_records(pairs: Columns, expected: npt.NDArray[Any]) -> None:
    p1 = structured(pairs[:4], [])
    p2 = structured(pairs[4:], [])
    assert np.array_equal(parkin.invariant_mass(p1, p2), expected)


def test_packed_records_with_unaligned_fields(
    pairs: Columns, expected: npt.NDArray[Any]
) -> None:
    # A one-byte field in front leaves every float64 field unaligned, 33
    # bytes from the next record's.
    p1 = structured(pairs[:4], [("charge", "i1")])
    p2 = structured(pairs[4:], [("charge", "i1")])
    assert p1.dtype.itemsize == 33
    assert not p1["pt"].flags.aligned
    assert np.array_equal(parkin.invariant_mass(p1, p2), expected)


def test_mappings(pairs: Columns, expected: npt.NDArray[Any]) -> None:
    p1 = mapping_of(pairs[:4])
    p2 = {**mapping_of(pairs[4:]), "charge": np.ones(3)}
    assert np.array_equal(parkin.invariant_mass(p1, p2), expected)


class LabelledParticles(parkin.Particles):
    """Particles that carry a label besides their components."""

    def __init__(self, *, label: str, **components: Any) -> None:
        super().__init__(**components)
        self.label = label


def test_subclass_of_particles(pairs: Columns, expected: npt.NDArray[Any]) -> None:
    p1 = LabelledParticles(label="first", **mapping_of(pairs[:4]))
    p2 = LabelledParticles(label="second", **mapping_of(pairs[4:]))
    assert np.array_equal(parkin.invariant_mass(p1, p2), expected)
    assert np.array_equal(parkin.invariant_mass(p1, particles_of(pairs[4:])), expected)


def test_layouts_together(pairs: Columns, expected: npt.NDArray[Any]) -> None:
    records = structured(pairs[:4], []).view(np.recarray)
    assert np.array_equal(
        parkin.invariant_mass(records, particles_of(pairs[4:])), expected
    )
    assert np.array_equal(
        parkin.invariant_mass(mapping_of(pairs[:4]), structured(pairs[4:], [])),
        expected,
    )


# --- Threads ----------------------------------------------------------------


@pytest.mark.parametrize("simd", INSTRUCTION_SETS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_result_does_not_depend_on_threads_or_block_size(simd: str, dtype: Any) -> None:
    # 10 007 elements: 39 whole chunks of the map loop and a partial one.
    columns = sample_pairs("generic", 10_007, dtype, seed=5)
    with parkin.execution(simd=simd, threads=1):
        reference = mass_of(columns)
    for threads in (1, 2, 3, 8, 33):
        for block_size in (1, 256, 1_000, 4_096, 10**9):
            with parkin.execution(simd=simd, threads=threads, block_size=block_size):
                assert np.array_equal(mass_of(columns), reference), (
                    threads,
                    block_size,
                )


# --- Refused input ----------------------------------------------------------


def test_refused_argument_is_named() -> None:
    particles = particles_of(sample_pairs("generic", 4, np.float64, seed=15)[:4])
    with pytest.raises(TypeError, match="p1 must be a Particles, a structured array"):
        parkin.invariant_mass([1.0, 2.0], particles)
    with pytest.raises(TypeError, match="p2 must be .* without fields"):
        parkin.invariant_mass(particles, np.zeros((4, 4)))
    cartesian = {name: np.zeros(4) for name in ("px", "py", "pz", "E")}
    with pytest.raises(ValueError, match="p2 has no field for the transverse momentum"):
        parkin.invariant_mass(particles, cartesian)
    integers = {name: np.zeros(4, dtype="i8") for name in FIELD_NAMES}
    with pytest.raises(TypeError, match="field 'pt' of p1 must be float32 or float64"):
        parkin.invariant_mass(integers, particles)


def test_number_of_sets_that_cannot_be_computed() -> None:
    particles = particles_of(sample_pairs("generic", 4, np.float64, seed=18)[:4])
    with pytest.raises(TypeError, match="needs at least one set of particles"):
        parkin.invariant_mass()
    with pytest.raises(ValueError, match="takes one or two sets of particles, got 3"):
        parkin.invariant_mass(particles, particles, particles)


def test_particles_that_do_not_match() -> None:
    columns = sample_pairs("generic", 4, np.float64, seed=16)[:4]
    particles = particles_of(columns)
    shorter = particles_of(tuple(column[:3] for column in columns))
    with pytest.raises(
        ValueError, match="p1 and p2 must have one dtype and one length"
    ):
        parkin.invariant_mass(particles, shorter)
    single = particles_of(tuple(column.astype("f4") for column in columns))
    with pytest.raises(
        ValueError, match="p1 and p2 must have one dtype and one length"
    ):
        parkin.invariant_mass(particles, single)
