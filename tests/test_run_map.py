"""parkin._core.run_map: the columns it borrows, its threads and what it refuses."""

import multiprocessing
import sys
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from parkin import _core

FLOAT_DTYPES = (np.float32, np.float64)
INSTRUCTION_SETS = tuple(_core.supported_instruction_sets())

# The kernel the tests run: the mass of a pair of particles, which reads the
# transverse momentum, pseudorapidity, azimuthal angle and mass of each.
KERNEL = "pair_mass_ptetaphim"
COMPONENT_NAMES = ("pt", "eta", "phi", "mass")

# Elements a thread takes at a time where a test does not vary it: sixteen
# chunks of the map loop.
BLOCK_SIZE = 4_096

Columns = tuple[npt.NDArray[Any], ...]


def run_into(
    out: npt.NDArray[Any],
    columns: Columns,
    *,
    kernel: str = KERNEL,
    thread_count: int = 1,
    block_size: int = BLOCK_SIZE,
    instruction_set: str | None = None,
) -> None:
    """Call run_map; the one place that knows its signature."""
    _core.run_map(kernel, columns, out, thread_count, block_size, instruction_set)


def pair_masses(columns: Columns, **settings: Any) -> npt.NDArray[Any]:
    """The mass of each pair of eight columns, four per particle."""
    out = np.empty(columns[0].shape[0], dtype=columns[0].dtype)
    run_into(out, columns, **settings)
    return out


def sample_columns(count: int, dtype: Any, seed: int) -> Columns:
    """Eight columns of unrelated particles, the same for one seed."""
    rng = np.random.default_rng(seed)
    columns = []
    for _ in range(2):
        columns += [
            10 ** rng.uniform(-1, 3, count),
            rng.uniform(-4, 4, count),
            rng.uniform(-np.pi, np.pi, count),
            10 ** rng.uniform(-3, 2, count),
        ]
    return tuple(np.ascontiguousarray(column, dtype=dtype) for column in columns)


def packed_records(columns: Columns) -> npt.NDArray[Any]:
    """Four columns as records with a one-byte field in front of them."""
    fields = [("charge", "i1")] + [(name, columns[0].dtype) for name in COMPONENT_NAMES]
    records = np.zeros(columns[0].shape[0], dtype=np.dtype(fields))
    for name, column in zip(COMPONENT_NAMES, columns, strict=True):
        records[name] = column
    return records


def fields_of(records: npt.NDArray[Any]) -> Columns:
    return tuple(records[name] for name in COMPONENT_NAMES)


@pytest.fixture
def columns() -> Columns:
    """Columns that span several chunks of the map loop and end in a partial one."""
    return sample_columns(2_003, np.float64, seed=1)


@pytest.fixture
def expected(columns: Columns) -> npt.NDArray[Any]:
    return pair_masses(columns)


# --- Values -----------------------------------------------------------------


@pytest.mark.parametrize("instruction_set", INSTRUCTION_SETS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_columns_reach_the_kernel_in_their_order(
    instruction_set: str, dtype: Any
) -> None:
    # Rows: massless and back to back (m = 2 pT); both at rest (m = m1 + m2).
    first = (np.array([45.0, 0.0]), np.zeros(2), np.zeros(2), np.array([0.0, 3.0]))
    second = (
        np.array([45.0, 0.0]),
        np.zeros(2),
        np.array([np.pi, 0.0]),
        np.array([0.0, 4.0]),
    )
    pairs = tuple(column.astype(dtype) for column in first + second)
    masses = pair_masses(pairs, instruction_set=instruction_set)
    assert masses.dtype == dtype
    np.testing.assert_allclose(masses, [90.0, 7.0], rtol=4 * np.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_empty_columns(dtype: Any) -> None:
    masses = pair_masses(tuple(np.empty(0, dtype=dtype) for _ in range(8)))
    assert masses.shape == (0,)


# --- Borrowed columns -------------------------------------------------------


def test_strided_and_reversed_views(
    columns: Columns, expected: npt.NDArray[Any]
) -> None:
    interleaved = [np.repeat(column, 3) for column in columns]
    every_third = tuple(column[::3] for column in interleaved)
    assert np.array_equal(pair_masses(every_third), expected)

    backwards = tuple(column[::-1] for column in columns)
    assert np.array_equal(pair_masses(backwards), expected[::-1])


def test_repeated_value_with_zero_stride(columns: Columns) -> None:
    count = columns[0].shape[0]
    massless = list(columns)
    massless[3] = np.zeros(count)
    massless[7] = np.zeros(count)
    broadcast = list(columns)
    broadcast[3] = np.broadcast_to(np.float64(0.0), (count,))
    broadcast[7] = np.broadcast_to(np.float64(0.0), (count,))
    assert broadcast[3].strides == (0,)
    assert np.array_equal(pair_masses(tuple(broadcast)), pair_masses(tuple(massless)))


def test_unaligned_fields_of_packed_records(
    columns: Columns, expected: npt.NDArray[Any]
) -> None:
    # The one-byte field leaves every float64 field unaligned, 33 bytes from
    # the next record's.
    first = packed_records(columns[:4])
    second = packed_records(columns[4:])
    assert first.dtype.itemsize == 33
    assert not first["pt"].flags.aligned
    assert np.array_equal(pair_masses(fields_of(first) + fields_of(second)), expected)


def test_read_only_and_memory_mapped_columns(
    columns: Columns, expected: npt.NDArray[Any], tmp_path: Path
) -> None:
    records_path = tmp_path / "particles.bin"
    first = packed_records(columns[:4])
    first.tofile(records_path)
    mapped = np.memmap(records_path, dtype=first.dtype, mode="r")
    assert not mapped.flags.writeable
    masses = pair_masses(fields_of(mapped) + columns[4:])
    # Windows cannot remove a file while it is mapped.
    del mapped
    assert np.array_equal(masses, expected)


def test_columns_are_read_in_place() -> None:
    # tracemalloc sees the allocations of Python and NumPy. Reading the
    # fields of packed records must allocate nothing of the size of a column.
    count = 500_000
    columns = sample_columns(count, np.float64, seed=4)
    fields = fields_of(packed_records(columns[:4])) + fields_of(
        packed_records(columns[4:])
    )
    out = np.empty(count)

    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        run_into(out, fields)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak - before < out.nbytes // 10
    assert np.array_equal(out, pair_masses(columns))


# --- Threads ----------------------------------------------------------------


@pytest.mark.parametrize("instruction_set", INSTRUCTION_SETS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_result_does_not_depend_on_threads_or_block_size(
    instruction_set: str, dtype: Any
) -> None:
    # 10 007 elements: 39 whole chunks of the map loop and a partial one.
    columns = sample_columns(10_007, dtype, seed=5)
    reference = pair_masses(columns, instruction_set=instruction_set)
    for thread_count in (1, 2, 3, 8, 33):
        for block_size in (1, 256, 1_000, 4_096, 10**9):
            masses = pair_masses(
                columns,
                thread_count=thread_count,
                block_size=block_size,
                instruction_set=instruction_set,
            )
            assert np.array_equal(masses, reference), (thread_count, block_size)


def test_concurrent_calls_from_several_python_threads() -> None:
    samples = [sample_columns(20_011, np.float64, seed=seed) for seed in range(6, 14)]
    references = [pair_masses(columns) for columns in samples]

    def compute(columns: Columns) -> npt.NDArray[Any]:
        return pair_masses(columns, thread_count=4, block_size=256)

    for _ in range(5):
        with ThreadPoolExecutor(max_workers=len(samples)) as executor:
            results = list(executor.map(compute, samples))
        for result, reference in zip(results, references, strict=True):
            assert np.array_equal(result, reference)


def _masses_on_four_threads(columns: Columns) -> npt.NDArray[Any]:
    return pair_masses(columns, thread_count=4, block_size=256)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no fork")
@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_forked_child_computes_on_its_own_threads() -> None:
    columns = sample_columns(20_011, np.float64, seed=14)
    # Start the parent's threads first: the child inherits the pool object
    # without them.
    reference = _masses_on_four_threads(columns)
    with multiprocessing.get_context("fork").Pool(1) as pool:
        from_child = pool.apply_async(_masses_on_four_threads, (columns,)).get(
            timeout=120
        )
    assert np.array_equal(from_child, reference)


# --- Refused arguments ------------------------------------------------------


def test_columns_that_cannot_be_read_as_they_are(columns: Columns) -> None:
    out = np.empty(columns[0].shape[0])

    def with_first(column: npt.NDArray[Any]) -> Columns:
        return (column, *columns[1:])

    with pytest.raises(TypeError, match="every input must hold values of the type"):
        run_into(out, with_first(columns[0].astype(np.float32)))
    with pytest.raises(TypeError, match="every input must hold values of the type"):
        run_into(out, with_first(columns[0].astype(np.int64)))
    with pytest.raises(TypeError, match="every input must hold values of the type"):
        run_into(out, with_first(columns[0].astype(columns[0].dtype.newbyteorder())))
    with pytest.raises(ValueError, match="one-dimensional and as long as out"):
        run_into(out, with_first(columns[0][:-1]))
    with pytest.raises(ValueError, match="one-dimensional and as long as out"):
        run_into(out, with_first(columns[0].reshape(1, -1)))
    with pytest.raises(ValueError, match="reads 8 columns, got 7"):
        run_into(out, columns[:7])


def test_output_that_cannot_be_written(columns: Columns) -> None:
    count = columns[0].shape[0]
    with pytest.raises(TypeError, match="out must hold float32 or float64"):
        run_into(np.empty(count, dtype=np.int64), columns)
    with pytest.raises(ValueError, match="out must be one-dimensional, contiguous"):
        run_into(np.empty(2 * count)[::2], columns)
    read_only = np.empty(count)
    read_only.flags.writeable = False
    with pytest.raises((BufferError, ValueError)):
        run_into(read_only, columns)


def test_settings_that_cannot_be_used(columns: Columns) -> None:
    out = np.empty(columns[0].shape[0])
    with pytest.raises(ValueError, match="unknown map kernel 'no_such_kernel'"):
        run_into(out, columns, kernel="no_such_kernel")
    with pytest.raises(ValueError, match="unsupported instruction set 'avx1024'"):
        run_into(out, columns, instruction_set="avx1024")
    with pytest.raises(ValueError, match="thread_count must be at least 1"):
        run_into(out, columns, thread_count=0)
    with pytest.raises(ValueError, match="block_size must be at least 1"):
        run_into(out, columns, block_size=0)
