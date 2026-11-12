"""parkin.ExecutionSettings, parkin.execution and the environment variables."""

import dataclasses
import inspect
import threading

import pytest

import parkin
from parkin import _core, _execution

SCALAR = "scalar"
WIDEST_INSTRUCTION_SET = _core.supported_instruction_sets()[-1]
ENVIRONMENT_VARIABLES = {
    "threads": "PARKIN_THREADS",
    "simd": "PARKIN_SIMD",
    "block_size": "PARKIN_BLOCK_SIZE",
}


def test_defaults() -> None:
    defaults = parkin.ExecutionSettings()
    assert defaults == parkin.ExecutionSettings(
        threads=None, simd=None, block_size=4096
    )
    assert defaults.thread_count == _execution.usable_cpu_count() >= 1
    assert parkin.ExecutionSettings(threads=3).thread_count == 3


def test_supported_instruction_sets_start_with_scalar() -> None:
    instruction_sets = _core.supported_instruction_sets()
    assert instruction_sets[0] == SCALAR
    assert len(set(instruction_sets)) == len(instruction_sets)
    for instruction_set in instruction_sets:
        assert parkin.ExecutionSettings(simd=instruction_set).simd == instruction_set


@pytest.mark.parametrize(
    ("settings", "error", "message"),
    [
        ({"threads": 0}, ValueError, "threads must be at least 1, got 0"),
        ({"threads": -2}, ValueError, "threads must be at least 1, got -2"),
        ({"threads": 2.0}, TypeError, "threads must be an integer, got float"),
        ({"threads": True}, TypeError, "threads must be an integer, got bool"),
        ({"block_size": 0}, ValueError, "block_size must be at least 1, got 0"),
        ({"block_size": "4096"}, TypeError, "block_size must be an integer, got str"),
        (
            {"block_size": None},
            TypeError,
            "block_size must be an integer, got NoneType",
        ),
        ({"simd": "avx9000"}, ValueError, "simd must be one of scalar.*got 'avx9000'"),
        ({"simd": 2}, TypeError, "simd must be a string or None, got int"),
    ],
)
def test_invalid_settings(
    settings: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        parkin.ExecutionSettings(**settings)
    with pytest.raises(error, match=message), parkin.execution(**settings):
        pass


def test_settings_cannot_be_changed_in_place() -> None:
    settings = parkin.ExecutionSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.threads = 2  # type: ignore[misc]


# --- The block --------------------------------------------------------------


def test_block_changes_settings_and_restores_them() -> None:
    outside = parkin.get_execution()
    with parkin.execution(threads=2, simd=SCALAR) as active:
        assert active == parkin.get_execution()
        assert (active.threads, active.simd) == (2, SCALAR)
        assert active.block_size == outside.block_size
    assert parkin.get_execution() == outside


def test_nested_blocks_keep_the_settings_they_do_not_name() -> None:
    with parkin.execution(threads=2, block_size=512):
        with parkin.execution(threads=5) as inner:
            assert (inner.threads, inner.block_size) == (5, 512)
        with parkin.execution() as unchanged:
            assert (unchanged.threads, unchanged.block_size) == (2, 512)
        assert parkin.get_execution().threads == 2


def test_none_inside_a_block_restores_the_automatic_choice() -> None:
    with parkin.execution(threads=2, simd=SCALAR, block_size=512):
        with parkin.execution(threads=None, simd=None) as automatic:
            assert automatic.threads is None
            assert automatic.thread_count == _execution.usable_cpu_count()
            assert automatic.simd is None
            assert automatic.block_size == 512
        assert parkin.get_execution().threads == 2


def test_settings_are_restored_after_an_exception() -> None:
    outside = parkin.get_execution()
    with pytest.raises(RuntimeError, match="inside"), parkin.execution(threads=2):
        raise RuntimeError("inside")
    assert parkin.get_execution() == outside


def test_block_applies_to_the_thread_that_enters_it() -> None:
    outside = parkin.get_execution()
    seen_elsewhere: list[parkin.ExecutionSettings] = []
    with parkin.execution(threads=2, block_size=512):
        other = threading.Thread(
            target=lambda: seen_elsewhere.append(parkin.get_execution())
        )
        other.start()
        other.join()
    assert seen_elsewhere == [outside]


def test_block_takes_every_setting_by_keyword() -> None:
    parameters = inspect.signature(parkin.execution).parameters
    fields = dataclasses.fields(parkin.ExecutionSettings)
    assert list(parameters) == [field.name for field in fields]
    for parameter in parameters.values():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is ...


# --- The environment --------------------------------------------------------


def test_every_setting_has_an_environment_variable() -> None:
    declared = {
        field.name: field.metadata[_execution._SETTING].environment_variable
        for field in dataclasses.fields(parkin.ExecutionSettings)
    }
    assert declared == ENVIRONMENT_VARIABLES


def test_environment_variables_set_the_fields() -> None:
    from_environment = _execution.execution_settings_from_environment
    assert from_environment({}) == parkin.ExecutionSettings()
    empty = dict.fromkeys(ENVIRONMENT_VARIABLES.values(), "")
    assert from_environment({**empty, "HOME": "/x"}) == parkin.ExecutionSettings()
    assert from_environment(
        {"PARKIN_THREADS": "6", "PARKIN_SIMD": SCALAR, "PARKIN_BLOCK_SIZE": "1024"}
    ) == parkin.ExecutionSettings(threads=6, simd=SCALAR, block_size=1024)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {"PARKIN_THREADS": "many"},
            "environment variable PARKIN_THREADS: 'many' is not an integer",
        ),
        (
            {"PARKIN_BLOCK_SIZE": "1e3"},
            "environment variable PARKIN_BLOCK_SIZE: '1e3' is not an integer",
        ),
        (
            {"PARKIN_THREADS": "0"},
            "environment variable PARKIN_THREADS: threads must be at least 1, got 0",
        ),
        (
            {"PARKIN_SIMD": "avx9000"},
            "environment variable PARKIN_SIMD: simd must be one of scalar",
        ),
        (
            {"PARKIN_THREADS": "2", "PARKIN_BLOCK_SIZE": "-1"},
            "environment variable PARKIN_BLOCK_SIZE: block_size must be at least 1",
        ),
    ],
)
def test_invalid_environment_variables(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _execution.execution_settings_from_environment(environment)


def test_environment_gives_the_settings_outside_any_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from_environment = parkin.ExecutionSettings(threads=2, simd=SCALAR)
    monkeypatch.setattr(_execution, "_environment_settings", from_environment)
    assert parkin.get_execution() == from_environment
    with parkin.execution(simd=WIDEST_INSTRUCTION_SET) as active:
        assert active == parkin.ExecutionSettings(
            threads=2, simd=WIDEST_INSTRUCTION_SET
        )
