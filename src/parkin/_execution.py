"""The execution settings: how a call is run, not what it computes."""

import os
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, replace
from types import EllipsisType
from typing import Any

from parkin import _core

# Defaults.
#
# Elements a thread takes at a time. Set from the pair-mass kernel: one block
# of this size takes several times longer than waking a thread, so an array of
# a few thousand elements already gains from a second thread, and the
# throughput of large arrays on 4 to 10 threads is the same for every block
# size from 2**12 to 2**18.
DEFAULT_BLOCK_SIZE = 4096

# Key of the _Setting in the metadata of a field of settings.
_SETTING = "parkin_setting"


def usable_cpu_count() -> int:
    """Count the CPUs this process may run on.

    The count follows the CPU affinity of the process where the system
    reports one, so it respects the cores a batch system has assigned.

    Returns
    -------
    int
        Number of usable CPUs, at least 1.
    """
    if sys.version_info >= (3, 13):
        count = os.process_cpu_count()
    elif sys.platform == "linux":
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return count or 1


@dataclass(frozen=True)
class _Setting:
    """What a field of settings has besides its name, type and default.

    The environment variable, the check and the merge of a ``with`` block
    are all read from the fields, so a setting is described in one place.

    Parameters
    ----------
    environment_variable : str
        The variable that sets the field when parkin is imported.
    parse : Callable[[str], object]
        Reads the text of that variable. It raises ValueError for text it
        cannot read.
    check : Callable[[str, object], None]
        Takes the name of the field and a value, and raises TypeError or
        ValueError when the field cannot hold the value.
    """

    environment_variable: str
    parse: Callable[[str], object]
    check: Callable[[str, object], None]


def _setting(
    default: object,
    *,
    environment_variable: str,
    parse: Callable[[str], object],
    check: Callable[[str, object], None],
) -> Any:
    """Declare a field of settings with its default and its `_Setting`."""
    return field(
        default=default,
        metadata={_SETTING: _Setting(environment_variable, parse, check)},
    )


def _parse_int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{text!r} is not an integer") from None


def _check_positive_int(name: str, value: object) -> None:
    # bool is a subclass of int, and True is not a count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")


def _check_thread_count(name: str, value: object) -> None:
    if value is not None:
        _check_positive_int(name, value)


def _check_instruction_set(name: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None, got {type(value).__name__}")
    supported = _core.supported_instruction_sets()
    if value not in supported:
        raise ValueError(f"{name} must be one of {', '.join(supported)}, got {value!r}")


@dataclass(frozen=True)
class ExecutionSettings:
    """How every parkin function runs a computation.

    These settings decide the threads and the instructions a call uses.
    They do not decide what it computes.

    Parameters
    ----------
    threads : int or None, default None
        Largest number of threads a call uses. ``None`` stands for the number
        of CPUs the process may run on.
    simd : str or None, default None
        Instruction set the kernels run with: ``"scalar"``, or the name of a
        SIMD instruction set that parkin was compiled for and the processor
        has, such as ``"sse2"`` or ``"neon64"``. ``None`` stands for the
        widest of them.
    block_size : int, default 4096
        Number of elements a thread takes at a time. An array of at most
        this many elements is computed on the calling thread alone.

    Raises
    ------
    TypeError
        If a setting has the wrong type.
    ValueError
        If ``threads`` or ``block_size`` is not positive, or ``simd`` is not
        a supported instruction set.

    See Also
    --------
    execution : Change settings for the calls made inside a ``with`` block.
    get_execution : Return the settings in effect.

    Notes
    -----
    For a given ``simd``, results do not depend on ``threads`` or
    ``block_size``. Results of two instruction sets can differ in the last
    bits, because each has its own implementation of the math functions.

    The environment variables ``PARKIN_THREADS``, ``PARKIN_SIMD`` and
    ``PARKIN_BLOCK_SIZE`` set the fields of those names when parkin is
    imported.
    """

    threads: int | None = _setting(
        None,
        environment_variable="PARKIN_THREADS",
        parse=_parse_int,
        check=_check_thread_count,
    )
    simd: str | None = _setting(
        None,
        environment_variable="PARKIN_SIMD",
        parse=str,
        check=_check_instruction_set,
    )
    block_size: int = _setting(
        DEFAULT_BLOCK_SIZE,
        environment_variable="PARKIN_BLOCK_SIZE",
        parse=_parse_int,
        check=_check_positive_int,
    )

    def __post_init__(self) -> None:
        for setting in fields(self):
            setting.metadata[_SETTING].check(setting.name, getattr(self, setting.name))

    @property
    def thread_count(self) -> int:
        """The number of threads ``threads`` stands for."""
        return usable_cpu_count() if self.threads is None else self.threads


def execution_settings_from_environment(
    environment: Mapping[str, str],
) -> ExecutionSettings:
    """Build the execution settings that the environment variables ask for.

    Parameters
    ----------
    environment : Mapping[str, str]
        The environment variables. Each field of `ExecutionSettings` is set
        by the variable its declaration names; a variable that is unset or
        empty leaves its field at the default.

    Returns
    -------
    ExecutionSettings
        The defaults, with the fields the environment sets.

    Raises
    ------
    ValueError
        If a variable does not hold a valid value for its field. The message
        names the variable.
    """
    values: dict[str, Any] = {}
    for setting in fields(ExecutionSettings):
        description: _Setting = setting.metadata[_SETTING]
        text = environment.get(description.environment_variable, "")
        if not text:
            continue
        try:
            value = description.parse(text)
            description.check(setting.name, value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"environment variable {description.environment_variable}: {error}"
            ) from None
        values[setting.name] = value
    return ExecutionSettings(**values)


_environment_settings = execution_settings_from_environment(os.environ)
_active_settings: ContextVar[ExecutionSettings] = ContextVar("parkin_execution")


def get_execution() -> ExecutionSettings:
    """Return the execution settings in effect.

    Returns
    -------
    ExecutionSettings
        The settings of the innermost enclosing `execution` block or,
        outside any block, the defaults with the fields that the environment
        variables set.
    """
    return _active_settings.get(_environment_settings)


@contextmanager
def execution(
    *,
    threads: int | None | EllipsisType = ...,
    simd: str | None | EllipsisType = ...,
    block_size: int | EllipsisType = ...,
) -> Iterator[ExecutionSettings]:
    """Change execution settings for the calls made inside a ``with`` block.

    Parameters
    ----------
    threads : int or None, optional
        Largest number of threads a call uses. ``None`` stands for the number
        of CPUs the process may run on.
    simd : str or None, optional
        Instruction set the kernels run with; see `ExecutionSettings`.
        ``None`` stands for the widest one available.
    block_size : int, optional
        Number of elements a thread takes at a time.

    Yields
    ------
    ExecutionSettings
        The settings in effect inside the block: those in effect outside
        it, with the settings given here. A setting left out keeps its
        value.

    Raises
    ------
    TypeError, ValueError
        If a setting is not valid; see `ExecutionSettings`.

    Notes
    -----
    The block applies to the thread, or the asyncio task, that enters it.
    Other threads keep their own settings.

    Examples
    --------
    >>> import parkin
    >>> with parkin.execution(threads=1) as active:
    ...     active.threads
    1
    """
    given: dict[str, Any] = {
        "threads": threads,
        "simd": simd,
        "block_size": block_size,
    }
    active = replace(
        get_execution(),
        **{name: value for name, value in given.items() if value is not ...},
    )
    token = _active_settings.set(active)
    try:
        yield active
    finally:
        _active_settings.reset(token)
