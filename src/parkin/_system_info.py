"""A description of the installed build and of the host it runs on."""

from dataclasses import dataclass
from importlib.metadata import version

from parkin import _core


@dataclass(frozen=True)
class SystemInfo:
    """The installed build of parkin and the SIMD support of the host.

    Attributes
    ----------
    parkin_version : str
        Version of the installed package.
    compiler : str
        Id and version of the C++ compiler that built the compiled core.
    xsimd_version : str
        Version of the xsimd headers the compiled core was built with.
    cpu_features : tuple[str, ...]
        SIMD instruction sets the running CPU supports, from the oldest to
        the newest of each processor family.
    """

    parkin_version: str
    compiler: str
    xsimd_version: str
    cpu_features: tuple[str, ...]

    def __str__(self) -> str:
        features = ", ".join(self.cpu_features) or "none detected"
        return (
            f"parkin {self.parkin_version}\n"
            f"compiler: {self.compiler}\n"
            f"xsimd: {self.xsimd_version}\n"
            f"CPU features: {features}"
        )


def system_info() -> SystemInfo:
    """Describe the installed build and the SIMD support of the host.

    Returns
    -------
    SystemInfo
        The package version, the compiler and xsimd version of the compiled
        core, and the SIMD instruction sets of the running CPU.

    Examples
    --------
    >>> import parkin
    >>> print(parkin.system_info())  # doctest: +SKIP
    parkin 0.1.0
    compiler: GNU 14.2.0
    xsimd: 14.3.0
    CPU features: sse2, sse3, ssse3, sse4.1, sse4.2, avx, fma3, avx2
    """
    return SystemInfo(
        parkin_version=version("parkin"),
        compiler=_core.compiler(),
        xsimd_version=_core.xsimd_version(),
        cpu_features=tuple(_core.detected_cpu_features()),
    )
