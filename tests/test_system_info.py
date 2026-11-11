"""The description of the installed build and of the host."""

import dataclasses
import platform
import re
from importlib.metadata import version

import pytest

import parkin
from parkin import SystemInfo, system_info

# Instruction set every CPU of a processor family supports, keyed by the names
# platform.machine() gives that family. SSE2 is part of the x86-64 baseline
# (https://gitlab.com/x86-psABIs/x86-64-ABI), and xsimd reports neon64 for
# every AArch64 build
# (https://github.com/xtensor-stack/xsimd/blob/14.3.0/include/xsimd/config/xsimd_cpu_features_arm.hpp).
BASELINE_FEATURE_BY_MACHINE = {
    "x86_64": "sse2",
    "amd64": "sse2",
    "arm64": "neon64",
    "aarch64": "neon64",
}


def test_package_version_is_the_installed_version() -> None:
    assert parkin.__version__ == version("parkin")
    assert system_info().parkin_version == version("parkin")


def test_compiler_names_an_id_and_a_version() -> None:
    compiler_id, compiler_version = system_info().compiler.split(" ", maxsplit=1)
    assert compiler_id
    assert compiler_version[0].isdigit()


def test_xsimd_version_has_three_numeric_parts() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", system_info().xsimd_version)


def test_cpu_features_are_unique_names() -> None:
    features = system_info().cpu_features
    assert isinstance(features, tuple)
    assert all(isinstance(feature, str) and feature for feature in features)
    assert len(set(features)) == len(features)


def test_cpu_features_include_the_baseline_of_the_processor_family() -> None:
    baseline_feature = BASELINE_FEATURE_BY_MACHINE.get(platform.machine().lower())
    if baseline_feature is None:
        pytest.skip(f"no baseline recorded for {platform.machine()}")
    assert baseline_feature in system_info().cpu_features


def test_system_info_is_immutable() -> None:
    info = system_info()
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.compiler = "other"  # type: ignore[misc]


def test_report_lists_every_field() -> None:
    info = SystemInfo(
        parkin_version="1.2.3",
        compiler="GNU 14.2.0",
        xsimd_version="14.3.0",
        cpu_features=("sse2", "avx2"),
    )
    assert str(info) == (
        "parkin 1.2.3\n"
        "compiler: GNU 14.2.0\n"
        "xsimd: 14.3.0\n"
        "CPU features: sse2, avx2"
    )


def test_report_says_when_no_cpu_feature_is_detected() -> None:
    info = SystemInfo(
        parkin_version="1.2.3",
        compiler="GNU 14.2.0",
        xsimd_version="14.3.0",
        cpu_features=(),
    )
    assert str(info).endswith("CPU features: none detected")
