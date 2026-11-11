"""Kinematic observables for high-energy physics on columnar arrays."""

from importlib.metadata import version

from parkin._system_info import SystemInfo, system_info

__version__ = version("parkin")

__all__ = ["SystemInfo", "__version__", "system_info"]
