"""High-performance computation of particle kinematics on array data."""

from importlib.metadata import version

from parkin._system_info import SystemInfo, system_info

__version__ = version("parkin")

__all__ = ["SystemInfo", "__version__", "system_info"]
