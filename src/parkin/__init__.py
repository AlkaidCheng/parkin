"""High-performance computation of particle kinematics on array data."""

from importlib.metadata import version

from parkin._execution import ExecutionSettings, execution, get_execution
from parkin._system_info import SystemInfo, system_info
from parkin.particles import Particles, as_particles

__version__ = version("parkin")

__all__ = [
    "ExecutionSettings",
    "Particles",
    "SystemInfo",
    "__version__",
    "as_particles",
    "execution",
    "get_execution",
    "system_info",
]
