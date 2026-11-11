"""High-performance computation of particle kinematics on array data."""

from importlib.metadata import version

from parkin._system_info import SystemInfo, system_info
from parkin.particles import Particles, as_particles

__version__ = version("parkin")

__all__ = [
    "Particles",
    "SystemInfo",
    "__version__",
    "as_particles",
    "system_info",
]
