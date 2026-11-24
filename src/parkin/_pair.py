"""Observables of a system of particles."""

import numpy as np

from parkin import _core
from parkin._execution import get_execution
from parkin.particles import Particles, as_particles
from parkin.typing import FloatArray, ParticlesLike

# Name of the compiled kernel, as cpp/parkin/cpu/map_kernels.cpp registers it.
PAIR_MASS_PT_ETA_PHI_MASS_KERNEL = "pair_mass_ptetaphim"


def invariant_mass(*particles: ParticlesLike) -> FloatArray:
    """Compute the invariant mass of systems of particles.

    Element ``i`` of the result is the mass of the system formed by particle
    ``i`` of every argument. One argument gives the mass of each of its
    particles, and two give the mass of each pair.

    Parameters
    ----------
    *particles : Particles, structured numpy.ndarray or Mapping[str, numpy.ndarray]
        The particles of each system, one or two sets: each a `Particles`,
        or a layout that `as_particles` reads. Messages call them ``p1`` and
        ``p2``. Two sets must have one length and one dtype.

    Returns
    -------
    numpy.ndarray
        The masses, one per system, in a new array with the dtype of the
        inputs and in the unit of their transverse momenta.

    Raises
    ------
    TypeError
        If no argument is given, or one is refused by `as_particles`.
    ValueError
        If an argument is refused by `as_particles`, if two differ in dtype
        or length, or if more than two are given.

    See Also
    --------
    Particles : The format the particles are computed on.
    as_particles : The layouts an argument may have.
    execution : Set the number of threads and the instruction set.

    Notes
    -----
    Angles are in radians. The formula that is evaluated, its derivation, the
    conventions for inputs such as a negative mass, and the measured accuracy
    are in the documentation:
    https://alkaidcheng.github.io/parkin/physics/observables/pair-invariant-mass/

    Examples
    --------
    Two massless particles, back to back in the transverse plane:

    >>> import numpy as np
    >>> import parkin
    >>> zero = np.array([0.0])
    >>> p1 = parkin.Particles(pt=np.array([45.0]), eta=zero, phi=zero, mass=zero)
    >>> p2 = parkin.Particles(
    ...     pt=np.array([45.0]), eta=zero, phi=np.array([np.pi]), mass=zero
    ... )
    >>> parkin.invariant_mass(p1, p2)
    array([90.])

    One set of particles has the masses it is given with:

    >>> parkin.invariant_mass(p1)
    array([0.])
    """
    arguments = _arguments_as_particles(*particles)
    if len(arguments) == 1:
        return np.array(arguments[0].mass)

    first, second = arguments
    masses: FloatArray = np.empty(len(first), dtype=first.dtype)
    active = get_execution()
    _core.run_map(
        PAIR_MASS_PT_ETA_PHI_MASS_KERNEL,
        (
            first.pt,
            first.eta,
            first.phi,
            first.mass,
            second.pt,
            second.eta,
            second.phi,
            second.mass,
        ),
        masses,
        active.thread_count,
        active.block_size,
        active.simd,
    )
    return masses


def _argument_names(count: int) -> tuple[str, ...]:
    """What messages call the arguments: p1, p2 and so on."""
    return tuple(f"p{position}" for position in range(1, count + 1))


def _arguments_as_particles(*arguments: ParticlesLike) -> tuple[Particles, ...]:
    """Read the arguments of invariant_mass as particles of one dtype and one length."""
    if not arguments:
        raise TypeError("invariant_mass needs at least one set of particles")
    if len(arguments) > 2:
        raise ValueError(
            f"invariant_mass takes one or two sets of particles, got {len(arguments)}"
        )
    names = _argument_names(len(arguments))
    particles = tuple(
        as_particles(argument, argument_name=name)
        for argument, name in zip(arguments, names, strict=True)
    )
    first = particles[0]
    for name, other in zip(names[1:], particles[1:], strict=True):
        if first.dtype != other.dtype or len(first) != len(other):
            raise ValueError(
                f"p1 and {name} must have one dtype and one length: p1 has "
                f"{first.dtype} and {len(first)} elements, {name} has "
                f"{other.dtype} and {len(other)} elements"
            )
    return particles
