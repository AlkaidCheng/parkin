"""The standard format of particle data, and the conversion of input into it."""

from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np

from parkin.typing import FloatArray, ParticlesLike

__all__ = ["Particles", "as_particles"]

# Element types the compiled core reads without conversion, in native byte
# order.
_SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))

# The components of the (pT, eta, phi, mass) coordinates, in the order of the
# keywords of Particles: the keyword, a description for messages, and the
# field names the component is recognised by in a structured array or a
# mapping. The names are those vector accepts: its momentum names and the
# generic names they stand for (_repr_momentum_to_generic in
# https://github.com/scikit-hep/vector/blob/v1.8.1/src/vector/_methods.py).
_PT_ETA_PHI_MASS_COMPONENTS = (
    ("pt", "transverse momentum", ("pt", "rho")),
    ("eta", "pseudorapidity", ("eta",)),
    ("phi", "azimuthal angle", ("phi",)),
    ("mass", "mass", ("mass", "m", "M", "tau")),
)


class Particles:
    """Four-momenta of particles, one array per component.

    This is the format every function of parkin computes on. A function
    passes each of its arguments through `as_particles`, so it also takes
    the layouts that function reads.

    Parameters
    ----------
    pt : numpy.ndarray
        Transverse momenta.
    eta : numpy.ndarray
        Pseudorapidities.
    phi : numpy.ndarray
        Azimuthal angles, in radians.
    mass : numpy.ndarray
        Masses, in the unit of the transverse momenta. A negative mass
        denotes a spacelike four-momentum.

    Raises
    ------
    TypeError
        If a component is not a NumPy array, is a masked array, or is not
        float32 or float64 in native byte order.
    ValueError
        If a component is not one-dimensional, or the components differ in
        dtype or length.

    See Also
    --------
    as_particles : Read particles from a structured array or a mapping.

    Notes
    -----
    The components are given by keyword and are one-dimensional arrays of
    one length and one dtype. Nothing is copied or converted: the object
    holds a view of each array, so it keeps their memory alive and sees a
    later change of their values.

    An array may have any strides and alignment. One whose elements are
    adjacent in memory is read fastest.

    `Particles` may be subclassed to carry further data and methods. A
    subclass passes the components on to ``Particles.__init__`` and is taken
    wherever a `Particles` is: `as_particles` returns it as it is, and a
    function computes on the components it was built with. They cannot be
    replaced afterwards.

    Examples
    --------
    >>> import numpy as np
    >>> import parkin
    >>> muons = parkin.Particles(
    ...     pt=np.array([45.0, 38.2]),
    ...     eta=np.array([0.3, -1.2]),
    ...     phi=np.array([0.1, 2.9]),
    ...     mass=np.array([0.106, 0.106]),
    ... )
    >>> muons
    <Particles (pt, eta, phi, mass), length 2, float64>
    >>> muons.pt
    array([45. , 38.2])

    A subclass that also carries the charges, built from the components of
    other particles:

    >>> class Muons(parkin.Particles):
    ...     def __init__(self, *, charge, **components):
    ...         super().__init__(**components)
    ...         self.charge = charge
    >>> Muons(charge=np.array([-1, 1]), **muons.components)
    <Muons (pt, eta, phi, mass), length 2, float64>
    """

    __slots__ = ("_columns",)

    _columns: tuple[FloatArray, FloatArray, FloatArray, FloatArray]

    def __init__(
        self, *, pt: FloatArray, eta: FloatArray, phi: FloatArray, mass: FloatArray
    ) -> None:
        self._columns = _standard_columns((pt, eta, phi, mass), _keywords())

    @classmethod
    def _from_labelled_columns(
        cls, columns: Sequence[object], labels: Sequence[str]
    ) -> Self:
        """Build particles from components that messages call by ``labels``."""
        particles = cls.__new__(cls)
        particles._columns = _standard_columns(columns, labels)
        return particles

    @property
    def pt(self) -> FloatArray:
        """Transverse momenta."""
        return self._columns[0]

    @property
    def eta(self) -> FloatArray:
        """Pseudorapidities."""
        return self._columns[1]

    @property
    def phi(self) -> FloatArray:
        """Azimuthal angles, in radians."""
        return self._columns[2]

    @property
    def mass(self) -> FloatArray:
        """Masses."""
        return self._columns[3]

    @property
    def components(self) -> dict[str, FloatArray]:
        """The components by name, in the order of the keywords.

        The dict is a new one on each call, and its arrays are the views
        the particles hold.
        """
        return dict(zip(_keywords(), self._columns, strict=True))

    @property
    def dtype(self) -> np.dtype[np.floating[Any]]:
        """Element type of every component."""
        return self._columns[0].dtype

    def __len__(self) -> int:
        return len(self._columns[0])

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} ({', '.join(_keywords())}), "
            f"length {len(self)}, {self.dtype}>"
        )


def as_particles(data: ParticlesLike, /, *, argument_name: str = "data") -> Particles:
    """Read particles from the layout they are stored in, without copying.

    Parameters
    ----------
    data : Particles, structured numpy.ndarray or Mapping[str, numpy.ndarray]
        The particles, in one of these layouts:

        - a `Particles`, which is returned as it is;
        - a mapping from field name to a one-dimensional array;
        - a one-dimensional structured array, whose fields are the
          components.

        In a mapping and in a structured array the components are
        recognised by field name: ``pt`` or ``rho``; ``eta``; ``phi``;
        ``mass``, ``m``, ``M`` or ``tau``. Exactly one name of each
        component must be present. Other fields are ignored.
    argument_name : str, default "data"
        What messages call ``data``. A function that takes particles passes
        the name of its own argument, so that an error names it.

    Returns
    -------
    Particles
        The particles, holding a view of each component.

    Raises
    ------
    TypeError
        If ``data`` has none of these layouts, or a component is refused by
        `Particles`.
    ValueError
        If a component has no field or more than one, or the components are
        refused by `Particles`.

    Notes
    -----
    The fields of a structured array are interleaved in memory, so each
    component is a strided view. It is read in place, which takes longer
    than reading separate arrays, the more so the more other fields a record
    holds.

    Examples
    --------
    >>> import numpy as np
    >>> import parkin
    >>> records = np.array(
    ...     [(45.0, 0.3, 0.1, 0.106), (38.2, -1.2, 2.9, 0.106)],
    ...     dtype=[("pt", "f8"), ("eta", "f8"), ("phi", "f8"), ("m", "f8")],
    ... )
    >>> muons = parkin.as_particles(records)
    >>> muons.mass
    array([0.106, 0.106])
    >>> np.shares_memory(muons.mass, records)
    True
    """
    if isinstance(data, Particles):
        return data
    if isinstance(data, np.ndarray):
        if data.dtype.names is None:
            raise TypeError(
                f"{argument_name} must be a Particles, a structured array or a "
                f"mapping of arrays, got an array of dtype {data.dtype} without "
                "fields"
            )
        available_names: tuple[Any, ...] = data.dtype.names
    elif isinstance(data, Mapping):
        available_names = tuple(data)
    else:
        raise TypeError(
            f"{argument_name} must be a Particles, a structured array or a "
            f"mapping of arrays, got {type(data).__name__}"
        )

    columns = []
    labels = []
    for _, description, accepted_names in _PT_ETA_PHI_MASS_COMPONENTS:
        present_names = [name for name in accepted_names if name in available_names]
        if len(present_names) != 1:
            problem = "has no field" if not present_names else "has several fields"
            raise ValueError(
                f"{argument_name} {problem} for the {description}: expected "
                f"exactly one of {_quoted(accepted_names)}, found "
                f"{_quoted(available_names) or 'no fields'}"
            )
        field_name = present_names[0]
        columns.append(data[field_name])
        labels.append(f"field {field_name!r} of {argument_name}")
    return Particles._from_labelled_columns(columns, labels)


def _keywords() -> tuple[str, ...]:
    """The keywords of Particles, in order."""
    return tuple(keyword for keyword, _, _ in _PT_ETA_PHI_MASS_COMPONENTS)


def _standard_columns(
    columns: Sequence[object], labels: Sequence[str]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Check the four components of particles and take a view of each.

    A view of the base array class leaves a memory map or a record array
    behind and cannot be reshaped through the caller's array.
    """
    views: list[FloatArray] = []
    for column, label in zip(columns, labels, strict=True):
        if not isinstance(column, np.ndarray):
            raise TypeError(
                f"{label} must be a NumPy array, got {type(column).__name__}; "
                "parkin reads arrays in place and does not convert other types"
            )
        if isinstance(column, np.ma.MaskedArray):
            raise TypeError(
                f"{label} is a masked array, and parkin does not read masks; "
                "pass the values to compute on as a plain array"
            )
        if column.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"{label} must be float32 or float64 in native byte order, got "
                f"{column.dtype.str!r}; parkin reads arrays in place and does "
                "not convert them"
            )
        if column.ndim != 1:
            raise ValueError(
                f"{label} must be one-dimensional, got {column.ndim} dimensions"
            )
        views.append(column.view(np.ndarray))

    first = views[0]
    for view, label in zip(views[1:], labels[1:], strict=True):
        if view.dtype != first.dtype or view.shape != first.shape:
            raise ValueError(
                "the components of particles must have one dtype and one "
                f"length: {labels[0]} has {first.dtype} and {first.shape[0]} "
                f"elements, {label} has {view.dtype} and {view.shape[0]} elements"
            )
    return views[0], views[1], views[2], views[3]


def _quoted(names: tuple[Any, ...]) -> str:
    return ", ".join(repr(name) for name in names)
