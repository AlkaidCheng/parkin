# parkin

parkin (particle kinematics) is a Python library of the kinematic observables
used in high-energy physics, from the four-momentum of a single particle to
the shape of a whole event:

- **Particles**: four-momentum components in Cartesian and collider
  coordinates (transverse momentum, pseudorapidity, rapidity, azimuthal angle,
  mass); Lorentz boosts and rotations.
- **Systems of particles**: invariant and transverse masses, angular
  separations, decay angles, momentum balance, Mandelstam and Dalitz
  variables.
- **Jets**: sequential-recombination algorithms (kT, Cambridge/Aachen,
  anti-kT); jet shapes and substructure (N-subjettiness, energy correlation
  functions, grooming).
- **Events**: scalar sums (HT), missing transverse momentum, event shapes
  (thrust, sphericity, Fox-Wolfram moments).
- **Tracks and calorimeter clusters**: helix and impact parameters, track
  extrapolation, cluster moments, isolation.

parkin works on columnar data: arrays whose values lie in contiguous memory,
either flat or jagged, with a variable number of entries per event. Such
arrays may live in memory, as NumPy and Awkward arrays do, or be memory-mapped
from a file. They are read in place, and the computation runs in a C++ core
that is vectorised with SIMD and parallel over threads.

The package is at the start of its development. It builds the compiled core
and reports how it was built and which SIMD instruction sets the host
supports; no observable is implemented yet.

## Installation

parkin is installed from its repository:

```
pip install git+https://github.com/AlkaidCheng/parkin.git
```

The build needs Python 3.11 or later, a C++20 compiler and Git. pip provides
the remaining build tools (scikit-build-core, nanobind, and CMake 3.26 or
later when the system has none). The xsimd headers are taken from an installed
copy of version 14.3.0 or later and are downloaded otherwise.

## Usage

```python
import parkin

print(parkin.system_info())
```

```
parkin 0.1.0
compiler: GNU 14.2.0
xsimd: 14.3.0
CPU features: sse2, sse3, ssse3, sse4.1, sse4.2, avx, fma3, avx2
```

`system_info()` returns a `SystemInfo` with the fields `parkin_version`,
`compiler`, `xsimd_version` and `cpu_features`. Include its printed form in a
bug report.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
