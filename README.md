# parkin

parkin (particle kinematics) is a Python library for computing the kinematic
observables of high-energy physics, from single-particle four-momenta to jet
clustering, jet substructure and event shapes. Its C++ core reads columnar
arrays in place, whether in memory or memory-mapped, and is accelerated with
SIMD, multithreading and CUDA.

## Scope

Observables are grouped by the physics object they describe:

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

The input is columnar: arrays whose values lie in contiguous memory, flat or
jagged with a variable number of entries per event. NumPy and Awkward arrays
are of this kind, and so are arrays memory-mapped from a file.

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
