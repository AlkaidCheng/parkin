# parkin

parkin (particle kinematics) is a high-performance Python library for the
computation of kinematic observables, from invariant masses, angular
separations and Lorentz transformations to jet clustering, jet substructure
and event shapes. Its C++ core combines zero-copy access to contiguous array
data, such as NumPy, Awkward and memory-mapped arrays, with SIMD vectorisation
and multithreading on CPUs and CUDA acceleration on GPUs.

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

| Requirement | Supported |
|---|---|
| Python | 3.11 or later |
| NumPy | 1.23 or later, including NumPy 2 |
| C++ compiler (build only) | C++20; GCC 10 or later is tested |

The build also needs Git. pip provides the remaining build tools, and the
xsimd headers are downloaded when no recent copy is installed.

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
