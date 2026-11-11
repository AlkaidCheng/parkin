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

The package is at the start of its development. So far it computes the
invariant mass of pairs of particles given in (pT, eta, phi, mass)
coordinates as NumPy arrays, on CPUs, with SSE2 on x86-64 and NEON on 64-bit
Arm.

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

A function takes sets of particles and returns one value per particle or per
pair. A set of particles is a `Particles`: one array per component, given by
keyword. `invariant_mass` gives the mass of each pair formed by two sets:

```python
import numpy as np
import parkin

mu1 = parkin.Particles(
    pt=np.array([45.0, 38.2]),
    eta=np.array([0.3, -1.2]),
    phi=np.array([0.1, 2.9]),
    mass=np.array([0.106, 0.106]),
)
mu2 = parkin.Particles(
    pt=np.array([41.5, 27.9]),
    eta=np.array([-0.4, -1.0]),
    phi=np.array([3.0, -0.4]),
    mass=np.array([0.106, 0.106]),
)

print(parkin.invariant_mass(mu1, mu2))
```

```
[91.18459273 65.41600255]
```

The components are one-dimensional NumPy arrays of one length, all float32 or
all float64, and a result has their type. `phi` is in radians and `mass` is in
the unit of `pt`. A `Particles` holds a view of each array and copies nothing,
so an array may be a slice or a memory-mapped file. parkin does not convert an
array of another type; it raises an error.

`Particles` and `as_particles` live in the module `parkin.particles`, and the
types for annotations, `ParticlesLike` and `FloatArray`, in `parkin.typing`.
`Particles` may be subclassed to carry further data and methods: a subclass is
taken wherever a `Particles` is, and `components` gives the arrays of any
particles by name:

```python
class Muons(parkin.Particles):
    def __init__(self, *, charge, **components):
        super().__init__(**components)
        self.charge = charge

muons = Muons(charge=np.array([-1, 1]), **mu1.components)

print(muons)
print(parkin.invariant_mass(muons, mu2))
```

```
<Muons (pt, eta, phi, mass), length 2, float64>
[91.18459273 65.41600255]
```

Particles stored in another layout need not be rearranged. Every function
converts its arguments with `as_particles`, which reads these layouts in
place:

| Layout | How it is read |
|---|---|
| `Particles` | as it is |
| Mapping from field name to array | each array |
| Structured array | each field, as a strided view |

In a mapping and in a structured array the components are recognised by field
name. Exactly one name of each row must be present; other fields are ignored.

| Component | Field names |
|---|---|
| Transverse momentum | `pt`, `rho` |
| Pseudorapidity | `eta` |
| Azimuthal angle | `phi` |
| Mass | `mass`, `m`, `M`, `tau` |

```python
muon = [("pt", "f8"), ("eta", "f8"), ("phi", "f8"), ("m", "f8"), ("charge", "i1")]
mu1 = np.array([(45.0, 0.3, 0.1, 0.106, -1), (38.2, -1.2, 2.9, 0.106, 1)], dtype=muon)
mu2 = np.array([(41.5, -0.4, 3.0, 0.106, 1), (27.9, -1.0, -0.4, 0.106, -1)], dtype=muon)

print(parkin.invariant_mass(mu1, mu2))
print(parkin.as_particles(mu1))
```

```
[91.18459273 65.41600255]
<Particles (pt, eta, phi, mass), length 2, float64>
```

The fields of a structured array are interleaved in memory, so they take
longer to read than separate arrays, the more so the more other fields a
record holds.

`system_info()` describes the installed build and the host. Include its
printed form in a bug report:

```python
print(parkin.system_info())
```

```
parkin 0.1.0
compiler: GNU 14.2.0
xsimd: 14.3.0
CPU features: sse2, sse3, ssse3, sse4.1, sse4.2, avx, fma3, avx2
```

It returns a `SystemInfo` with the fields `parkin_version`, `compiler`,
`xsimd_version` and `cpu_features`.

## Configuration

The execution settings decide how a call is run, not what it computes:

| Setting | Environment variable | Default | Meaning |
|---|---|---|---|
| `threads` | `PARKIN_THREADS` | every CPU the process may run on | Largest number of threads a call uses. |
| `simd` | `PARKIN_SIMD` | the widest set available | Instruction set the computation runs with: `scalar`, or a SIMD set the build and the processor support (`sse2` on x86-64, `neon64` on 64-bit Arm). |
| `block_size` | `PARKIN_BLOCK_SIZE` | 4096 | Number of elements a thread takes at a time. An array of at most this many elements is computed on the calling thread alone. |

The environment variables are read when parkin is imported.
`parkin.execution` changes settings for the calls made inside a `with` block,
in the thread that enters it, and `parkin.get_execution()` returns the
settings in effect:

```python
with parkin.execution(threads=4):
    masses = parkin.invariant_mass(mu1, mu2)
```

For a given instruction set, a result does not depend on the number of threads
or on the block size. Two instruction sets can differ in the last bits of a
result, because each has its own implementation of the math functions.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
