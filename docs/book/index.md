---
title: parkin
summary: The documentation of parkin, a library for the computation of kinematic observables on array data.
---

## About parkin

parkin (particle kinematics) is a high-performance Python library for the
computation of kinematic observables, from invariant masses, angular
separations and Lorentz transformations to jet clustering, jet substructure
and event shapes. Its C++ core combines zero-copy access to contiguous array
data, such as NumPy, Awkward and memory-mapped arrays, with SIMD vectorisation
and multithreading on CPUs and CUDA acceleration on GPUs.

The library is at the start of its development and has no release yet. Its
source and the installation instructions are in the
[repository](https://github.com/AlkaidCheng/parkin).

## About this documentation

The documentation states what the library computes and how. Each observable
has a section of its own, which gives its definition, the formula in the form
that is evaluated, the derivation of that form, and the conventions for the
inputs that a definition leaves open. A section is added when its observable
is implemented, so the documentation grows with the library.

Sections, equations and definitions are numbered, so that any statement can be
referred to precisely.
