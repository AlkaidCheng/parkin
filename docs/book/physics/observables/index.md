---
title: Kinematic observables
summary: Quantities computed from the four-momenta of particles, one section for each, with its definition, its derivation and its conventions.
references:
  pdg2024: "S. Navas et al. (Particle Data Group), *Review of Particle Physics*, Phys. Rev. D **110** (2024) 030001, [doi:10.1103/PhysRevD.110.030001](https://doi.org/10.1103/PhysRevD.110.030001), review *Kinematics*."
---

A kinematic observable is a quantity computed from the four-momenta of one or
more particles, such as a mass, an angle or a separation. Each section of this
chapter treats one observable: it gives the definition, derives the form of
the formula that the library evaluates, and lists the conventions for the
inputs that the definition does not cover. The sections share the notation
below, which is that of the Particle Data Group [@pdg2024].

Momenta, energies and masses are given in one unit, with the speed of light
set to one, and the metric has the signature $(+,-,-,-)$. A particle has the
four-momentum $p = (E, \vec{p})$ with the energy $E$ and the momentum
$\vec{p} = (p_x, p_y, p_z)$, and its mass $m$ is given by

$$
m^2 = p^2 = E^2 - |\vec{p}|^2 . \label{eq:mass-shell}
$$

A four-momentum is timelike for $m^2 > 0$, lightlike for $m^2 = 0$ and
spacelike for $m^2 < 0$. The four-momentum of a free particle is timelike or
lightlike.

The $z$ axis is the beam axis of a collider. The momentum transverse to it
has the magnitude $\pT$, the transverse momentum, and the direction $\phi$,
the azimuthal angle. The pseudorapidity $\eta$ gives the direction relative to
the beam, in terms of the polar angle $\theta$ between the momentum and the
$z$ axis:

$$
\begin{aligned}
& \pT = \sqrt{p_x^2 + p_y^2} ,
\narrowsep \phi = \operatorname{atan2}(p_y, p_x) ,
\narrowsep \eta = \operatorname{asinh} \frac{p_z}{\pT} = -\ln \tan \frac{\theta}{2} .
\end{aligned}
\label{eq:collider-coordinates}
$$

Conversely,

$$
\begin{aligned}
& p_x = \pT \cos\phi ,
\narrowsep p_y = \pT \sin\phi ,
\narrowsep p_z = \pT \sinh\eta ,
\end{aligned}
\label{eq:cartesian-components}
$$

and, because $\cosh^2\eta - \sinh^2\eta = 1$, the magnitude of the momentum is

$$
|\vec{p}| = \pT \cosh\eta . \label{eq:momentum-magnitude}
$$

Angles are in radians.
