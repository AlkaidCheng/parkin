---
title: Invariant mass of a pair of particles
summary: The mass of the system that two particles form, and the form of its formula that keeps its precision for collinear and for light pairs.
references:
  pdg2024: "S. Navas et al. (Particle Data Group), *Review of Particle Physics*, Phys. Rev. D **110** (2024) 030001, [doi:10.1103/PhysRevD.110.030001](https://doi.org/10.1103/PhysRevD.110.030001), review *Kinematics*."
  goldberg1991: "D. Goldberg, *What every computer scientist should know about floating-point arithmetic*, ACM Comput. Surv. **23** (1991) 5, [doi:10.1145/103162.103163](https://doi.org/10.1145/103162.103163)."
  sterbenz1974: "P. H. Sterbenz, *Floating-Point Computation*, Prentice-Hall, Englewood Cliffs (1974)."
  brun1997root: "R. Brun and F. Rademakers, *ROOT: an object oriented data analysis framework*, Nucl. Instrum. Meth. A **389** (1997) 81, [doi:10.1016/S0168-9002(97)00048-X](https://doi.org/10.1016/S0168-9002(97)00048-X)."
  chopra2025vector: "S. Chopra, H. Schreiner, E. Rodrigues, J. Eschle and J. Pivarski, *Vector: JIT-compilable mathematical manipulations of ragged Lorentz vectors*, J. Open Source Softw. **10** (2025) 7791, [doi:10.21105/joss.07791](https://doi.org/10.21105/joss.07791)."
---

The invariant mass of a pair of particles is the mass of the system that the
two form, that is, the magnitude of the sum of their four-momenta
[@pdg2024]. It has the same value in every inertial frame, and for the two
products of a two-body decay it is the mass of the parent. This section
derives the form of the formula that the library evaluates for two particles
given by their transverse momentum, pseudorapidity, azimuthal angle and mass.

## Definition

!!! definition "Invariant mass of a pair"
    The invariant mass $m_{12}$ of two particles with the four-momenta $p_1$
    and $p_2$ is given by

    $$
    \begin{aligned}
    & m_{12}^2 = (p_1 + p_2)^2
    \narrowbreak = (E_1 + E_2)^2 - |\vec{p}_1 + \vec{p}_2|^2 .
    \end{aligned}
    \label{eq:pair-mass-definition}
    $$

Expanding the square and using $\eqref{eq:mass-shell}$ for each particle gives

$$
\begin{aligned}
& m_{12}^2 = m_1^2 + m_2^2
\narrowbreak + 2 \left( E_1 E_2 - \vec{p}_1 \cdot \vec{p}_2 \right) .
\end{aligned}
\label{eq:pair-mass-expanded}
$$

The masses of the two particles enter as they are. Everything that depends on
how the particles move relative to each other is in the last term, the
Minkowski product $p_1 \cdot p_2 = E_1 E_2 - \vec{p}_1 \cdot \vec{p}_2$. It is
the sum of two terms,

$$
\begin{aligned}
& E_1 E_2 - \vec{p}_1 \cdot \vec{p}_2
\narrowbreak = \left( E_1 E_2 - |\vec{p}_1| |\vec{p}_2| \right)
\narrowbreak + \left( |\vec{p}_1| |\vec{p}_2| - \vec{p}_1 \cdot \vec{p}_2 \right) .
\end{aligned}
\label{eq:pair-mass-two-terms}
$$

The first is the energy term, which vanishes for two massless particles. The
second is the angular term, which equals
$|\vec{p}_1| |\vec{p}_2| (1 - \cos\theta_{12})$ for the opening angle
$\theta_{12}$ of the pair and vanishes for collinear particles. Neither term
is negative for timelike or lightlike momenta: the first because
$E \ge |\vec{p}|$, the second by the Cauchy-Schwarz inequality.

## The angular term

Let $\Delta\eta = \eta_1 - \eta_2$ and $\Delta\phi = \phi_1 - \phi_2$ be the
separations of the two particles. With the Cartesian components of
$\eqref{eq:cartesian-components}$ and the magnitude of
$\eqref{eq:momentum-magnitude}$,

$$
\begin{aligned}
\vec{p}_1 \cdot \vec{p}_2 &= \pTi{1} \pTi{2}
\bigl[ \cos\Delta\phi \narrowbreak + \sinh\eta_1 \sinh\eta_2 \bigr] , \\
|\vec{p}_1| |\vec{p}_2| &= \pTi{1} \pTi{2} \cosh\eta_1 \cosh\eta_2 .
\end{aligned}
\label{eq:pair-products}
$$

Their difference contains
$\cosh\eta_1 \cosh\eta_2 - \sinh\eta_1 \sinh\eta_2$, which is
$\cosh\Delta\eta$ by the addition theorem of the hyperbolic cosine. The
angular term is therefore

$$
\begin{aligned}
& |\vec{p}_1| |\vec{p}_2| - \vec{p}_1 \cdot \vec{p}_2
\narrowbreak = \pTi{1} \pTi{2} \left( \cosh\Delta\eta - \cos\Delta\phi \right) .
\end{aligned}
\label{eq:angular-term-difference}
$$

Both functions in the bracket are close to one for a small separation, and
only their difference from one matters. The identities
$\cosh x - 1 = 2 \sinh^2 (x/2)$ and $1 - \cos x = 2 \sin^2 (x/2)$ remove the
common part:

$$
\begin{aligned}
& |\vec{p}_1| |\vec{p}_2| - \vec{p}_1 \cdot \vec{p}_2
\narrowbreak = 2 \, \pTi{1} \pTi{2}
\Bigl( \sinh^2 \frac{\Delta\eta}{2} \narrowbreak + \sin^2 \frac{\Delta\phi}{2} \Bigr) .
\end{aligned}
\label{eq:angular-term}
$$

## The energy term

Let $d$ be the excess of the energy of a particle over the magnitude of its
momentum. Since $m^2 = (E - |\vec{p}|)(E + |\vec{p}|)$ by
$\eqref{eq:mass-shell}$,

$$
d = E - |\vec{p}| = \frac{m^2}{E + |\vec{p}|} .
\label{eq:energy-excess}
$$

Writing each energy as $E = |\vec{p}| + d$ turns the energy term into

$$
\begin{aligned}
& E_1 E_2 - |\vec{p}_1| |\vec{p}_2|
\narrowbreak = |\vec{p}_1| \, d_2 + |\vec{p}_2| \, d_1 + d_1 d_2 .
\end{aligned}
\label{eq:energy-term}
$$

## The formula as evaluated

Inserting $\eqref{eq:angular-term}$ and $\eqref{eq:energy-term}$ into
$\eqref{eq:pair-mass-expanded}$ by way of $\eqref{eq:pair-mass-two-terms}$ gives

$$
\begin{aligned}
m_{12}^2 = {} & m_1^2 + m_2^2
\narrowbreak + 2 \left( |\vec{p}_1| \, d_2 + |\vec{p}_2| \, d_1 + d_1 d_2 \right) \\
& {} + 4 \, \pTi{1} \pTi{2}
\Bigl( \sinh^2 \frac{\Delta\eta}{2} \narrowbreak + \sin^2 \frac{\Delta\phi}{2} \Bigr) ,
\end{aligned}
\label{eq:pair-mass}
$$

where each particle contributes

$$
\begin{aligned}
& |\vec{p}| = \pT \cosh\eta ,
\narrowsep E = \sqrt{m^2 + |\vec{p}|^2} ,
\narrowsep d = \frac{m^2}{E + |\vec{p}|} .
\end{aligned}
\label{eq:pair-mass-inputs}
$$

This is the form the library evaluates. It is the definition
$\eqref{eq:pair-mass-definition}$ rearranged, with no approximation.

## Special cases

Massless particles
:   Here $d_1 = d_2 = 0$, and with $\eqref{eq:angular-term-difference}$ the
    formula becomes the one usually quoted for a pair of jets or of photons,
    $m_{12}^2 = 2 \, \pTi{1} \pTi{2} \left( \cosh\Delta\eta - \cos\Delta\phi \right)$.

A small separation
:   For massless particles with $|\Delta\eta| \ll 1$ and $|\Delta\phi| \ll 1$,
    $m_{12}^2 \approx \pTi{1} \pTi{2} \, \Delta R^2$ with
    $\Delta R^2 = \Delta\eta^2 + \Delta\phi^2$. The mass of a collimated
    massless pair is proportional to its angular separation.

A particle paired with itself
:   The angular term vanishes and the energy term is
    $2 |\vec{p}| d + d^2 = d (E + |\vec{p}|) = m^2$, so that
    $m_{12}^2 = 4 m^2$ and $m_{12} = 2m$.

One particle at rest
:   For $\pTi{1} = 0$ the first particle has $|\vec{p}_1| = 0$ and
    $d_1 = m_1$, and the formula becomes the fixed-target expression
    $m_{12}^2 = m_1^2 + m_2^2 + 2 m_1 E_2$.

## Precision in floating-point arithmetic

The definition $\eqref{eq:pair-mass-definition}$ is exact, but evaluated in
floating-point arithmetic it subtracts two numbers of the size of the squared
energy of the pair, $E^2 = (E_1 + E_2)^2$. Each of them carries a rounding
error of the relative size $\epsilon$, the machine epsilon, which is
$1.2 \times 10^{-7}$ in single precision and $2.2 \times 10^{-16}$ in double
precision. Their difference therefore has an absolute error near
$\epsilon E^2$ whatever its own size, and the relative error of the squared
mass is near $\epsilon E^2 / m_{12}^2$. This loss is known as catastrophic
cancellation [@goldberg1991]. For the two photons from the decay of a neutral
pion of 50 GeV, $m_{12}^2 / E^2 \approx 7 \times 10^{-6}$, and in single
precision the squared mass would be wrong at the level of a percent.

Equation $\eqref{eq:pair-mass}$ avoids the cancellation in three places:

1. Every term is non-negative for timelike and lightlike momenta. A sum of
   non-negative numbers loses no digits, so its relative error is that of its
   terms, a small multiple of $\epsilon$.
2. The excess $d$ is computed as the quotient in $\eqref{eq:energy-excess}$ and
   never as the difference $E - |\vec{p}|$. For a light particle $d$ is about
   $m^2 / 2|\vec{p}|$, which can lie far below the rounding error of the
   energy.
3. The only subtractions left are $\Delta\eta$ and $\Delta\phi$, which are
   differences of the inputs themselves. The difference of two floating-point
   numbers within a factor of two of each other is exact [@sterbenz1974], so a
   small separation is known as precisely as the inputs give it. The squared
   sines of $\eqref{eq:angular-term}$ then carry that precision into the result,
   which the bracket of $\eqref{eq:angular-term-difference}$ would not.

!!! remark "What the formula cannot recover"
    The result is precise for the inputs as they are given. The mass of a
    nearly collinear pair is proportional to its separation, so a rounding
    error already present in the stored angles appears in the mass in full.
    In particular, two azimuthal angles on either side of $\pm\pi$ differ by
    nearly $2\pi$, and that difference is rounded at the size of $2\pi$, not
    at the size of the separation.

## Conventions

The definition leaves some inputs open. The library treats them as ROOT
[@brun1997root] and vector [@chopra2025vector] do.

| Input | Convention |
|---|---|
| Units | Transverse momenta and masses share one unit, which is also that of the result. |
| Azimuthal angles | Any range. Only $\sin^2 (\Delta\phi / 2)$ enters, which has the period $2\pi$ in $\Delta\phi$. |
| A negative mass $m$ | Denotes a spacelike four-momentum with the squared mass $-m^2$. In $\eqref{eq:pair-mass}$ and $\eqref{eq:pair-mass-inputs}$, $m^2$ then stands for $m \lvert m \rvert$. |
| A spacelike four-momentum with $-m^2 > \lvert\vec{p}\rvert^2$ | Has no real energy and is given $E = 0$. |
| A four-momentum of zero | Has $d = 0$. |
| A negative $m_{12}^2$ | Gives the mass $-\sqrt{-m_{12}^2}$. |
| A NaN in any component | Gives NaN. |

Table: Conventions for the inputs that the definition of the pair mass leaves open.
