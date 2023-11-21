# Hartree-Fock (WIP)

## Gaussian Basis
Solve the Hartree-Fock Roothaan equations using a Gaussian basis set. The Gaussian basis set requires having to solve for Gaussian integrals, where the primary reference for solving these integrals is [this document](https://joshuagoings.com/2017/04/28/integrals/) by Joshua Goings.

In order to find a suitable Gaussian basis, the script `gaussian_slater_fit.py` fits a set of Gaussians to the Hydrogen Slater orbitals and saves the data to the file `orbitals.json`. The Slater orbitals are referenced from [Wikipedia](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics) and [Hyperphysics](http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/hydwf.html).

[These notes](http://vergil.chemistry.gatech.edu/notes/hf-intro/hf-intro.html) and [video series](https://www.youtube.com/watch?v=qcYxyP_SDLU) by David Sherrill that introduces Hartree-Fock were very helpful for learning it for the first time. The book <i>Modern Quantum Chemistry</i> by Attila Szabo was an invaluable resource as well.

## Point Basis
The Hartree-Fock equations for atoms are solved by exploiting spherical symmetry to reduce the equations to 1D and then are discretized on a uniform grid of points. This is following chapter 23 of the book <i>Applied Computational Physics</i> by Joseph Boudreau and Eric Swanson. To remember how to compute the Coulomb integrals given spherical symmetry, example 2.7 in <i>Introduction to Electrodynamics</i> by David J. Griffiths was a good reminder. These calculations were also done separately on a non-uniform grid introduced in [exercise 5.1](https://books.google.ca/books?id=flolqBpoJeEC&lpg=PA94&ots=sykRIPQppz&dq=computational%20physics%20thijssen%20chapter%205&pg=PA109) of <i>Computational Physics</i> by Jos Thijssen.
