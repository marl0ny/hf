# Hartree-Fock (WIP)

## Finite Differences
The Hartree-Fock equations for atoms are solved by first exploiting spherical symmetry to reduce the equations to 1D, and then a finite difference discretization is done where the resulting equations are solved using numerical matrix methods. This is following chapter 23 of the book <i>Applied Computational Physics</i> by Joseph Boudreau and Eric Swanson.  A non uniform grid is used for the discretization, which is introduced in [exercise 5.1](https://books.google.ca/books?id=flolqBpoJeEC&lpg=PA94&ots=sykRIPQppz&dq=computational%20physics%20thijssen%20chapter%205&pg=PA109) of <i>Computational Physics</i> by Jos Thijssen. To remember how to compute the Coulomb integrals given spherical symmetry, example 2.7 in <i>Introduction to Electrodynamics</i> by David J. Griffiths was a good reminder. The beginning of the Hatree-Fock iteration starts with the orbitals for the Hydrogen-like atoms, which are referenced from [Wikipedia](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics) and [Hyperphysics](http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/hydwf.html). 

## Gaussian Basis
Solve the Hartree-Fock Roothaan equations using basis functions that are composed of Gaussian functions. This requires the evaluation of Gaussian integrals, where [this document](https://joshuagoings.com/2017/04/28/integrals/) by Joshua Goings is the primary reference.

For some of the molecules, the [Computational Chemistry Comparison and Benchmark DataBase](https://cccbdb.nist.gov/exp2x.asp) was consulted to get the experimental geometry. Reference tables of theoretical [computed values](https://cccbdb.nist.gov/energy1x.asp) were used as well.

For a first introduction to the Hartree-Fock method, [these notes](http://vergil.chemistry.gatech.edu/notes/hf-intro/hf-intro.html) and [video series](https://www.youtube.com/watch?v=qcYxyP_SDLU) by David Sherrill were very helpful. The book <i>Modern Quantum Chemistry</i> by Attila Szabo was an invaluable resource as well.
