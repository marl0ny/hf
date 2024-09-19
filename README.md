# Hartree-Fock (WIP)

## Finite Differences
The Hartree-Fock equations for single atoms are solved by first exploiting spherical symmetry to reduce the equations to 1D, and then a finite difference discretization is applied to them. These are subsequently solved using numerical matrix methods. This follows chapter 23 of <i>Applied Computational Physics</i> by Joseph Boudreau and Eric Swanson.  A non uniform grid is used for the discretization, which is introduced in [exercise 5.1](https://books.google.ca/books?id=flolqBpoJeEC&lpg=PA94&ots=sykRIPQppz&dq=computational%20physics%20thijssen%20chapter%205&pg=PA109) of <i>Computational Physics</i> by Jos Thijssen. 

To remember how to compute the Coulomb integrals given spherical symmetry, at least for the simple s-type orbitals example 2.7 in <i>Introduction to Electrodynamics</i> by David J. Griffiths was a good reminder. However, when dealing with orbitals with higher angular momenta the full spherical harmonics expansion of the Coulomb potential was necessary, which can for example be found in equation 23.39 in the Boudreau and Swanson book mentioned previously. From here the form of the repulsion-exchange integrals are then obtained by using Sympy's [spherical harmonics](https://docs.sympy.org/latest/modules/functions/special.html#spherical-harmonics) and [Gaunt](https://docs.sympy.org/latest/modules/physics/wigner.html#sympy.physics.wigner.gaunt) functions.

The initial input to start the Hatree-Fock iteration are the orbitals for the Hydrogen-like atoms, which are referenced from [Wikipedia](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics) and [Hyperphysics](http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/hydwf.html). 

The resulting finite difference orbitals are then fitted with Gaussian functions, where these Gaussians are used to construct the basis functions used in the next method described.

## Gaussian Basis
Solve the Hartree-Fock Roothaan equations using basis functions that are composed of Gaussian functions. This requires the evaluation of Gaussian integrals, where [this post](https://joshuagoings.com/2017/04/28/integrals/) by Joshua Goings is the primary reference.

For some of the molecules, the [Computational Chemistry Comparison and Benchmark DataBase](https://cccbdb.nist.gov/exp2x.asp) was consulted to get the experimental geometry. Reference tables of theoretical [computed values](https://cccbdb.nist.gov/energy1x.asp) were used as well.

For a first introduction to the Hartree-Fock method, [these notes](http://vergil.chemistry.gatech.edu/notes/hf-intro/hf-intro.html) and [video series](https://www.youtube.com/watch?v=qcYxyP_SDLU) by David Sherrill were very helpful. The book <i>Modern Quantum Chemistry</i> by Attila Szabo was an invaluable resource as well.

## Build Instructions
Currently this project has only been tested on Linux and MacOS. For running the examples found in the `finite_difference`
directory, the only requirements is Python3.9 or greater, as well as those modules listed in the
requirements.txt file. Install these by invoking `python3 -m pip install -r requirements.txt `.

Additional steps must be taken to be able to run the examples found in the `gaussian_basis` directory, because they depend on an additional extension that is written in C++ instead of Python. First these packages need to be installed in your system: the [Clang compiler](https://clang.llvm.org/)
and [Boost Math library](https://www.boost.org/doc/libs/?view=category_math) (Boost Math is required for using the [Hypergeometric 1F1](https://live.boost.org/doc/libs/master/libs/math/doc/html/math_toolkit/hypergeometric/hypergeometric_1f1.html) function for calculating integrals involving the Coulomb potential). For MacOS Boost Math must be installed using [HomeBrew](https://brew.sh/). Numpy may also need to be installed locally in your home directory, as the path used for building the extension assumes this. You can then build the required extension by invoking `python3 -m setup build_ext`.
