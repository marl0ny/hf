# Hartree-Fock (WIP)

## Finite Differences - Solving the Hartree-Fock Equations for Single Atoms
The Hartree-Fock equations for an atom are solved by first exploiting spherical symmetry to reduce the equations to 1D, then a finite difference discretization is applied to them. These are subsequently solved using numerical matrix methods. This follows chapter 23 of <i>Applied Computational Physics</i> by Joseph Boudreau and Eric Swanson.  A non uniform grid is used in the discretization, which is introduced in [exercise 5.1](https://books.google.ca/books?id=flolqBpoJeEC&lpg=PA94&ots=sykRIPQppz&dq=computational%20physics%20thijssen%20chapter%205&pg=PA109) of <i>Computational Physics</i> by Jos Thijssen. 

Evaluating integrals whose integrand contains the product of the Coulomb potential with the orbitals are necessary to solve the Hartree-Fock equations. For finding these Coulomb integrals given spherical symmetry, an intuitive and clear method is given in example 2.7 of <i>Introduction to Electrodynamics</i> by David J. Griffiths, at least for the s-type orbitals. However, when dealing with orbitals with higher angular momenta the full spherical harmonics expansion of the Coulomb potential is necessary, which can for example be found in equation 23.39 in the Boudreau and Swanson book mentioned previously. From here the form of the repulsion-exchange integrals are then obtained by using Sympy's [spherical harmonics](https://docs.sympy.org/latest/modules/functions/special.html#spherical-harmonics) and [Gaunt](https://docs.sympy.org/latest/modules/physics/wigner.html#sympy.physics.wigner.gaunt) functions.

The initial input to start the Hatree-Fock iteration are the orbitals for the Hydrogen-like atoms, which are referenced from [Wikipedia](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics) and [Hyperphysics](http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/hydwf.html). The resulting atomic orbitals of this finite difference Hartree-Fock algorithm are then fitted with Gaussian functions, where these Gaussians are used to construct the basis functions used in the next method.

## Gaussian Basis
Solve the Hartree-Fock Roothaan equations using basis functions that are composed of Gaussians. This requires the evaluation of Gaussian integrals, where [this post](https://joshuagoings.com/2017/04/28/integrals/) by Joshua Goings is the primary reference. Unlike in the previous finite difference implementation, spherical symmetry is not assumed, so this method is not restricted to single atoms and can be applied in general to molecular systems.

For some of the molecules, the [Computational Chemistry Comparison and Benchmark DataBase](https://cccbdb.nist.gov/exp2x.asp) is consulted to get the experimental geometry. Reference tables of theoretical [computed values](https://cccbdb.nist.gov/energy1x.asp) are used as well.

##
For a first introduction to the Hartree-Fock method, [these notes](http://vergil.chemistry.gatech.edu/notes/hf-intro/hf-intro.html) and [video series](https://www.youtube.com/watch?v=qcYxyP_SDLU) by David Sherrill are incredibly helpful. The book <i>Modern Quantum Chemistry</i> by Attila Szabo is an invaluable resource as well.

## Build Instructions
Currently this project has only been tested on Linux and MacOS. For running the examples found in the `finite_difference`
directory, the only requirements are Python3.9 or greater, and those modules listed in the
`requirements.txt` file. Install these by invoking `python3 -m pip install -r requirements.txt `.

Additional steps must be taken to be able to run the examples found in the `gaussian_basis` directory, because they depend on an additional extension that is written in C++ instead of Python. First these packages need to be installed in your system: the [Clang compiler collection](https://clang.llvm.org/)
and [Boost Math library](https://www.boost.org/doc/libs/?view=category_math) (Boost Math is required because it has the [Hypergeometric 1F1](https://live.boost.org/doc/libs/master/libs/math/doc/html/math_toolkit/hypergeometric/hypergeometric_1f1.html) function for calculating integrals involving the Coulomb potential). For MacOS the Clang compiler comes with the XCode command line tools, and the Boost math libraries must be installed using [Homebrew](https://brew.sh/). For Linux please refer to your package manager to install these dependencies. For example, in Debian-based systems using apt-get: `sudo apt-get install clang lldb lld libboost-math-dev`. You can then build the required extension by invoking `python3 -m setup build_ext`.

## Todo List
- [ ] Gaussian basis functions for orbitals higher than 2p
- [ ] Proper geometry optimization to find the lowest energy configuration, instead of relying on the experimental geometry or brute-force iterating over some possible ones
- [ ] Post Hartree-Fock methods
- [ ] Time-dependent Hartree-Fock equations