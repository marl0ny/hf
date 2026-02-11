# Hartree-Fock (WIP)

## 1. Using Finite Differences to Solve the Hartree-Fock Equations for Simple Atomic Systems
The Hartree-Fock equations for single atom multi-electron systems are evaluated by first exploiting spherical symmetry to reduce the equations to 1D, discretizing these equations using finite differences, then subsequently solving them using numerical matrix methods. A more detailed explanation of this approach is found in chapter 23 of <i>Applied Computational Physics</i> by Joseph Boudreau and Eric Swanson.  A non-uniform grid is used in the discretization process, which is introduced in [exercise 5.1](https://books.google.ca/books?id=flolqBpoJeEC&lpg=PA94&ots=sykRIPQppz&dq=computational%20physics%20thijssen%20chapter%205&pg=PA109) of the book <i>Computational Physics</i> by Jos Thijssen. 

Solving the Hartree-Fock equations requires evaluating somewhat challenging integrals that contain the product of orbitals with their Coulomb interaction. For finding these Coulomb integrals with spherical symmetry assumed, a good introduction is example 2.7 of <i>Introduction to Electrodynamics</i> by David J. Griffiths. While the approach used in this pedagogical example is simple and conspicuously intuitive, it unfortunately only works for s-type orbitals. When confronting orbitals with higher angular momenta, the full spherical harmonics expansion of the Coulomb potential is necessary, which may be found, for example, in equation 23.39 in the Boudreau and Swanson book mentioned previously. From here the integrals are algebraically evaluated on a computer using Sympy's [Gaunt](https://docs.sympy.org/latest/modules/physics/wigner.html#sympy.physics.wigner.gaunt) and [spherical harmonics](https://docs.sympy.org/latest/modules/functions/special.html#spherical-harmonics) functions.

The initial input orbitals to kick-start the Hatree-Fock iteration are the exact solutions to the single electron Hydrogen-like atoms, which are obtained from [Wikipedia](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics) and [Hyperphysics](http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/hydwf.html). The final output atomic orbitals are then fitted with Gaussian functions, where these Gaussians are used to construct the basis functions in the next method.

## 2. Gaussian Basis
Solve the Hartree-Fock Roothaan equations using a set of basis functions composed of a combination of Gaussian primitives. This requires the evaluation of difficult Gaussian integrals, where [this article](https://joshuagoings.com/2017/04/28/integrals/) by Joshua Goings is used to get their solutions. Unlike in the previous finite difference implementation, spherical symmetry is not assumed, so this method is not restricted to single atoms and can be applied in general to molecular systems.

For some of the molecules, the [Computational Chemistry Comparison and Benchmark DataBase](https://cccbdb.nist.gov/exp2x.asp) is used to get the experimental geometry. Reference tables of theoretical [computed values](https://cccbdb.nist.gov/energy1x.asp) are consulted as well.

### Some General Resources and References
For a first introduction to the Hartree-Fock method, [these notes](http://vergil.chemistry.gatech.edu/notes/hf-intro/hf-intro.html) and [video series](https://www.youtube.com/watch?v=qcYxyP_SDLU) by David Sherrill are incredibly helpful. The book <i>Modern Quantum Chemistry</i> by Attila Szabo is an invaluable resource as well.

## Build Instructions
Currently this project has only been tested on Linux and MacOS. For running the two example scripts `closed_shell_atoms.py` and `unrestricted_open_shell_atoms.py` found inside the `./finite_difference`
directory, the only requirements are Python3.9 or greater, and the modules listed in the
`requirements.txt` file. Install these modules by invoking `python3 -m pip install -r requirements.txt`.

Additional steps must be taken to run the example scripts found in the `./gaussian_basis` directory, as they depend on an additional extension module that is written in C++ instead of Python (note that the source files for this extension are found in the `./gaussian_basis/gaussian_basis/src` directory). First these system packages must be installed: the [Clang compiler collection](https://clang.llvm.org/)
and [Boost Math library](https://www.boost.org/doc/libs/?view=category_math) (Boost Math is required because its [Hypergeometric 1F1](https://live.boost.org/doc/libs/master/libs/math/doc/html/math_toolkit/hypergeometric/hypergeometric_1f1.html) function is used in calculating those Gaussian integrals that involve the Coulomb potential). For MacOS the Clang compiler comes with the official XCode command line tools provided by Apple, and the Boost math libraries must be installed using [Homebrew](https://brew.sh/). For Linux, installing these dependencies differ based on which distribution and package manager is being used. For example, when using Debian-based systems with apt-get, use the following command: `sudo apt-get install clang lldb lld libboost-math-dev`. After installing these necessary prerequisite packages, build the required extension module by invoking `python3 -m setup build_ext`. Running the examples in the `./gaussian_basis` directory should now work. For example, to visualize the probability density of the electrons in a water molecule enter the following command while in the `./gaussian_basis` directory: `python3 water.py`.



## Todo List
- [ ] Proper geometry optimization to find the lowest energy configuration, instead of relying on the experimental geometry or brute-force iterating over some possible ones
- [ ] Gaussian basis functions for orbitals higher than 2p
- [ ] Post Hartree-Fock methods
- [ ] Time-dependent Hartree-Fock equations