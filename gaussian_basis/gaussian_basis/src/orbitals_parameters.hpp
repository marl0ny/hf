#include "vec3.hpp"
#include <memory>
#include <vector>
#include <string>

#ifndef _ORBITALS_PARAMETERS_
#define _ORBITALS_PARAMETERS_


/*Class for encapsulating the parameters for a Gaussian primitive.
This is not meant to be used by itself, but as part of a composition
in the BasisFunctionParameters, OrbitalParameters, etc classes.

This does not include the position or the angular number of the primitive,
which are stored in those classes which this forms the composition of.

*/
struct PrimitiveParameters {
    double coefficient;
    double exponent;
};

/* Does not include its position or the angular numbers
of each of the primitives.*/
class BasisFunctionParameters {
    double coefficient;
    std::vector<PrimitiveParameters> primitives;
    public:
    auto begin() const {return primitives.begin();};
    auto end() const {return primitives.end();};
    void set_coefficient(double coefficient);
    void add_primitive(const PrimitiveParameters &p);
    PrimitiveParameters get_primitive(int i) const;
    PrimitiveParameters operator[](int i) const;
    double get_coefficient() const;
    int number_of_primitives() const;

};

class OrbitalParameters {
    char name[3];
    std::vector<BasisFunctionParameters> basis_functions;
    public:
    auto begin() const {return basis_functions.begin();};
    auto end() const {return basis_functions.end();};
    OrbitalParameters();
    void set_name(const std::string &s);
    void add_basis_function(const BasisFunctionParameters &f);
    std::string get_name() const;
    int get_multiplicity() const;
    int get_angular_number() const;
    BasisFunctionParameters get_basis_function(int i) const;
    BasisFunctionParameters operator[](int i) const;
    int number_of_primitives() const;
    int multiplicity_adjusted_number_of_primitives() const;
    int number_of_basis_functions() const;
    int multiplicity_adjusted_number_of_basis_functions() const;
};

class AtomicOrbitalsParameters {
    spatial::Vec3 position;
    std::vector<OrbitalParameters> orbital_parameters;
    public:
    auto begin() const {return orbital_parameters.begin();};
    auto end() const {return orbital_parameters.end();};
    spatial::Vec3 get_position() const;
    void set_position(const spatial::Vec3 &position);
    void add_orbital(const OrbitalParameters &o);
    OrbitalParameters get_orbital(int i) const;
    OrbitalParameters operator[](int i) const;
    int number_of_primitives() const;
    int multiplicity_adjusted_number_of_primitives() const;
    int number_of_basis_functions() const;
    int multiplicity_adjusted_number_of_basis_functions() const;
    int number_of_orbitals() const;
    int multiplicity_adjusted_number_of_orbitals() const;
    void print() const;
};

class MolecularOrbitalsParameters {
    std::vector<AtomicOrbitalsParameters> atomic_orbitals_parameters;
    public:
    auto begin() const {return atomic_orbitals_parameters.begin();};
    auto end() const {return atomic_orbitals_parameters.end();};
    void add_atomic_orbitals(const AtomicOrbitalsParameters &a);
    AtomicOrbitalsParameters get_atomic_orbitals(int i) const;
    const std::vector<AtomicOrbitalsParameters> &get_all_atomic_orbitals() const;
    AtomicOrbitalsParameters operator[](int i) const;
    int number_of_primitives() const;
    int multiplicity_adjusted_number_of_primitives() const;
    int number_of_basis_functions() const;
    int multiplicity_adjusted_number_of_basis_functions() const;
    int number_of_orbitals() const;
    int multiplicity_adjusted_number_of_orbitals() const;

};

AtomicOrbitalsParameters get_atomic_orbitals_parameters(
    const spatial::Vec3 &position, const std::string &parameters_file);

AtomicOrbitalsParameters get_atomic_orbitals_parameters_from_file_name(
    const spatial::Vec3 &position, const std::string &parameters_file);

#endif
