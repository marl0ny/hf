#include "orbitals_parameters.hpp"
#include "parse_subset_of_json.hpp"
#include "gaussian3d.hpp"
#include "basis_function.hpp"

#include  <fstream>
#include <iostream>

using namespace spatial;
using namespace parse_subset_of_json;

void
BasisFunctionParameters::set_coefficient(double coefficient) {
    this->coefficient = coefficient;
}

void 
BasisFunctionParameters::add_primitive(const PrimitiveParameters &p) {
    this->primitives.push_back(p);
}

PrimitiveParameters
BasisFunctionParameters::get_primitive(int i) const {
    return this->primitives[i];
}

PrimitiveParameters
BasisFunctionParameters::operator[](int i) const {
    return get_primitive(i);
}

int
BasisFunctionParameters::number_of_primitives() const {
    return this->primitives.size();
}

double
BasisFunctionParameters::get_coefficient() const {
    return this->coefficient;
}

OrbitalParameters::OrbitalParameters() {
    this->basis_functions = {};
    for (int i = 0; i < 3; i++) 
        name[i] = '\0';
}

void
OrbitalParameters::set_name(const std::string &s) {
    for (int i = 0; i < 3; i++)
        name[i] = s[i];
}

void
OrbitalParameters::add_basis_function(const BasisFunctionParameters &f) {
    this->basis_functions.push_back(f);
}

std::string
OrbitalParameters::get_name() const {
    return std::string(this->name);
}

int
OrbitalParameters::get_multiplicity() const {
    switch (name[1]) {
        case 's': return 1;
        case 'p': return 3;
        case 'd': return 5;
        case 'f': return 7;
    }
    return 0;
}

int
OrbitalParameters::get_angular_number() const {
    switch (name[1]) {
        case 's': return 0;
        case 'p': return 1;
        case 'd': return 2;
        case 'f': return 3;
    }
    return 0;
}

BasisFunctionParameters
OrbitalParameters::get_basis_function(int i) const {
    return this->basis_functions[i];
}

BasisFunctionParameters
OrbitalParameters::operator[](int i) const {
    return get_basis_function(i);
}

int
OrbitalParameters::number_of_primitives() const {
    int number_of_primitives = 0;
    for (const auto &basis_func: basis_functions)
        number_of_primitives += basis_func.number_of_primitives();
    return number_of_primitives;
}

int OrbitalParameters::
multiplicity_adjusted_number_of_primitives() const {
    return number_of_primitives()*get_multiplicity();
}

int
OrbitalParameters::number_of_basis_functions() const {
    return this->basis_functions.size();
}

int OrbitalParameters::
multiplicity_adjusted_number_of_basis_functions() const {
    return this->basis_functions.size()*get_multiplicity();
}

Vec3
AtomicOrbitalsParameters::get_position() const {
    return this->position;
}

void
AtomicOrbitalsParameters::set_position(const Vec3 &position) {
    this->position = position;
}

void
AtomicOrbitalsParameters::add_orbital(const OrbitalParameters &o) {
    this->orbital_parameters.push_back(o);
}

OrbitalParameters
AtomicOrbitalsParameters::get_orbital(int i) const {
    return this->orbital_parameters[i];
}

OrbitalParameters
AtomicOrbitalsParameters::operator[](int i) const {
    return get_orbital(i);
}

int
AtomicOrbitalsParameters::number_of_primitives() const {
    int number_of_primitives = 0;
    for (const auto& orbital: orbital_parameters)
        number_of_primitives += orbital.number_of_primitives();
    return number_of_primitives;
}

int AtomicOrbitalsParameters::
multiplicity_adjusted_number_of_primitives() const {
    int number_of_primitives = 0;
    for (const auto& orbital: orbital_parameters)
        number_of_primitives 
            += orbital.multiplicity_adjusted_number_of_primitives();
    return number_of_primitives;
}

int
AtomicOrbitalsParameters::number_of_basis_functions() const {
    int number_of_basis_functions = 0;
    for (const auto &orbital: orbital_parameters)
        number_of_basis_functions += orbital.number_of_basis_functions();
    return number_of_basis_functions;
}

int AtomicOrbitalsParameters::
multiplicity_adjusted_number_of_basis_functions() const {
    int number_of_basis_functions = 0;
    for (auto &orbital: this->orbital_parameters)
        number_of_basis_functions 
            += orbital.multiplicity_adjusted_number_of_basis_functions();
    return number_of_basis_functions;
}

int
AtomicOrbitalsParameters::number_of_orbitals() const {
    return this->orbital_parameters.size();
}

int
AtomicOrbitalsParameters::multiplicity_adjusted_number_of_orbitals() const {
    int number_of_orbitals = 0;
    for (const auto &orbital: this->orbital_parameters)
        number_of_orbitals += orbital.get_multiplicity();
    return number_of_orbitals;
}

void
AtomicOrbitalsParameters::print() const {
    for (int i = 0; i < this->number_of_orbitals(); i++) {
        OrbitalParameters o = this->get_orbital(i);
        std::cout << o.get_name() << std::endl;
        for (int j = 0; j < o.number_of_basis_functions(); j++) {
            BasisFunctionParameters b = o.get_basis_function(j);
            std::cout << "coefficient: " << b.get_coefficient() << std::endl;
            for (int k = 0; k < b.number_of_primitives(); k++) {
                if (k == 0)
                    std::cout << "\tcoefficient\texponent\n";
                PrimitiveParameters p = b.get_primitive(k);
                std::cout << "\t" << p.coefficient;
                std::cout << "\t" << p.exponent << std::endl;
            }
        }
    }
}

void MolecularOrbitalsParameters
::add_atomic_orbitals(const AtomicOrbitalsParameters &a) {
    this->atomic_orbitals_parameters.push_back(a);
}

AtomicOrbitalsParameters MolecularOrbitalsParameters
::get_atomic_orbitals(int i) const {
    return this->get_atomic_orbitals(i);
}

const std::vector<AtomicOrbitalsParameters> &MolecularOrbitalsParameters
::get_all_atomic_orbitals() const {
    return this->atomic_orbitals_parameters;
}

int MolecularOrbitalsParameters::number_of_primitives() const {
    int number_of_primitives = 0;
    for (const AtomicOrbitalsParameters& 
         atomic_orbitals: atomic_orbitals_parameters)
        number_of_primitives
            += atomic_orbitals.number_of_primitives();
    return number_of_primitives;
}

int MolecularOrbitalsParameters::
multiplicity_adjusted_number_of_primitives() const {
    int number_of_primitives = 0;
    for (const AtomicOrbitalsParameters& 
         atomic_orbitals: this->atomic_orbitals_parameters)
        for (const OrbitalParameters& orbital: atomic_orbitals) {
            int multiplicity = orbital.get_multiplicity();
            int orbital_primitive_count = orbital.number_of_primitives();
            number_of_primitives += multiplicity*orbital_primitive_count;
        }
    return number_of_primitives;
}

int MolecularOrbitalsParameters::number_of_basis_functions() const {
    int number_of_basis_functions = 0;
    for (const AtomicOrbitalsParameters& 
         atomic_orbitals: this->atomic_orbitals_parameters)
        number_of_basis_functions 
            += atomic_orbitals.number_of_basis_functions();
    return number_of_basis_functions;
}

int MolecularOrbitalsParameters
::multiplicity_adjusted_number_of_basis_functions() const {
    int number_of_basis_functions = 0;
    for (const AtomicOrbitalsParameters& 
         atomic_orbitals: this->atomic_orbitals_parameters)
        for (const OrbitalParameters &orbital: atomic_orbitals) {
            int multiplicity = orbital.get_multiplicity();
            int orbital_basis_func_count
                 = orbital.number_of_basis_functions();
            number_of_basis_functions 
                += multiplicity*orbital_basis_func_count;
        }
    return number_of_basis_functions;
}


int MolecularOrbitalsParameters::number_of_orbitals() const {
    int number_of_orbitals = 0;
    for (const AtomicOrbitalsParameters& 
         atomic_orbitals: atomic_orbitals_parameters)
        number_of_orbitals 
            += atomic_orbitals.number_of_orbitals();
    return number_of_orbitals;
}

int MolecularOrbitalsParameters::
multiplicity_adjusted_number_of_orbitals() const {
    int number_of_orbitals = 0;
    for (const AtomicOrbitalsParameters& 
         atomic_orbitals: atomic_orbitals_parameters)
        number_of_orbitals 
            += atomic_orbitals.multiplicity_adjusted_number_of_orbitals();
    return number_of_orbitals;
}


static std::string open_file(std::string filename) {
    std::fstream file{std::string(filename), std::fstream::in};
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filename << ".\n";
        return "";
    }
    std::string file_contents = "";
    char c;
    while ((c = file.get()) && !file.eof())
        file_contents += c;
    file.close();
    return file_contents;
}

AtomicOrbitalsParameters get_atomic_orbitals_parameters(
    const Vec3 &position, const string &contents
) {
    Bracket b = parse(contents);
    AtomicOrbitalsParameters atomic_orbitals {};
    for (auto &orbital_name: b.keys()) {
        OrbitalParameters orbital_parameters {};
        orbital_parameters.set_name(orbital_name);
        std::vector<Bracket> basis_func_list = b[orbital_name].bracket_list;
        for (auto &basis_func: basis_func_list) {
            BasisFunctionParameters basis_func_parameters {};
            basis_func_parameters.set_coefficient(
                basis_func["coefficient"].number);
            {
                Bracket primitives = basis_func["primitives"].bracket;
                std::vector<double> primitive_coefficients
                    = primitives["coefficients"].number_list;
                std::vector<double> exponents
                    = primitives["exponents"].number_list;
                for (int i = 0; i < primitive_coefficients.size(); i++) {
                    PrimitiveParameters primitive_parameters {
                        .coefficient=primitive_coefficients[i],
                        .exponent=exponents[i],
                    };
                    basis_func_parameters.add_primitive(
                        primitive_parameters
                    );
                }
            }
            orbital_parameters.add_basis_function(
                basis_func_parameters);
        }
        atomic_orbitals.add_orbital(orbital_parameters);
    }
    return atomic_orbitals;
}

AtomicOrbitalsParameters get_atomic_orbitals_parameters_from_file_name(
    const Vec3 &position, const string &parameters_file) {
    string contents = open_file(parameters_file);
    return get_atomic_orbitals_parameters(position, contents);
}


