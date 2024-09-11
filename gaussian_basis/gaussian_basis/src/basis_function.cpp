#include "basis_function.hpp"


const Gaussian3D &BasisFunction::operator[](int index) const {
    return this->primitives[index];
}

bool BasisFunction::operator==(const BasisFunction &w) const {
    return (this->count == w.count && this->primitives == w.primitives);
}

