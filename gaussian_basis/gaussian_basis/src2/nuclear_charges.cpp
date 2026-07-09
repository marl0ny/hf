#include "nuclear_charges.hpp"

int NuclearChargesArray::size() const {
    return m_nuclear_charges.size();
}

int NuclearChargesArray::strength(int k) const {
    return m_nuclear_charges[k].strength;
}

spatial::Vector NuclearChargesArray::location(int k) const {
    return m_nuclear_charges[k].position;
}