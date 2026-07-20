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

NuclearChargesArray::NuclearChargesArray(
    std::vector<NuclearCharge> charges) {
    m_nuclear_charges = {};
    for (const NuclearCharge &e: charges)
        this->push_back(e);
}

void NuclearChargesArray::push_back(NuclearCharge c) {
    m_nuclear_charges.push_back({
        .position=c.position,
        .strength=c.strength
    });
}

double NuclearChargesArray::get_energy() {
    double sum = 0.0;
    for (int i = 0; i < this->size(); i++) {
        for (int j = i + 1; j < this->size(); j++) {
            double q_i = this->strength(i);
            double q_j = this->strength(j);
            spatial::Vector r = this->location(i) - this->location(j);
            sum += q_i*q_j/std::sqrt(dot(r, r));
        }
    }
    return sum;
}