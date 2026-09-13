#include "spatial.hpp"
#include <vector>


#ifndef _NUCLEAR_CHARGES_
#define _NUCLEAR_CHARGES_

struct NuclearCharge {
    spatial::Vector position;
    int strength;
};

class NuclearChargesArray {
    std::vector<NuclearCharge> m_nuclear_charges;
    public:
    NuclearChargesArray(std::vector<NuclearCharge> charges);
    int size() const;
    int strength(int) const;
    spatial::Vector location(int) const;
    void push_back(NuclearCharge c);
    double get_energy();
    spatial::Vector get_center() const;
    double furthest_from_center() const;
};

#endif