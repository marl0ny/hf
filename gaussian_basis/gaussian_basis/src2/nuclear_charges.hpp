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
    int size() const;
    int strength(int) const;
    spatial::Vector location(int) const;
};

#endif