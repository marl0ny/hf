#include "nuclear_charges.hpp"
#include "orbitals_description.hpp"

#ifndef _SIMPLE_SYSTEM_
#define _SIMPLE_SYSTEM_

class System {
    NuclearChargesArray m_nuclear_charges;
    std::vector<orbital_description_data::PositionedOrbitalsData>
        m_atomic_orbitals;
    unsigned int m_electron_count;
    unsigned int m_up_count, m_down_count;
    public:
    System();
    const NuclearChargesArray &get_nuclear_charges() const;
    const std::vector<orbital_description_data::PositionedOrbitalsData>
        & get_atomic_orbital_description_list() const;
    void add_atom(int atomic_number, spatial::Vector position);
    void clear();
    unsigned int get_up_count() const;
    unsigned int get_down_count() const;
    unsigned int electron_count() const;
};

#endif