#include "simple_system.hpp"

#include "nuclear_charges.hpp"
#include "orbitals_description.hpp"
#include "atomic_data.hpp"

#include <vector>

System::System(): 
m_nuclear_charges({}), m_atomic_orbitals({}),
m_electron_count(0), m_up_count(0), m_down_count(0) {

}

const NuclearChargesArray &System::get_nuclear_charges() const {
    return m_nuclear_charges;
}

const std::vector<orbital_description_data::PositionedOrbitalsData>
    & System::get_atomic_orbital_description_list() const {
    return m_atomic_orbitals;
}

void System::add_atom(int atomic_number, spatial::Vector position) {
    int u_count = 0, d_count = 0;
    orbital_description_data::OrbitalsData descr = 
    get_atomic_description(
        atomic_number, u_count, d_count);
    if (atomic_number == 1) {
        u_count = 1, d_count = 0;
    }
    m_electron_count += atomic_number;
    orbital_description_data::PositionedOrbitalsData 
        element {descr, position};
    m_atomic_orbitals.push_back(element);
    m_nuclear_charges.push_back(
        {.position=position, .strength=(int)atomic_number});
    int total_count = m_up_count + m_down_count + u_count + d_count;
    m_down_count = total_count / 2;
    m_up_count = total_count - m_down_count;
}

void System::clear() {
    this->m_nuclear_charges = NuclearChargesArray({});
    this->m_atomic_orbitals 
        = std::vector<orbital_description_data::PositionedOrbitalsData> {};
    this->m_electron_count = 0;
    this->m_up_count = 0;
    this->m_down_count = 0;
}

unsigned int System::electron_count() const {
    return this->m_electron_count;
}

unsigned int System::get_up_count() const {
    return this->m_up_count;
}

unsigned int System::get_down_count() const {
    return this->m_down_count;
}