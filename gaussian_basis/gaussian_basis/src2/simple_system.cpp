#include "simple_system.hpp"

#include "nuclear_charges.hpp"
#include "orbitals_description.hpp"
#include "atomic_data.hpp"

#include <vector>

static OrbitalsData get_atomic_description(int z, int &u_count, int &d_count) {
    OrbitalsData descr;
    switch (z) {
        case 1:
        u_count = 1, d_count = 0;
        descr = atomic_data_descriptions::ORB_1P1E_1S31;
        break;
        case 2:
        u_count = 1, d_count = 1;
        descr = atomic_data_descriptions::ORB_2P2E_1S4;
        break;
        case 3:
        u_count = 2, d_count = 1;
        descr = atomic_data_descriptions::ORB_3P3E_1S4_2S4;
        break;
        case 4:
        u_count = 2, d_count = 2;
        descr = atomic_data_descriptions::ORB_4P4E_1S4_2S4;
        break;
        case 5:
        u_count = 3, d_count = 2;
        descr = atomic_data_descriptions::ORB_5P5E_1S4_2S4_2P4;
        break;
        case 6:
        u_count = 4, d_count = 2;
        descr = atomic_data_descriptions::ORB_6P6E_1S5_2S5_2P5;
        break;
        case 7:
        u_count = 5, d_count = 2;
        descr = atomic_data_descriptions::ORB_7P7E_1S4_2S4_2P4;
        break;
        case 8:
        u_count = 5, d_count = 3;
        descr = atomic_data_descriptions::ORB_8P8E_1S5_2S5_2P5;
        break;
        case 9:
        u_count = 5, d_count = 4;
        descr = atomic_data_descriptions::ORB_9P9E_1S4_2S4_2P4;
        break;
        case 10:
        u_count = 5, d_count = 5;
        descr = atomic_data_descriptions::ORB_10P10E_1S6_2S6_2P6;
        break;
        case 11:
        u_count = 6, d_count = 5;
        descr = atomic_data_descriptions::ORB_11P11E_1S4_2S4_2P4_3S4;
        break;
        case 12:
        u_count = 6, d_count = 6;
        descr = atomic_data_descriptions::ORB_12P12E_1S4_2S4_2P4_3S4;
        break;
        case 13:
        u_count = 7, d_count = 6;
        descr = atomic_data_descriptions::ORB_13P13E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 14:
        u_count = 8, d_count = 6;
        descr = atomic_data_descriptions::ORB_14P14E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 15:
        u_count = 9, d_count = 6;
        descr = atomic_data_descriptions::ORB_15P15E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 16:
        u_count = 9, d_count = 7;
        descr = atomic_data_descriptions::ORB_16P16E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 17:
        u_count = 9, d_count = 8;
        descr = atomic_data_descriptions::ORB_17P17E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 18:
        u_count = 9, d_count = 9;
        descr = atomic_data_descriptions::ORB_18P18E_1S4_2S4_2P4_3S4_3P4;
        break;
        case 19:
        u_count = 10, d_count = 9;
        descr = atomic_data_descriptions::ORB_19P19E_1S4_2S4_2P4_3S4_3P4_4S4;
        break;
        case 20:
        u_count = 10, d_count = 10;
        descr = atomic_data_descriptions::ORB_20P20E_1S4_2S4_2P4_3S4_3P4_4S4;
        break;
        case 21:
        u_count = 11, d_count = 10;
        descr = atomic_data_descriptions::ORB_21P21E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 22:
        u_count = 12, d_count = 10;
        descr = 
            atomic_data_descriptions::ORB_22P22E_1S4_2S4_2P4_3S4_3P4_3D211_4S211;
        break;
        case 23:
        u_count = 13, d_count = 10;
        descr = 
            atomic_data_descriptions::ORB_23P23E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 24:
        u_count = 14, d_count = 10;
        descr = 
            atomic_data_descriptions::ORB_24P24E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 25:
        u_count = 15, d_count = 10;
        descr = 
            atomic_data_descriptions::ORB_25P25E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 26:
        u_count = 15, d_count = 11;
        descr = 
            atomic_data_descriptions::ORB_26P26E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 27:
        u_count = 15, d_count = 12;
        descr = 
            atomic_data_descriptions::ORB_27P27E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 28:
        u_count = 15, d_count = 13;
        descr = 
            atomic_data_descriptions::ORB_28P28E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 29:
        u_count = 15, d_count = 14;
        descr = 
            atomic_data_descriptions::ORB_29P29E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 30:
        u_count = 15, d_count = 15;
        descr = 
            atomic_data_descriptions::ORB_30P30E_1S6_2S6_2P6_3S6_3P6_3D6_4S6;
        break;
        case 31:
        u_count = 16, d_count = 15;
        descr = 
            atomic_data_descriptions::ORB_31P31E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 32:
        u_count = 17, d_count = 15;
        descr = 
            atomic_data_descriptions::ORB_32P32E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 33:
        u_count = 18, d_count = 15;
        descr = 
            atomic_data_descriptions::ORB_33P33E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 34:
        u_count = 18, d_count = 16;
        descr = 
            atomic_data_descriptions::ORB_34P34E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 35:
        u_count = 18, d_count = 17;
        descr = 
            atomic_data_descriptions::ORB_35P35E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
        case 36:
        u_count = 18, d_count = 18;
        descr = 
            atomic_data_descriptions::ORB_36P36E_1S6_2S6_2P6_3S6_3P6_3D6_4S6_4P6;
        break;
    }
    return descr;
}

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