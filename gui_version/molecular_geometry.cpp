#include "molecular_geometry.hpp"
#include <cmath>

using namespace molecular_geometry;


static spatial::Vector rotate(
    spatial::Vector r, double angle, 
    spatial::Vector axis, spatial::Vector offset) {
    #define MUL(a, b)\
        {a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],\
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],\
        a[0]*b[2] + a[2]*b[0] + a[3]*b[1] - a[1]*b[3],\
        a[0]*b[3] + a[3]*b[0] + a[1]*b[2] - a[2]*b[1]} 
    double c = cos(angle/2.0), s = sin(-angle/2.0);
    double norm = sqrt(
        axis.x*axis.x + axis.y*axis.y + axis.z*axis.z);
    double rot[4] = {c, s*axis.x/norm, s*axis.y/norm, s*axis.z/norm};
    double rot_inv[4] = {rot[0], -rot[1], -rot[2], -rot[3]};
    double q0[4] = {1.0, 
        r[0] - offset[0], r[1] - offset[1], r[2] - offset[2]};
    double q0_rot[4] = MUL(q0, rot);
    double qf[4] = MUL(rot_inv, q0_rot);
    #undef MUL
    return spatial::Vector{
        .t=0.0,
        .x=qf[1] + offset[0],
        .y=qf[2] + offset[1],
        .z=qf[3] + offset[2]};
}


void MolecularGeometry::create_empty_m_positions() {
    m_positions = {{}};
    // Only goes up to Calcium, for now.
    for (int _ = 1; _ <= 20; _++)
        m_positions.push_back({});
}

MolecularGeometry::MolecularGeometry() {
    this->create_empty_m_positions();
}

MolecularGeometry::MolecularGeometry(
    AtomicSymbol atomic_symbol,
    const spatial::Vector &position
) {
    this->create_empty_m_positions();
    m_positions[int(atomic_symbol)].push_back(position);
}

MolecularGeometry::MolecularGeometry(
    AtomicSymbol atomic_symbol, 
    const std::vector<spatial::Vector> &positions
) {
    this->create_empty_m_positions();
    for (auto &position: positions)
        m_positions[int(atomic_symbol)].push_back(position);
}

MolecularGeometry::MolecularGeometry(
    std::vector<AtomicSymbol> atomic_symbols, 
    const std::vector<std::vector<spatial::Vector>> &positions_list) {
    this->create_empty_m_positions();
    int i = 0;
    for (AtomicSymbol &atomic_symbol: atomic_symbols) {
        for (auto &position: positions_list[i])
            m_positions[int(atomic_symbol)].push_back(position);
        i++;
    }
}

MolecularGeometry
::MolecularGeometry(const std::map<AtomicSymbol, 
    std::vector<spatial::Vector>> &atom_positions) {
    for (auto &e: atom_positions) {
        AtomicSymbol atomic_symbol = e.first;
        std::vector<spatial::Vector> positions = e.second;
        for (auto &position: positions)
            m_positions[int(atomic_symbol)].push_back(position);
    }
}

void MolecularGeometry
::add_atom(AtomicSymbol atomic_symbol, const spatial::Vector &position) {
    m_positions[int(atomic_symbol)].push_back(position);
}

std::vector<spatial::Vector> MolecularGeometry
::operator[](AtomicSymbol atomic_symbol) const {
    return m_positions[int(atomic_symbol)];
}

MolecularGeometry MolecularGeometry
::operator+(const spatial::Vector &translate_position) const {
    MolecularGeometry res{};
    for (int n = 1; n < m_positions.size(); n++) {
        for (auto &position: m_positions[n])
            res.m_positions[n].push_back(
                position + translate_position);
    }
    return res;
}

MolecularGeometry MolecularGeometry
::operator-(const spatial::Vector &translate_position) const {
    MolecularGeometry res{};
    for (int n = 1; n < m_positions.size(); n++) {
        for (auto &position: m_positions[n])
            res.m_positions[n].push_back(
                position - translate_position);
    }
    return res;
}

MolecularGeometry MolecularGeometry
::operator+(const MolecularGeometry &geom) const {
    MolecularGeometry res{};
    for (int n = 1; n < m_positions.size(); n++) {
        for (auto &position: m_positions[n])
            res.m_positions[n].push_back(position);
        for (auto &position: geom.m_positions[n])
            res.m_positions[n].push_back(position);
    }
    return res;
}

MolecularGeometry MolecularGeometry
::rotate(
    double angle, 
    const spatial::Vector &rotation_axis, 
    const spatial::Vector &offset) const {
    MolecularGeometry res{};
    for (int n = 1; n < m_positions.size(); n++) {
        for (auto &position: m_positions[n])
            res.m_positions[n].push_back(
                ::rotate(position, angle, rotation_axis, offset)
            );
    }
    return res;
}

NuclearChargesArray MolecularGeometry
::get_nuclear_configuration() const {
    NuclearChargesArray config({});
    for (int atomic_number = 1; 
        atomic_number < m_positions.size(); atomic_number++) {
        for (auto &position: m_positions[atomic_number])
            config.push_back({
                .position=position,
                .strength=atomic_number
            });
    }
    return config;
}


molecular_geometry::MolecularGeometry make_oh(
    double oh_length, const spatial::Vector &axis
) {
    return MolecularGeometry({
        {AtomicSymbol::H, {oh_length*axis}},
        {AtomicSymbol::O, {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}}},
    });
}

molecular_geometry::MolecularGeometry
    make_co(double co_length, const spatial::Vector &axis=spatial::Vector{.x=0.0, .y=0.0, .z=1.0}) {
    return MolecularGeometry ({
        {AtomicSymbol::C, {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}}},
        {AtomicSymbol::O, {co_length*axis}},
    });
}

molecular_geometry::MolecularGeometry
make_ch2(double ch1, double ch2, double hch_angle,
    const spatial::Vector &axis1, const spatial::Vector &axis2) {
    spatial::Vector a1{axis1}, a2{axis2};
    return MolecularGeometry(
        {
            {AtomicSymbol::C, {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}}},
            {AtomicSymbol::H, 
                {
                    ch1*(a1*cos(hch_angle/2.0) + a2*sin(hch_angle/2.0)),
                    ch2*(a1*cos(hch_angle/2.0) - a2*sin(hch_angle/2.0))
                }
            }
        }
    );
}

molecular_geometry::MolecularGeometry
make_ch3(double ch1, double ch2, double ch3, double ach_angle,
    const spatial::Vector &axis1, const spatial::Vector &axis2) {
    spatial::Vector a{axis1}, x{axis2};
    spatial::Vector y = cross(a, x);
    spatial::Vector p1 = x;
    double phi = 2.0*(3.141592653589793)/3.0;
    spatial::Vector p2 = x*cos(phi) + y*sin(phi);
    spatial::Vector p3 = x*cos(2.0*phi) + y*sin(2.0*phi);
    return MolecularGeometry(
        {
            {AtomicSymbol::C, {{.t=0.0, .x=0.0, .y=0.0, .z=0.0}}},
            {AtomicSymbol::H, {
                {ch1*(a*cos(ach_angle) + p1*sin(ach_angle))},
                {ch2*(a*cos(ach_angle) + p2*sin(ach_angle))},
                {ch3*(a*cos(ach_angle) + p3*sin(ach_angle))}
            }}
        }
    );
}
