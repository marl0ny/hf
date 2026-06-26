#include "vec3.hpp"
#include "nuclear.hpp"
#include <vector>
#include <map>

#ifndef _MOLECULAR_GEOMETRY_
#define _MOLECULAR_GEOMETRY_

namespace molecular_geometry {

    using namespace spatial;

    enum class AtomicSymbol {
        H=1, HE=2, 
        LI=3, BE=4, B=5, C=6, N=7, O=8, F=9, NE=10,
        NA=11, MG=12, AL=13, SI=14, P=15, S=16, CL=17, AR=18,
        K=19, CA=20,
    };

    class MolecularGeometry {
        public:
        MolecularGeometry();
        MolecularGeometry(
            AtomicSymbol atomic_symbol,
            const Vec3 &position);
        MolecularGeometry(
            AtomicSymbol atomic_symbol, 
            const std::vector<Vec3> &positions);
        MolecularGeometry(
            std::vector<AtomicSymbol> atomic_symbols, 
            const std::vector<std::vector<Vec3>> &positions_list);
        MolecularGeometry(const std::map<AtomicSymbol, std::vector<Vec3>>&);
        void add_atom(AtomicSymbol atomic_symbol, const Vec3 &position);
        std::vector<Vec3>operator[](AtomicSymbol atomic_symbol) const;
        MolecularGeometry operator+(const Vec3 &position) const;
        MolecularGeometry operator-(const Vec3 &position) const;
        MolecularGeometry operator+(const MolecularGeometry &geom) const;
        MolecularGeometry rotate(
            double angle, const Vec3 &rotation_axis,
            const Vec3 &offset=Vec3{.x=0.0, .y=0.0, .z=0.0}) const;
        NuclearConfiguration get_nuclear_configuration() const;
        private:
        std::vector<std::vector<Vec3>> m_positions;
        void create_empty_m_positions();
    };

    MolecularGeometry
    make_oh(double oh_length, const Vec3 &axis=Vec3{.x=0.0, .y=0.0, .z=1.0});

    MolecularGeometry
    make_co(double co_length, const Vec3 &axis=Vec3{.x=0.0, .y=0.0, .z=1.0});

    MolecularGeometry
    make_ch2(double ch1, double ch2, double hch_angle,
        const Vec3 &axis1, const Vec3 &axis2);

    MolecularGeometry
    make_ch3(double ch1, double ch2, double ch3, double ach_angle,
        const Vec3 &axis1, const Vec3 &axis2);

}

#endif
