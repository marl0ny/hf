#include "orbitals_parameters.hpp"
#include "vec3.hpp"
#include "orbitals.hpp"

#include <iostream>

using namespace spatial;
using namespace molecular_geometry;

#include <string>
#include <fstream>
#include <iostream>

static std::string open_file(std::string filename) {
    std::fstream file{std::string(filename), std::fstream::in};
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filename << ".\n";
        return "";
    }
    std::string file_contents = "";
    char c;
    while ((c = file.get()) && !file.eof())
        file_contents += c;
    file.close();
    return file_contents;
}

int main() {
    AtomicOrbitalsParameters a 
        = get_atomic_orbitals_parameters_from_file_name(
        spatial::Vec3{.x=0.0, .y=0.0, .z=0.0}, 
        "../../../data/7p7e_1s6_2s42_2p42.json");
    a.print();
    std::string o_data = open_file("../../../data/7p7e_1s6_2s42_2p42.json");
    std::string h_data = open_file("../../../data/1p1e_1s4.json");
    molecular_geometry::MolecularGeometry geom;
    geom.add_atom(AtomicSymbol::H, Vec3{.x=1.0, .y=0.0, .z=0.0});
    geom.add_atom(AtomicSymbol::H, Vec3{.x=0.0, .y=1.0, .z=0.0});
    geom.add_atom(AtomicSymbol::O, Vec3{.x=0.0, .y=0.0, .z=0.0});
    Orbitals o = Orbitals::from_geometry_and_atomic_json_data(
        geom, 
        {
            {AtomicSymbol::H, h_data},
            {AtomicSymbol::O, o_data}
        });
    o.print();
    std::vector<Nuclear> nuclear_config = geom.get_nuclear_configuration();
    for (Nuclear n: nuclear_config) {
        std::cout << n.charge << std::endl;
        std::cout << n.position.x << ", " << n.position.y << 
            ", " << n.position.z << std::endl;
    }
    return 0;
}
