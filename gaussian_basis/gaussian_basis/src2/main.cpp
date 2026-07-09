#include "basis_function_array.hpp"
#include "orbitals_description.hpp"

static unsigned int get_angular_number(const std::string &orbital_letter) {
    if (orbital_letter.substr(1) == "s") {
        return 0;
    }
    else if (orbital_letter.substr(1) == "p") {
        return 1;
    }
    else if (orbital_letter.substr(1) == "d") {
        return 2;
    }
    else if (orbital_letter.substr(1) == "d") {
        return 3;
    }
    return 0;
}

static unsigned int get_angular_multiplicity(const std::string &orbital_letter) {
    if (orbital_letter.substr(1) == "s") {
        return 1;
    }
    else if (orbital_letter.substr(1) == "p") {
        return 3;
    }
    else if (orbital_letter.substr(1) == "d") {
        return 5;
    }
    else if (orbital_letter.substr(1) == "d") {
        return 7;
    }
    return 0;
}

int main() {
    orbital_description_data::OrbitalsData h {

    }; 
    orbital_description_data::OrbitalsData o 
    {{
        { "1s", {
            {1.0, {
                {-1.565145837078264, -3.2236688634591735, -5.06767755890338, -4.97157485330118, -1.5385215642168175},
                {4917.464401508277, 639.5663856621088, 128.2768175526814, 32.356355456120546, 9.19912353939672}}
            },
        }},
        { "2s", {
            {0.3175179837964322, {
                {-2.9789387460278784, -5.597631221888869, -5.544100372008324},
                {1120.6258122308013, 110.93703793713162, 19.448441289134422}}
            },
            {0.7085814239470662, {
                {0.8450702001437524},
                {1.254999617721712}}
            },
            {0.3814201499948026, {
                {0.3417065041192091},
                {0.37525302696178114}}
            },
        }},
        { "2p", {
            {0.33895869962327, {
                {10.56860018785741, 11.773609092593366, 8.621957795319085},
                {104.96875904945789, 18.566810994814183, 4.783861677077649}}
            },
            {0.5147121761201113, {
                {2.0890150948130195},
                {1.3576917911330364}}
            },
            {0.3777298776428878, {
                {0.4004605146792502},
                {0.36215600491247746}}
            },
        }},
    }};
    BasisFunctionArray arr = orbital_description_data::get_basis_function_array(
        o, spatial::Vector{.x=0.0, 0.0, 0.0});
    arr.print();
    orbital_description_data::PositionedOrbitalsData q {
        o, {}};
    return 0;
}