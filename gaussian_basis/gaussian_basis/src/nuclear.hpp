#include "vec3.hpp"
#include <vector>

#ifndef _NUCLEAR_
#define _NUCLEAR_


struct Nuclear {
    unsigned long charge;
    spatial::Vec3 position;
};

using NuclearConfiguration = std::vector<Nuclear>;


#endif

