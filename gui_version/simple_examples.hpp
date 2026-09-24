#include <map>
#include "spatial.hpp"

#ifndef _SIMPLE_EXAMPLES_
#define _SIMPLE_EXAMPLES_


void closed_shell_element_example(int z);

void unrestricted_element_example(int z, bool verbose=false);

void h2_example();

void h2o_example();

void benzene_example();

void co2_example();

void o2_example();

void simple_system(
    std::vector<std::pair<unsigned int, spatial::Vector>> &atoms,
    bool closed, unsigned int iter_count);


#endif