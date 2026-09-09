#include "simple_examples.hpp"

#include <iostream>


int main() {
    struct timespec frame_time[2];
    clock_gettime(CLOCK_MONOTONIC, &frame_time[0]);
    // array_helpers::test1();
    // array_helpers::test2();
    // array_helpers::test3();
    // array_helpers::test4();
    // array_helpers::test5();
    // array_helpers::test6();
    // array_helpers::test7();
    // array_helpers::test8();
    // array_helpers::test9();
    // array_helpers::test10();
    // array_helpers::test11();
    // array_helpers::test12();
    // h2_example();
    h2o_example();
    // benzene_example();
    // co2_example();
    // o2_example();
    // for (int i = 1; i <= 20; i++) {
    //     printf("Atomic number: %d:\n", i);
    //     // printf("Closed:\n");
    //     // closed_shell_element_example(i);
    //     printf("Unrestricted Open:\n");
    //     unrestricted_element_example(i);
    //     puts("############################################################");
    // }
    clock_gettime(CLOCK_MONOTONIC, &frame_time[1]);
    double delta_t = frame_time[1].tv_sec - frame_time[0].tv_sec;
    std::cout << "Time taken: " << delta_t << "s \n";
    return 0;
}