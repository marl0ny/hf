/* See, for example, pg. 429 to 430 of
"An Introduction to Computer Simulation Methods"
by Harvey Gould, Jan Tobochnik, and Wolfgang Christian for a general 
outline of the Metropolis algorithm.

Gould H., Tobochnik J., Christian W., "Numerical and Monte Carlo Methods,"
in <i>An Introduction to Computer Simulation Methods</i>,
2016, ch 11., pg 406-444.

*/ 
#include "metropolis.hpp"
#include <random>


typedef std::vector<double> Arr1D;

MetropolisResultInfo metropolis(
    Arr1D &configs, const Arr1D &x0, const Arr1D &delta,
    double (* dist_func)(const Arr1D &x, void *params),
    int steps, void *params
    ) {
    std::random_device rand_device;
    std::default_random_engine rand_engine (rand_device());
    std::uniform_real_distribution<double> rand(0.0, 1.0);
    Arr1D x_curr(x0);
    int size = x0.size();
    Arr1D x_next(size);
    double prob_curr = dist_func(x_curr, params);
    int accepted_count = 1;
    int rejection_count = 0;
    if (configs.size() != size*steps)
        configs.resize(size*steps);
    for (int step_count = 0; step_count < steps; step_count++) {
        for (int k = 0; k < size; k++) {
            configs[step_count*size + k] = x_curr[k];
            x_next[k] = x_curr[k] + delta[k]*(rand(rand_engine) - 0.5);
        }
        double prob_next = dist_func(x_next, params);
        if (prob_next >= prob_curr ||
            rand(rand_engine) <= prob_next/prob_curr) {
            prob_curr = prob_next;
            x_curr = x_next;
            accepted_count++;
        } else {
            rejection_count++;
        }
        
    }
    return {
        .accepted_count=accepted_count,
        .rejection_count=rejection_count};
}