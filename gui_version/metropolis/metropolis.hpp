/* See, for example, pg. 429 to 430 of
"An Introduction to Computer Simulation Methods"
by Harvey Gould, Jan Tobochnik, and Wolfgang Christian for a general 
outline of the Metropolis algorithm.

Gould H., Tobochnik J., Christian W., "Numerical and Monte Carlo Methods,"
in <i>An Introduction to Computer Simulation Methods</i>,
2016, ch 11., pg 406-444.

*/ 
#include <vector>


struct MetropolisResultInfo {
    int accepted_count, rejection_count;
};

MetropolisResultInfo metropolis(
    std::vector<double> &configs, 
    const std::vector<double> &x0,
    const std::vector<double> &delta,
    double (* dist_func)(const std::vector<double> &x, void *params),
    int steps, void *params);
