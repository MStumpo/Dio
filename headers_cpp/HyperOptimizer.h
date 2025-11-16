#pragma once
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include "HyperParameters.h"

struct HyperOptimizer {
    std::vector<HyperParameters> params;
    std::vector<double> scores;

    std::mt19937 rng;
    std::normal_distribution<double> dist{0.0, 1.0};

    HyperOptimizer(const HyperParameters& hyperparams, double init_step = 3);

    HyperParameters propose(size_t N = 5, double var = 0.1);
    void update(const HyperParameters& candidate, double score);
};
