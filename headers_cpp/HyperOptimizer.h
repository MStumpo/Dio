#pragma once
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include "HyperParameters.h"

struct HyperOptimizer {
    std::vector<std::pair<HyperParameters, double>> hist;

    std::mt19937 rng;
    std::normal_distribution<double> dist{0.0, 1.0};

    HyperOptimizer(): rng(std::random_device{}()) {}
    HyperParameters propose(size_t N = 5);
    void update(const HyperParameters& candidate, double score);
};
