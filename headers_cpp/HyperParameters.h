#pragma once
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdint>

struct HyperParameters {
    size_t NEURON_SIZE = 10;
    double lr = 0.01;
    double reg = 0.01;
    double entropy_factor = 1.0;
    double decay = 0.1;
    double u_decay = 0.1;
    double determinism = 0.5;
    double firing_value = 1.0;

    std::vector<uint8_t> log_scale = {true, true, false, false, false, false, false};
    std::vector<std::pair<double,double>> limits = {
        {1e-3,1.0},{1e-3,1.0},{-2.0,2.0},{0.00,1.0},{0.00,1.0},{0.0,1.0},{-3.0,3.0}
    };


    size_t SIZE = 7;

    HyperParameters(size_t s = 10, double l = 0.01, double r = 0.001, double e = 1.0, double d = 0.01,
                    double u = 0.01, double det = 0.5, double f = 1.0)
        : NEURON_SIZE(s), lr(l), reg(r), entropy_factor(e), decay(d), u_decay(u), determinism(det), firing_value(f)
    {}

    double& operator[](size_t i);
    size_t size() const { return SIZE; }
};
