#pragma once
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdint>

struct HyperParameters {
    size_t NEURON_SIZE = 10;

    double lr = 0.1;
    double reg = 0.1;
    double entropy_factor = 1.0;
    double decay = 0.1;
    double u_decay = 0.1;
    double determinism = 0.1;
    double firing_value = 1;
    double contrib_factor = 1.0;
    double alpha = 0.5; //yes I am a double alpha

    std::vector<uint8_t> log_scale = {true, true, false, false, false, false, false, false, false};
    std::vector<std::pair<double,double>> limits = {
        {1e-7,1.0},{0.015,1.0},{-2.0,2.0},{0.0,1.0},{0.0,1.0},{0.0,1.0},{-3.0,3.0}, {0.0,1.0}, {0.0, 100.0}
    };


    size_t SIZE = 9;

    HyperParameters(size_t s = 10, double l = 0.01, double r = 0.001, double e = 1.0, double d = 0.01,
                    double u = 0.01, double det = 0.5, double f = 1.0, double contrib = 1.0, double alph = 0.5)
        : NEURON_SIZE(s), lr(l), reg(r), entropy_factor(e), decay(d), u_decay(u), determinism(det), firing_value(f), contrib_factor(contrib), alpha(alph)
    {}

    size_t size() const { return SIZE; }
    double& operator[](size_t i) {
        switch(i) {
            case 0: return lr;
            case 1: return reg;
            case 2: return entropy_factor;
            case 3: return decay;
            case 4: return u_decay;
            case 5: return determinism;
            case 6: return firing_value;
            case 7: return contrib_factor;
            case 8: return alpha;
            default: throw std::out_of_range("Invalid HyperParameter index");
        }
    }

};
