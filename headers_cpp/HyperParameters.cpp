#include "HyperParameters.h"
#include <stdexcept>
#include <cmath>

double& HyperParameters::operator[](size_t i) {
    switch(i) {
        case 0: return lr;
        case 1: return reg;
        case 2: return entropy_factor;
        case 3: return decay;
        case 4: return u_decay;
        case 5: return determinism;
        case 6: return firing_value;
        case 7: return contrib_factor;
        default: throw std::out_of_range("Invalid HyperParameter index");
    }
}
