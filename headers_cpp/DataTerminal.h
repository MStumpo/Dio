#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>

//#include "Network.h"

struct Neuron;

using namespace std;

using NeuronPointer = std::shared_ptr<Neuron>;

struct DataTerminal
{
    int id;
    vector<NeuronPointer> coordinates = {};
    size_t size;
    bool calibration;
    vector<uint8_t> values; //Same indexes as coords, meant to be updated constantly with time
    bool clamped;
    void updateValues(vector<uint8_t> new_vals);
    DataTerminal(int i, size_t s, bool c) :
    id(i), size(s), calibration(c), values(vector<uint8_t>(s, 0)){}
};
