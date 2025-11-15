#pragma once
#include <vector>
#include <memory>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <cstdio>
#include "Network.h"
#include "DataTerminal.h"

struct Network;
struct Neuron;
struct Edge;
using NeuronPointer = std::shared_ptr<Neuron>;
using EdgePointer = std::shared_ptr<Edge>;
using namespace std;

struct SharedNetwork {
    vector<NeuronPointer> neurons;
    vector<EdgePointer> edges;
    vector<Network*> sub_networks;
    vector<DataTerminal> terminals;

    // Node / Edge creation
    NeuronPointer makeNeuron(uint8_t v = false, Network* n = nullptr);
    EdgePointer makeEdge(const NeuronPointer& s, const NeuronPointer& d, double v = 0.0);

    // Merge neurons
    void mergeNeuron(NeuronPointer& dominant, NeuronPointer& recessive);

    // Dynamics
    void updateTrace();
    void neuronFiring();
    void clampData();
};
