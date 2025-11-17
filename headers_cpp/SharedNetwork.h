#pragma once
#include <vector>
#include <memory>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <cstdio>
#include <string>

#include "DataTerminal.h"
#include "DatasetManager.h"
#include "HyperParameters.h"
#include "HyperOptimizer.h"

using NeuronPointer = std::shared_ptr<Neuron>;
using EdgePointer = std::shared_ptr<Edge>;
struct DatasetManager;
struct Network;


using namespace std;

struct SharedNetwork {
    vector<NeuronPointer> neurons;
    vector<EdgePointer> edges;
    vector<Network*> sub_networks;
    vector<DataTerminal> terminals;
    std::unique_ptr<DatasetManager> data_manager;

    SharedNetwork(int time_window=10 , int null_window=10);
    NeuronPointer makeNeuron(uint8_t v, Network* n);
    EdgePointer makeEdge(const NeuronPointer& s, const NeuronPointer& d, double v);
    void makeSubNetwork(HyperParameters& hp);

    void mergeNeuron(NeuronPointer& dominant, NeuronPointer& recessive);

    // Dynamics
    void updateTrace();
    void neuronFiring();
    void clampData();
    void runDataset(int iterations, double test_fraction);
};
