#pragma once
#include <vector>
#include <memory>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <cstdio>
#include <string>

#include "DataTerminal.h"
#include "HyperParameters.h"
#include "HyperOptimizer.h"

struct Network;
struct Neuron;
struct Edge;
struct DataManager;
using NeuronPointer = std::shared_ptr<Neuron>;
using EdgePointer = std::shared_ptr<Edge>;


using namespace std;

struct SharedNetwork {
    vector<NeuronPointer> neurons;
    vector<EdgePointer> edges;
    vector<unique_ptr<Network>> sub_networks;
    unique_ptr<DataManager> data_manager;

    SharedNetwork();
    ~SharedNetwork();

    NeuronPointer makeNeuron(uint8_t v, Network* n);
    EdgePointer makeEdge(const NeuronPointer& s, const NeuronPointer& d, double v);
    void makeSubNetwork(HyperParameters& hp);
    void mergeNeuron(NeuronPointer& dominant, NeuronPointer& recessive);
    void mergeNeuronsFromMatrix(vector<vector<int>> matrix, bool overlap);

    // Dynamics
    void updateTrace();
    void resetRandom();
    void neuronFiring();
    void clampData();
    void runDataset(int iterations, int train_window, int test_window, int null_window, int optimize_period, int verb);
    void runNethackOnline(int n_games, int verb);
    void runPlayground(size_t iterations, size_t window, bool optimize = false, int verb = 1);
    void runLogicTest(size_t iterations, size_t window, size_t rest_time, vector<vector<vector<uint8_t>>> dataset = {{}}, int optimize_period = -1);
};
