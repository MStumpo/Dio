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
#include "NethackManager.h"
#include "HyperParameters.h"
#include "HyperOptimizer.h"


using NeuronPointer = std::shared_ptr<Neuron>;
using EdgePointer = std::shared_ptr<Edge>;
struct DatasetManager;
struct NethackManager;
struct Network;


using namespace std;

struct SharedNetwork {
    vector<NeuronPointer> neurons;
    vector<EdgePointer> edges;
    vector<unique_ptr<Network>> sub_networks;
    unique_ptr<DatasetManager> data_manager;
    unique_ptr<NethackManager> nh;

    NeuronPointer makeNeuron(uint8_t v, Network* n);
    EdgePointer makeEdge(const NeuronPointer& s, const NeuronPointer& d, double v);
    void makeSubNetwork(HyperParameters& hp);
    void createDatasetManager(string path);
    void createNethackManager(vector<int> input_indexes, int output_index);
    void mergeNeuron(NeuronPointer& dominant, NeuronPointer& recessive);
    void mergeNeuronsFromMatrix(vector<vector<int>> matrix, bool overlap);

    // Dynamics
    void updateTrace();
    void resetRandom();
    void neuronFiring();
    void clampData(bool is_nh);
    void runDataset(int iterations, int train_window, int test_window, int null_window, int optimize_period, int verb);
    void runNethackOnline(int n_games, int verb);
};
