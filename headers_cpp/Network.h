
#pragma once
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include "HyperOptimizer.h"
#include "HyperParameters.h"

class Network;
struct SharedNetwork;

struct Neuron {
    uint8_t value;
    double trace;
    std::vector<Network*> members;
    Neuron(uint8_t v = false, double t = 0.0, Network* n = nullptr)
        : value(v), trace(t), members({n}) {};
};
using NeuronPointer = std::shared_ptr<Neuron>;

struct Edge {
    NeuronPointer sender;
    NeuronPointer destination;
    double value;
    double U = 0.0;
    Edge(NeuronPointer s, NeuronPointer d, double v = 0.0, double u = 0.0)
        : sender(std::move(s)), destination(std::move(d)), value(v), U(u) {}
};

using EdgePointer = std::shared_ptr<Edge>;


class Network {
    private:
    SharedNetwork* shared;

    class AdjMatrix {
    private:
        Network& parent;
        std::vector<std::vector<EdgePointer>> data;

        std::vector<double> colEntropy();

    public:
        AdjMatrix(Network& parent_network);
        void initialize();
        void updateAdj();
        const std::vector<std::vector<EdgePointer>>& getData() const { return data; }
    };

    public:
    AdjMatrix adj;

public:
    HyperParameters hp;
    HyperOptimizer opt;
    std::vector<NeuronPointer> neurons;

    Network(SharedNetwork* s, HyperParameters& hp_arg);

    bool operator[](size_t i) const;
    size_t size() const;

    // printing / utility functions
    void printAdjMatrix(int width = 1, int decimals = 2);
    void printUMatrix(int width = 1, int decimals = 2);
    void printNetwork(const std::vector<int>& pos = {}, bool new_line = true);

    const AdjMatrix& getAdjMatrix() const { return adj; }
};
