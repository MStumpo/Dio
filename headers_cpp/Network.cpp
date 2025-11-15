#include "Network.h"
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <algorithm>

using namespace std;

// ------------------- AdjMatrix -------------------
Network::AdjMatrix::AdjMatrix(Network& parent_network)
    : parent(parent_network) {}

vector<double> Network::AdjMatrix::colEntropy() {
    size_t N = data.size();
    vector<double> entropy(N, 0.0);

    for (size_t col = 0; col < N; col++) {
        for (size_t row = 0; row < N; row++) {
            if (data[row][col]->value != 0.0 && data[row][col] != nullptr)
                entropy[col] += -abs(data[row][col]->value) * log(abs(data[row][col]->value));
        }
        entropy[col] /= N;
    }
    return entropy;
}

void Network::AdjMatrix::initialize() {
    random_device rnd_device;
    mt19937 rng(rnd_device());
    uniform_real_distribution<double> unif(-1.0, 1.0);
    auto gen = [&]() { return unif(rng); };

    size_t N = parent.neurons.size();
    data.resize(N);
    for (size_t i = 0; i < N; i++) {
        data[i].resize(N);
        for (size_t j = 0; j < N; j++) {
            auto e = make_shared<Edge>(parent.neurons[i], parent.neurons[j], gen());
            data[i][j] = e;
            parent.shared.edges.push_back(e);
        }
    }
}

void Network::AdjMatrix::updateAdj() {
    size_t N = data.size();
    vector<double> E = colEntropy();

    #pragma omp for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if(data[i][j] != nullptr){
                data[i][j]->value += parent.hp.lr * (parent.neurons[j]->value * data[i][j]->U *
                    pow(E[j], parent.hp.entropy_factor) - parent.hp.reg * data[i][j]->value *
                    pow(parent.neurons[i]->trace, 2));
                data[i][j]->value = max(-1.0, min(1.0, data[i][j]->value));
            }
        }
    }
}

// ------------------- Network -------------------
Network::Network(SharedNetwork& s, const HyperParameters& hp_arg)
    : shared(s), adj(*this), hp(hp_arg), opt(hp_arg) {

    for (size_t i = 0; i < hp_arg.SIZE; i++) {
        NeuronPointer n = make_shared<Neuron>(false, 0.0, this);
        neurons.push_back(n);
        s.neurons.push_back(n);
    }

    adj.initialize();
    s.sub_networks.push_back(this);
}

bool Network::operator[](size_t i) const { return neurons[i] != nullptr; }
size_t Network::size() const { return neurons.size(); }

// ------------------- Printing -------------------
void Network::printAdjMatrix(int width, int decimals) {
    printf("\n");
    auto& adjData = adj.getData();
    for (size_t i = 0; i < adjData.size(); ++i) {
        for (size_t j = 0; j < adjData[i].size(); ++j) {
            if (adjData[i][j]->value > 0) printf(" ");
            printf("%-*.*f ", width, decimals, adjData[i][j]->value);
        }
        printf("\n");
    }
}

void Network::printUMatrix(int width, int decimals) {
    // Assuming you have a U matrix somewhere (not defined in your snippet)
}

void Network::printNetwork(const vector<int>& pos, bool new_line) {
    for (size_t b = 0; b < neurons.size(); b++) {
        printf("%d", neurons[b] ? true : false);
        for (auto p : pos) {
            if (b == static_cast<size_t>(p)) printf("|");
        }
    }
    if (new_line) printf("\n");
}
