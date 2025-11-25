#include "Network.h"
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <algorithm>

#include "SharedNetwork.h"

using namespace std;

// ------------------- AdjMatrix -------------------
Network::AdjMatrix::AdjMatrix(Network& parent_network)
    : parent(parent_network) {}

vector<double> Network::AdjMatrix::colEntropy() {
    size_t N = data.size();
    vector<double> entropy(N, 0.0);

    for (size_t col = 0; col < N; col++) {
        for (size_t row = 0; row < N; row++) {
            if (data[row][col]->value != 0.0)
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
            parent.shared->edges.push_back(e);
        }
    }
}

void Network::AdjMatrix::updateAdj() {
    size_t N = data.size();
    vector<double> E = colEntropy();

    //#pragma omp for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            data[i][j]->value += parent.hp.lr * (parent.neurons[j]->value * data[i][j]->U * pow(E[j], parent.hp.entropy_factor) - parent.hp.reg * data[i][j]->value * pow(parent.neurons[i]->trace, 2));
            data[i][j]->value = max(-1.0, min(1.0, data[i][j]->value));
            //printf("\nDEBUG GGG %d, %d, %f, %f",parent.neurons[i]->value, parent.neurons[j]->value, data[i][j]->U, parent.hp.u_decay);
        }
    }
}


// ------------------- Network -------------------
Network::Network(SharedNetwork* s, HyperParameters& hp_arg)
    : shared(s), adj(*this), hp(hp_arg) {

    for (size_t i = 0; i < hp_arg.NEURON_SIZE; i++) {
        NeuronPointer n = make_shared<Neuron>(false, 0.0, this);
        neurons.push_back(n);

        shared->neurons.push_back(n);
    }

    adj.initialize();
}

size_t Network::size() const { return neurons.size(); }

void Network::printNetwork(const vector<int>& pos, bool new_line) {
    for (size_t b = 0; b < neurons.size(); b++) {
        printf("%d", neurons[b]->value ? true : false); //this doesn't
        for (auto p : pos) {
            if (b == static_cast<size_t>(p)) printf("|");
        }
    }
    if (new_line) printf("\n");
}

string Network::networkString(){
    string s = "";
    for(size_t b = 0; b < neurons.size(); b++){
        s.append(neurons[b]->value ? "1" : "0");
    }
    return s;
}

string Network::adjString(){
    string s = "";
    int width = 5;
    int prec  = 2;

    for (auto &r : adj.data) {
        s.append("\n");
        for (auto &c : r) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%*.*f", width, prec, c->value);
            s.append(buf);
        }
    }
    return s;

}
