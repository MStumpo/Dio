#include "SharedNetwork.h"
#include <ctime>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdio>

using namespace std;


#include "Network.h"


using NeuronPointer = std::shared_ptr<Neuron>;
using EdgePointer = std::shared_ptr<Edge>;

SharedNetwork::SharedNetwork(int time_window = 10, int null_window = 10) {
    data_manager = std::make_unique<DatasetManager>(this, "", null_window, time_window);
};

// ---------------- Neuron / Edge creation ----------------
NeuronPointer SharedNetwork::makeNeuron(uint8_t v, Network* n) {
    auto p = make_shared<Neuron>(v, 0.0, n);
    neurons.push_back(p);
    return p;
}

EdgePointer SharedNetwork::makeEdge(const NeuronPointer& s, const NeuronPointer& d, double v) {
    auto p = make_shared<Edge>(s, d, v);
    edges.push_back(p);
    return p;
}

// ---------------- Merge neurons ---------------- IN THIS CASE edges aren't removed from AdjMatrix because I still want both networks to be able to modify the merged neuron
void SharedNetwork::mergeNeuron(NeuronPointer& dominant, NeuronPointer& recessive) {
    unordered_set<Network*> seen(dominant->members.begin(), dominant->members.end());
    for (auto* ptr : recessive->members) {
        if (seen.insert(ptr).second) {
            dominant->members.push_back(ptr);
        }
    }

    edges.erase(
        std::remove_if(edges.begin(), edges.end(), [&](auto& e) {
            return (e->sender == dominant && e->destination == recessive || e->sender == recessive && e->destination == dominant);
        }),
        edges.end()
    );

    for (auto& e : edges) {
        if (e->sender == recessive) e->sender = dominant;
        if (e->destination == recessive) e->destination = dominant;
    }

    // Remove recessive neuron from list
    neurons.erase(remove(neurons.begin(), neurons.end(), recessive), neurons.end());

    // Update reference
    recessive = dominant;
}

void SharedNetwork::makeSubNetwork(HyperParameters& hp){
    auto new_net = std::make_unique<Network>(this,hp);
    sub_networks.push_back(new_net.get());
}

// ---------------- Dynamics ---------------- Sender neuron's original net controls trace decay
void SharedNetwork::updateTrace() {
    for (auto& n : neurons) {
        n->trace = n->trace * (1 - exp(-n->members[0]->hp.decay)) + n->members[0]->hp.decay * n->value;
    }

    for (auto& e : edges) {
        e->U = e->U * (1 - exp(-e->sender->members[0]->hp.u_decay)) +
               e->sender->members[0]->hp.u_decay * e->sender->trace * 2 * (e->destination->value - 0.5);
    }
}

void SharedNetwork::neuronFiring() {
    mt19937 gen(random_device{}());
    uniform_real_distribution<double> unif(0.0, 1.0);
    vector<double> newStates(neurons.size(), 0.0);

    for (auto& e : edges) {
        if (e->sender->value) {
            size_t idx = distance(neurons.begin(), find(neurons.begin(), neurons.end(), e->destination));
            if (idx < newStates.size()) {
                newStates[idx] += (unif(gen) > e->value)
                                  ? (e->value > 0 ? -1 : 1)
                                  : e->sender->members[0]->hp.determinism * e->value;
            }
        }
    }

    for (size_t i = 0; i < newStates.size(); i++) {
        neurons[i]->value = (newStates[i] >= neurons[i]->members[0]->hp.firing_value) ? 1 : 0; //maybe firing val is sum of firing vals from both networks?
    }
}

void SharedNetwork::clampData(){
    for(DataTerminal terminal : terminals){
        if(terminal.clamped){
            for(int i = 0; i < terminal.coordinates.size(); i++){
                terminal.coordinates[i]->value = terminal.values[i];
            }
        }
    }
}

void SharedNetwork::runDataset(int iterations, string p = ""){
    if(p == ""){
        printf("ASSUMING DATASETMANAGER'S DATASET");
    }else{
        data_manager->path = p;
    }
    for(int t = 0; t < iterations; t++){
        //TODO
    }
}