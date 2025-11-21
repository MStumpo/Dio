#include "SharedNetwork.h"
#include <ctime>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdio>
#include <string>
#include <format>
using namespace std;


#include "Network.h"


using NeuronPointer = std::shared_ptr<Neuron>;
using EdgePointer = std::shared_ptr<Edge>;

void progress_bar(int current, int total, string message = "") {
    int barWidth = 50;
    float progress = (float)current / total;

    printf(" \033[A \33[2K \r");
    int pos = (int)(barWidth * progress);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    if(message == ""){
        printf("] %f %%", (progress * 100));
    }else{
        printf("] %f%% %s ", (progress * 100), message.c_str());
    }
    fflush(stdout); // make sure it prints immediately
}

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


void SharedNetwork::createDatasetManager(string path) {
    data_manager = std::move(std::make_unique<DatasetManager>(this, path));
};

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
    sub_networks.push_back(std::make_unique<Network>(this, hp));
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

    for (auto& e : edges) {
        if (e->sender->value) {
            /* //Dunno what chatgpt was on about here this is some weird shit
            size_t idx = distance(neurons.begin(), find(neurons.begin(), neurons.end(), e->destination));
            if (idx < newStates.size()) {
                newStates[idx] += (unif(gen) < e->value)
                                  ? (e->value > 0 ? -1 : 1)
                                  : e->sender->members[0]->hp.determinism * e->value;
            }*/
            e->destination->buf += (unif(gen) < e->value)
                                  ? (e->value > 0 ? -1 : 1)
                                  : e->sender->members[0]->hp.determinism * e->value;
        }
    }

    for (auto& neuron : neurons) {
        neuron->value = (neuron->buf >= neuron->members[0]->hp.firing_value) ? 1 : 0; //maybe firing val is sum of firing vals from both networks?
        neuron->buf = 0.0;
    }
}

void SharedNetwork::clampData(){
    for(DataTerminal& terminal : terminals){
        if(terminal.clamped){
            for(int i = 0; i < terminal.coordinates.size(); i++){
                terminal.coordinates[i]->value = terminal.values[i];
            }
        }
    }
}

void SharedNetwork::runDataset(int iterations, int train_window, int test_window, int null_window =0, int optimize_period = -1, int verb = 0){
    //Verbose: 0- nothing 1- scores 3 - network 4- network + adj
    //I don't give a flying rat's ass if it's not implemented yet I am tired, busy and apparently autistic

    vector<double> current_scores(data_manager->score_calculators.size(), 0.0);
    string message;
    for(int i = 0; i < iterations; i++){
        message = "";
        if(verb >= 3){
            message.append(" ADJ MATRICES: \n");
            for(auto& net : sub_networks) message.append(format(" {}|", net->adjString()));
        }else if(verb >= 2){
	    message.append(" NETS: ");
 	    for(auto& net : sub_networks) message.append(format(" {}|", net->networkString()));
       }else if(verb >= 1){
            message.append(" SCORES: ");
            for(double score : current_scores) message.append(format(" {:} : " ,score/((1+i%optimize_period)*test_window)));
        }
        progress_bar(i, iterations, message);
	for(int t = 0; t < train_window + null_window + test_window; t++ ){
            if(t == 0) for(DataTerminal& terminal : terminals) terminal.clamped = true;
            if(t == train_window) for(DataTerminal& terminal : terminals) terminal.clamped = false;
            if(t == train_window + null_window)  for(DataTerminal& terminal : terminals) if(!terminal.calibration) terminal.clamped = false;
            clampData();
        	neuronFiring();
        	updateTrace();
        	for(auto& net : sub_networks){
        		net->adj.updateAdj();
        	}


            if(t >= train_window + null_window){
                for(int s = 0; s < current_scores.size(); s++){
		              current_scores[s] += data_manager->score_calculators[s].score(); //don't forget to avg later
                }
            }
        }

        if(i%optimize_period == 0){
            for(int s = 0; s < current_scores.size(); s++){
                current_scores[s] /= optimize_period*test_window; //don't forget to avg later
                for(auto& target : data_manager->score_calculators[s].targets){
                    target->opt.update(target->hp, current_scores[s]); //PROBLEM: TARGETS MUST NOT REPEAT-> TODO: sum same-target scores
                    target->hp = target->opt.propose();
                }
            }
            current_scores = vector<double>(data_manager->score_calculators.size(), 0.0);
        }
        data_manager->updateCurrentValues();
    }

}
