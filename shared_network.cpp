#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "network.cpp"

using namespace std;

struct Neuron {
	uint8_t value;
	double trace;
	vector<Network*> members;
	Neuron(uint8_t v = false, double t = 0.0, Network* n)
		: value(v), trace(t), members({n}) {}
};
using NeuronPointer = shared_ptr<Neuron>;

struct Edge {
    NeuronPointer sender;
    NeuronPointer destination;
    double value;
    double U = 0.0;
    Edge(NeuronPointer s, NeuronPointer d, double v= 0.0, double u=0.0)
        : sender(move(s)), destination(move(d)), value(v),U(u) {}
};
using EdgePointer = shared_ptr<Edge>;

class SharedNetwork{
	vector<NeuronPointer> neurons;
	vector<EdgePointer> edges;
	vector<Network*> sub_networks;

    EdgePointer makeEdge(const NeuronPointer& s, const NeuronPointer& d, double v = 0.0) {
        auto p = make_shared<Edge>(s,d,v);
        edges.push_back(p);
        return p;
    }

    NeuronPointer makeNode(uint8_t v = false, Network* n = nullptr) {
        auto p = make_shared<Neuron>(v, 0.0, n);
        neurons.push_back(p);
        return p;
    }

    void mergeNeuron(NeuronPointer& dominant, NeuronPointer& recessive){
    	//Dominant's value is maintained over recessive's 
    	//Merge members of 
    	//Merge pointers
    	//If edge exists delete it
    	//set edges between the two networks connecting to the dominant (if that's not automatic already???)
		unordered_set<Network*> seen(dominant->members.begin(), dominant->members.end());
		for(auto* ptr : recessive->members){
			if(seen.insert(ptr).second){
				dominant->members.push_back(ptr);
			}
		}
		edges.erase(
			    remove_if(edges.begin(), edges.end(), [&](auto& e) {
			        return (e->sender == dominant && e->destination == recessive) || (e->sender == recessive && e->destination == dominant);  // remove synapses between now merged neurons
			    }),
			    edges.end()
		);

		for (auto& e : edges) {
        	if (e->sender == recessive) e->sender = dominant;
        	if (e->destination == recessive) e->destination = dominant;
    	}

    	neurons.erase(remove(neurons.begin(), neurons.end(), recessive), neurons.end());
		recessive = dominant;
    }

    void updateTrace(){
    	for(auto& n : neurons){
    		n->trace = n->trace*(1-exp(-n->members[0]->hp.decay)) + n->members[0]->hp.decay*n->value;
    	}
    	for(auto& e : edges){
    		e->U = e->U*(1-exp(-e->sender->members[0]->hp.u_decay)) + e->sender->members[0]->hp.u_decay*e->sender->trace*2*(e->destination->value - 0.5);
    	}
    }

    void neuronFiring(){
    	mt19937 gen(random_device{}());
		uniform_real_distribution<double> unif(0.0,1.0);
		vector<double> newStates(neurons.size(), 0.0);
		for(int i = 0; i < edges.size(); i++){
			if(edges[i]->sender->value) newStates[i] += (unif(gen) > edges[i]->value) ? (edges[i]->value > 0 ? -1 : 1) : edges[i]->sender->members[0]->hp.determinism*edges[i]->value;
		}
		for(int i = 0; i < newStates.size(); i++){
			neurons[i]->value = (newStates[i] >= neurons[i]->members[0]->hp.firing_value) ? 1 : 0;
		}
    }
}


//   	FILE* f = []{ char t[32], path[64]; time_t tt=time(nullptr); strftime(t, sizeof t, "%j_%H.%M", localtime(&tt)); snprintf(path, sizeof path, "outputs/%s.txt", t); return fopen(path, "w"); }();
