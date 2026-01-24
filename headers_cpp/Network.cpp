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

vector<double> Network::AdjMatrix::colEntropy(){ // + col average
    double range = 2; // -1 to 1

    size_t N = data.size();
    int n_bins = ceil(sqrt(data.size())); //rule o thomb

    vector<double> entropy(N,0.0);
    for (size_t col = 0; col < N; col++){
    	vector<double> counts(n_bins, 0); // counts[i] is between min + i*range/n_bins and min + (i+1)*range/n_bins <-- i = floor((x-min)*n_bins/range)
        double sum = 0.0;
    	for (size_t row = 0; row < N; row++){
            int idx = floor((data[row][col]->value + 1) * n_bins / range);
            if(idx == n_bins) idx = n_bins -1;
            counts[idx]++;
            sum += (data[row][col]->value);
        }
    	for(int count : counts){
    		double p = double(count)/N;
    		if(p > 0) entropy[col] -= p*log2(p);
    	}
        entropy[col] /= log2(n_bins);
    }
    return entropy;
}

pair<vector<vector<double>>,vector<double>> Network::AdjMatrix::entropyAndContribution() { // first: (I-M)^-1 - I second: colEntropy
    int n = data.size();
    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> I(n, vector<double>(n));

    double range = 2;
    int n_bins = ceil(sqrt(n));
    vector<double> entropy(n, 0.0);

    // Build A = I - M + fill entropy
    for (size_t col = 0; col < n; col++) {
        I[col][col] = 1.0;
        vector<double> counts(n_bins,0);
        double sum = 0.0;
        for (size_t row = 0; row < n; row++) {
            double Mij = round(data[row][col]->value*100)/100;
            A[row][col] = (row == col ? 1.0 : 0.0) - parent.hp.contrib_factor*Mij;
            int idx = floor((Mij + 1 )*n_bins/range);
            if(idx == n_bins) idx = n_bins -1;
            counts[idx]++;
            sum += Mij;
        }
        for(int count : counts){
            double p = double(count)/n;
            if(p > 0) entropy[col] -= p*log2(p);
        }
    }

    for(double& e : entropy) e /= log2(n_bins);

    // Gauss–Jordan
    for (int col = 0; col < n; col++) {
        int pivot = col;
        for (int r = col + 1; r < n; r++)
            if (fabs(A[r][col]) > fabs(A[pivot][col])) pivot = r;

        swap(A[col], A[pivot]);
        swap(I[col], I[pivot]);

        double div = A[col][col];
        for (int j = 0; j < n; j++) {
            A[col][j] /= div;
            I[col][j] /= div;
        }

        for (int r = 0; r < n; r++) {
            if (r == col){
    		    I[r][col] -= 1;
    	    	continue;
    	    }
            double f = A[r][col];
            for (int j = 0; j < n; j++) {
                A[r][j] -= f * A[col][j];
                I[r][j] -= f * I[col][j];
            }
        }
    }
    return make_pair(I, entropy);
}
//_____________________
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

    auto A = entropyAndContribution();
    vector<vector<double>> C = A.first;
    vector<double> E = A.second;
    //vector<double> E = colEntropy();

    //#pragma omp for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            //data[i][j]->value += parent.hp.lr*(data[i][j]->destination->value * data[i][j]->U  - parent.hp.reg*C[i][j]*pow(E[j], parent.hp.entropy_factor)*(pow(data[i][j]->destination->trace,2) + data[i][j]->sender->value));


            data[i][j]->value += parent.hp.lr*(data[i][j]->destination->value*data[i][j]->U/C[i][j] - 
                parent.hp.reg*(data[i][j]->destination->trace + pow(E[j], parent.hp.entropy_factor)));


            data[i][j]->value = max(-1.0, min(1.0, data[i][j]->value));

	        //printf("DEBUG %f \n", E[j]);
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
