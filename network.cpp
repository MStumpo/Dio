#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <variant>
#include <stdexcept>
#include "adjMatrix.cpp"
using namespace std;


//NOTE ABOUT vector<bool>: it returns a proxy element so setting refs causes bad behavior. copying and setting seems to be fine and for printing use vector[i] ? true : false
class Network{

    private:
        vector<vector<bool>> neuronHistory;
        vector<bool> neurons;
        AdjacencyMatrix adjMatrix;
        double lr = 0.001;
        double reg  =0.0001;
        int timeWindow = 10;
        double tau_pos = 2;
        double tau_neg = 1;
        double decay = 0.99;
        int kernel_size = 2;
        bool kernelNormalization = false;
        double determinism = 0.0;
        double firing_value = 1.0;
        double entropyFactor = 1.0;
        bool verbose = false;

        random_device rd;
        mt19937 gen;
        //uniform_real_distribution<double> unif;


        pair<double,double> STDPUpdate(const vector<bool>& history_i, const vector<bool>& history_j) { //history_i or j must be vector<bool>s where the higher indexes are more recent

            if(history_i.size() != history_j.size()){
                throw runtime_error("Neuron history sizes must be the same (and higher index is closer to the present)");
            }

            pair<double, double> total_update = make_pair(0.0,0.0); //first: i->j second j->i

            int size = static_cast<int>(history_i.size());

            // Iterate over the spike history of the two neurons, since it's pushback t=0 is oldest and t=history_i.size() -1 is most recent
            for (int t1 = 0; t1 < size; t1++) {
                for (int t2 = 0; t2 < size; t2++) { 
                    if (history_i[t1] && history_j[t2]) {
                        double time_diff = t2 - t1;  // Calculate time difference
                        if (time_diff > 0) {
                            total_update.first += exp(-abs(time_diff) / tau_pos) * pow(decay, size - t2 -1);
                            total_update.second -= exp(-abs(time_diff) / tau_neg) * pow(decay, size - t2 -1);
                        }
                        else if (time_diff < 0) {
                            total_update.first -= exp(-abs(time_diff) / tau_neg) * pow(decay, size - t1 -1);
                            total_update.second += exp(-abs(time_diff) / tau_pos) * pow(decay, size - t1 -1);
                        }
                    }
                }
            }
            return total_update;
        }

        vector<vector<double>> getCorrelationMatrix() {
            size_t numNeurons = neuronHistory[0].size();   // Number of neurons (columns in neuronHistory)
            size_t numTimesteps = neuronHistory.size();    // Number of timesteps (rows in neuronHistory)

            vector<vector<double>> corrMatrix(numNeurons, vector<double>(numNeurons, 0.0));

            // added by womane with chagpt help
            vector<vector<bool>> neuronSpikes(numNeurons, vector<bool>(numTimesteps, false));
            for (int t = 0; t < numTimesteps; t++) {
                for (int n = 0; n < numNeurons; n++) {
                    neuronSpikes[n][t] = neuronHistory[t][n];
                }
            }

            // Calculate cross-correlation between each pair of neurons (i, j)
            pair<double, double> update;
            for (int i = 0; i < numNeurons; i++) {
                for (int j = 0; j < i; j++) {
                    if (i != j) { 
                        update = STDPUpdate(neuronSpikes[i], neuronSpikes[j]);
                        corrMatrix[i][j] = update.first;
                        corrMatrix[j][i] = update.second;
                    } else {
                        corrMatrix[i][j] = 1.0;
                    }
                }
            }
            return corrMatrix;
        }


    public:

        bool operator[](size_t i) const {
            return neurons[i];
        };

        size_t size() const { return neurons.size(); }

        //Network(int neuronSize, int timeWindow = 10, double lr = 0.001, double reg=0.001, double tau_pos=2.0, double tau_neg = 1.0, double decay = 0.95,
        // , int kernel_size=2, bool kernelNormalization=false) : adjMatrix(neuronSize){

        Network(vector<pair<string, variant<int, double, bool>>> networkArgs) : adjMatrix(get<int>(networkArgs[0].second)), gen(rd()){

            for(auto& pair : networkArgs){
                if(pair.first == "--neuron-size"){
                    neurons.assign(get<int>(pair.second), false);
                    printf("\nneurons : %d", get<int>(pair.second));
                }else if(pair.first == "--time-window"){
                    this->timeWindow = get<int>(pair.second);
                    printf("\ntimeWindow: %d", this->timeWindow);
                }else if(pair.first == "--lr"){
                    this->lr = get<double>(pair.second);
                    printf("\nlr: %f", this->lr);
                }
                else if(pair.first == "--reg"){
                    this->reg = get<double>(pair.second);
                    printf("\nreg: %f", this->reg);
                }
                else if(pair.first == "--tau-pos"){
                    this->tau_pos = get<double>(pair.second);
                    printf("\ntau_pos: %f", this->tau_pos);
                }
                else if(pair.first =="--tau-neg"){
                    this->tau_neg = get<double>(pair.second);
                    printf("\ntau_neg: %f", this->tau_neg);
                } 
                else if(pair.first =="--decay"){
                    this->decay = get<double>(pair.second);
                    printf("\ndecay: %f", this->decay);
                }
                else if(pair.first == "--kernel-size"){
                    this->kernel_size = get<int>(pair.second);
                    printf("\nk_size: %d", this->kernel_size);
                }
                else if(pair.first == "--kernel-normalization"){
                    this->kernelNormalization = get<bool>(pair.second); 
                    printf("\nk_norm: %d", this->kernelNormalization);
                }
                else if(pair.first == "--determinism"){
                    this->determinism = get<double>(pair.second); 
                    printf("\ndeterminism: %f", this->determinism);
                }
                else if(pair.first == "--firing-value"){
                    this->firing_value = get<double>(pair.second); 
                    printf("\nfiring_value: %f", this->firing_value);
                }
                else if(pair.first == "--entropy-factor"){
                    this->entropyFactor = get<double>(pair.second); 
                    printf("\nentropy-factor: %f", this->entropyFactor);
                }else if(pair.first == "--verbose"){
                    this->verbose = get<bool>(pair.second); 
                    printf("\nentropy-factor: %d", this->verbose);
                }
            }
        }

        void neuronFiring(){

            uniform_real_distribution<double> unif(0.0,1.0);

            // Vector of new states, of the size of neuron vector
            //vector<int> newStates(neurons.size(), 0);  // Vector to store updated states
            vector<double> newStates(neurons.size(), 0.0);

            for (int i=0; i < adjMatrix.cols(); i++){
                if(neurons[i]){
                    for(int j=0; j < adjMatrix.rows(); j++){
                        if(unif(gen) < abs(adjMatrix[i][j])){
                            newStates[j] += ((adjMatrix[i][j] > 0)? 1.0 : -1.0)*(1-determinism);
                        }
                            newStates[j] += adjMatrix[i][j]*determinism;
                    }
                }
            }
            for (int i = 0; i < newStates.size(); i++) {
                neurons[i] = (newStates[i] >= firing_value) ? true : false;
            }
        }

        void train(vector<pair<vector<bool>, vector<bool>>> dataset, int epochs = 1){
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                clearNeuronHistory();
                for(const auto& datapoint : dataset){
                    for(int timestep = 0; timestep < timeWindow; timestep++){
                        for(int i = 0; i < datapoint.first.size(); i++){
                            neurons[i] = datapoint.first[i];
                        }
                        neuronFiring();
                        for(int i = 0; i < datapoint.second.size(); i++){
                            neurons[neurons.size() - datapoint.second.size() + i] = datapoint.second[i];
                        }
                        storeNeuronStates(neurons);
                        adjMatrix.updateAdj(getCorrelationMatrix(),1, lr, reg, kernel_size, kernelNormalization, entropyFactor);
                    }
                }
            }
        }
        void test(vector<pair<vector<bool>, vector<bool>>> dataset, int epochs = 1){

            double score;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                score = 0;
                for(const auto& datapoint : dataset){
                    for(int timestep = 0; timestep < timeWindow; timestep++){
                        for(int i = 0; i < datapoint.first.size(); i++){
                            neurons[i] = datapoint.first[i];
                        }
                        neuronFiring();
                        for(int i = 0; i < datapoint.second.size(); i++){
                            if(neurons[neurons.size() - datapoint.second.size() + i] == datapoint.second[i]){
                                score += 1/((double)datapoint.second.size()*timeWindow*(double)dataset.size());
                            }else{
                                score -= 1/((double)datapoint.second.size()*timeWindow*(double)dataset.size());
                            }
                        }
                    }
                }
                printf("\nEPOCH %d, score: %f", epoch, score);
            }
            printf("\n");

        }

        void validate(vector<bool> sample, vector<bool> target, int iterations = 1){

            for(int t = 0; t < iterations; t++){
                for(int i = 0; i < sample.size(); i++){
                    neurons[i] = sample[i];
                }
                neuronFiring();
                printf("\n TIMESTEP %d OUTPUT: ", t);
                for(int i = 0; i < target.size(); i++){
                    printf("%d", neurons[neurons.size() - target.size() + i] ? true : false);
                }
            }
            printf("\n TARGET: ");
            for(int i = 0; i < target.size(); i++){
                printf("%d", target[i] ? true : false);
            }
            printf("\n");
        }

        void storeNeuronStates(const vector<bool> newStates) {
            neuronHistory.push_back(newStates);
        }
        void clearNeuronHistory(){
            neuronHistory.clear();
        }
        void printAdjMatrix(int width=1, int decimals=2) {
            printf("\n");
            for (int i = 0; i < adjMatrix.rows(); ++i) {
                for (int j = 0; j < adjMatrix.cols(); ++j) {
                if (adjMatrix[i][j] > 0) {
                    printf(" ");  // This adds a space before negative numbers
                }
                    printf("%-*.*f ", width, decimals, adjMatrix[i][j]);
                }
                printf("\n");
            }
        }
};