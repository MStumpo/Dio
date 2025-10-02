#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <variant>
#include <stdexcept>
#include <omp.h>
#include "adjMatrix.cpp"
using namespace std;


//NOTE ABOUT vector<bool>: it returns a proxy element so setting refs causes bad behavior. copying and setting seems to be fine and for printing use vector[i] ? true : false
class Network{

    private:
        vector<bool> neurons;
        vector<double> tracePre;
        vector<double> tracePost;

        AdjacencyMatrix adjMatrix;
        double lr = 0.001;
        double reg  =0.0001;
        int timeWindow = 10;
        int null_window = 10;
        double decayPre = 0.01;
        double decayPost = 0.01;
        int kernel_size = 2;
        bool kernelNormalization = false;
        double determinism = 0.0;
        double firing_value = 1.0;
        double entropyFactor = 1.0;
        bool col_only = false;
        bool verbose = false;
        double pos_lr = 0.0001;
        double neg_lr = 0.00001;

        //uniform_real_distribution<double> unif;

        void updateTrace(const vector<bool> spikes, bool pre=true){

            if(pre){
                for(int i = 0; i < spikes.size(); i++){
                    tracePre[i] = (1.0-decayPre)*tracePre[i] + (spikes[i] ? 1.0 : -1.0);
                    tracePost[i] *= (1.0-decayPost);
                }
            }else{
                for(int i = 0; i < spikes.size(); i++){
                    tracePost[i] += (spikes[i] ? 1.0 : -1.0);
                }
            }
        }

    public:

        bool operator[](size_t i) const {
            return neurons[i];
        };

        size_t size() const { return neurons.size(); }

        Network(vector<pair<string, variant<int, double, bool>>> networkArgs) : adjMatrix(get<int>(networkArgs[0].second)){

            for(auto& pair : networkArgs){
                if(pair.first == "--neuron-size"){
                    neurons.assign(get<int>(pair.second), false);
                    tracePre = vector<double>(get<int>(pair.second), 0.0);
                    tracePost = vector<double>(get<int>(pair.second), 0.0);
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
                else if(pair.first =="--decayPre"){
                    this->decayPre = get<double>(pair.second);
                    printf("\ndecay: %f", this->decayPre);
                }
                else if(pair.first =="--decayPost"){
                    this->decayPost = get<double>(pair.second);
                    printf("\ndecay: %f", this->decayPre);
                }
                else if(pair.first == "--pos-amplitude"){
                    this->pos_lr = get<double>(pair.second); 
                    printf("\npos-amplitude: %f", this->pos_lr);
                }
                else if(pair.first == "--neg-amplitude"){
                    this->neg_lr = get<double>(pair.second); 
                    printf("\nneg_lr: %f", this->neg_lr);
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
                }
                else if(pair.first == "--col-only"){
                    this->col_only = get<bool>(pair.second); 
                    printf("\nentropy-col-only: %d", this->col_only);
                }
                else if(pair.first == "--verbose"){
                    this->verbose = get<bool>(pair.second); 
                    printf("\nVerbose: %d", this->verbose);
                }
                else if(pair.first == "--null-window"){
                    this->null_window = get<int>(pair.second);
                    printf("\nnull_window: %d", this->null_window);
                }
            }
        }

        void neuronFiring(){

            mt19937 gen(random_device{}() + omp_get_thread_num());

            uniform_real_distribution<double> unif(0.0,1.0);

            vector<double> newStates(neurons.size(), 0.0);

            #pragma omp parallel for
            for (int i=0; i < adjMatrix.cols(); i++){
                if(neurons[i]){
                    for(int j=0; j < adjMatrix.rows(); j++){
                        double add = 0.0;
                        if(unif(gen) < abs(adjMatrix[i][j])){
                            add += ((adjMatrix[i][j] > 0)? 1.0 : -1.0)*(1.0-determinism);
                        }
                        add += adjMatrix[i][j]*determinism;
                        if(add != 0.0){
                            #pragma omp atomic
                            newStates[j] += add;
                        }
                    }
                }
            }

            #pragma omp parallel for
            for (int i = 0; i < newStates.size(); i++) {
                neurons[i] = (newStates[i] >= firing_value) ? true : false;
            }
        }

        void runFull(vector<pair<vector<bool>, vector<bool>>> dataset, vector<pair<vector<bool>, vector<bool>>> dataset_test, int epochs=10, bool ds_shuffle=true){
            

            int score = 0;
            int scoreC = 0;
            for(int epoch = 0; epoch < epochs; epoch++){
                if(ds_shuffle){
                    mt19937 g(random_device{}());
                    shuffle(dataset.begin(), dataset.end(), g);
                    shuffle(dataset_test.begin(), dataset_test.end(), g);
                }

                for(const auto& datapoint:dataset){
                    for(int timestep = 0; timestep < null_window+timeWindow; timestep ++){
                        if(timestep >= null_window){
                            for(int i = 0; i < datapoint.first.size(); i++){
                                neurons[i] = datapoint.first[i];
                            }
                            for(int i = 0; i < datapoint.second.size(); i++){
                                neurons[neurons.size() - datapoint.second.size()+i] = datapoint.second[i];
                            }
                        }
                        for(int b = 0; b < neurons.size(); b++){
                            printf("%d", neurons[b] ? true : false);
                            if(b == datapoint.first.size()-1){
                                printf("|");
                            }else if(b == neurons.size() - datapoint.second.size()-1){
                                printf("|");
                            }
                        }
                        if(timestep < null_window){
                            printf("    Epoch %d, null", epoch);
                        }else{
                            printf("    Epoch %d, training", epoch);
                        }
                        printf(" score: %f \n", (double)score/scoreC);
                        updateTrace(neurons, true);
                        neuronFiring();
                        updateTrace(neurons, false);
                        adjMatrix.updateAdj(neurons, tracePre, tracePost,1, reg, kernel_size, kernelNormalization, entropyFactor, col_only, pos_lr, neg_lr);
                    }
                }

                for(const auto& datapoint:dataset_test){
                    for(int timestep = 0; timestep < null_window+timeWindow; timestep ++){
                        if(timestep >= null_window){
                            for(int i = 0; i < datapoint.first.size(); i++){
                                neurons[i] = datapoint.first[i];
                            }
                        }
                        neuronFiring();
                        if(timestep >= null_window){
                            for(int i = 0; i < datapoint.second.size(); i++){
                                if(neurons[neurons.size() - datapoint.second.size() + i] == datapoint.second[i]){
                                    score ++;
                                }else{
                                    score --;
                                }
                                scoreC++;
                            }
                        }
                        for(int b = 0; b < neurons.size(); b++){
                            printf("%d", neurons[b] ? true : false);
                            if(b == datapoint.first.size()-1){
                                printf("|");
                            }else if(b == neurons.size() - datapoint.second.size()-1){
                                printf("|");
                            }
                        }
                        if(timestep < null_window){
                            printf("    Epoch %d, null\n", epoch);
                        }else{
                            printf("    Epoch %d, testing, (", epoch);
                            for(int b = 0; b < datapoint.second.size(); b++) printf("%d", datapoint.second[b] ? 1:0);
                            printf(")");
                        }
                        printf(" score: %f\n", (double)score/scoreC);

                    }
                }
            }

            }


        void train(vector<pair<vector<bool>, vector<bool>>> dataset, int epochs = 1){

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                if(verbose) printf("\n Epoch %d", epoch+1);
                for(const auto& datapoint : dataset){
                    for(int timestep = 0; timestep < null_window; timestep ++){
                        neuronFiring();
                    }
                    for(int timestep = 0; timestep < timeWindow; timestep++){
                        for(int i = 0; i < datapoint.first.size(); i++){
                            neurons[i] = datapoint.first[i];
                        }
                        neuronFiring();
                        for(int i = 0; i < datapoint.first.size(); i++){
                            neurons[i] = datapoint.first[i];
                        }
                        for(int i = 0; i < datapoint.second.size(); i++){
                            neurons[neurons.size() - datapoint.second.size() + i] = datapoint.second[i];
                        }
                        adjMatrix.updateAdj(neurons, tracePre, tracePost,1, reg, kernel_size, kernelNormalization, entropyFactor, col_only, pos_lr, neg_lr);
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
                    for(int timestep = 0; timestep < null_window; timestep ++) neuronFiring();
                    for(int timestep = 0; timestep < timeWindow; timestep++){
                        for(int i = 0; i < datapoint.first.size(); i++){
                            neurons[i] = datapoint.first[i];
                        }
                        neuronFiring();
                        for(int i = 0; i < datapoint.second.size(); i++){
                            if(neurons[neurons.size() - datapoint.second.size() + i] == datapoint.second[i]){
                                score += 1.0/((double)datapoint.second.size()*timeWindow*(double)dataset.size());
                            }else{
                                score -= 1.0/((double)datapoint.second.size()*timeWindow*(double)dataset.size());
                            }
                        }
                    }
                }
                printf("\nEPOCH %d, score: %f", epoch+1, score);
            }
            printf("\n");

        }

        void validate(vector<bool> sample, vector<bool> target, int iterations = NULL){

            if(iterations == NULL){
                iterations = timeWindow;
            }
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