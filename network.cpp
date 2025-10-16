#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <variant>
#include <stdexcept>
#include <omp.h>
#include <cstdio>

#include "adjMatrix.cpp"
using namespace std;


//NOTE ABOUT vector<bool>: it returns a proxy element so setting refs causes bad behavior. copying and setting seems to be fine and for printing use vector[i] ? true : false
//AdjMatrix[i][j] is i->j , row: sender , col: receiver
class Network{

    private:
        vector<bool> neurons;
        vector<double> trace_pre;
        vector<double> trace_post;

        AdjacencyMatrix adjMatrix;
        double reg  =0.0001;
        int timeWindow = 10;
        int null_window = 10;
        double decay = 0.01;
	double trace_increase = 0.8;
        double determinism = 0.5;
        double firing_value = 1.0;
        double pos_lr = 0.0001;
        double neg_lr = 0.00001;
        double path_decay = 0.1;

        //uniform_real_distribution<double> unif;

        void updateTrace(const vector<bool> spikes, bool pre= true){

            if(pre){
                for(int i = 0; i < spikes.size(); i++){
                    trace_pre[i] = min((1.0-decay)*trace_pre[i] + (spikes[i] ? trace_increase : 0.0)*(1-trace_pre[i]), 1.0); 
                }
            }else{
                for(int i = 0; i < spikes.size(); i++){
                    trace_post[i] = min((1.0-decay)*trace_post[i] + (spikes[i] ? trace_increase : 0.0)*(1-trace_post[i]), 1.0); 
                }
            }
        }

        void neuronFiring(){

            mt19937 gen(random_device{}());
            uniform_real_distribution<double> unif(0.0,1.0);
            vector<double> newStates(neurons.size(), 0.0);
            for (int i=0; i < adjMatrix.rows(); i++){
                if(neurons[i]){
                    for(int j=0; j < adjMatrix.cols(); j++){
                        newStates[j] += adjMatrix[i][j]*(determinism + (unif(gen) < abs(adjMatrix[i][j]) ? 1 : 0)*((1/abs(adjMatrix[i][j])) - determinism));
                    }
                }
            }
            for (int i = 0; i < newStates.size(); i++) {
                neurons[i] = (newStates[i] >= firing_value) ? true : false;
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
                    trace_pre = vector<double>(get<int>(pair.second), 0.0);
                    trace_post = vector<double>(get<int>(pair.second), 0.0);
                    printf("\nneurons : %d", get<int>(pair.second));
                }else if(pair.first == "--time-window"){
                    this->timeWindow = get<int>(pair.second);
                    printf("\ntimeWindow: %d", this->timeWindow);
                }else if(pair.first == "--pos-lr"){
                    this->pos_lr = get<double>(pair.second);
                    printf("\npos_lr: %f", this->pos_lr);
                }else if(pair.first == "--neg-lr"){
                    this->neg_lr = get<double>(pair.second);
                    printf("\nneg_lr: %f", this->neg_lr);
                }
                else if(pair.first == "--reg"){
                    this->reg = get<double>(pair.second);
                    printf("\nreg: %f", this->reg);
                }
                else if(pair.first =="--decay"){
                    this->decay = get<double>(pair.second);
                    printf("\ndecay: %f", this->decay);
                }
               else if(pair.first == "--path-decay"){
                    this->path_decay = get<double>(pair.second);
                    printf("\npath_decay: %f", this->path_decay);
                }
                else if(pair.first == "--determinism"){
                    this->determinism = get<double>(pair.second); 
                    printf("\ndeterminism: %f", this->determinism);
                }
                else if(pair.first == "--firing-value"){
                    this->firing_value = get<double>(pair.second); 
                    printf("\nfiring_value: %f", this->firing_value);
                }
                else if(pair.first == "--null-window"){
                    this->null_window = get<int>(pair.second);
                    printf("\nnull_window: %d", this->null_window);
                }
		else if(pair.first == "--trace-increase"){
		    this->trace_increase = get<double>(pair.second);
		    printf("\ntrace_increase: %f", this->trace_increase);
		}

            }
        }

        void runFull(vector<pair<vector<bool>, vector<bool>>> dataset, vector<pair<vector<bool>, vector<bool>>> dataset_test, int epochs=10, bool ds_shuffle=true){
            
            FILE* f = fopen("outputs/output.txt", "w");

            int score = 0;
            int scoreC = 0;
            for(int epoch = 0; epoch < epochs; epoch++){

                if(ds_shuffle){
                    mt19937 g(random_device{}());
                    shuffle(dataset.begin(), dataset.end(), g);
                    shuffle(dataset_test.begin(), dataset_test.end(), g);
                }

                for(const auto& datapoint:dataset){
                    for(int timestep = 0; timestep < null_window+timeWindow;timestep ++){
                        if(timestep >= null_window){
                            for(int i = 0; i < datapoint.first.size(); i++){
                                neurons[i] = datapoint.first[i];
                            }
                            for(int i = 0; i < datapoint.second.size(); i++){
                                neurons[neurons.size() - datapoint.second.size()+i] = datapoint.second[i];
                            }
                        }
                        updateTrace(neurons, true);
                        neuronFiring();
                        updateTrace(neurons, false);
                        adjMatrix.updateAdj(neurons, trace_pre, trace_post, reg, pos_lr, neg_lr, path_decay);

                        printNetwork({static_cast<int>(datapoint.first.size()-1), static_cast<int>(neurons.size() - datapoint.second.size()-1)});

                        if(timestep < null_window){
                            printf("|%d|null    ", epoch);
                        }else{
                            printf("|%d|training", epoch);

                        }
                        printf("|%f\n", (double) score/scoreC);
                    }
                }

                for(const auto& datapoint:dataset_test){
                    for(int timestep = 0; timestep < null_window+timeWindow; timestep ++){
                        if(timestep >= null_window){
                            for(int i = 0; i < datapoint.first.size(); i++){
                                neurons[i] = datapoint.first[i];
                            }
                        }

                        //updateTrace(neurons, true);
                        neuronFiring();
                        //updateTrace(neurons, true);
                        //adjMatrix.updateAdj(neurons, trace_pre, trace_post, reg, pos_lr, neg_lr);
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

                        printNetwork({static_cast<int>(datapoint.first.size()-1), static_cast<int>(neurons.size() - datapoint.second.size()-1)});

                        if(timestep < null_window){
                            printf("|%d|null    |%f|\n", epoch, (double) score/scoreC);
                        }else{
                            printf("|%d|testing |%f|", epoch, (double) score/scoreC);
                            for(int b = 0; b < datapoint.second.size(); b++) printf("%d", datapoint.second[b] ? 1:0);
                            printf("\n");
                        }
                    }
                }
            }

            }



        //TRAIN AND TEST ARE OUTDATED
        void train(vector<pair<vector<bool>, vector<bool>>> dataset, int epochs = 1){

            for (int epoch = 0; epoch < epochs; epoch++)
            {
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
                        adjMatrix.updateAdj(neurons, trace_pre, trace_post, reg, pos_lr, neg_lr, path_decay);
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
        void printNetwork(vector<int> pos, bool new_line = false){

            for(int b = 0; b < neurons.size(); b++){
                printf("%d", neurons[b] ? true : false);

                for(int p = 0; p < pos.size(); p++){
                    if(b == pos[p]){
                        printf("|");
                        continue;
                    }
                }
            }
        
        }
};
