#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <variant>
#include <stdexcept>
#include <omp.h>
#include <cstdio>

#include "adjMatrix.cpp"
#include "hyperparam.cpp"
using namespace std;


//NOTE ABOUT vector<bool>: it returns a proxy element so setting refs causes bad behavior. copying and setting seems to be fine and for printing use vector[i] ? true : false
//AdjMatrix[i][j] is i->j , row: sender , col: receiver
class Network{

    private:
        vector<uint8_t> neurons;
        vector<double> trace;
	vector<vector<double>> U;

        AdjacencyMatrix adjMatrix;
        double reg  =0.0001;
        int timeWindow = 10;
        int null_window = 10;
        double decay = 0.01;
	   double U_decay = 0.01;
	   double lr = 0.01;
        double determinism = 0.5;
        double firing_value = 1.0;

        //uniform_real_distribution<double> unif;

        void updateTrace(const vector<uint8_t> spikes){
		for(int i=0; i < trace.size(); i++){
			trace[i] = trace[i]*(1-exp(-decay)) + decay*spikes[i];
		}
		for(int i=0; i < U.size(); i++){
			U[i][i] = U[i][i]*(1-exp(-U_decay)) + U_decay*trace[i]*2*(spikes[i] -0.5);
			for(int j=0; j < i; j++){
				U[i][j] = U[i][j]*(1-exp(-U_decay)) + U_decay*trace[i]*2*(spikes[j] -0.5);
				U[j][i] = U[j][i]*(1-exp(-U_decay)) + U_decay*trace[j]*2*(spikes[i] -0.5);
			}
		}
        }

        void neuronFiring(){

            mt19937 gen(random_device{}());
            uniform_real_distribution<double> unif(0.0,1.0);
            vector<double> newStates(neurons.size());

            for (int j=0; j < adjMatrix.cols(); j++){
                newStates[j] = (neurons[j] ? 1.0 : 0.0);
                for(int i=0; i < adjMatrix.rows(); i++){
                    if (neurons[i]) newStates[j] += (unif(gen) > adjMatrix[i][j]) ? (adjMatrix[i][j] > 0 ? -1 : 1) : determinism*adjMatrix[i][j];
                }
		        neurons[j] = (newStates[j] >= firing_value) ? 1 : 0;
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
                    this->neurons.assign(get<int>(pair.second), 0);
                    trace = vector<double>(get<int>(pair.second), 0.0);
                    U = vector<vector<double>>(get<int>(pair.second), vector<double>(get<int>(pair.second),0.0));
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
                else if(pair.first == "--decay"){
                    this->decay = get<double>(pair.second);
                    printf("\ndecay: %f", this->decay);
                }
		else if(pair.first == "--u-decay"){
		    this->U_decay = get<double>(pair.second);
		    printf("\nU-decay: %f", this->U_decay);
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
		else{
		   printf("\n!!!! UNKNOWN ARG PASSED TO NETWORK !!!! : %s ", pair.first.c_str());
		}
            }
        }

        void runFull(vector<pair<vector<uint8_t>, vector<uint8_t>>> dataset, vector<pair<vector<uint8_t>, vector<uint8_t>>> dataset_test, int epochs=10, bool ds_shuffle=true, bool optimize = true){

            FILE* f = fopen("outputs/output.txt", "w");

            int score = 0;
            int scoreC = 0;
		//optimizable specs: time_window, null_window, decay, U_decay,lr, reg, determinism, firing_value
		vector<ParamSpec> specs = { //min, max, log_scale, is_int
			{1, 50, false, true},
			{0, 10, false, true},
			{0, 1, false, false},
			{0, 1, false, false},
			{0.00000000001, 1, true, false},
			{0.00000000001, 1, true, false},
			{0,1, false, false},
			{0,1, false, false}
		};
		HyperOptimizer opt(specs);
		auto cand = opt.propose();

            for(int epoch = 0; epoch < epochs; epoch++){
                if(ds_shuffle){
                    mt19937 g(random_device{}());
                    shuffle(dataset.begin(), dataset.end(), g);
                    shuffle(dataset_test.begin(), dataset_test.end(), g);
                }
		if(optimize){
			auto cand = opt.propose();
			this->timeWindow = (int) cand[0];
			this->null_window = (int) cand[1];
			this->decay = cand[2];
			this->U_decay = cand[3];
			this->lr = cand[4];
			this->reg = cand[5];
			this->determinism = cand[6];
			this->firing_value = cand[7];
		}

                for(const auto& datapoint:dataset){
                    for(int timestep = 0; timestep < null_window+timeWindow;timestep ++){
                        if(timestep >= null_window) for(int i = 0; i < datapoint.first.size(); i++) neurons[i] = datapoint.first[i];
                        
                        neuronFiring();
                        //This part is so the trace knows it
                        if(timestep >= null_window) for(int i = 0; i < datapoint.first.size(); i++) neurons[i] = datapoint.first[i];
            			for(int i = 0; i < datapoint.second.size(); i++) neurons[neurons.size() - datapoint.second.size()+i] = datapoint.second[i];
                        updateTrace(neurons);
                        adjMatrix.updateAdj(neurons, trace, U, reg, lr);

                        printNetwork({static_cast<int>(datapoint.first.size()-1), static_cast<int>(neurons.size() - datapoint.second.size()-1)});
                        if(timestep < null_window){
                            printf("|%d|null    ", epoch);
                        }else{
                            printf("|%d|training", epoch);

                        }
                        printf("|%f\n", (double) score/scoreC);
                        //printUMatrix();
                    }
                }

		score = 0;
		scoreC = 0;
                for(const auto& datapoint:dataset_test){
                    for(int timestep = 0; timestep < null_window+timeWindow; timestep ++){
                        if(timestep >= null_window) for(int i = 0; i < datapoint.first.size(); i++)neurons[i] = datapoint.first[i];
                        
                        neuronFiring();
                         //This part is so the trace knows it
                        if(timestep >= null_window) for(int i = 0; i < datapoint.first.size(); i++) neurons[i] = datapoint.first[i];
                        updateTrace(neurons);
                        //adjMatrix.updateAdj(neurons, trace, U, reg, lr);
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
		if(optimize){
			opt.update(cand, (double)score/scoreC);
			for(double d : cand) printf("%f||", d);
			printf("\n");
		}
            }
        }

        void validate(vector<uint8_t> sample, vector<uint8_t> target, int iterations = -1){

            if(iterations == -1){
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
        void printUMatrix(int width=1, int decimals=2) {
            for (int i = 0; i < U.size(); ++i) {
                for (int j = 0; j < U[i].size(); ++j) {
                if (U[i][j] > 0) {
                    printf(" ");  // This adds a space before negative numbers
                }
                    printf("%-*.*f ", width, decimals, U[i][j]);
                }
                printf("\n");
            }
            printf("\n");
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
