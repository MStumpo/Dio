#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <variant>
#include <stdexcept>
#include <omp.h>
#include <cstdio>
#include <ctime>

//AdjMatrix[i][j] is i->j , row: sender , col: receiver
class Network{

private:
	SharedNetwork& shared;
	vector<NeuronPointer> neurons;
	HyperOptimizer opt;

	class AdjMatrix{
	private:
		Network& parent;
		vector<vector<EdgePointer>> data;
		vector<double> colEntropy(){
		  	size_t N = data.size();
		  	vector<double> entropy(N, 0.0);

		  	for(int col = 0; col < N; col++){
		  		for(int row=0; row < N; row++){
		  			if(data[row][col]->value != 0.0) entropy[col] += -abs(data[row][col]->value)*log(abs(data[row][col]->value));
		  		}
		  		entropy[col] /=N;
		  	}

		  	return entropy;
		  }
	public:

		AdjMatrix(Network& parent_network) : parent(parent_network){}

		void initialize(){
			mt19937 rng(rnd_device());
			uniform_real_distribution<double> unif(-1.0, 1.0);
			auto gen = [&](){ return unif(rng); };
			data.resize(parent.neurons.size());
			for(size_t i = 0; i < parent.neurons.size(); i++){
				data[i].resize(parent.neurons.size());
				for(size_t j = 0; j < parent.neurons.size(); j++){
					auto e = make_shared<Edge>(parent.neurons[i], parent.neurons[j], gen());
					data[i][j] = e;
					parent.shared.edges.push_back(e);
				}
			}
		}
		void updateAdj(){
				size_t N = data.size();
				vector<double> E = colEntropy();

				#pragma omp for collapse(2) 
				for(int i = 0; i < N; i++){
					for(int j=0; j < N; j++){
         			//printf("%f, %f, %d\n",data[0][2], trace[0], spikes[2] ? 1 : 0);
		         	//THIS ONE IS THE ONE THAT GOT THE GOOD RESULTS :3333
		         	//data[i][j] += lr*((spikes[j] ? 1.0 : 0.0)*U[i][j]*pow(E[j], entropy_factor) - reg*data[i][j]*pow(trace[i],2));

						data[i][j]->value += lr*(parent.neurons[j]->value*data[i][j]->U*pow(E[j],parent.hp.entropy_factor) - parent.hp.reg*data[i][j]*pow(parent.hp.neurons[i]->trace,2));
						data[i][j]->value = max(-1.0,min(1.0,data[i][j]->value));
					}
				}
			}
	}
	AdjMatrix adj;
public:
	HyperParameters hp;
	bool operator[](size_t i) const {
		return neurons[i];
	};

	size_t size() const { return neurons.size(); }

	Network(SharedNetwork& s, const HyperParameters& hp_arg) : shared(s), adj(*this), hp(hp_arg), opt(hp_arg){

		for(size_t i = 0; i < SIZE; i++){
			NeuronPointer n = make_shared<Neuron>(false,0.0, this);
			neurons.push_back(n);
			s.neurons.push_back(n);
		}
		adj.initialize();
		s.sub_networks.push_back(this);

   }
void printAdjMatrix(int width=1, int decimals=2) {
	printf("\n");
	for (int i = 0; i < adjMatrix.data.size(); ++i) {
		for (int j = 0; j < adjMatrix.data[j].size(); ++j) {
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
