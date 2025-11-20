#include <cstdlib>  // For std::stoi and std::stod
#include <variant>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include "SharedNetwork.h"
#include "HyperParameters.h"
#include "DatasetManager.h"



//PROBLEM: TARGETS MUST NOT REPEAT-> TODO: sum same-target scores  SHaredNet::runDataset
//Convention: merge neurons at the end of the vector (neurons[size -i]) and apply terminals to the beggining (neurons[i]), so you're less likely to have merged neurons in terminals (which wouldn't break everything but the terminal would still only refer to the predesignated network)
using namespace std;
int main(int argc, char *argv[]){

	int TRAIN_WINDOW = 30;
	int TEST_WINDOW = 30;
	int NULL_WINDOW = 30;
	int TOTAL_ITERATIONS = 100000;
	int OPTIMIZE_ITERATIONS = 100;

	HyperParameters hp1;
	HyperParameters hp2;
	hp1.NEURON_SIZE = 30;
	hp2.NEURON_SIZE = 50;
	string PATH = "datasets/toy_dataset.csv";
	SharedNetwork shared_net;

	shared_net.makeSubNetwork(hp1);
	shared_net.makeSubNetwork(hp2);

	shared_net.createDatasetManager(PATH);
	shared_net.data_manager->createScoreRule({0},{1.0},{0,1});

	for(int i = 0; i < 7; i++){
		shared_net.mergeNeuron(shared_net.sub_networks[0]->neurons[hp1.NEURON_SIZE-1-i], shared_net.sub_networks[1]->neurons[hp2.NEURON_SIZE-1-i]);
	}

	shared_net.runDataset(TOTAL_ITERATIONS, TRAIN_WINDOW, TEST_WINDOW, NULL_WINDOW, OPTIMIZE_ITERATIONS, 2);




	return 0;
}
