#include <cstdlib>  // For std::stoi and std::stod
#include <variant>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include "SharedNetwork.h"
#include "Network.h"
#include "HyperParameters.h"
#include "DatasetManager.h"

using namespace std;
int main(int argc, char *argv[]){

	int TIME_WINDOW = 30;
	int NULL_WINDOW = 30;
	HyperParameters hp1;
	HyperParameters hp2;
	hp1.NEURON_SIZE = 10;
	hp2.NEURON_SIZE = 30;
	string PATH = "datasets/toy_dataset.csv";

	printf("WEWEWEWEW\n");

	SharedNetwork shared_net(TIME_WINDOW, NULL_WINDOW);

	printf("WAWA1\n");

	shared_net.makeSubNetwork(hp1);
	shared_net.makeSubNetwork(hp2);

	shared_net.data_manager = make_unique<DatasetManager>(&shared_net, PATH, NULL_WINDOW, TIME_WINDOW);

	shared_net.data_manager->createScoreRule({0,1},{0.0,1.0},{0,1});


	//printf("%d _____ \n", shared_net.sub_networks[0]->neurons[hp1.NEURON_SIZE-1]->value);

	printf("WEWE\n");

	for(int i = 0; i < 5; i++){
		shared_net.mergeNeuron(shared_net.sub_networks[0]->neurons[hp1.NEURON_SIZE-1-i], shared_net.sub_networks[1]->neurons[hp2.NEURON_SIZE-1-i]);
	}





	return 0;
}
