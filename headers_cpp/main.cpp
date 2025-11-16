#include <cstdlib>  // For std::stoi and std::stod
#include <variant>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include "SharedNetwork.h"
#include "Network.h"
#include "HyperParameters.h"

using namespace std;
int main(int argc, char *argv[]){

	int TIME_WINDOW = 30;
	int NULL_WINDOW = 30;
	HyperParameters hp1;
	HyperParameters hp2;
	hp1.NEURON_SIZE = 10;
	hp2.NEURON_SIZE = 30;
	SharedNetwork shared_net(TIME_WINDOW, NULL_WINDOW);
	string PATH = "datasets/toy_dataset.csv";
	shared_net.makeSubNetwork(hp1);
	shared_net.makeSubNetwork(hp2);



	printf("%d _____ \n", shared_net.sub_networks[0]->neurons[hp1.NEURON_SIZE-1]->value);

	printf("WEWE\n");

	for(int i = 0; i < 3; i++){
		shared_net.mergeNeuron(shared_net.sub_networks[0]->neurons[hp1.NEURON_SIZE-1-i], shared_net.sub_networks[1]->neurons[hp2.NEURON_SIZE-1-i]);
	}




	return 0;
}
