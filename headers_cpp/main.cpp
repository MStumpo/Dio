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

	int TIME_WINDOW = 30;
	int NULL_WINDOW = 30;
	HyperParameters hp1;
	HyperParameters hp2; 
	hp1.NEURON_SIZE = 10;
	hp2.NEURON_SIZE = 30;
	string PATH = "datasets/toy_dataset.csv";
	SharedNetwork shared_net;

	shared_net.makeSubNetwork(hp1);
	shared_net.makeSubNetwork(hp2);

	shared_net.createDatasetManager(PATH);


	shared_net.data_manager->createScoreRule({1},{1.0},{0,1}); //Remember, it's the terminals that are used to calculate the score, the weights and the target networks

	for(int i = 0; i < 5; i++){
		shared_net.mergeNeuron(shared_net.sub_networks[0]->neurons[hp1.NEURON_SIZE-1-i], shared_net.sub_networks[1]->neurons[hp2.NEURON_SIZE-1-i]);
	}


	shared_net.runDataset(1000, 10, 10, 0, 10);




	return 0;
}
