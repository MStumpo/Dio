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

	int TRAIN_WINDOW = 15;
	int TEST_WINDOW = 15;
	int NULL_WINDOW = 3;
	int TOTAL_ITERATIONS = 1000000;
	int OPTIMIZE_ITERATIONS = 1000;

	HyperParameters hp1;
	HyperParameters hp2;
	hp1.NEURON_SIZE = 15;
	hp2.NEURON_SIZE = 15;
	string PATH = "datasets/papa_gpt_generated.csv";
	SharedNetwork shared_net;

	shared_net.makeSubNetwork(hp1);
	shared_net.makeSubNetwork(hp2);

	shared_net.createDatasetManager(PATH);
	shared_net.data_manager->createScoreRule({1},{1.0},{0,1}); // createScoreRule(vector<int> TERMINAL ----> ids, vector<double> weights, vector<int> eval_ids
	shared_net.data_manager->terminals[0]->calibration = true;


	for(int i = 0; i < 5; i++){
		shared_net.mergeNeuron(shared_net.sub_networks[0]->neurons[hp1.NEURON_SIZE-1-i], shared_net.sub_networks[1]->neurons[hp2.NEURON_SIZE-1-i]);
	}

	shared_net.runDataset(TOTAL_ITERATIONS, TRAIN_WINDOW, TEST_WINDOW, NULL_WINDOW, OPTIMIZE_ITERATIONS, 6969);




	return 0;
}
