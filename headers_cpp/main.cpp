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


	/* //UNcomment this but comment the rest if you want to see dataset action
	int TRAIN_WINDOW = 15;
	int TEST_WINDOW = 15;
	int NULL_WINDOW = 3;
	int TOTAL_ITERATIONS = 10000000;
	int OPTIMIZE_ITERATIONS = 10000;

	HyperParameters hp1;
	HyperParameters hp2;
	hp1.NEURON_SIZE = 18;
	hp2.NEURON_SIZE = 14;
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
	*/

		//Nethack screen bits is 5x5x2 = 50
	HyperParameters hp1;
	HyperParameters hp2;
	HyperParameters hp3;
	HyperParameters hp4;
	hp1.NEURON_SIZE = 20;
	hp2.NEURON_SIZE = 20;
	hp3.NEURON_SIZE = 20;
	hp4.NEURON_SIZE = 15;


	vector<int> INPUT_INDEXES = {0,1,2};
	int OUTPUT_INDEX = 3;

	vector<vector<int>> MERGE_MATRIX = {{0, 5, 0, 5},
										{0, 0, 5,  5},
										{0, 0, 0,  5},
										{0, 0, 0,   0}};

	SharedNetwork shared_net;

	shared_net.makeSubNetwork(hp1);
	shared_net.makeSubNetwork(hp2);

	shared_net.makeSubNetwork(hp3);
	shared_net.makeSubNetwork(hp4);

	shared_net.mergeNeuronsFromMatrix(MERGE_MATRIX, false);

	shared_net.createNethackManager(INPUT_INDEXES, OUTPUT_INDEX);

	shared_net.runNethackOnline(300000, 2);
	return 0;
}
