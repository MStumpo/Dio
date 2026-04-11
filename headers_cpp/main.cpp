#include <cstdlib>  // For std::stoi and std::stod
#include <variant>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include "SharedNetwork.h"
#include "DataManager.h"
#include "HyperParameters.h"



//PROBLEM: TARGETS MUST NOT REPEAT-> TODO: sum same-target scores  SHaredNet::runDataset
//Convention: merge neurons at the end of the vector (neurons[size -i]) and apply terminals to the beggining (neurons[i]), so you're less likely to have merged neurons in terminals (which wouldn't break everything but the terminal would still only refer to the predesignated network)
using namespace std;
int main(int argc, char *argv[]){
	
	/*
	//UNcomment this but comment the rest if you want to see dataset action
	int TRAIN_WINDOW = 15;
	int TEST_WINDOW = 15;
	int NULL_WINDOW = 3;
	int TOTAL_ITERATIONS = 10000000;
	int OPTIMIZE_ITERATIONS = 2000;

	HyperParameters hp1;
	HyperParameters hp2;
	hp1.NEURON_SIZE = 20;
	hp2.NEURON_SIZE = 20;
	string PATH = "datasets/papa_gpt_generated.csv";
	SharedNetwork shared_net;

	shared_net.makeSubNetwork(hp1);
	shared_net.makeSubNetwork(hp2);

	shared_net.data_manager->makeDatasetManager(PATH);
	shared_net.data_manager->createScoreRule({1},{1.0},{0,1}); // createScoreRule(vector<int> TERMINAL ----> ids, vector<double> weights, vector<int> eval_ids
	shared_net.data_manager->terminals[0]->calibration = true;


	for(int i = 0; i < 5; i++){
		shared_net.mergeNeuron(shared_net.sub_networks[0]->neurons[hp1.NEURON_SIZE-1-i], shared_net.sub_networks[1]->neurons[hp2.NEURON_SIZE-1-i]);
	}

	shared_net.runDataset(TOTAL_ITERATIONS, TRAIN_WINDOW, TEST_WINDOW, NULL_WINDOW, OPTIMIZE_ITERATIONS, 6969);
	
	/*	//Nethack screen bits is 5x5x2 = 50
	HyperParameters hp1;
	HyperParameters hp2;
	HyperParameters hp3;
	HyperParameters hp4;
	HyperParameters hp5;

	hp1.NEURON_SIZE = 50;
	hp2.NEURON_SIZE = 50;
	hp3.NEURON_SIZE = 50;
	hp4.NEURON_SIZE = 30;
	hp5.NEURON_SIZE = 50;


	vector<int> INPUT_INDEXES = {0,1,2};
	int OUTPUT_INDEX = 3;

	SharedNetwork shared_net;

	shared_net.makeSubNetwork(hp1);
	shared_net.makeSubNetwork(hp2);

	shared_net.makeSubNetwork(hp3);
	shared_net.makeSubNetwork(hp4);
	shared_net.makeSubNetwork(hp5);


	vector<vector<int>> MERGE_MATRIX = {{0, 5, 0, 5, 5},
										{0, 0, 5, 5, 5},
										{0, 0, 0, 5, 5},
										{0, 0, 0, 0, 5},
										{0, 0, 0, 0, 0}};

	shared_net.mergeNeuronsFromMatrix(MERGE_MATRIX, false);

	shared_net.data_manager->makeNethackManager(INPUT_INDEXES, OUTPUT_INDEX);

	shared_net.runNethackOnline(300000, 2);
	*/

	vector<HyperParameters> hp = {HyperParameters(30), HyperParameters(30), HyperParameters(30)};

	SharedNetwork shared_net;

	for(HyperParameters& haho : hp) shared_net.makeSubNetwork(haho);
	
	vector<vector<int>> MERGE_MATRIX = {{0, 5, 5}, 
										{0, 0, 5},
										{0, 0, 0}};


	shared_net.mergeNeuronsFromMatrix(MERGE_MATRIX, false);



	DataTerminal t0 = DataTerminal(0, 4, false);
	DataTerminal t1 = DataTerminal(1, 8, false);
	DataTerminal t2 = DataTerminal(2, 4, false);
	DataTerminal t3 = DataTerminal(3, 4, false);
	DataTerminal t4 = DataTerminal(4, 5, true);
	DataTerminal t5 = DataTerminal(5, 6, false);
	DataTerminal t6 = DataTerminal(6, 4, false);
	DataTerminal t7 = DataTerminal(7,2,false);
	t4.values = {1,0,1,0,0};
	t4.calibration = true;

	t0.coordinates = vector<NeuronPointer>(shared_net.sub_networks[0]->neurons.begin(), shared_net.sub_networks[0]->neurons.begin() + 4);
	t1.coordinates = vector<NeuronPointer>(shared_net.sub_networks[0]->neurons.begin(), shared_net.sub_networks[0]->neurons.begin() + 8); //t0 is contained in t1
	t2.coordinates = vector<NeuronPointer>(shared_net.sub_networks[1]->neurons.begin(), shared_net.sub_networks[1]->neurons.begin() + 4);
	t3.coordinates = vector<NeuronPointer>(shared_net.sub_networks[2]->neurons.begin(), shared_net.sub_networks[2]->neurons.begin() + 4);
	t4.coordinates = vector<NeuronPointer>(shared_net.sub_networks[2]->neurons.begin()+5, shared_net.sub_networks[2]->neurons.begin() + 10);
	t5.coordinates = vector<NeuronPointer>(shared_net.sub_networks[1]->neurons.begin(), shared_net.sub_networks[1]->neurons.begin() + 6);
	t6.coordinates = vector<NeuronPointer>(shared_net.sub_networks[1]->neurons.begin()+7, shared_net.sub_networks[1]->neurons.begin() + 7+4);
	t7.coordinates = vector<NeuronPointer>(shared_net.sub_networks[0]->neurons.begin()+4+1, shared_net.sub_networks[0]->neurons.begin() + 4+2);
    
	for(DataTerminal t : {t0,t1,t2,t3, t4, t5,t6,t7}){
		shared_net.data_manager->terminals.push_back(make_unique<DataTerminal>(move(t))); //yes I know there's no point in assigning ids if everything just mentions them by vector index but uhm uuuhm uuuuuuh
	}

    //Switch(DataTerminal* transmit, vector<vector<uint8_t>> trig, bool clamps = false, double rew = 0): 
	using Switch = DataManager::Playground::Switch;
	vector<Switch> myswitches= {
		Switch(shared_net.data_manager->terminals[0].get(), {{0,0,1,1}}, true, 0.3),
		Switch(shared_net.data_manager->terminals[1].get(), {{0,0,1,1,0,0,1,1}, {0,0,1,1,1,1,0,0}}, false, 0.5),
		Switch(shared_net.data_manager->terminals[2].get(), {{0,1,0,0}, {1,0,0,0}}, true, 0.2)
	};

	shared_net.data_manager->makePlayground();


	get<DataManager::Playground>(shared_net.data_manager->data_source).switches = myswitches;


	shared_net.runPlayground(1000, 10000, true, 2);
	
	return 0;
}
