#include "DatasetManager.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>


using namespace std;

DatasetManager::DatasetManager(SharedNetwork* net, string p) : shared_network(net), path(p){

    ifstream file(path);
    //Csv rules: while in same sub_net, see if data_id is always increasing, if data_id resets to 0 in the same sub_net then assign a new Terminal
    //THIS ALGO DOES NOT REMEMBER PREV_SUB_NET_NEURONS FROM PREVIOUSLY ACCESSED NETS SO PLEASE WRITE DATASETS SEQUENTIALLY TO AVOID OVERRIDES
    string line;
    int prev_sub_net_neuron = 0;
    int prev_sub_net = 0;
    while(getline(file, line)){
        stringstream ss(line);
        string sub_net;
        string data_id;
        string shuff;
        string values_string;
        vector<uint8_t> values;

        getline(ss, sub_net, ',');
        getline(ss, data_id, ',');
        getline(ss, shuff, ',');
        getline(ss, values_string, ',');

        if(sub_net == "sub_net") continue;

        for(char c : values_string) values.push_back(c == '1' ? true : false);

        if(stoi(data_id) == 0){
            createNewTerminal(terminals.size(), values.size(), false);

            dataset.push_back(vector<vector<uint8_t>>({}));
            shuffle.push_back((shuff == "1" ? 1 : 0));

            if(stoi(sub_net) != prev_sub_net) prev_sub_net_neuron = 0;

            for(int i = 0; i < values.size(); i++) terminals[terminals.size()-1]->coordinates.push_back(shared_network->sub_networks[stoi(sub_net)]->neurons[prev_sub_net_neuron + i]);

            prev_sub_net_neuron += values.size();
            prev_sub_net = stoi(sub_net);
        }
        dataset[dataset.size()-1].push_back(values); //always update to latest terminal dataset is read [TERMINALID][DATA_INDEX][BIT]
    }
}

void DatasetManager::createNewTerminal(int id, size_t size, bool calibration){
    terminals.push_back(make_unique<DataTerminal>(id, size, calibration));
}

void DatasetManager::createScoreRule(vector<int> ids, vector<double> weights, vector<int> eval_ids){
    vector<DataTerminal*> targets;
    vector<Network*> eval_targets;
    for(int id : ids){
        for(unique_ptr<DataTerminal>& t : terminals){
            if(t->id == id) targets.push_back(t.get()); //this condition is redundant if ID is equal to index but oh well
        }
    }
    for(int id : eval_ids){
        eval_targets.push_back(shared_network->sub_networks[id].get()); //for some reason this works but &shared_net->sub_net[id] doesn't
    }
    score_calculators.push_back(ScoreCalculator(targets, weights, eval_targets));
}

void DatasetManager::updateCurrentValues(){
    int rand_idx = rand()%dataset[0].size(); //we need to assume all terminals have the same number of indices  
    for(int i = 0; i < dataset.size(); i++){//[terminal ID][data index][bit]
        if(!shuffle[i]){ terminals[i]->updateValues(dataset[i][(current_iteration+1)%dataset[i].size()]);}
        else {
            terminals[i]->updateValues(dataset[i][rand_idx]);
        }
    }
    current_iteration++;
}

double ScoreCalculator::score(){
    double final_score = 0.0;
    double final_weights = 0.0;
    for(int i = 0; i < terminal_ptrs.size(); i++){
        if(!terminal_ptrs[i]->clamped && weights[i] != 0.0){
            for(int j = 0; j < terminal_ptrs[i]->size; j++) final_score += weights[i]*(terminal_ptrs[i]->coordinates[j]->value == terminal_ptrs[i]->values[j] ? 1.0 : -1.0)/((double) terminal_ptrs[i]->size);
	    final_weights += weights[i];
        }
    }
    return final_score/(final_weights);
}
