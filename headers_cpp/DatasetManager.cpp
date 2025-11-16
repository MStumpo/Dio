#include "DatasetManager.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>


using namespace std;


void DataTerminal::updateValues(vector<uint8_t> new_vals){
	for(int i = 0; i < values.size(); i++) values[i] = new_vals[i];
}

double ScoreCalculator::score(){
 	double final_score = 0.0;
    double final_weights = 0.0;
    for(int i = 0; i < terminals.size(); i++){
        if(!terminals[i]->clamped){
            for(int j = 0; j < terminals[i]->size; j++) final_score += (terminals[i]->coordinates[j]->value == terminals[i]->values[j] ? 1.0 : -1.0);
            final_weights += weights[i];
        }
    }
    return final_score/final_weights;
}

DatasetManager::DatasetManager(SharedNetwork* net, string p, int null, int time) : shared_network(net), path(p), null_window(null), time_window(time){
    ifstream file(path);
    //Csv rules: while in same sub_net, see if data_id is always increasing, if data_id resets to 0 in the same sub_net then assign a new Terminal
    //THIS ALGO DOES NOT REMEMBER PREV_SUB_NET_NEURONS FROM PREVIOUSLY ACCESSED NETS SO PLEASE WRITE DATASETS SEQUENTIALLY TO AVOID OVERRIDES
    string line;
    int prev_data_id = 1;
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
        if(prev_data_id > 0 && stoi(data_id) == 0){
            createNewTerminal(terminals.size()-1, values.size(), false);
            dataset.push_back(vector<vector<uint8_t>>({}));
            shuffle.push_back((shuff == "1" ? 1 : 0));
            //Assign new sub_net slots
            if(stoi(sub_net) == prev_sub_net){
                terminals[terminals.size()-1].coordinates.assign(
                    shared_network->sub_networks[prev_sub_net]->neurons.begin() + prev_sub_net_neuron,
                     shared_network->sub_networks[prev_sub_net]->neurons.begin() + prev_sub_net_neuron + values.size());
                prev_sub_net_neuron += values.size();
            }else{
                prev_sub_net_neuron = 0;
                prev_sub_net = stoi(sub_net);
            }
        }
        dataset[dataset.size()-1].push_back(values);
        prev_data_id = stoi(data_id);
    }
}

void DatasetManager::createNewTerminal(int id, size_t size, bool calibration){
    terminals.push_back(DataTerminal(id, size, calibration));
}

void DatasetManager::createScoreRule(vector<int> ids, vector<double> weights, vector<int> eval_ids){
    vector<DataTerminal*> targets;
    vector<Network*> eval_targets;
    for(int id : ids){
        for(DataTerminal& t : terminals){
            if(t.id == id){ //this whole loop is redundant if ID is equal to index but oh well
                targets.push_back(&t);
            }
        }
    }
    for(int id : eval_ids){
        eval_targets.push_back(shared_network->sub_networks[id]); //for some reason this works but &shared_net->sub_net[id] doesn't
    }
    score_calculators.push_back(ScoreCalculator(targets, weights, eval_targets));
}

void DatasetManager::updateCurrentValues(){
    for(int i = 0; i < dataset.size(); i++){//[terminal ID][data index][bit]
        if(!shuffle[i]){ terminals[i].updateValues(dataset[i][current_iteration+1]);}
        else {terminals[i].updateValues(dataset[i][rand()%dataset[i].size()]);}
    }
    current_iteration++;
}