#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "Network.h"
using namespace std;

struct Neuron;
using NeuronPointer = std::shared_ptr<Neuron>;

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