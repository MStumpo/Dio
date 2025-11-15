#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "Network.h"
using namespace std;

struct Neuron;
using NeuronPointer = std::shared_ptr<Neuron>;


struct DataTerminal
{
    int id;
    vector<NeuronPointer> coordinates;
    vector<uint8_t> values; //Same indexes as coords, meant to be updated constantly with time
    size_t size;
    bool clamped;
    bool calibration;

    void updateValues(vector<uint8_t> new){
        for(int i = 0; i < values.size(); i++) values[i] = new[i];
    }
};

struct ScoreCalculator
{
    vector<DataTerminal*> terminals;
    vector<double> weights;
    double score(){
        double final_score = 0.0;
        double final_weights = 0.0;
        for(int i = 0; i < terminals.size(); i++){
            if(!terminals[i]->clamped){
                for(NeuronPointer point : coordinates) final_score += (point->value == terminals[i]->value ? 1.0 : -1.0);
                final_weights += weights[i];
            }
        }
        return final_score/final_weights;
    }
};