#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>


#include "Network.h"
#include "SharedNetwork.h"
#include "HyperOptimizer.h"
#include "HyperParameters.h"
#include "DataTerminal.h"

using namespace std;

struct ScoreCalculator
{
    vector<DataTerminal*> terminals;
    vector<double> weights;
    vector<Network*> targets;
    double score();
    ScoreCalculator(vector<DataTerminal*> ts, vector<double> w, vector<Network*> ts2) : terminals(ts), weights(w), targets(ts2) {};
};

struct DatasetManager
{
    SharedNetwork* shared_network;
    vector<DataTerminal> terminals;
    vector<vector<vector<uint8_t>>> dataset; //[terminal ID][data index][bit]
    vector<bool> shuffle; //Same index as terminal ID
    vector<ScoreCalculator> score_calculators;
    int current_iteration = 0;
    string path;
    DatasetManager(SharedNetwork* net, string p);
    void createNewTerminal(int id, size_t size, bool calibration);
    void createScoreRule(vector<int> ids, vector<double> weights, vector<int> eval_ids); //first ids is terminal
    void updateCurrentValues();
};
