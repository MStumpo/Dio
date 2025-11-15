#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "Network.h"
using namespace std;

struct Neuron;
using NeuronPointer = std::shared_ptr<Neuron>;
struct SharedNetwork;
struct Network; 
struct DataTerminal
{
    int id;
    vector<NeuronPointer> coordinates;
    size_t size;
    bool calibration;
    vector<uint8_t> values; //Same indexes as coords, meant to be updated constantly with time
    bool clamped;
    void updateValues(vector<uint8_t> new_vals);
    DataTerminal(int i, size_t s, bool c) :
    id(i), size(s), calibration(c){}
};

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
    int null_window;
    int time_window;
    int current_iteration = 0;
    string path;
    DatasetManager(SharedNetwork* net, string p, int null, int time);
    void createNewTerminal(int id, size_t size, bool calibration);
    void createScoreRule(vector<int> ids, vector<double> weights, vector<int> eval_ids);
    void updateCurrentValues();
};