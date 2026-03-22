#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>
#include <variant>

#include "Network.h"
#include "HyperOptimizer.h"
#include "HyperParameters.h"
#include "DataTerminal.h"

using namespace std;

struct DataManager
{
    DataManager(SharedNetwork* net);
    SharedNetwork* shared_network;
    vector<unique_ptr<DataTerminal>> terminals;
    void createNewTerminal(int id, size_t size, bool calibration);

    struct ScoreCalculator
    {
        vector<DataTerminal*> terminal_ptrs;
        vector<double> weights;
        vector<Network*> targets;
        double score();
        ScoreCalculator(vector<DataTerminal*> ts, vector<double> w, vector<Network*> ts2) : terminal_ptrs(ts), weights(w), targets(ts2) {}; //calculates score of neuron values (unclamped) vs terminal values 
    };
    vector<ScoreCalculator> score_calculators;
    void createScoreRule(vector<int> ids, vector<double> weights, vector<int> eval_ids); //first ids is terminal

    struct Dataset{
        Dataset(DataManager* manager, string p);
        vector<vector<vector<uint8_t>>> dataset; //[terminal ID][data index][bit]
        vector<bool> shuffle; //Same index as terminal ID
        int current_iteration = 0;
        void updateCurrentValues(DataManager* manager);
    };

    struct NethackManager{
        const int ROWS = 24;
        const int COLS = 80;
        const int N = 50;
        const int BITS_PER_CELL = 2;
        const int TIMEOUTMAX = 20;

        const double K = 10000;

        const vector<char> actions = {'.',',','h','f','j','k','n','l','y','u','b','q','d','>','<'};

        const vector<char> visible_cells = {'#', '$', '|', '@', 'a', 'f'};

        const int BITS_PER_ACTION = 4;

        inline int bit_idx(int r, int c, int b) {
            //this is for a grid view
            return (r * N + c) * BITS_PER_CELL + b;
        }
        NethackManager(DataManager* manager, vector<int> input_indexes, int output_index);
        void launchNetHack();
        void step(DataManager* manager);                    
        void sendAction(DataManager* manager); 
        void initPTY();
        void parseScreen();
        void resetGame();
        bool checkDeath();
        void readScreen();
        vector<int> input_nets;
        int output_net;

        double getScore();
        // PTY / process
        int master_fd;
        pid_t nh_pid;
        bool watch = true;
        int turn_count = 0;
        int timeout = 0;
        string buffer;
        // cached binary state
        vector<uint8_t> screen_bits;
    };

    struct Playground{
        //some terminals will always be clamped, some terminals will respond to outputs from the network and change others, some will give a reward signal
        //switch terminals: one terminal will have a set of trigger values that will cause another one to assume a set of values and possibly clamp
        //switch objects: (switch terminal ; activation value) -> (receiver terminal ; output value)
        //Since one neuron can be referenced by multiple terminals then we can have multiple switches with overlapping terminals for AND/OR logic
        struct Switch{
            size_t transmitter_id;
            vector<vector<uint8_t>> triggers;
            size_t receiver_id;
            vector<uint8_t> signal;
            bool clamper = false; //if the receiver terminal wasn't clamped before it becomes clamped when the switch is flipped; this is permanent until reset
            double reward = 0;
            bool slip = false; //slippery switches turn back off when the terminal stops matching
            bool flipped = false;
            Switch(size_t transmitter, size_t receiver, vector<vector<uint8_t>> trig, vector<uint8_t> output, bool slippery = false, bool clamps = false, double rew = 0): 
            transmitter_id(transmitter), receiver_id(receiver), triggers(trig), signal(output), clamper(clamps), reward(rew), slip(slippery) {};
        };

        vector<Switch> switches;

        Playground();
        double reward();
        void reset(DataManager* manager);
        void applySwitches(DataManager* manager);
    };

    variant<monostate, Dataset, NethackManager, Playground> data_source;
    void makeDatasetManager(string p);
    void makeNethackManager(vector<int> input_indexes, int output_index);
    void makePlayground();
};
