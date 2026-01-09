#pragma once
#include "DataTerminal.h"
#include "SharedNetwork.h"

#include <vector>
#include <memory>
#include <cstdint>
#include <string>

using namespace std;

struct NethackManager {
    NethackManager(SharedNetwork* net, vector<int> input_indexes, int output_index);

    void launchNetHack();
    void step();                    
    void sendAction(); 
    void initPTY();
    bool parseScreen(string screen);
    void resetGame();
    bool checkDeath();
    string readScreen();
    vector<int> input_nets;
    int output_net;

    double getScore();

    SharedNetwork* shared_network;

    vector<unique_ptr<DataTerminal>> terminals;

    // PTY / process
    int master_fd;
    pid_t nh_pid;

    bool watch = true;
    int turn_count = 0;

    // terminal emulation
    struct VTerm* vt;
    struct VTermScreen* vts;

    // cached binary state
    vector<uint8_t> screen_bits;
};
