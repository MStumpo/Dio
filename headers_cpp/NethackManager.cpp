#include "NethackManager.h"

#include <pty.h>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <fcntl.h>
#include <thread>
#include <chrono>
#include <regex>

using namespace std;

constexpr int ROWS = 24;
constexpr int COLS = 80;
//constexpr int N = 20; //These represent the whole vision now so if you want a grid or square better make it a perf square or at least not a prime
constexpr int N = 23; //you know what fuck you
constexpr int BITS_PER_CELL = 2;

constexpr double K = 10000;

const vector<char> actions = {'.',',','h','f','j','k','n','l','y','u','b','q','d','>','<'};

const vector<char> visible_cells = {'#', '$', '|', '@', 'a', 'f'};

constexpr int BITS_PER_ACTION = 4;

static NethackManager* g_mgr = nullptr;

inline int bit_idx(int r, int c, int b) {
    //this is for a grid view
    return (r * N + c) * BITS_PER_CELL + b;
}

NethackManager::NethackManager(
    SharedNetwork* net,
    vector<int> input_indexes,
    int output_index
)
    : shared_network(net),
      input_nets(input_indexes),
      output_net(output_index),
      master_fd(-1),
      nh_pid(-1)
{
    screen_bits.resize(N * BITS_PER_CELL, 0);


    //These distribute the screen bits throughout the input nets as equally as possible
    for (int i = 0; i < input_nets.size(); i++) {
        terminals.push_back(make_unique<DataTerminal>(
            input_nets[i],
            (screen_bits.size() / input_nets.size()) +
                (i == 0) * screen_bits.size() % input_nets.size(),
            false
        ));

        for (int n = 0;
             n < (screen_bits.size() / input_nets.size()) +
                     (i == 0) * screen_bits.size() % input_nets.size();
             n++) {

            terminals.back()->coordinates.push_back(
                shared_network->sub_networks[input_nets[i]]->neurons[n]
            );
            terminals.back()->clamped = true;
        }
    }

    terminals.push_back(make_unique<DataTerminal>(
        output_net,
        BITS_PER_ACTION,
        false
    ));

    for (int n = 0; n < BITS_PER_ACTION; n++) {
        terminals.back()->coordinates.push_back(
            shared_network
                ->sub_networks[output_index]
                ->neurons[
                    shared_network
                        ->sub_networks[output_index]
                        ->hp.NEURON_SIZE - 1 - n
                ]
        );
    }

    terminals.back()->clamped = false;
}

string stripAnsi(const string& input) {
    static const regex ansi("\x1B\\[[0-9;]*[A-Za-z]");
    return regex_replace(input, ansi, " ");
}

void NethackManager::launchNetHack() {
    fflush(nullptr);
    setenv("TERM", "xterm", 1);

    nh_pid = forkpty(&master_fd, nullptr, nullptr, nullptr);
    if (nh_pid < 0) {
        write(
            2,
            "the child failed\n",
            sizeof("the child failed\n") - 1
        );
        perror("forkpty failed");
        _exit(1);
    }

    if (nh_pid == 0) {
        write(2, "child alive\n", sizeof("child alive\n") - 1);
        execl("/usr/bin/nethack", "nethack", nullptr);
        _exit(1);
    }
    fcntl(master_fd, F_SETFL, O_NONBLOCK);


    write(master_fd, "y", 1);
    write(master_fd, "\n", 1);
}

void NethackManager::resetGame() {
    if (master_fd >= 0) {
        close(master_fd);
        master_fd = -1;
    }

    if (nh_pid > 0) {
        int status;
        while (waitpid(nh_pid, &status, 0) == -1 && errno == EINTR) {}
        nh_pid = -1;
    }

    fill(screen_bits.begin(), screen_bits.end(), 0);
    turn_count = 0;
    buffer.clear();  
    launchNetHack();
}

void NethackManager::readScreen() {
    char tmp[4096];
    buffer.clear();
    ssize_t n;
    while ((n = read(master_fd, tmp, sizeof(tmp))) > 0) {
        buffer.append(tmp, n);
    }
}

void NethackManager::step() {

    readScreen();

    vector<uint8_t> prev_bits = screen_bits;

    parseScreen();

    if(prev_bits == screen_bits){
        write(master_fd, " ", 1);
        write(master_fd, ".", 1);
        write(master_fd, "\n", 1);
    }else {
        turn_count++;
        for (int i = 0; i < input_nets.size(); i++) {
            terminals[i]->updateValues(
                vector<uint8_t>(
                    screen_bits.begin() + terminals[i]->size * i,
                    screen_bits.begin() + terminals[i]->size * (i + 1)
                )
            );
        }
    }   
}

bool NethackManager::checkDeath() {
    if (buffer.find("REST") != string::npos
        && buffer.find("PEACE") != string::npos) {
        return true;
    }
    return false;
}

void NethackManager::parseScreen() {
    fill(screen_bits.begin(), screen_bits.end(), 0); 
    //this just reads the first N visible characters in the buffer and passes them to the screen bits
    int idx = 0;
    vector<uint8_t> current_cell(BITS_PER_CELL, 0);
    for(int i = buffer.size()-1; i >= 0 && idx < N*BITS_PER_CELL; i--){ //read it backwards to be compatible with a non-clearing buffer
        if(buffer[i] == ' ') continue;

        for(int a = 0; a < visible_cells.size(); a++){
            if(buffer[i] == visible_cells[a]){
                generate(current_cell.rbegin(), current_cell.rend(), [n = a % (1 << BITS_PER_CELL)]() mutable { auto b = n & 1; n >>= 1; return b; });
                //People are using LLMs for all the wrong reasons like look at this shit this is a oneliner for binary(a%BITSPERCELL) 
                for(int b = 0; b < BITS_PER_CELL; b++){
                    screen_bits[idx] = current_cell[b];
                    idx++;
                }
                break;
            }
        }
    }
}

double NethackManager::getScore() {
    regex r(R"((\d+)\s*Au)");
    smatch m;
    regex_search(buffer, m, r);
    double score = stod(m[1]);

    return (score + 1)/(score + K/turn_count) ;
}

void NethackManager::sendAction() {
    int code = 0;

    for (int b = 0; b < BITS_PER_ACTION; b++) {
        code |= terminals.back()->coordinates[b]->value << b;
    }

    int action_idx = code % actions.size();
    char action = actions[action_idx];

    write(master_fd, &action, 1);
}

