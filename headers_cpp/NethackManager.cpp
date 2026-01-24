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
constexpr int N = 5;
constexpr int HALF = N / 2;
constexpr int BITS_PER_CELL = 2;

constexpr double K = 10000;

const vector<char> actions = {'.',',','h','f','j','k','n','l','y','u','b','q','d','>','<'};
constexpr int BITS_PER_ACTION = 4;

static NethackManager* g_mgr = nullptr;

inline int bit_idx(int r, int c, int b) {
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
    screen_bits.resize(N * N * BITS_PER_CELL, 0);

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
    return regex_replace(input, ansi, "");
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

    ssize_t n;
    while ((n = read(master_fd, tmp, sizeof(tmp))) > 0) {
        buffer.append(tmp, n);
    }
}

void NethackManager::step() {

    buffer.clear();
    readScreen();
    
    
    if(!parseScreen()){ 
        write(master_fd, " ", 1);
        write(master_fd, ".", 1);

        this_thread::sleep_for(20ms);
    }
    
    if (watch) {
        write(STDOUT_FILENO, buffer.c_str(), buffer.size());
        write(STDOUT_FILENO, "\n", 1);
    }

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

bool NethackManager::checkDeath() {
    if (buffer.find("REST") != string::npos
        && buffer.find("PEACE") != string::npos) {
        return true;
    }
    return false;
}

bool NethackManager::parseScreen() {
    //fill(screen_bits.begin(), screen_bits.end(), 0); //if this happens then a parallel process might receive 0s when it shouldn't

    vector<string> lines;
    size_t pos = 0, next;

    while ((next = buffer.find('\n', pos)) != string::npos) {
        lines.push_back(buffer.substr(pos, next - pos));
        pos = next + 1;
    }

    if (pos < buffer.size())
        lines.push_back(buffer.substr(pos));

    buffer.erase(0, pos);

    vector<string> neg_triggers = {"--More--", "."};
    if(lines.empty()) return false;

    for(string trigger : neg_triggers) if(lines[0].find(trigger) != string::npos) return false;

    int player_r = -1;
    int player_c = -1;

    for (int r = 0; r < lines.size() && player_r == -1; r++) {
        for (int c = 0; c < lines[r].size(); c++) {
            if (lines[r][c] == '@') {
                player_r = r;
                player_c = c;
                break;
            }
        }
    }

    if(player_r == -1) return false;

    for (int dr = -HALF; dr <= HALF; dr++) {
        for (int dc = -HALF; dc <= HALF; dc++) {
            int rr = player_r + dr;
            int cc = player_c + dc;

            uint8_t b0 = 0, b1 = 0; //DEFAULT 00 for any characters
            //TODO do this for multiple bits per cell

            if (rr >= 0 && rr < lines.size() &&
                cc >= 0 && cc < lines[rr].size()) {

                char ch = lines[rr][cc];
                if (ch == '#' || ch == '$')      { b0 = 1; b1 = 0; }
                else if (ch == '|' || ch == '-') { b0 = 0; b1 = 1; }
                else if (ch != '.'){ b0 = 1; b1 = 1; }
            }

            int w_r = dr + HALF;
            int w_c = dc + HALF;

            screen_bits[bit_idx(w_r, w_c, 0)] = b0;
            screen_bits[bit_idx(w_r, w_c, 1)] = b1;
        }
    }

    return true;
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

