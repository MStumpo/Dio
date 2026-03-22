#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <regex>

#include <pty.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <sys/wait.h>

#include "DataManager.h"
#include "SharedNetwork.h"
using namespace std;

DataManager::DataManager(SharedNetwork* net) : shared_network(net){}

DataManager::Dataset::Dataset(DataManager* manager, string p){

    ifstream file(p);
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
            manager->createNewTerminal(manager->terminals.size(), values.size(), false);

            dataset.push_back(vector<vector<uint8_t>>({}));
            shuffle.push_back((shuff == "1" ? 1 : 0));

            if(stoi(sub_net) != prev_sub_net) prev_sub_net_neuron = 0;

            for(int i = 0; i < values.size(); i++) manager->terminals[manager->terminals.size()-1]->coordinates.push_back(manager->shared_network->sub_networks[stoi(sub_net)]->neurons[prev_sub_net_neuron + i]);

            prev_sub_net_neuron += values.size();
            prev_sub_net = stoi(sub_net);
        }
        dataset[dataset.size()-1].push_back(values); //always update to latest terminal dataset is read [TERMINALID][DATA_INDEX][BIT]
    }
}

void DataManager::createNewTerminal(int id, size_t size, bool calibration){
    terminals.push_back(make_unique<DataTerminal>(id, size, calibration));
}

void DataManager::createScoreRule(vector<int> ids, vector<double> weights, vector<int> eval_ids){
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

void DataManager::Dataset::updateCurrentValues(DataManager* manager){
    int rand_idx = rand()%dataset[0].size(); //we need to assume all terminals have the same number of indices  
    for(int i = 0; i < dataset.size(); i++){//[terminal ID][data index][bit]
        if(!shuffle[i]){ manager->terminals[i]->updateValues(dataset[i][(current_iteration+1)%dataset[i].size()]);}
        else {
            manager->terminals[i]->updateValues(dataset[i][rand_idx]);
        }
    }
    current_iteration++;
}

double DataManager::ScoreCalculator::score(){
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

DataManager::NethackManager::NethackManager(
    DataManager* manager,
    vector<int> input_indexes,
    int output_index
)
    : input_nets(input_indexes),
      output_net(output_index),
      master_fd(-1),
      nh_pid(-1)
{
    screen_bits.resize(N * BITS_PER_CELL, 0);


    //These distribute the screen bits throughout the input nets as equally as possible so it's not gonna be plugged in createNewTerminal
    for (int i = 0; i < input_nets.size(); i++) {
        manager->terminals.push_back(make_unique<DataTerminal>(
            input_nets[i],
            (screen_bits.size() / input_nets.size()) +
                (i == 0) * screen_bits.size() % input_nets.size(),
            false
        ));

        for (int n = 0;
             n < (screen_bits.size() / input_nets.size()) +
                     (i == 0) * screen_bits.size() % input_nets.size();
             n++) {

            manager->terminals.back()->coordinates.push_back(
                manager->shared_network->sub_networks[input_nets[i]]->neurons[n]
            );
            manager->terminals.back()->clamped = true;
        }
    }

    manager->terminals.push_back(make_unique<DataTerminal>(
        output_net,
        BITS_PER_ACTION,
        false
    ));

    for (int n = 0; n < BITS_PER_ACTION; n++) {
        manager->terminals.back()->coordinates.push_back(
            manager->shared_network
                ->sub_networks[output_index]
                ->neurons[
                    manager->shared_network
                        ->sub_networks[output_index]
                        ->hp.NEURON_SIZE - 1 - n
                ]
        );
    }

    manager->terminals.back()->clamped = false;
}

string stripAnsi(const string& input) {
    static const regex ansi("\x1B\\[[0-9;]*[A-Za-z]");
    return regex_replace(input, ansi, " ");
}

void DataManager::NethackManager::launchNetHack() {
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

void DataManager::NethackManager::resetGame() {
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
    timeout = 0;
    buffer.clear();  
    launchNetHack();
}

void DataManager::NethackManager::readScreen() {
    char tmp[4096*2];
    ssize_t n;
    while ((n = read(master_fd, tmp, sizeof(tmp))) > 0) {
        buffer.append(tmp, n);
    }
}

void DataManager::NethackManager::step(DataManager* manager) {

    readScreen();

    vector<uint8_t> prev_bits = screen_bits;

    parseScreen();

    if(prev_bits == screen_bits){
        write(master_fd, " ", 1);
        write(master_fd, ".", 1);
        write(master_fd, "\n", 1);
        timeout++;
    }else{
        timeout = 0;
        turn_count++;
        for (int i = 0; i < input_nets.size(); i++) {
            manager->terminals[i]->updateValues(
                vector<uint8_t>(
                    screen_bits.begin() + manager->terminals[i]->size * i,
                    screen_bits.begin() + manager->terminals[i]->size * (i + 1)
                )
            );
        }
    }   
}

bool DataManager::NethackManager::checkDeath() {
    if ((buffer.find("REST") != string::npos
        && buffer.find("PEACE") != string::npos) || timeout >= TIMEOUTMAX) {
        return true;
    }
    return false;
}

void DataManager::NethackManager::parseScreen() {
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

double DataManager::NethackManager::getScore() {
    regex r(R"((\d+)\s*Au)");
    smatch m;
    regex_search(buffer, m, r);
    double score = (m.size() >= 1) ? stod(m[1]) : -1;

    return (score + 1)/(score + K/turn_count) ;
}

void DataManager::NethackManager::sendAction(DataManager* manager) {
    int code = 0;

    for (int b = 0; b < BITS_PER_ACTION; b++) {
        code |= manager->terminals.back()->coordinates[b]->value << b;
    }

    int action_idx = code % actions.size();
    char action = actions[action_idx];

    write(master_fd, &action, 1);
}

double DataManager::Playground::reward(){
    double final_reward = 0;
    for(Switch s : switches) if(s.flipped) final_reward += s.reward;

    return min(max(final_reward, 1.0),-1.0);
}

void DataManager::Playground::applySwitches(DataManager* manager){
    for(Switch s : switches){
        if((!s.slip && s.flipped)) continue;
        bool activated = true;
        for(vector<uint8_t> vals : s.triggers){
            activated = true;
            for(int b = 0; b < manager->terminals[s.transmitter_id]->size; b++)if(manager->terminals[s.transmitter_id]->coordinates[b]->value != vals[b]){
                activated = false;
                break;
            };
            if(activated){
                manager->terminals[s.receiver_id]->values = s.signal;
                if(s.clamper) manager->terminals[s.receiver_id]->clamped = true;
                s.flipped = true;
                break;
            }else if(s.slip){
                if(s.clamper) manager->terminals[s.receiver_id]->clamped = false;
                s.flipped = false;
            };
        }
    }
}

void DataManager::Playground::reset(DataManager* manager){
    for(Switch s : switches){
        s.flipped = false;
        if(s.clamper) manager->terminals[s.receiver_id]->clamped = false;
        fill(manager->terminals[s.receiver_id]->values.begin(), manager->terminals[s.receiver_id]->values.end(), false);
    }
}

DataManager::Playground::Playground(){
    switches = {};
}

void DataManager::makeDatasetManager(string p){
    data_source.emplace<Dataset>(this, p);
}
void DataManager::makeNethackManager(vector<int> input_indexes, int output_index){
    data_source.emplace<NethackManager>(this, input_indexes, output_index);
}
void DataManager::makePlayground(){
    data_source.emplace<Playground>();
}