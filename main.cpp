
#include "network.cpp"
#include <cstdlib>  // For std::stoi and std::stod
#include <variant>
#include <fstream>


vector<pair<vector<bool>, vector<bool>>> readDatasetFile(const string& filename){
    vector<pair<vector<bool>, vector<bool>>> dataset;
    ifstream infile(filename);
    string line;
    while(getline(infile,line)){
        dataset.emplace_back();
        bool first = true;
        for(char c : line){
            if(c == '-' || c == '.' || c == '|'){
                first = false;
                continue;
            }
            if(c != '0' && c != '1') continue;

            if(first){
                dataset.back().first.push_back(c == '1');
            }else{
                dataset.back().second.push_back(c == '1');
            }
        }
    }   
    return dataset;
}

int main(int argc, char *argv[]){

    using Args = variant<int, double, bool>;

    vector<pair<string, Args>> networkArgs = {{"--neuron-size", 30}, {"--time-window", 10}, {"--reg", 0.001}, {"--pos-lr", 0.01},
                                                 {"--neg-lr", 0.01}, {"--decay",0.05}, {"--path-decay", 0.1},
                                                  {"--determinism", 0.5},{"--firing-value", 1.0},{"--null-window", 0}
                                                   };

    int epochs = 1;
    vector<pair<vector<bool>, vector<bool>>> dataset;
    vector<pair<vector<bool>, vector<bool>>> dataset_test;


    for (int i = 1; i < argc; i++) {
        for(int j = 0; j <networkArgs.size(); j++){
            if(string(argv[i]) == networkArgs[j].first){
                visit([&](auto& val){
                    using T = decay_t<decltype(val)>;
                    if constexpr (is_same_v<T, int>){
                        networkArgs[j].second = stoi(argv[i+1]);
                        i++;
                    }else if constexpr (is_same_v<T, double>){
                        networkArgs[j].second = stod(argv[i+1]);
                        i++;
                    }else if constexpr (is_same_v<T, bool>){
                        networkArgs[j].second = (string(argv[i+1]) == "true" || string(argv[i+1]) == "1");
                        i++;
                    }
                }, networkArgs[j].second);
            }
        }
        if(string(argv[i]) == "--train-epochs" || string(argv[i]) == "--epochs"){
            epochs = stoi(argv[i+1]);
            printf("\ntrain_epochs: %d", epochs);

            i++;
        }else if (string(argv[i]) == "--dataset"){
            dataset = readDatasetFile(argv[i+1]);
            i++;
        }else if (string(argv[i]) == "--dataset-test"){
            dataset_test = readDatasetFile(argv[i+1]);
            i++;
        }
    }

    if(dataset_test.empty()){
        dataset_test = dataset;
        printf("ASSUMING TEST = TRAIN");
    }


	Network network = Network(networkArgs);

    network.runFull(dataset, dataset_test, epochs);

    //network.printAdjMatrix();

	return 0;
}