
#include "network.cpp"
#include <cstdlib>  // For std::stoi and std::stod
#include <variant>


std::vector<bool> randomBinarySequence(int numBits, double p = 0.5){
    std::vector<bool> binary(numBits, false);

    uniform_real_distribution<double> unif(0,+1);
	random_device rnd_device;

    for(int i = 0; i < numBits; i++){
        if(unif(rnd_device) < p){
            binary[i] = true;
        }
    }
    return binary;
}

vector<pair<vector<bool>, vector<bool>>> generateRandomDataset(int datasetSize=16, int inputSize=5,int outputSize=5, double p=0.5){

	vector<pair<vector<bool>, vector<bool>>> dataset;
    vector<bool> inputVector(inputSize, false);

	for(int i = 0; i < datasetSize; i++){
		dataset.push_back({inputVector, randomBinarySequence(outputSize,p)});
        for(int j = inputSize - 1; j >= 0; j--){
            if(!inputVector[j]){
                inputVector[j] = true;
                break;
            }else{
                inputVector[j] = false;
            }
        }
	}
	return dataset;
}

int main(int argc, char *argv[]){

    using Args = variant<int, double, bool>;

    vector<pair<string, Args>> networkArgs = {{"--neuron-size", 30}, {"--time-window", 10}, {"--lr", 0.001}, {"--reg", 0.001}, {"--tau-pos", 1.0},
                                                 {"--tau-neg", 2.0}, {"--decay",0.95}, {"--entropy-factor", 1.0}, {"--kernel-size", 2},
                                                  {"--kernel-normalization", false}, {"--determinism", 0.0}, {"--firing-value", 1.0}, {"--verbose", false}};


    int train_epochs = 1;
    int test_epochs  =1;
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
                        networkArgs[j].second = (argv[i+1] == "true" || argv[i+1] == "1");
                        i++;
                    }
                }, networkArgs[j].second);
            }
        }
        if(string(argv[i]) == "--train-epochs" || string(argv[i]) == "--epochs"){
            train_epochs = stoi(argv[i+1]);
            printf("\ntrain_epochs: %d", train_epochs);

            i++;
        }else if (string(argv[i]) == "--test-epochs"){
            test_epochs = stoi(argv[i+1]);

            printf("\ntest_epochs: %d", test_epochs);

            i++;
        }
    }

    vector<pair<vector<bool>, vector<bool>>> dataset;

    /*
    dataset.push_back(make_pair(vector<bool>{false, false}, vector<bool>{false}));
    dataset.push_back(make_pair(vector<bool>{false, true}, vector<bool>{false}));
    dataset.push_back(make_pair(vector<bool>{true, false}, vector<bool>{false}));
    dataset.push_back(make_pair(vector<bool>{true, true}, vector<bool>{true}));
    dataset.push_back(make_pair(vector<bool>{true, true}, vector<bool>{true}));
    dataset.push_back(make_pair(vector<bool>{true, true}, vector<bool>{true}));//repetition of these is to avoid a bias for the final answer
    */
    
    dataset = generateRandomDataset();

	Network network = Network(networkArgs);

	network.train(dataset,train_epochs);	

    
    network.printAdjMatrix();

	network.test(dataset, test_epochs);
    //network.validate(dataset[0].first,dataset[0].second, 30);
    //network.validate(dataset[3].first,dataset[3].second, 30);



	return 0;
}