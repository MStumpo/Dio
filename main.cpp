
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

bool XNOR(vector<bool> input){
    for(int i = 1; i < input.size(); i++){
        if(input[i] != input[i-1]) return false;
    }
    return true;
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
                                                  {"--kernel-normalization", false}, {"--determinism", 0.0}, {"--firing-value", 1.0}, {"--verbose", false},
                                                  {"--row-only", false}};


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


    //For this I'll make a 1st bit switch: 0-XNOR; 1-random vector, and the rest are "payload" bits

    /*
    int dataset_size = 30;

    int payload_size = 5;
    int output_size = 3;

    for(int i = 0; i < dataset_size; i++){
        vector<bool> input = randomBinarySequence(1);
        vector<bool> input2 = randomBinarySequence(payload_size);
        vector<bool> output;
        if(!input[0]){
            output = vector<bool>(output_size, XNOR(input2));
        }else{
            output = randomBinarySequence(output_size);
        }
        input.insert(input.end(),input2.begin(),input2.end());
        dataset.push_back(make_pair(input, output));
    }
    //this is for generating code for putting it back as a custom pre-made dataset
    for(auto& datapoint : dataset){
        printf("\n dataset.push_back(make_pair(vector<bool>{");
        for(auto wawa : datapoint.first){
            printf("%d,", wawa ? 1 : 0);
        }
        printf("},vector<bool>{");
        for(auto wawa : datapoint.second){
            printf("%d,", wawa ? 1 : 0);
        }
        printf("}));");
    }
    */
    //I decided on a custom dataset because 1. Both input and output may be random and thus this may not a function 2. nxfalse or nxtrue is a 0.5^n chance and it'd barely contain positive XNOR examples
     dataset.push_back(make_pair(vector<bool>{1,0,0,1,0,1},vector<bool>{0,1,0}));
     dataset.push_back(make_pair(vector<bool>{0,1,0,1,1,1},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{1,0,1,1,1,1},vector<bool>{1,0,0}));
     dataset.push_back(make_pair(vector<bool>{0,0,0,0,0,0},vector<bool>{1,1,1}));
     dataset.push_back(make_pair(vector<bool>{0,0,0,0,1,1},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{1,0,1,0,0,0},vector<bool>{1,0,1}));
     dataset.push_back(make_pair(vector<bool>{1,1,1,0,1,0},vector<bool>{0,0,1}));
     dataset.push_back(make_pair(vector<bool>{1,0,1,0,1,1},vector<bool>{1,1,0}));
     dataset.push_back(make_pair(vector<bool>{1,0,0,1,1,1},vector<bool>{1,0,0}));
     dataset.push_back(make_pair(vector<bool>{0,1,0,1,1,0},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{1,0,1,1,0,0},vector<bool>{0,1,1}));
     dataset.push_back(make_pair(vector<bool>{0,0,0,0,0,0},vector<bool>{1,1,1}));
     dataset.push_back(make_pair(vector<bool>{0,1,1,1,1,1},vector<bool>{1,1,1}));
     dataset.push_back(make_pair(vector<bool>{1,1,1,1,1,1},vector<bool>{0,1,0}));
     dataset.push_back(make_pair(vector<bool>{0,0,0,0,0,0},vector<bool>{1,1,1}));
     dataset.push_back(make_pair(vector<bool>{0,0,0,1,1,1},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{1,1,1,1,1,0},vector<bool>{1,1,1}));
     dataset.push_back(make_pair(vector<bool>{0,1,1,0,1,1},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{0,1,1,1,1,1},vector<bool>{1,1,1}));
     dataset.push_back(make_pair(vector<bool>{0,1,1,1,1,0},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{1,1,1,1,0,1},vector<bool>{1,0,1}));
     dataset.push_back(make_pair(vector<bool>{0,0,1,0,1,0},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{0,1,1,1,1,1},vector<bool>{1,1,1}));
     dataset.push_back(make_pair(vector<bool>{0,1,1,0,0,1},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{1,0,0,1,0,0},vector<bool>{0,1,1}));
     dataset.push_back(make_pair(vector<bool>{0,1,0,0,0,1},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{0,0,0,1,0,0},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{1,0,1,1,0,1},vector<bool>{1,1,0}));
     dataset.push_back(make_pair(vector<bool>{1,0,0,0,0,0},vector<bool>{0,1,0}));
     dataset.push_back(make_pair(vector<bool>{0,1,1,0,0,0},vector<bool>{0,0,0}));
     dataset.push_back(make_pair(vector<bool>{0,0,0,0,0,0},vector<bool>{1,1,1}));
     dataset.push_back(make_pair(vector<bool>{1,1,1,0,1,1},vector<bool>{1,1,1}));



	Network network = Network(networkArgs);

	network.train(dataset,train_epochs);	

    
    network.printAdjMatrix();

	network.test(dataset, test_epochs);

    /*
    for(auto& datum : dataset){
        network.validate(datum.first,datum.second, 10);
    }*/
    //network.validate(dataset[0].first,dataset[0].second, 10);
    //network.validate(dataset[3].first,dataset[3].second, 10);



	return 0;
}