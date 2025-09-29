

//For now I'm just dumping code here to clear up main where I can just read the datasets from txt files

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

    //Adding 5 new XOR inputs for it to test
    for(int i = 0; i < 5; i++){
        vector<bool> input = {false};
        vector<bool> input2 = randomBinarySequence(5);
        vector<bool> output;
        if(!input[0]){
            output = vector<bool>(3, XNOR(input2));
        }else{
            output = randomBinarySequence(3);
        }
        input.insert(input.end(),input2.begin(),input2.end());
        dataset.push_back(make_pair(input, output));
    }