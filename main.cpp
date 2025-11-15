#include <cstdlib>  // For std::stoi and std::stod
#include <variant>
#include <fstream>
#include <chrono>
#include <thread>


//TODO::
/*
-FIX HYPERPARAMS IN NETWORKARGS CODE
-FIX HYPERPARAMS IN HYPERPARAM OPTIMIZER (move struct there?)
-GET PARENT NETWORK FROM NEURON OR EDGE FOR HYPERPARAMS IN FIRING AND TRACE UPDATE
--DECIDE WHAT HP TO CHOOSE DO IF A NEURON IS MERGED 
*/

int main(int argc, char *argv[]){


    //NETWORKS ARGS IS OBSOLETE
    int epochs = 1;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if(string(argv[i]) == "--epochs"){
            epochs = stoi(argv[i+1]);
            printf("\ntrain_epochs: %d", epochs);
            i++;
        }else if (string(argv[i]) == "--dataset"){
            dataset = readDatasetFile(argv[i+1]);
            i++;
        }else if (string(argv[i]) == "--dataset-test"){
            dataset_test = readDatasetFile(argv[i+1]);
            i++;
        }else if (string(argv[i]) == "--optimize"){
	       optimize = stoi(argv[i+1]);
           i++;
    	}else if(string(argv[i]) == "--verbose"){
    		verbose = true;
    	}
    	else{
    	    printf("\n!!!!! UNKNOWN COMMAND !!!!!\n");
                printf("%s\n", string(argv[i]).c_str());
    	}
    }

    if(dataset_test.empty()){
        dataset_test = dataset;
        printf("ASSUMING TEST = TRAIN");
    }

	return 0;
}
