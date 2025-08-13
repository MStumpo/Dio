#include <vector>
#include <cmath>
#include <random>
#include <algorithm>



using namespace std;

class AdjacencyMatrix{
	private:
		vector<vector<double>> data;

		vector<vector<double>> crossKernelMatrixEntropy(int kernelSize = 2, bool normalization = false){
			
			// H = - sum(p*log(p))
		    // H_norm = H*e/(n*log(e)) where n is total analytes (2*kernel_size +2)
		    //normalization is of the probabilities of adjMatrix [-1 , 1] -> [0,1], not H_norm
		    //since our ps are [-1,1] we can do either (p+1)/2 normalization or sum(|p|*log(|p|))
		    //these two strats can be switched around via the normalization bool

		    vector<vector<double>> entropy(data.size(), vector<double>(data[0].size(), 0.0));
			double *row_buffer[kernelSize*2 +1];
			int a = 0;
			for(int i =0; i < data.size(); i++){

				//column shift and row reset because we're on the other side of the matrix now
				for(int x = 0; x < kernelSize*2 +1; x++){
					row_buffer[x] = (x < kernelSize) ? &data[i][x] : NULL;
				}
				a = kernelSize;

				for(int j = 0; j < data[i].size(); j++){

					int n = 0;

					//condition to prevent out of bounds attributing: stops at edge of matrix
					if(j+kernelSize < data[i].size()){
						row_buffer[a] = &data[i][j+kernelSize];
					}else{
						row_buffer[a] = NULL; 
					}
					a = (a+1)%(kernelSize*2 + 1);


					//Fetching from the cols since they're always new in a colshift
		            for(int k = 1; i+k < data.size() && k <= kernelSize; k++){
		            	if(normalization){
		            		entropy[i][j] += -((data[i+k][j]+1)/2)*log((data[i+k][j]+1)/2);
		            	}else{
		            		entropy[i][j] += -abs(data[i+k][j])*log(abs(data[i+k][j]));
		            	}
		            	n++;
		            }
		           	for(int k = 1; i-k >= 0 && k <= kernelSize; k++){
		            	if(normalization){
		            		entropy[i][j] += -((data[i-k][j]+1)/2)*log((data[i-k][j]+1)/2);
		            	}else{
		            		entropy[i][j] += -abs(data[i-k][j])*log(abs(data[i-k][j]));
		            	}
		            	n++;
		            }

		            /*Going to the buffer :) note: I'm not doing if the coordinate = ij because I took it off the col loop and I want the actual value  
		            * to be included in the entropy calculation and it's already in the buffer
		            */
		            for(auto& buffer : row_buffer){

		            	if(buffer != NULL){
		            		double val = *buffer;
			            	if(normalization && val != -1){
			                    entropy[i][j] += -((val+1)/2)*log((val+1)/2);
			                }else if(!normalization && val != 0) {
			                    entropy[i][j] += -abs(val)*log(abs(val));
			                }
			                n++;
		            	}
		            }

		           	entropy[i][j] *= 2.718281846/(log(2.718281846) * n);

				}

			}
			return entropy;
		}

	public:

    	vector<double> operator[](size_t col){
    		return data[col];
    	}

    	size_t cols() const { return data[0].size(); }
    	size_t rows() const { return data.size(); }


		AdjacencyMatrix(size_t neuronSize){
			uniform_real_distribution<double> unif(-1,+1);
			random_device rnd_device;
		    auto gen = [&](){
               return unif(rnd_device);
           	};

            data = vector<vector<double>>(neuronSize, vector<double>(neuronSize));
            for(auto& a : data){
            	generate(a.begin(), a.end(), gen);
            }

		}

        void updateAdj(vector<vector<double>> crossCorrelation, double reward = 1, double lr = 0.001, double reg = 0.001, int kernelSize = 2,  bool kernelNormalization=false, double entropyFactor = 1){
            vector<vector<double>> entropy(data.size(), vector<double>(data[0].size(), 1));


            if (entropyFactor != 0.0){
				entropy = crossKernelMatrixEntropy(kernelSize, kernelNormalization);

            }
            for (int i=0; i < data.size(); i++){
                for(int j=0; j < data[i].size(); j++){

                    data[i][j] += lr*crossCorrelation[i][j]*reward*pow(entropy[i][j],entropyFactor) - reg*data[i][j];

                    data[i][j] =  max(-1.0, min(1.0, data[i][j]));  // Clip

                    //printf("[%d][%d], cross_corr: %f, entropy: %f, data: %f\n", i,j,crossCorrelation[i][j],entropy[i][j],data[i][j]);

                }
            }
        }
	
};