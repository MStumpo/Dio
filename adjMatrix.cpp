#include <vector>
#include <cmath>
#include <random>
#include <algorithm>



using namespace std;

class AdjacencyMatrix{
	private:
		vector<vector<double>> data;



		// neg_lr multiplication
		vector<vector<double>> matMul(const vector<vector<double>>& A, const vector<vector<double>>& B) {
		    size_t N = A.size();
		    vector<vector<double>> C(N, std::vector<double>(N, 0.0));

		    for (size_t i = 0; i < N; ++i) {
		        for (size_t j = 0; j < N; ++j) {
		            for (size_t k = 0; k < N; ++k) {
		                C[i][j] += A[i][k] * B[k][j];
		            }
		        }
		    }
		    return C;
		}

		// Matrix addition
		vector<vector<double>> matAdd(const vector<vector<double>>& A, const vector<vector<double>>& B) {
		    size_t N = A.size();
		    vector<vector<double>> C(N, std::vector<double>(N, 0.0));

		    for (size_t i = 0; i < N; ++i) {
		        for (size_t j = 0; j < N; ++j) {
		            C[i][j] = A[i][j] + B[i][j];
		        }
		    }
		    return C;
		}

		// Scalar multiplication
		vector<vector<double>> scalarMul(const vector<vector<double>>& A, double scalar) {
		    size_t N = A.size();
		    vector<vector<double>> C(N, std::vector<double>(N, 0.0));

		    for (size_t i = 0; i < N; ++i) {
		        for (size_t j = 0; j < N; ++j) {
		            C[i][j] = scalar * A[i][j];
		        }
		    }
		    return C;
		}

		// Compute T = alpha*W + alpha^2*W^2 + ... up to K terms
		vector<vector<double>> computeTotalContribution(const vector<vector<double>>& W, double alpha = 0.1, int K = 10) {
		    size_t N = W.size();

		    vector<vector<double>> T(N, std::vector<double>(N, 0.0));
		    vector<vector<double>> W_power = W; // W^1

		    for (int k = 1; k <= K; ++k) {
		        double factor = pow(alpha, k);
		        T = matAdd(T, scalarMul(W_power, factor));
		        W_power = matMul(W_power, W); // W^(k+1)

		    }
		    double max_val = 0.0;

		    for(const auto& he : T){
		    	for(const auto& val : he){
		    		max_val = max(max_val, fabs(val));
		    	}
		    }


		    return scalarMul(T, 1/max_val);
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

        void updateAdj(const vector<bool> spikes,const vector<double> trace_pre, const vector<double> trace_post, double reg = 0.001,
        	double pos_lr = 0.001, double neg_lr = 0.001){

        	vector<vector<double>> T = computeTotalContribution(data);
         for(int i = 0; i < data.size(); i++){
         		for(int j=0; j < data[i].size(); j++){

         			//printf("\n %f, %f",data[i][j],T[i][j]);

         			data[i][j] += (pos_lr*trace_pre[i]*spikes[j]*(1-data[i][j]) - neg_lr*spikes[i]*trace_post[j]*(1+data[i][j]))/T[i][j] - reg*data[i][j];


         			//printf("---> %f",data[i][j]);


         			data[i][j] = max(-1.0,min(1.0,data[i][j]));
            	}
            }
        }
	
};