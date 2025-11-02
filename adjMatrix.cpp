#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

class AdjacencyMatrix{
	private:
		vector<vector<double>> data;

		//These here assume square matrices but ya know
		vector<double> colAverages(const vector<vector<double>>& matrix){
			int n = matrix.size();

			vector<double> sums(n, 0.0);

			for(int col = 0; col < n; col++){
				for(int row = 0; row < n; row++){
					sums[col] += matrix[row][col];
				}
				sums[col] /= n;
			}
			return sums;
		}

	    vector<vector<double>> getMinor(const vector<vector<double>>& matrix, int row, int col) {
	    int n = matrix.size();
	    vector<vector<double>> minor;
	    for (int i = 0; i < n; i++) {
	        if (i == row) continue;
	        vector<double> minorRow;
	        for (int j = 0; j < n; j++) {
	            if (j == col) continue;
	            minorRow.push_back(matrix[i][j]);
	        }
	        minor.push_back(minorRow);
	    }
	    return minor;
		}

		// Recursive determinant calculation
		double determinant(const vector<vector<double>>& matrix) {
		    int n = matrix.size();

		    double det = 0.0;
		    for (int col = 0; col < n; col++) {
		        double cofactor = pow(-1, col) * matrix[0][col] * determinant(getMinor(matrix, 0, col));
		        det += cofactor;
		    }
		    return det;
		}


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
		vector<vector<double>> computeTotalContribution(const vector<vector<double>>& W, double path_decay = 0.1, int K = 10) {
		    size_t N = W.size();
		    vector<vector<double>> T(N, vector<double>(N, 0.0));
		    vector<vector<double>> W_power = W; // W^1
		    double factor = 1.0;

		    for (int k = 1; k <= K; ++k) {
		        T = matAdd(T, scalarMul(W_power, factor));
		        W_power = matMul(W_power, W); // W^(k+1)
		        factor *= path_decay;
		    }
		    double max_val = 0.0;

		    for(const auto& he : T){
		    	for(const auto& val : he){
		    		max_val = max(max_val, fabs(val));
		    	}
		    }
		    return scalarMul(T, 1/max_val);
		}
		vector<double> colEntropy(const vector<vector<double>>& W){
			size_t N = W.size();
			vector<double> entropy(N, 0.0);

			for(int col = 0; col < N; col++){
				for(int row=0; row < N; row++){
					if(W[row][col] != 0.0) entropy[col] += -abs(W[row][col])*log(abs(W[row][col]));
				}
				entropy[col] /=N;
			}

			return entropy;
		}

		vector<double> U_squared(const vector<vector<double>>& U){
			size_t N = U.size();
			vector<double> U_sq(N, 0.0);
			for(int i = 0; i < N; i++){
				U_sq[i] += pow(U[i][i],2);
				for(int j = 0; j < i; j++){
					U_sq[i] += pow(U[j][i],2);
					U_sq[j] += pow(U[i][j],2);
				}
				U_sq[i] /= N;
			}
			return U_sq;
		}
	public:

    	vector<double>& operator[](size_t row){
    		return data[row];
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

      void updateAdj(const vector<uint8_t> spikes ,const vector<double> trace, const vector<vector<double>> U, double reg, double lr){
		//vector<double> U_sq = U_squared(U);
      //vector<vector<double>> C = computeTotalContribution(data, 0.5, 5);
         for(int i = 0; i < data.size(); i++){
         		for(int j=0; j < data[i].size(); j++){
         			//printf("%f, %f, %d\n",data[0][2], trace[0], spikes[2] ? 1 : 0);

		         	//data[i][j] = data[i][j]*(1-reg*trace[i]) + lr*spikes[j]*(U[i][j] - sqrt(U_sq[j]));

		         	data[i][j] += lr*(U[i][j] - reg*data[i][j]*pow(trace[i],2)); 
						data[i][j] = max(-1.0,min(1.0,data[i][j]));
            	}
            }
      }
};
