#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;

    struct HyperParameters {
        double lr = 0.01;
        double reg = 0.001;
        double entropy_factor = 1.0;
        double decay = 0.01;
        double u_decay = 0.01;
        double determinism = 0.5;
        double firing_value = 1.0;

        const vector<bool> log_scale = {true, true, false, false, false, false, false};
        const vector<pair<double,double>> limits = {{0.00000000001,1},{0.00000000001,1},{-5,5}, {0,1}, {0,1}, {0,1}, {-3,3}}; 

        size_t SIZE = 100;

        double& operator[](size_t i) {
        switch(i) {
            case 0: return lr;
            case 1: return reg;
            case 2: return entropy_factor;
            case 3: return decay;
            case 4: return u_decay;
            case 5: return determinism;
            case 6: return firing_value;
            default: throw out_of_range("Invalid HyperParameter index");
        }
        size_t size() const { return 6; }
        HyperParameters(double l = 0.01, double r = 0.001, double e = 1.0, double d = 0.01, double u = 0.01, double det = 0.5, double f = 1.0):
            lr(l), reg(r), entropy_factor(e), decay(d), u_decay(u), determinism(det), firing_value(f)
        {}
    };

    struct HyperOptimizer {
        vector<HyperParameters> params;
        vector<double> scores;
        mt19937 rng;
        normal_distribution<double> dist{0.0, 1.0};

        HyperOptimizer(HyperParameters& hyperparams, double init_step = 3)
            : rng(random_device{}())
        {
            params.push_back(hyperparams);
        }

        HyperParameters propose(size_t N = 5, double var = 0.1) {
            vector<size_t> indices(scores.size());
            iota(indices.begin(), indices.end(), 0);
            sort(indices.begin(), indices.end(), [&](size_t i, size_t j){
                return scores[i] > scores[j];
            });
            HyperParameters estimated;
            double total_score = 0.0;
            for (size_t k = 0; k < N; ++k) total_score += scores[indices[k]];
            for (size_t p = 0; p < estimated.SIZE; ++p) {
                double weighted_sum = 0.0;
                if(estimated.log_scale[p]) weighted_sum = 1.0;
                for (size_t k = 0; k < N; ++k) {
                    weighted_sum += params[indices[k]][p] * (scores[indices[k]] / total_score);
                }
                estimated[p] = clip(weighted_sum    + dist(rng)*(estimated.limits[p].second - estimated.limits[p].first)*N/(total_score), estimated.limits[p].first, estimated.limits[p].second);
                if(estimated.log_scale[p]) estimated[p] = log(estimated[p]);
            }
            return estimated;
        }

        void update(const HyperParameters candidate, double score) {
            params.push_back(candidate);
            scores.push_back(score);
        }
    };
