#include "HyperOptimizer.h"
#include <cmath>
#include <algorithm>

using namespace std;



HyperParameters HyperOptimizer::propose(size_t N) {

    vector<size_t> indices(hist.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&](size_t i, size_t j){
        return hist[i].second > hist[j].second;
    });

    HyperParameters estimated;
    double total_score = 0.0; //ASSUME MAX SCORE IS 1.0 for noise calculation
    for(size_t k = 0; k < N && k < hist.size(); ++k) total_score += hist[indices[k]].second;

    for(size_t p = 0; p < estimated.SIZE; ++p) {
        double weighted_sum = 0.0;
        bool is_log = estimated.log_scale[p];

        for(size_t k = 0; k < N && k < hist.size(); ++k) {
            weighted_sum += (is_log ? log(hist[indices[k]].first[p]) : hist[indices[k]].first[p]) * (hist[indices[k]].second / total_score);
        }

        // add noise and clip to limits
        double noise = 0.1*dist(rng)*(1.0 - total_score)*(is_log ? log(estimated.limits[p].second) - log(estimated.limits[p].first): estimated.limits[p].second - estimated.limits[p].first); //be careful with the baseline here

        weighted_sum += noise;

        if(is_log) weighted_sum = exp(weighted_sum);
        weighted_sum = max(estimated.limits[p].first, min(weighted_sum, estimated.limits[p].second));


        estimated[p] = weighted_sum;
    }

    return estimated;
}

void HyperOptimizer::update(const HyperParameters& candidate, double score) {
    hist.push_back(std::make_pair(candidate, score));
}
