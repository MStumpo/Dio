#include "HyperOptimizer.h"
#include <cmath>
#include <algorithm>

using namespace std;

HyperOptimizer::HyperOptimizer(const HyperParameters& hyperparams, double init_step)
    : rng(random_device{}())
{
    params.push_back(hyperparams);
}

HyperParameters HyperOptimizer::propose(size_t N, double var) {
    if(scores.empty()) return params.back(); // fallback

    vector<size_t> indices(scores.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&](size_t i, size_t j){
        return scores[i] > scores[j];
    });

    HyperParameters estimated;
    double total_score = 0.0;
    for(size_t k = 0; k < N && k < scores.size(); ++k) total_score += scores[indices[k]];

    for(size_t p = 0; p < estimated.SIZE; ++p) {
        double weighted_sum = 0.0;
        if(estimated.log_scale[p]) weighted_sum = 1.0;

        for(size_t k = 0; k < N && k < scores.size(); ++k) {
            weighted_sum += params[indices[k]][p] * (scores[indices[k]] / total_score);
        }

        // add noise and clip to limits
        double noise = dist(rng) * (estimated.limits[p].second - estimated.limits[p].first) * N / total_score;
        weighted_sum = max(estimated.limits[p].first, min(weighted_sum + noise, estimated.limits[p].second));

        if(estimated.log_scale[p]) weighted_sum = log(weighted_sum);

        estimated[p] = weighted_sum;
    }

    return estimated;
}

void HyperOptimizer::update(const HyperParameters& candidate, double score) {
    params.push_back(candidate);
    scores.push_back(score);
}
