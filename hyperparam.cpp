    #include <vector>
    #include <random>
    #include <cmath>
    #include <algorithm>

    struct ParamSpec {
        double min_val;
        double max_val;
        bool log_scale; // true = log search space
        bool is_int;    // true = integer parameter
    };

    struct HyperOptimizer {
        std::vector<ParamSpec> specs;
        std::vector<double> params;      // current best parameters (actual scale)
        std::vector<double> step_sizes;  // adaptive step sizes (in transformed space)
        double best_score;
        std::mt19937 rng;
        std::normal_distribution<double> dist{0.0, 1.0};

        HyperOptimizer(const std::vector<ParamSpec>& param_specs, double init_step = 3)
            : specs(param_specs),
              params(param_specs.size()),
              step_sizes(param_specs.size(), init_step),
              best_score(0),
              rng(std::random_device{}())
        {
            // Initialize parameters to midpoints of ranges
            for (size_t i = 0; i < specs.size(); ++i) {
                if (specs[i].log_scale)
                    params[i] = std::exp(std::log(specs[i].min_val) +
                                         0.5 * std::log(specs[i].max_val / specs[i].min_val));
                else
                    params[i] = (specs[i].min_val + specs[i].max_val) / 2.0;
                if (specs[i].is_int)
                    params[i] = std::round(params[i]);
            }
        }

        std::vector<double> propose() {
            std::vector<double> candidate = params;
            for (size_t i = 0; i < specs.size(); ++i) {
                double value = specs[i].log_scale ? std::log(candidate[i]) : candidate[i];
                //double value = specs[i];
        	value += dist(rng) * step_sizes[i]*(specs[i].log_scale ? std::exp(specs[i].max_val - specs[i].min_val) : specs[i].max_val -  specs[i].min_val);

                candidate[i] = std::clamp((specs[i].log_scale ? std::exp(candidate[i] + value) : candidate[i] + value), specs[i].min_val, specs[i].max_val);
                if (specs[i].is_int) candidate[i] = std::round(candidate[i]);

            }
            return candidate;
        }

        void update(const std::vector<double>& candidate, double score) {
            score = std::clamp(score, -1.0, 1.0);
            if (score > best_score) {
                params = candidate;
                best_score = score;
            }
            for (int i = 0; i < step_sizes.size(); i++){
                step_sizes[i] = min(max(step_sizes[i]*(1.0- score + best_score),10e-6), 3.0);
    	    }
	    best_score *= 0.95;
        }
    };
