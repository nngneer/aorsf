/**
 * Standalone test for survival forest to debug Python binding issue.
 */
#include <iostream>
#include <vector>
#include <memory>

#include "arma_config.h"
#include "globals.h"
#include "Data.h"
#include "ForestSurvival.h"
#include "Output.h"
#include "Interrupts.h"
#include "RMath.h"

using namespace aorsf;

int main() {
    std::cout << "Testing ForestSurvival standalone..." << std::endl;

    // Initialize output handler (use console output for standalone test)
    class ConsoleOutput : public OutputHandler {
    public:
        void print(const std::string& msg) override { std::cout << msg << std::flush; }
        void println(const std::string& msg) override { std::cout << msg << std::endl; }
    };
    OutputManager::set_handler(std::make_shared<ConsoleOutput>());

    // Initialize default stat distributions (important!)
    StatManager::set_distributions(std::make_shared<DefaultStatDistributions>());

    // Create data (similar to Python test)
    arma::arma_rng::set_seed(42);

    arma::uword n_samples = 100;  // Larger sample size
    arma::uword n_features = 5;

    arma::mat x = arma::randn<arma::mat>(n_samples, n_features);

    // Create survival outcome (time, status)
    arma::vec time = arma::randu<arma::vec>(n_samples) * 100 + 1;  // Time between 1-101
    arma::vec status(n_samples);
    for (arma::uword i = 0; i < n_samples; ++i) {
        status[i] = (arma::randu() > 0.3) ? 1.0 : 0.0;  // 70% events
    }

    // Sort by time (required for survival analysis)
    arma::uvec sort_idx = arma::sort_index(time);
    time = time(sort_idx);
    status = status(sort_idx);
    x = x.rows(sort_idx);

    arma::mat y(n_samples, 2);
    y.col(0) = time;
    y.col(1) = status;

    arma::vec w = arma::ones<arma::vec>(n_samples);

    std::cout << "Data created:" << std::endl;
    std::cout << "  x shape: " << x.n_rows << "x" << x.n_cols << std::endl;
    std::cout << "  y shape: " << y.n_rows << "x" << y.n_cols << std::endl;
    std::cout << "  First 5 times: " << y(arma::span(0, 4), 0).t();
    std::cout << "  First 5 status: " << y(arma::span(0, 4), 1).t();
    std::cout << "  N events: " << arma::sum(status) << std::endl;

    // Create Data object
    auto data = std::make_unique<Data>(x, y, w);
    std::cout << "Data object created" << std::endl;

    // Create forest - leaf_min_events and split_min_events MUST be >= 2
    // because compute_max_leaves() divides by (split_min_events - 1)
    arma::vec pred_horizon({50.0});
    auto forest = std::make_unique<ForestSurvival>(2.0, 2.0, pred_horizon);
    std::cout << "ForestSurvival created" << std::endl;

    // Create tree seeds
    std::vector<int> tree_seeds = {1};

    // Empty partial dependence containers
    std::vector<arma::mat> pd_x_vals;
    std::vector<arma::uvec> pd_x_cols;
    arma::vec pd_probs;

    // Initialize forest
    std::cout << "Calling forest->init()..." << std::endl;
    try {
        forest->init(
            std::move(data),
            tree_seeds,
            1,    // n_tree
            3,    // mtry
            true, // sample_with_replacement
            0.632,// sample_fraction
            true, // grow_mode
            VI_NONE,
            0.01, // vi_max_pvalue
            5.0,  // leaf_min_obs
            SPLIT_LOGRANK,
            10.0, // split_min_obs
            0.0,  // split_min_stat
            5,    // split_max_cuts
            3,    // split_max_retry
            LC_RANDOM_COEFS,
            1e-9, // lincomb_eps
            20,   // lincomb_iter_max
            true, // lincomb_scale
            0.5,  // lincomb_alpha
            0,    // lincomb_df_target
            0,    // lincomb_ties_method
            nullptr, // lincomb_callback
            PRED_RISK,
            false, // pred_mode
            true,  // pred_aggregate
            PD_NONE,
            pd_x_vals,
            pd_x_cols,
            pd_probs,
            false, // oobag_pred
            EVAL_NONE,
            1,    // oobag_eval_every
            nullptr, // oobag_callback
            1,    // n_thread
            4     // verbosity - very high for debugging
        );
        std::cout << "forest->init() completed" << std::endl;

        // Run forest
        std::cout << "Calling forest->run()..." << std::endl;
        forest->run(false);
        std::cout << "forest->run() completed" << std::endl;

        std::cout << "SUCCESS!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
