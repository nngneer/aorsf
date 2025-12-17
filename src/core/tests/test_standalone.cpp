/**
 * @brief Standalone compilation test for aorsf core.
 *
 * This test verifies that the core library can be compiled and linked
 * without any R/Rcpp dependencies.
 */

#include <iostream>
#include <cassert>

// Include arma_config.h first to ensure proper Armadillo configuration
#include "arma_config.h"

#include "Data.h"
#include "Forest.h"
#include "ForestSurvival.h"
#include "ForestClassification.h"
#include "ForestRegression.h"
#include "globals.h"
#include "Exceptions.h"
#include "Output.h"
#include "Interrupts.h"
#include "RMath.h"

using namespace aorsf;

void test_data_creation() {
    arma::mat x(100, 5, arma::fill::randn);
    arma::mat y(100, 2);
    y.col(0) = arma::randu<arma::vec>(100) * 100;  // time
    y.col(1) = arma::randi<arma::vec>(100, arma::distr_param(0, 1));  // status
    arma::vec w(100, arma::fill::ones);

    Data data(x, y, w);

    assert(data.n_rows == 100);
    assert(data.n_cols_x == 5);
    assert(data.n_cols_y == 2);

    std::cout << "test_data_creation: PASSED" << std::endl;
}

void test_globals() {
    assert(DEFAULT_N_TREE == 500);
    assert(DEFAULT_LEAF_MIN_OBS == 5);
    assert(DEFAULT_SPLIT_MIN_OBS == 10);

    std::cout << "test_globals: PASSED" << std::endl;
}

void test_output_system() {
    // Test that output system works without R
    class TestOutput : public OutputHandler {
    public:
        std::string captured;
        void print(const std::string& msg) override { captured += msg; }
        void println(const std::string& msg) override { captured += msg + "\n"; }
    };

    auto handler = std::make_shared<TestOutput>();
    OutputManager::set_handler(handler);

    AORSF_OUT.println("test message");

    assert(handler->captured == "test message\n");

    // Reset to silent
    OutputManager::set_handler(nullptr);

    std::cout << "test_output_system: PASSED" << std::endl;
}

void test_interrupt_system() {
    // Test that interrupt system works without R
    class TestInterrupt : public InterruptHandler {
    public:
        bool should_interrupt = false;
        bool check() override { return should_interrupt; }
    };

    auto handler = std::make_shared<TestInterrupt>();
    InterruptManager::set_handler(handler);

    assert(InterruptManager::check() == false);

    handler->should_interrupt = true;
    assert(InterruptManager::check() == true);

    // Reset to no-op
    InterruptManager::set_handler(nullptr);

    std::cout << "test_interrupt_system: PASSED" << std::endl;
}

void test_exceptions() {
    try {
        throw invalid_argument_error("test error");
    } catch (const aorsf_error& e) {
        assert(std::string(e.what()) == "test error");
    }

    std::cout << "test_exceptions: PASSED" << std::endl;
}

void test_stat_distributions() {
    // Test that stat distributions work without R
    // Using default approximations
    double p = AORSF_STAT.pt(1.96, 100);
    assert(p > 0.97 && p < 0.98);  // Should be ~0.974

    double pchi = AORSF_STAT.pchisq(3.84, 1);
    assert(pchi > 0.94 && pchi < 0.96);  // Should be ~0.95

    // Test infinity constants
    assert(AORSF_POS_INF > 1e308);
    assert(AORSF_NEG_INF < -1e308);

    std::cout << "test_stat_distributions: PASSED" << std::endl;
}

void test_forest_creation() {
    // Test that forest classes can be instantiated
    arma::vec pred_horizon({30, 60, 90});
    ForestSurvival fs(1.0, 1.0, pred_horizon);
    ForestClassification fc(3);
    ForestRegression fr;

    std::cout << "test_forest_creation: PASSED" << std::endl;
}

int main() {
    std::cout << "Running standalone aorsf core tests..." << std::endl;
    std::cout << "======================================" << std::endl;

    test_globals();
    test_data_creation();
    test_output_system();
    test_interrupt_system();
    test_exceptions();
    test_stat_distributions();
    test_forest_creation();

    std::cout << "======================================" << std::endl;
    std::cout << "All standalone tests passed!" << std::endl;

    return 0;
}
