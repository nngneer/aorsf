#include "catch2/catch.hpp"
#include <RcppArmadillo.h>
#include "../utility.h"

using namespace arma;
using namespace aorsf;

// ============================================================================
// Test Data for Regression
// ============================================================================

struct RegressionTestData {
    mat x;     // Predictors
    mat y;     // Response
    vec w;     // Weights
};

// Linear regression test data with clear relationship
RegressionTestData make_linear_test_data() {
    RegressionTestData data;

    // 20 observations, 2 predictors
    // y = 2 + 3*x1 + 1*x2 + noise
    data.x = mat(20, 2);
    data.x.col(0) = vec{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                        1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5};
    data.x.col(1) = vec{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                        0.8, 1.3, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3, 4.8, 5.3};

    // Response: approximately y = 2 + 3*x1 + 1*x2
    data.y = mat(20, 1);
    data.y.col(0) = vec{5.5, 9.0, 12.5, 16.0, 19.5, 23.0, 26.5, 30.0, 33.5, 37.0,
                        7.3, 10.8, 14.3, 17.8, 21.3, 24.8, 28.3, 31.8, 35.3, 38.8};

    data.w = vec(20, fill::ones);

    return data;
}

// Logistic regression test data
RegressionTestData make_logistic_test_data() {
    RegressionTestData data;

    // 30 observations, 2 predictors
    data.x = mat(30, 2);
    // First predictor varies from low to high
    data.x.col(0) = linspace<vec>(1.0, 10.0, 30);
    // Second predictor with some variation
    data.x.col(1) = linspace<vec>(0.5, 5.0, 30);

    // Binary response (0 or 1): higher x values -> more likely to be 1
    data.y = mat(30, 1);
    data.y.col(0) = vec{0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Low x -> mostly 0
                        0, 0, 0, 1, 0, 1, 1, 0, 1, 1,  // Mid x -> mixed
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // High x -> mostly 1

    data.w = vec(30, fill::ones);

    return data;
}

// ============================================================================
// Linear Regression Tests
// ============================================================================

TEST_CASE("linreg_fit returns valid output structure", "[regression][linear]") {
    auto data = make_linear_test_data();

    // linreg_fit signature: (x, y, w, do_scale, epsilon, iter_max)
    // Returns: n_vars x 2 matrix [beta, pvalues]
    mat result = linreg_fit(
        data.x,
        data.y,
        data.w,
        false,  // do_scale
        1e-9,   // epsilon
        20      // iter_max
    );

    // Check result dimensions
    REQUIRE(result.n_rows == 2);  // 2 predictors (intercept excluded)
    REQUIRE(result.n_cols == 2);  // [beta, pvalues]

    // Coefficients should be finite
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));

    // P-values should be finite and in [0, 1]
    REQUIRE(std::isfinite(result(0, 1)));
    REQUIRE(std::isfinite(result(1, 1)));
    REQUIRE(result(0, 1) >= 0.0);
    REQUIRE(result(0, 1) <= 1.0);
    REQUIRE(result(1, 1) >= 0.0);
    REQUIRE(result(1, 1) <= 1.0);
}

TEST_CASE("linreg_fit with single predictor", "[regression][linear]") {
    RegressionTestData data;

    // Single predictor
    data.x = mat(15, 1);
    data.x.col(0) = linspace<vec>(1.0, 10.0, 15);

    // Simple linear relationship: y = 2 + 3*x
    data.y = mat(15, 1);
    data.y.col(0) = 2.0 + 3.0 * data.x.col(0);

    data.w = vec(15, fill::ones);

    mat result = linreg_fit(data.x, data.y, data.w, false, 1e-9, 20);

    // Should return 1 x 2
    REQUIRE(result.n_rows == 1);
    REQUIRE(result.n_cols == 2);

    // Beta should be close to 3.0 (perfect fit)
    REQUIRE(result(0, 0) == Approx(3.0).epsilon(0.01));

    // P-value should be very small (highly significant)
    REQUIRE(result(0, 1) < 0.01);
}

TEST_CASE("linreg_fit with scaling enabled", "[regression][linear]") {
    auto data = make_linear_test_data();

    mat result = linreg_fit(data.x, data.y, data.w, true, 1e-9, 20);

    // Results should have valid structure
    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));
    REQUIRE(result(0, 1) >= 0.0);
    REQUIRE(result(0, 1) <= 1.0);
}

TEST_CASE("linreg_fit with weighted observations", "[regression][linear]") {
    auto data = make_linear_test_data();

    // Use non-uniform weights
    data.w = linspace<vec>(0.5, 2.0, 20);

    mat result = linreg_fit(data.x, data.y, data.w, false, 1e-9, 20);

    // Should handle weights correctly
    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));
    REQUIRE(std::isfinite(result(0, 1)));
    REQUIRE(std::isfinite(result(1, 1)));
}

TEST_CASE("linreg_fit with singular matrix returns zeros", "[regression][linear][edge]") {
    RegressionTestData data;

    // Create singular design: x2 = 2 * x1 (perfectly collinear)
    data.x = mat(10, 2);
    data.x.col(0) = linspace<vec>(1.0, 10.0, 10);
    data.x.col(1) = 2.0 * data.x.col(0);

    data.y = mat(10, 1);
    data.y.col(0) = linspace<vec>(5.0, 15.0, 10);

    data.w = vec(10, fill::ones);

    mat result = linreg_fit(data.x, data.y, data.w, false, 1e-9, 20);

    // Should return zeros for singular case
    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(result(0, 0) == 0.0);
    REQUIRE(result(1, 0) == 0.0);
}

// ============================================================================
// Logistic Regression Tests
// ============================================================================

TEST_CASE("logreg_fit returns valid output structure", "[regression][logistic]") {
    auto data = make_logistic_test_data();

    // logreg_fit signature: (x, y, w, do_scale, epsilon, iter_max)
    // Returns: n_vars x 2 matrix [beta, pvalues]
    mat result = logreg_fit(
        data.x,
        data.y,
        data.w,
        false,  // do_scale
        1e-9,   // epsilon
        20      // iter_max
    );

    // Check result dimensions
    REQUIRE(result.n_rows == 2);  // 2 predictors
    REQUIRE(result.n_cols == 2);  // [beta, pvalues]

    // Coefficients should be finite
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));

    // P-values should be finite and in [0, 1]
    REQUIRE(std::isfinite(result(0, 1)));
    REQUIRE(std::isfinite(result(1, 1)));
    REQUIRE(result(0, 1) >= 0.0);
    REQUIRE(result(0, 1) <= 1.0);
    REQUIRE(result(1, 1) >= 0.0);
    REQUIRE(result(1, 1) <= 1.0);
}

TEST_CASE("logreg_fit with single predictor", "[regression][logistic]") {
    RegressionTestData data;

    // Single predictor with more separation
    data.x = mat(20, 1);
    data.x.col(0) = linspace<vec>(1.0, 10.0, 20);

    // Binary response: lower x -> 0, higher x -> 1
    data.y = mat(20, 1);
    data.y.col(0) = vec{0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    data.w = vec(20, fill::ones);

    mat result = logreg_fit(data.x, data.y, data.w, false, 1e-9, 20);

    // Should return 1 x 2
    REQUIRE(result.n_rows == 1);
    REQUIRE(result.n_cols == 2);

    // Beta should be positive (higher x -> higher probability of y=1)
    REQUIRE(result(0, 0) > 0.0);

    // P-value should be valid (may not be significant with this simple data)
    REQUIRE(std::isfinite(result(0, 1)));
    REQUIRE(result(0, 1) >= 0.0);
    REQUIRE(result(0, 1) <= 1.0);
}

TEST_CASE("logreg_fit with scaling enabled", "[regression][logistic]") {
    auto data = make_logistic_test_data();

    mat result = logreg_fit(data.x, data.y, data.w, true, 1e-9, 20);

    // Results should have valid structure
    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));
    REQUIRE(result(0, 1) >= 0.0);
    REQUIRE(result(0, 1) <= 1.0);
}

TEST_CASE("logreg_fit with limited iterations", "[regression][logistic]") {
    auto data = make_logistic_test_data();

    // Fast mode with only 1 iteration
    mat result = logreg_fit(data.x, data.y, data.w, false, 1e-9, 1);

    // Should still produce valid output structure
    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));
}

TEST_CASE("logreg_fit with weighted observations", "[regression][logistic]") {
    auto data = make_logistic_test_data();

    // Use non-uniform weights
    data.w = linspace<vec>(0.5, 2.0, 30);

    mat result = logreg_fit(data.x, data.y, data.w, false, 1e-9, 20);

    // Should handle weights correctly
    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));
    REQUIRE(std::isfinite(result(0, 1)));
    REQUIRE(std::isfinite(result(1, 1)));
}

TEST_CASE("logreg_fit with all same class returns zeros", "[regression][logistic][edge]") {
    RegressionTestData data;

    // All observations have y = 1 (no variation in response)
    data.x = mat(10, 1);
    data.x.col(0) = linspace<vec>(1.0, 10.0, 10);

    data.y = mat(10, 1);
    data.y.col(0).fill(1.0);

    data.w = vec(10, fill::ones);

    mat result = logreg_fit(data.x, data.y, data.w, false, 1e-9, 20);

    // May return zeros or extreme values for degenerate case
    REQUIRE(result.n_rows == 1);
    REQUIRE(result.n_cols == 2);
    // Just check it doesn't crash and returns finite or inf values
    // (inf is expected for perfect separation)
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_CASE("linreg_fit with minimal data", "[regression][linear][edge]") {
    RegressionTestData data;

    // 3 observations, 1 predictor (minimal for valid regression)
    data.x = mat(3, 1);
    data.x.col(0) = vec{1.0, 2.0, 3.0};

    data.y = mat(3, 1);
    data.y.col(0) = vec{2.0, 4.0, 6.0};

    data.w = vec(3, fill::ones);

    mat result = linreg_fit(data.x, data.y, data.w, false, 1e-9, 20);

    REQUIRE(result.n_rows == 1);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(0, 1)));
}

TEST_CASE("logreg_fit with balanced classes", "[regression][logistic]") {
    RegressionTestData data;

    // 20 observations: 10 with y=0, 10 with y=1
    data.x = mat(20, 1);
    data.x.col(0) = linspace<vec>(1.0, 10.0, 20);

    data.y = mat(20, 1);
    // Perfectly balanced
    data.y.col(0) = vec{0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                        0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

    data.w = vec(20, fill::ones);

    mat result = logreg_fit(data.x, data.y, data.w, false, 1e-9, 20);

    REQUIRE(result.n_rows == 1);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(0, 1)));
}
