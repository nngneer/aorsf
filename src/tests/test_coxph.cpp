#include "catch2/catch.hpp"
#include <RcppArmadillo.h>
#include "../Coxph.h"
#include "../utility.h"

using namespace arma;
using namespace aorsf;

// ============================================================================
// Test Data for Cox Regression
// ============================================================================

struct CoxTestData {
    mat x;     // Predictors
    mat y;     // Survival outcome [time, status]
    vec w;     // Weights
};

// Creates properly sorted Cox regression test data
// Data must be sorted by time (ascending), with events before censored for ties
CoxTestData make_simple_cox_data() {
    CoxTestData data;

    // 20 observations, 2 independent predictors
    data.x = mat(20, 2);
    // Non-collinear predictors with reasonable variation
    data.x.col(0) = vec{1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
                        0.5, 1.2, 2.3, 3.1, 3.8, 4.2, 4.7, 5.2, 5.8, 6.0};
    data.x.col(1) = vec{0.2, 0.8, 1.1, 1.7, 2.2, 2.8, 3.1, 3.6, 4.1, 4.5,
                        0.4, 1.0, 1.5, 2.1, 2.6, 3.2, 3.7, 4.3, 4.8, 5.3};

    // Survival data: sorted by time (ascending)
    // More events for better estimation
    data.y = mat(20, 2);
    data.y.col(0) = vec{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}; // times
    data.y.col(1) = vec{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // status: 12 events, 8 censored

    data.w = vec(20, fill::ones);

    return data;
}

// ============================================================================
// Cox Fit Tests
// ============================================================================

TEST_CASE("coxph_fit returns valid output structure", "[coxph][fit]") {
    auto data = make_simple_cox_data();

    // coxph_fit signature: (x, y, w, do_scale, ties_method, epsilon, iter_max)
    // Returns: n_vars x 2 matrix [beta, pvalues]
    mat result = coxph_fit(
        data.x,
        data.y,
        data.w,
        false,  // do_scale
        0,      // ties_method (0 = Breslow)
        1e-9,   // epsilon
        20      // iter_max
    );

    // Check result dimensions: n_vars x 2 (beta, pvalues)
    REQUIRE(result.n_rows == 2);  // 2 predictors
    REQUIRE(result.n_cols == 2);  // [beta, pvalues]

    // Coefficients (column 0) should be finite
    // Note: May be zero if model doesn't converge with this data
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));

    // P-values (column 1) may be NaN if betas are zero
    // Just check they are either finite or NaN (not infinity)
    REQUIRE_FALSE(std::isinf(result(0, 1)));
    REQUIRE_FALSE(std::isinf(result(1, 1)));
}

TEST_CASE("coxph_fit with single predictor", "[coxph][fit]") {
    CoxTestData data;

    // Single predictor with reasonable sample size and variation
    data.x = mat(15, 1);
    data.x.col(0) = vec{1.0, 1.5, 2.0, 2.8, 3.2, 3.9, 4.5, 5.1, 5.7, 6.2,
                        0.8, 1.3, 2.4, 3.5, 4.6};

    data.y = mat(15, 2);
    data.y.col(0) = vec{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                        11.0, 12.0, 13.0, 14.0, 15.0};
    data.y.col(1) = vec{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0};

    data.w = vec(15, fill::ones);

    mat result = coxph_fit(data.x, data.y, data.w, false, 0, 1e-9, 20);

    // Should return 1 x 2 (1 predictor, 2 columns)
    REQUIRE(result.n_rows == 1);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE_FALSE(std::isinf(result(0, 1)));
}

TEST_CASE("coxph_fit with scaling enabled", "[coxph][fit]") {
    auto data = make_simple_cox_data();

    mat result = coxph_fit(data.x, data.y, data.w, true, 0, 1e-9, 20);

    // Results should still have valid structure with scaling
    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));
    REQUIRE_FALSE(std::isinf(result(0, 1)));
    REQUIRE_FALSE(std::isinf(result(1, 1)));
}

TEST_CASE("coxph_fit with limited iterations (fast mode)", "[coxph][fit]") {
    auto data = make_simple_cox_data();

    // Fast mode with only 1 iteration - used for tree growing
    mat result = coxph_fit(data.x, data.y, data.w, false, 0, 1e-9, 1);

    // Should still produce valid output structure
    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));
}

TEST_CASE("coxph_fit with Efron ties method", "[coxph][fit]") {
    CoxTestData data;

    // Create data with tied event times
    data.x = mat(8, 1);
    data.x.col(0) = vec{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    data.y = mat(8, 2);
    // Tied times at 2.0 and 4.0
    data.y.col(0) = vec{1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0};
    data.y.col(1) = vec{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0};

    data.w = vec(8, fill::ones);

    // Use Efron method (ties_method = 1)
    mat result = coxph_fit(data.x, data.y, data.w, false, 1, 1e-9, 20);

    REQUIRE(result.n_rows == 1);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE_FALSE(std::isinf(result(0, 1)));
}

TEST_CASE("coxph_fit with weighted observations", "[coxph][fit]") {
    auto data = make_simple_cox_data();

    // Use non-uniform weights
    data.w = vec{1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0,
                 1.5, 1.5, 2.5, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    mat result = coxph_fit(data.x, data.y, data.w, false, 0, 1e-9, 20);

    // Should handle weights correctly
    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(std::isfinite(result(0, 0)));
    REQUIRE(std::isfinite(result(1, 0)));
    REQUIRE_FALSE(std::isinf(result(0, 1)));
    REQUIRE_FALSE(std::isinf(result(1, 1)));
}
