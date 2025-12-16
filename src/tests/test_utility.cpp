#include "catch2/catch.hpp"
#include <RcppArmadillo.h>
#include "../utility.h"

using namespace arma;
using namespace aorsf;

// ============================================================================
// Test Data Generators
// ============================================================================

// Generate simple survival data for testing
struct SurvivalData {
    mat y;  // n x 2: [time, status]
    vec w;  // weights
    uvec g; // groups (0 or 1)
};

SurvivalData make_simple_survival_data() {
    SurvivalData data;

    // 10 observations: 5 in group 0, 5 in group 1
    data.y = mat(10, 2);
    data.y.col(0) = vec{1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5}; // times
    data.y.col(1) = vec{1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0}; // status

    data.w = vec(10, fill::ones);
    data.g = uvec{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    return data;
}

// Generate classification data
struct ClassificationData {
    mat y;  // n x 1: class labels (not one-hot)
    vec w;  // weights
    uvec g; // groups
};

ClassificationData make_simple_classification_data() {
    ClassificationData data;

    // 10 observations, binary classification (class labels: 0 or 1)
    data.y = mat(10, 1);
    data.y.col(0) = vec{0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

    data.w = vec(10, fill::ones);
    data.g = uvec{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    return data;
}

// ============================================================================
// Sanity Tests (from Phase 1)
// ============================================================================

TEST_CASE("Armadillo basic functionality", "[sanity]") {
    vec v = {1.0, 2.0, 3.0};
    REQUIRE(v.n_elem == 3);
    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 2.0);
    REQUIRE(v(2) == 3.0);
}

TEST_CASE("Basic arithmetic", "[sanity]") {
    REQUIRE(1 + 1 == 2);
    REQUIRE(2.0 * 3.0 == 6.0);
}

// ============================================================================
// Logrank Tests
// ============================================================================

TEST_CASE("compute_logrank with identical groups returns NaN", "[utility][logrank]") {
    auto data = make_simple_survival_data();

    // All in same group (variance will be 0, leading to NaN)
    data.g.fill(0);

    double logrank = compute_logrank(data.y, data.w, data.g);

    REQUIRE(std::isnan(logrank));
}

TEST_CASE("compute_logrank with no events returns NaN", "[utility][logrank]") {
    auto data = make_simple_survival_data();

    // All censored (no events, variance will be 0, leading to NaN)
    data.y.col(1).fill(0.0);

    double logrank = compute_logrank(data.y, data.w, data.g);

    REQUIRE(std::isnan(logrank));
}

TEST_CASE("compute_logrank returns positive value for different groups", "[utility][logrank]") {
    auto data = make_simple_survival_data();

    double logrank = compute_logrank(data.y, data.w, data.g);

    REQUIRE(logrank > 0.0);
    REQUIRE(std::isfinite(logrank));
}

// ============================================================================
// Gini Impurity Tests
// ============================================================================

TEST_CASE("compute_gini with pure node returns zero", "[utility][gini]") {
    auto data = make_simple_classification_data();

    // All in same class (all class 1)
    data.y.fill(1.0);

    double gini = compute_gini(data.y, data.w, data.g);

    REQUIRE(gini == Approx(0.0).margin(1e-10));
}

TEST_CASE("compute_gini with 50-50 split returns positive", "[utility][gini]") {
    ClassificationData data;

    // 4 observations, 50-50 split in each group
    data.y = mat(4, 1);
    data.y.col(0) = vec{0, 1, 0, 1};  // 50% class 0, 50% class 1

    data.w = vec(4, fill::ones);
    data.g = uvec{0, 0, 1, 1}; // Split into two groups

    double gini = compute_gini(data.y, data.w, data.g);

    REQUIRE(gini > 0.0);
    REQUIRE(std::isfinite(gini));
}

// ============================================================================
// Variance Reduction Tests
// ============================================================================

TEST_CASE("compute_var_reduction with identical values returns zero", "[utility][variance]") {
    vec y = vec(10, fill::value(5.0)); // All same value
    vec w = vec(10, fill::ones);
    uvec g = uvec{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    double var_red = compute_var_reduction(y, w, g);

    REQUIRE(var_red == Approx(0.0).margin(1e-10));
}

TEST_CASE("compute_var_reduction with perfect split returns positive", "[utility][variance]") {
    vec y = vec{1, 1, 1, 1, 1, 10, 10, 10, 10, 10}; // Clear split
    vec w = vec(10, fill::ones);
    uvec g = uvec{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}; // Groups match split

    double var_red = compute_var_reduction(y, w, g);

    REQUIRE(var_red > 0.0);
    REQUIRE(std::isfinite(var_red));
}

// ============================================================================
// C-statistic Tests
// ============================================================================

TEST_CASE("compute_cstat_surv with perfect predictions", "[utility][cstat]") {
    mat y = mat(5, 2);
    y.col(0) = vec{1, 2, 3, 4, 5}; // times
    y.col(1) = vec{1, 1, 1, 1, 1}; // all events

    vec w = vec(5, fill::ones);
    vec p = vec{5, 4, 3, 2, 1}; // Perfect risk order

    double cstat = compute_cstat_surv(y, w, p, true);

    REQUIRE(cstat == Approx(1.0).epsilon(0.01));
}

TEST_CASE("compute_cstat_surv with random predictions", "[utility][cstat]") {
    mat y = mat(5, 2);
    y.col(0) = vec{1, 2, 3, 4, 5};
    y.col(1) = vec{1, 1, 1, 1, 1};

    vec w = vec(5, fill::ones);
    vec p = vec{3, 1, 4, 2, 5}; // Random order

    double cstat = compute_cstat_surv(y, w, p, true);

    REQUIRE(cstat >= 0.0);
    REQUIRE(cstat <= 1.0);
}

// ============================================================================
// Scale X Tests
// ============================================================================

TEST_CASE("scale_x centers and scales correctly", "[utility][scale]") {
    mat x = mat(5, 2);
    x.col(0) = vec{1, 2, 3, 4, 5};
    x.col(1) = vec{10, 20, 30, 40, 50};

    vec w = vec(5, fill::ones);

    // scale_x modifies x in-place and returns transformation matrix (n_cols x 2)
    mat x_transforms = scale_x(x, w);

    // Check transformation matrix dimensions
    REQUIRE(x_transforms.n_rows == 2);  // 2 columns in x
    REQUIRE(x_transforms.n_cols == 2);  // [means, scales]

    // Check x is now centered (mean ≈ 0)
    REQUIRE(mean(x.col(0)) == Approx(0.0).margin(1e-10));
    REQUIRE(mean(x.col(1)) == Approx(0.0).margin(1e-10));

    // Check x is now scaled (sd ≈ 1)
    REQUIRE(stddev(x.col(0)) == Approx(1.0).epsilon(0.01));
    REQUIRE(stddev(x.col(1)) == Approx(1.0).epsilon(0.01));
}
