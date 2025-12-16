#include "catch2/catch.hpp"
#include <RcppArmadillo.h>
#include "../Data.h"

using namespace arma;
using namespace aorsf;

// ============================================================================
// Test Data Generators
// ============================================================================

// Create a simple Data object for testing
Data* make_test_data() {
    mat x = mat(5, 3);
    x << 1.0 << 2.0 << 3.0 << endr
      << 4.0 << 5.0 << 6.0 << endr
      << 7.0 << 8.0 << 9.0 << endr
      << 10.0 << 11.0 << 12.0 << endr
      << 13.0 << 14.0 << 15.0 << endr;

    mat y = mat(5, 2);
    y << 1.0 << 0.0 << endr
      << 2.0 << 1.0 << endr
      << 3.0 << 1.0 << endr
      << 4.0 << 0.0 << endr
      << 5.0 << 1.0 << endr;

    vec w = vec{1.0, 1.0, 2.0, 2.0, 1.0};

    return new Data(x, y, w);
}

// ============================================================================
// Construction and Initialization Tests
// ============================================================================

TEST_CASE("Data constructor initializes dimensions correctly", "[data][constructor]") {
    mat x = mat(10, 4);
    x.fill(1.0);
    mat y = mat(10, 2);
    y.fill(0.0);
    vec w = vec(10, fill::ones);

    Data data(x, y, w);

    REQUIRE(data.get_n_rows() == 10);
    REQUIRE(data.get_n_cols_x() == 4);
    REQUIRE(data.n_cols_y == 2);
    REQUIRE(data.has_weights == true);
}

TEST_CASE("Data constructor with empty weights", "[data][constructor]") {
    mat x = mat(5, 2);
    x.fill(1.0);
    mat y = mat(5, 1);
    y.fill(0.0);
    vec w; // Empty weights

    Data data(x, y, w);

    REQUIRE(data.get_n_rows() == 5);
    REQUIRE(data.has_weights == false);
}

// ============================================================================
// Getter Tests
// ============================================================================

TEST_CASE("Data getters return correct matrices", "[data][getters]") {
    auto data = make_test_data();

    mat& x = data->get_x();
    mat& y = data->get_y();
    vec& w = data->get_w();

    // Check dimensions
    REQUIRE(x.n_rows == 5);
    REQUIRE(x.n_cols == 3);
    REQUIRE(y.n_rows == 5);
    REQUIRE(y.n_cols == 2);
    REQUIRE(w.n_elem == 5);

    // Check some values
    REQUIRE(x(0, 0) == 1.0);
    REQUIRE(x(4, 2) == 15.0);
    REQUIRE(y(1, 1) == 1.0);
    REQUIRE(w(2) == 2.0);

    delete data;
}

// ============================================================================
// Submatrix Operation Tests
// ============================================================================

TEST_CASE("x_rows extracts correct rows", "[data][submatrix]") {
    auto data = make_test_data();

    uvec row_indices = {0, 2, 4};
    mat result = data->x_rows(row_indices);

    REQUIRE(result.n_rows == 3);
    REQUIRE(result.n_cols == 3);
    REQUIRE(result(0, 0) == 1.0);  // Row 0
    REQUIRE(result(1, 0) == 7.0);  // Row 2
    REQUIRE(result(2, 0) == 13.0); // Row 4

    delete data;
}

TEST_CASE("x_cols extracts correct columns", "[data][submatrix]") {
    auto data = make_test_data();

    uvec col_indices = {0, 2};
    mat result = data->x_cols(col_indices);

    REQUIRE(result.n_rows == 5);
    REQUIRE(result.n_cols == 2);
    REQUIRE(result(0, 0) == 1.0);  // Col 0
    REQUIRE(result(0, 1) == 3.0);  // Col 2
    REQUIRE(result(4, 1) == 15.0); // Row 4, Col 2

    delete data;
}

TEST_CASE("y_rows extracts correct rows", "[data][submatrix]") {
    auto data = make_test_data();

    uvec row_indices = {1, 3};
    mat result = data->y_rows(row_indices);

    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(result(0, 0) == 2.0); // Row 1, Col 0
    REQUIRE(result(0, 1) == 1.0); // Row 1, Col 1
    REQUIRE(result(1, 0) == 4.0); // Row 3, Col 0

    delete data;
}

TEST_CASE("x_submat extracts submatrix correctly", "[data][submatrix]") {
    auto data = make_test_data();

    uvec row_indices = {0, 2};
    uvec col_indices = {1, 2};
    mat result = data->x_submat(row_indices, col_indices);

    REQUIRE(result.n_rows == 2);
    REQUIRE(result.n_cols == 2);
    REQUIRE(result(0, 0) == 2.0);  // x(0, 1)
    REQUIRE(result(0, 1) == 3.0);  // x(0, 2)
    REQUIRE(result(1, 0) == 8.0);  // x(2, 1)
    REQUIRE(result(1, 1) == 9.0);  // x(2, 2)

    delete data;
}

TEST_CASE("w_subvec extracts weights correctly", "[data][submatrix]") {
    auto data = make_test_data();

    uvec indices = {0, 2, 4};
    vec result = data->w_subvec(indices);

    REQUIRE(result.n_elem == 3);
    REQUIRE(result(0) == 1.0);
    REQUIRE(result(1) == 2.0);
    REQUIRE(result(2) == 1.0);

    delete data;
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

TEST_CASE("x_submat_mult_beta computes linear combination", "[data][multiplication]") {
    auto data = make_test_data();

    uvec x_rows = {0, 1, 2};
    uvec x_cols = {0, 1, 2};
    vec beta = vec{1.0, 0.5, 0.25};

    vec result = data->x_submat_mult_beta(x_rows, x_cols, beta);

    // Expected: x[0,] * beta = 1.0*1.0 + 2.0*0.5 + 3.0*0.25 = 2.75
    //           x[1,] * beta = 4.0*1.0 + 5.0*0.5 + 6.0*0.25 = 8.0
    //           x[2,] * beta = 7.0*1.0 + 8.0*0.5 + 9.0*0.25 = 13.25
    REQUIRE(result.n_elem == 3);
    REQUIRE(result(0) == Approx(2.75));
    REQUIRE(result(1) == Approx(8.0));
    REQUIRE(result(2) == Approx(13.25));

    delete data;
}

TEST_CASE("x_submat_mult_beta with single row and column", "[data][multiplication]") {
    auto data = make_test_data();

    uvec x_rows = {2};
    uvec x_cols = {1};
    vec beta = vec{2.0};

    vec result = data->x_submat_mult_beta(x_rows, x_cols, beta);

    // Expected: x[2, 1] * 2.0 = 8.0 * 2.0 = 16.0
    REQUIRE(result.n_elem == 1);
    REQUIRE(result(0) == Approx(16.0));

    delete data;
}

TEST_CASE("x_submat_mult_beta with partial dependence substitution", "[data][multiplication]") {
    auto data = make_test_data();

    uvec x_rows = {0, 1};
    uvec x_cols = {0, 1, 2};
    vec beta = vec{1.0, 1.0, 1.0};
    vec pd_x_vals = vec{10.0};  // Replace column 1 with value 10.0
    uvec pd_x_cols = uvec{1};

    vec result = data->x_submat_mult_beta(x_rows, x_cols, beta, pd_x_vals, pd_x_cols);

    // Expected: For both rows, column 1 is replaced with 10.0
    // Row 0: 1.0*1.0 + 10.0*1.0 + 3.0*1.0 = 14.0
    // Row 1: 4.0*1.0 + 10.0*1.0 + 6.0*1.0 = 20.0
    REQUIRE(result.n_elem == 2);
    REQUIRE(result(0) == Approx(14.0));
    REQUIRE(result(1) == Approx(20.0));

    delete data;
}

TEST_CASE("x_submat_mult_beta with empty pd_x_cols", "[data][multiplication]") {
    auto data = make_test_data();

    uvec x_rows = {0};
    uvec x_cols = {0, 1};
    vec beta = vec{1.0, 2.0};
    vec pd_x_vals;
    uvec pd_x_cols;

    vec result = data->x_submat_mult_beta(x_rows, x_cols, beta, pd_x_vals, pd_x_cols);

    // Should fall back to normal multiplication
    // Expected: 1.0*1.0 + 2.0*2.0 = 5.0
    REQUIRE(result.n_elem == 1);
    REQUIRE(result(0) == Approx(5.0));

    delete data;
}

// ============================================================================
// Column Manipulation Tests
// ============================================================================

TEST_CASE("save_col and restore_col preserve column values", "[data][column_ops]") {
    auto data = make_test_data();

    // Save original column 1
    data->save_col(1);

    // Modify column 1
    data->fill_col(99.0, 1);

    mat& x = data->get_x();
    REQUIRE(x(0, 1) == 99.0);
    REQUIRE(x(4, 1) == 99.0);

    // Restore column 1
    data->restore_col(1);

    REQUIRE(x(0, 1) == 2.0);
    REQUIRE(x(1, 1) == 5.0);
    REQUIRE(x(4, 1) == 14.0);

    delete data;
}

TEST_CASE("fill_col sets all values in column", "[data][column_ops]") {
    auto data = make_test_data();

    data->fill_col(42.0, 2);

    mat& x = data->get_x();
    REQUIRE(x(0, 2) == 42.0);
    REQUIRE(x(1, 2) == 42.0);
    REQUIRE(x(2, 2) == 42.0);
    REQUIRE(x(3, 2) == 42.0);
    REQUIRE(x(4, 2) == 42.0);

    // Other columns unchanged
    REQUIRE(x(0, 0) == 1.0);
    REQUIRE(x(0, 1) == 2.0);

    delete data;
}

TEST_CASE("permute_col shuffles column values", "[data][column_ops]") {
    auto data = make_test_data();

    // Save original column
    data->save_col(0);
    mat& x = data->get_x();
    vec original = x.col(0);

    // Permute with fixed seed
    std::mt19937_64 rng(12345);
    data->permute_col(0, rng);

    vec permuted = x.col(0);

    // Values should be different from original (with high probability)
    // but contain the same elements
    vec sorted_original = sort(original);
    vec sorted_permuted = sort(permuted);

    REQUIRE(approx_equal(sorted_original, sorted_permuted, "absdiff", 1e-10));

    // Restore should bring back original order
    data->restore_col(0);
    REQUIRE(approx_equal(x.col(0), original, "absdiff", 1e-10));

    delete data;
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_CASE("Data with single row", "[data][edge_cases]") {
    mat x = mat(1, 2);
    x << 5.0 << 10.0 << endr;
    mat y = mat(1, 1);
    y << 1.0 << endr;
    vec w = vec{2.0};

    Data data(x, y, w);

    REQUIRE(data.get_n_rows() == 1);
    REQUIRE(data.get_n_cols_x() == 2);

    uvec row_idx = {0};
    uvec col_idx = {0, 1};
    mat result = data.x_submat(row_idx, col_idx);

    REQUIRE(result(0, 0) == 5.0);
    REQUIRE(result(0, 1) == 10.0);
}

TEST_CASE("Data with single column", "[data][edge_cases]") {
    mat x = mat(3, 1);
    x << 1.0 << endr << 2.0 << endr << 3.0 << endr;
    mat y = mat(3, 1);
    y.fill(0.0);
    vec w = vec(3, fill::ones);

    Data data(x, y, w);

    REQUIRE(data.get_n_cols_x() == 1);

    uvec col_idx = {0};
    mat result = data.x_cols(col_idx);

    REQUIRE(result.n_rows == 3);
    REQUIRE(result.n_cols == 1);
}
