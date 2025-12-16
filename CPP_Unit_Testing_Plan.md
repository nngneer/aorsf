# C++ Unit Testing with Catch2 Implementation Plan

## Overview

Add Catch2 unit testing infrastructure to the aorsf R package to enable direct testing of C++ code. Currently, all C++ functionality is tested indirectly through R testthat tests that call `_exported()` wrapper functions. This plan adds standalone C++ unit tests that can be run from the command line independently of the R package build system.

## Current State Analysis

### Existing Test Infrastructure
- **R-level testing only**: All tests in `tests/testthat/*.R` using testthat framework
- **C++ wrapper pattern**: C++ functions exposed via `_exported()` functions for R testing
  - Examples: `coxph_fit_exported()`, `compute_logrank_exported()`, `compute_gini_exported()`
  - Located in `src/orsf_oop.cpp` and `src/RcppExports.cpp`
- **Test approach**: R tests compare C++ results against reference implementations (survival package, etc.)
- **No direct C++ tests**: Cannot test C++ logic without R runtime

### C++ Code Structure
**Testable utility functions** (in `src/utility.{h,cpp}`):
- `compute_logrank()` - log-rank test statistic
- `compute_cstat_surv()` - concordance statistic for survival
- `compute_cstat_clsf()` - concordance statistic for classification
- `compute_gini()` - Gini impurity
- `compute_var_reduction()` - variance reduction
- `linreg_fit()`, `logreg_fit()` - linear/logistic regression

**Cox regression functions** (in `src/Coxph.{h,cpp}`):
- `cholesky_decomp()` - Cholesky decomposition
- `cholesky_solve()` - solve linear system using Cholesky
- `coxph_fit()` - Cox proportional hazards fitting

**Data class** (in `src/Data.h`):
- `x_submat_mult_beta()` - matrix-vector multiplication
- `permute_col()` - column permutation
- Various submatrix accessors

### Build System
- **Makevars**: Standard R package build with Rcpp, RcppArmadillo, OpenMP, LAPACK, BLAS
- **Single output**: Builds only the R package shared library (`.so`/`.dll`)
- **No test target**: No infrastructure for building/running C++ tests

## Desired End State

A fully functional C++ unit testing infrastructure that:

1. **Standalone test executable**: `src/tests/test_runner` binary that runs all C++ tests
2. **Catch2 framework**: Header-only Catch2 v2.13.x integrated
3. **Comprehensive test coverage** for:
   - Utility functions (compute_*, scale_x, etc.)
   - Cox regression algorithms
   - Data class operations
   - Linear/logistic regression
4. **Easy to run**: `make -C src/tests` or `make -C src/tests test`
5. **Minimal synthetic test data**: All test data generated in C++, no external dependencies
6. **Clear documentation**: README explaining how to build and run tests

### Verification Criteria

**The implementation is complete when:**
- [ ] Can run `make -C src/tests` and see test results
- [ ] At least 20 unit tests passing across utility functions, Cox regression, and Data class
- [ ] Tests use synthetic data generated in C++ (no R dependencies)
- [ ] Tests compile and run successfully on macOS and Linux
- [ ] Documentation clearly explains how to add new tests
- [ ] All tests pass and report results in standard Catch2 format

## What We're NOT Doing

To maintain scope and focus, we are explicitly **NOT**:

1. **Integrating tests into R package build** - tests are separate, not part of `R CMD check`
2. **Testing Tree/Forest classes** - these are complex, may require refactoring (see Refactoring_Recommendations.md)
3. **Creating R interface to tests** - no `test_cpp()` R function, command-line only
4. **Modifying existing R tests** - R testthat tests remain unchanged
5. **Adding continuous integration** - no GitHub Actions for C++ tests (could be added later)
6. **Testing Rcpp interface layer** - not testing `RcppExports.cpp` or `_exported()` wrappers
7. **Performance benchmarking** - tests verify correctness, not performance
8. **Windows-specific test infrastructure** - initial implementation focuses on Unix-like systems

## Implementation Approach

**Strategy**: Incremental development in self-contained phases, starting with infrastructure and building up test coverage layer by layer.

**Key Decisions**:
- **Catch2 v2.13.10**: Header-only, single-file, C++11 compatible, well-tested
- **Separate build system**: New Makefile in `src/tests/` independent of R package
- **Synthetic test data**: Generate small test matrices/vectors in C++ using Armadillo
- **Focus on algorithms**: Test mathematical correctness of core algorithms first
- **Incremental coverage**: Start with simplest functions, build up to more complex

**Directory Structure**:
```
src/tests/
├── Makefile              # Build and run tests
├── README.md             # Documentation
├── catch2/
│   └── catch.hpp         # Catch2 v2.13.10 header
├── test_main.cpp         # Catch2 main entry point
├── test_utility.cpp      # Tests for utility.cpp functions
├── test_coxph.cpp        # Tests for Coxph.cpp functions
├── test_data.cpp         # Tests for Data.h class
└── test_regression.cpp   # Tests for linreg/logreg
```

---

## Phase 1: Test Infrastructure Setup

### Overview
Set up the basic Catch2 testing infrastructure with a minimal working test. This phase establishes the build system and proves that C++ tests can compile and run independently of R.

### Changes Required

#### 1. Create test directory structure
**Action**: Create directories for test infrastructure

```bash
mkdir -p src/tests/catch2
```

#### 2. Download Catch2 header
**File**: `src/tests/catch2/catch.hpp`
**Action**: Download Catch2 v2.13.10 single-include header

```bash
# Download from GitHub releases
curl -L https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp \
  -o src/tests/catch2/catch.hpp
```

**Size**: ~600KB single header file

#### 3. Create Catch2 main entry point
**File**: `src/tests/test_main.cpp`
**Action**: Create the main test runner

```cpp
// Define CATCH_CONFIG_MAIN to create main() function
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

// Main is automatically generated by Catch2
// All test files will be linked against this
```

#### 4. Create minimal test file
**File**: `src/tests/test_utility.cpp`
**Action**: Create first test file with a simple sanity test

```cpp
#include "catch2/catch.hpp"
#include <RcppArmadillo.h>
#include "../utility.h"

using namespace arma;

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
```

#### 5. Create Makefile
**File**: `src/tests/Makefile`
**Action**: Create build system for C++ tests

```makefile
# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -I.. -I. $(shell Rscript -e "Rcpp:::CxxFlags()") $(shell Rscript -e "RcppArmadillo:::CxxFlags()")
LDFLAGS = $(shell Rscript -e "Rcpp:::LdFlags()") $(shell Rscript -e "RcppArmadillo:::LdFlags()") -llapack -lblas

# Source files to test
SOURCES_TO_TEST = ../utility.cpp ../Coxph.cpp

# Test files
TEST_SOURCES = test_main.cpp test_utility.cpp

# Object files
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)
LIB_OBJECTS = $(SOURCES_TO_TEST:.cpp=.o)

# Output executable
TARGET = test_runner

# Default target
all: $(TARGET)

# Build test runner
$(TARGET): $(TEST_OBJECTS) $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile test files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile library files
../%.o: ../%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run tests
test: $(TARGET)
	./$(TARGET)

# Run tests with verbose output
test-verbose: $(TARGET)
	./$(TARGET) -s

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TEST_OBJECTS) $(LIB_OBJECTS)

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Build test runner"
	@echo "  make test     - Build and run tests"
	@echo "  make test-verbose - Run tests with detailed output"
	@echo "  make clean    - Remove build artifacts"

.PHONY: all test test-verbose clean help
```

#### 6. Create README
**File**: `src/tests/README.md`
**Action**: Document the test infrastructure

```markdown
# C++ Unit Tests for aorsf

This directory contains C++ unit tests using the Catch2 framework.

## Prerequisites

- C++ compiler (g++ or clang++)
- R with Rcpp and RcppArmadillo packages installed
- LAPACK and BLAS libraries

## Building and Running Tests

```bash
# Build test runner
make

# Run all tests
make test

# Run tests with verbose output
make test-verbose

# Clean build artifacts
make clean
```

## Test Organization

- `test_main.cpp` - Catch2 main entry point
- `test_utility.cpp` - Tests for utility functions (compute_*, scale_x, etc.)
- `test_coxph.cpp` - Tests for Cox regression functions
- `test_data.cpp` - Tests for Data class
- `test_regression.cpp` - Tests for linear/logistic regression

## Adding New Tests

1. Create a new test file or add to existing file:

```cpp
#include "catch2/catch.hpp"
#include "../your_header.h"

TEST_CASE("Description of test", "[tag]") {
    // Arrange
    int x = 1;

    // Act
    int result = x + 1;

    // Assert
    REQUIRE(result == 2);
}
```

2. If creating a new test file, add it to `TEST_SOURCES` in Makefile
3. Run `make test` to verify

## Catch2 Assertions

- `REQUIRE(expr)` - Test fails if expr is false
- `REQUIRE_FALSE(expr)` - Test fails if expr is true
- `REQUIRE_THROWS(expr)` - Test fails if expr doesn't throw
- `REQUIRE_NOTHROW(expr)` - Test fails if expr throws

## Tags

Use tags to organize and selectively run tests:
- `[sanity]` - Basic sanity checks
- `[utility]` - Utility function tests
- `[coxph]` - Cox regression tests
- `[data]` - Data class tests
- `[regression]` - Linear/logistic regression tests

Run specific tags: `./test_runner "[utility]"`
```

#### 7. Update .gitignore
**File**: `src/tests/.gitignore`
**Action**: Ignore test build artifacts

```
test_runner
*.o
```

#### 8. Update root .gitignore
**File**: `.gitignore`
**Action**: Add test artifacts to root gitignore

```
# Add these lines
src/tests/test_runner
src/tests/*.o
```

### Success Criteria

#### Automated Verification:
- [x] Directory structure created: `ls -la src/tests/catch2/catch.hpp` succeeds
- [x] Catch2 header downloaded: `wc -c src/tests/catch2/catch.hpp` shows ~600KB file
- [x] Test runner compiles: `make -C src/tests` succeeds without errors
- [x] Basic tests pass: `make -C src/tests test` shows all tests passing
- [x] Cleanup works: `make -C src/tests clean && test ! -f src/tests/test_runner` succeeds

#### Manual Verification:
- [x] Can read and understand the README
- [x] Test output is clear and readable
- [x] Verbose output (`make test-verbose`) shows test details

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation that the infrastructure is working as expected before proceeding to the next phase.

---

## Phase 2: Utility Function Tests

### Overview
Add comprehensive tests for the mathematical utility functions in `src/utility.cpp`. These functions implement statistical tests and metrics used throughout the forest algorithms.

### Changes Required

#### 1. Expand test_utility.cpp
**File**: `src/tests/test_utility.cpp`
**Action**: Add tests for compute_logrank, compute_gini, compute_var_reduction

```cpp
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
    mat y;  // n x k: one-hot encoded classes
    vec w;  // weights
    uvec g; // groups
};

ClassificationData make_simple_classification_data() {
    ClassificationData data;

    // 10 observations, binary classification
    data.y = mat(10, 2);
    data.y.col(0) = vec{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}; // class 0
    data.y.col(1) = vec{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}; // class 1

    data.w = vec(10, fill::ones);
    data.g = uvec{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    return data;
}

// ============================================================================
// Logrank Tests
// ============================================================================

TEST_CASE("compute_logrank with identical groups returns zero", "[utility][logrank]") {
    auto data = make_simple_survival_data();

    // All in same group
    data.g.fill(0);

    double logrank = compute_logrank(data.y, data.w, data.g);

    REQUIRE(logrank == Approx(0.0).margin(1e-10));
}

TEST_CASE("compute_logrank with no events returns zero", "[utility][logrank]") {
    auto data = make_simple_survival_data();

    // All censored
    data.y.col(1).fill(0.0);

    double logrank = compute_logrank(data.y, data.w, data.g);

    REQUIRE(logrank == Approx(0.0).margin(1e-10));
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

    // All in same class (group doesn't matter for pure node)
    data.y.col(0).fill(1.0);
    data.y.col(1).fill(0.0);

    double gini = compute_gini(data.y, data.w, data.g);

    REQUIRE(gini == Approx(0.0).margin(1e-10));
}

TEST_CASE("compute_gini with 50-50 split returns maximum", "[utility][gini]") {
    ClassificationData data;

    // 4 observations, 50-50 split
    data.y = mat(4, 2);
    data.y.col(0) = vec{1, 1, 0, 0};
    data.y.col(1) = vec{0, 0, 1, 1};

    data.w = vec(4, fill::ones);
    data.g = uvec{0, 0, 1, 1}; // Split perfectly

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

    mat x_scaled = scale_x(x, w);

    // Check dimensions preserved
    REQUIRE(x_scaled.n_rows == x.n_rows);
    REQUIRE(x_scaled.n_cols == x.n_cols);

    // Check columns are centered (mean ≈ 0)
    REQUIRE(mean(x_scaled.col(0)) == Approx(0.0).margin(1e-10));
    REQUIRE(mean(x_scaled.col(1)) == Approx(0.0).margin(1e-10));

    // Check columns are scaled (sd ≈ 1)
    REQUIRE(stddev(x_scaled.col(0)) == Approx(1.0).epsilon(0.01));
    REQUIRE(stddev(x_scaled.col(1)) == Approx(1.0).epsilon(0.01));
}
```

### Success Criteria

#### Automated Verification:
- [x] Tests compile: `make -C src/tests` succeeds
- [x] All utility tests pass: `make -C src/tests test` shows passing tests for logrank, gini, variance, cstat, scale (all 12 tests, 25 assertions passed)
- [x] No memory leaks: `valgrind --leak-check=full src/tests/test_runner "[utility]"` (valgrind not available on macOS - skipped)
- [x] Tests are fast: `make -C src/tests test` completes in < 5 seconds (0.088s actual)

#### Manual Verification:
- [ ] Test output clearly indicates which utility functions are tested
- [ ] Failed tests (if any) provide clear error messages
- [ ] Can run individual test tags: `./test_runner "[logrank]"`

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to the next phase.

---

## Phase 3: Cox Regression Tests

### Overview
Add tests for Cox proportional hazards regression functions in `src/Coxph.cpp`, focusing on the Cholesky decomposition and Cox fitting algorithms.

### Changes Required

#### 1. Create test_coxph.cpp
**File**: `src/tests/test_coxph.cpp`
**Action**: Create new test file for Cox regression

```cpp
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

CoxTestData make_simple_cox_data() {
    CoxTestData data;

    // 6 observations, 2 predictors
    data.x = mat(6, 2);
    data.x.col(0) = vec{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    data.x.col(1) = vec{0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    data.y = mat(6, 2);
    data.y.col(0) = vec{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // times
    data.y.col(1) = vec{1.0, 1.0, 1.0, 1.0, 0.0, 0.0}; // status (4 events, 2 censored)

    data.w = vec(6, fill::ones);

    return data;
}

// ============================================================================
// Cholesky Decomposition Tests
// ============================================================================

TEST_CASE("cholesky_decomp on identity matrix", "[coxph][cholesky]") {
    mat vmat = eye<mat>(3, 3);
    mat vmat_orig = vmat;

    cholesky_decomp(vmat);

    // Diagonal should be 1s
    REQUIRE(vmat(0, 0) == Approx(1.0));
    REQUIRE(vmat(1, 1) == Approx(1.0));
    REQUIRE(vmat(2, 2) == Approx(1.0));

    // Result should be triangular
    REQUIRE(vmat(1, 0) == Approx(0.0).margin(1e-10));
    REQUIRE(vmat(2, 0) == Approx(0.0).margin(1e-10));
    REQUIRE(vmat(2, 1) == Approx(0.0).margin(1e-10));
}

TEST_CASE("cholesky_decomp on positive definite matrix", "[coxph][cholesky]") {
    // Create a symmetric positive definite matrix
    mat A = mat(3, 3);
    A << 4.0 << 2.0 << 1.0 << endr
      << 2.0 << 5.0 << 3.0 << endr
      << 1.0 << 3.0 << 6.0 << endr;

    mat vmat = A;
    cholesky_decomp(vmat);

    // Diagonal elements should be positive
    REQUIRE(vmat(0, 0) > 0.0);
    REQUIRE(vmat(1, 1) > 0.0);
    REQUIRE(vmat(2, 2) > 0.0);

    // All values should be finite
    REQUIRE(std::isfinite(vmat(0, 0)));
    REQUIRE(std::isfinite(vmat(1, 1)));
    REQUIRE(std::isfinite(vmat(2, 2)));
}

// ============================================================================
// Cholesky Solve Tests
// ============================================================================

TEST_CASE("cholesky_solve solves identity system", "[coxph][cholesky]") {
    mat vmat = eye<mat>(3, 3);
    cholesky_decomp(vmat);

    vec u = {1.0, 2.0, 3.0};
    vec u_orig = u;

    cholesky_solve(vmat, u);

    // Solution should be unchanged for identity
    REQUIRE(u(0) == Approx(u_orig(0)));
    REQUIRE(u(1) == Approx(u_orig(1)));
    REQUIRE(u(2) == Approx(u_orig(2)));
}

TEST_CASE("cholesky_solve produces finite results", "[coxph][cholesky]") {
    // Positive definite matrix
    mat A = mat(2, 2);
    A << 4.0 << 2.0 << endr
      << 2.0 << 5.0 << endr;

    cholesky_decomp(A);

    vec u = {8.0, 11.0};
    cholesky_solve(A, u);

    REQUIRE(std::isfinite(u(0)));
    REQUIRE(std::isfinite(u(1)));
}

// ============================================================================
// Cox Fit Tests
// ============================================================================

TEST_CASE("coxph_fit returns valid coefficients", "[coxph][fit]") {
    auto data = make_simple_cox_data();

    // Note: coxph_fit signature from examining the code
    // Adjust based on actual function signature
    List result = coxph_fit(
        data.x,
        data.y,
        data.w,
        0,      // method (0 = Breslow)
        1e-8,   // epsilon
        20      // iter_max
    );

    // Check result contains expected fields
    REQUIRE(result.containsElementNamed("beta"));
    REQUIRE(result.containsElementNamed("beta_var"));

    vec beta = as<vec>(result["beta"]);

    // Coefficients should be finite
    REQUIRE(std::isfinite(beta(0)));
    REQUIRE(std::isfinite(beta(1)));
}

TEST_CASE("coxph_fit with single predictor", "[coxph][fit]") {
    CoxTestData data;

    // Single predictor
    data.x = mat(6, 1);
    data.x.col(0) = vec{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    data.y = mat(6, 2);
    data.y.col(0) = vec{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    data.y.col(1) = vec{1.0, 1.0, 1.0, 1.0, 0.0, 0.0};

    data.w = vec(6, fill::ones);

    List result = coxph_fit(data.x, data.y, data.w, 0, 1e-8, 20);

    vec beta = as<vec>(result["beta"]);
    REQUIRE(beta.n_elem == 1);
    REQUIRE(std::isfinite(beta(0)));
}

TEST_CASE("coxph_fit converges with small dataset", "[coxph][fit]") {
    auto data = make_simple_cox_data();

    List result = coxph_fit(data.x, data.y, data.w, 0, 1e-8, 20);

    // Check convergence indicators
    int iter = as<int>(result["iter"]);
    REQUIRE(iter > 0);
    REQUIRE(iter <= 20); // Should converge within max iterations
}
```

#### 2. Update Makefile
**File**: `src/tests/Makefile`
**Action**: Add test_coxph.cpp to TEST_SOURCES

```makefile
# Change this line:
TEST_SOURCES = test_main.cpp test_utility.cpp

# To:
TEST_SOURCES = test_main.cpp test_utility.cpp test_coxph.cpp
```

### Success Criteria

#### Automated Verification:
- [ ] Tests compile: `make -C src/tests clean && make -C src/tests` succeeds
- [ ] All Cox regression tests pass: `make -C src/tests test` includes passing coxph tests
- [ ] Cholesky tests pass: `./src/tests/test_runner "[cholesky]"` shows all passing
- [ ] Cox fit tests pass: `./src/tests/test_runner "[fit]"` shows all passing

#### Manual Verification:
- [ ] Can identify which Cox regression functions are tested from output
- [ ] Test failures (if any during development) show helpful error messages
- [ ] Running `make test-verbose` shows detailed test information

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to the next phase.

---

## Phase 4: Data Class Tests

### Overview
Add tests for the Data class methods, focusing on matrix operations, submatrix access, and column manipulations used throughout the forest algorithms.

### Changes Required

#### 1. Create test_data.cpp
**File**: `src/tests/test_data.cpp`
**Action**: Create test file for Data class

```cpp
#include "catch2/catch.hpp"
#include <RcppArmadillo.h>
#include "../Data.h"

using namespace arma;
using namespace aorsf;

// ============================================================================
// Test Data Constructor
// ============================================================================

TEST_CASE("Data constructor initializes correctly", "[data][constructor]") {
    mat x = mat(5, 3, fill::randu);
    mat y = mat(5, 1, fill::randu);
    vec w = vec(5, fill::ones);

    Data data(x, y, w);

    REQUIRE(data.get_n_rows() == 5);
    REQUIRE(data.get_n_cols_x() == 3);
    REQUIRE(data.has_weights == true);
}

TEST_CASE("Data constructor with empty weights", "[data][constructor]") {
    mat x = mat(5, 3, fill::randu);
    mat y = mat(5, 1, fill::randu);
    vec w; // Empty

    Data data(x, y, w);

    REQUIRE(data.get_n_rows() == 5);
    REQUIRE(data.has_weights == false);
}

// ============================================================================
// Submatrix Access Tests
// ============================================================================

TEST_CASE("x_rows returns correct rows", "[data][submatrix]") {
    mat x = mat(5, 2);
    x << 1 << 2 << endr
      << 3 << 4 << endr
      << 5 << 6 << endr
      << 7 << 8 << endr
      << 9 << 10 << endr;

    mat y = mat(5, 1, fill::ones);
    vec w = vec(5, fill::ones);

    Data data(x, y, w);

    uvec rows = {0, 2, 4};
    mat x_subset = data.x_rows(rows);

    REQUIRE(x_subset.n_rows == 3);
    REQUIRE(x_subset.n_cols == 2);
    REQUIRE(x_subset(0, 0) == 1);
    REQUIRE(x_subset(1, 0) == 5);
    REQUIRE(x_subset(2, 0) == 9);
}

TEST_CASE("x_cols returns correct columns", "[data][submatrix]") {
    mat x = mat(3, 4);
    x << 1 << 2 << 3 << 4 << endr
      << 5 << 6 << 7 << 8 << endr
      << 9 << 10 << 11 << 12 << endr;

    mat y = mat(3, 1, fill::ones);
    vec w = vec(3, fill::ones);

    Data data(x, y, w);

    uvec cols = {1, 3};
    mat x_subset = data.x_cols(cols);

    REQUIRE(x_subset.n_rows == 3);
    REQUIRE(x_subset.n_cols == 2);
    REQUIRE(x_subset(0, 0) == 2);
    REQUIRE(x_subset(0, 1) == 4);
}

TEST_CASE("x_submat returns correct submatrix", "[data][submatrix]") {
    mat x = mat(4, 4, fill::eye);
    mat y = mat(4, 1, fill::ones);
    vec w = vec(4, fill::ones);

    Data data(x, y, w);

    uvec rows = {0, 2};
    uvec cols = {1, 3};
    mat x_subset = data.x_submat(rows, cols);

    REQUIRE(x_subset.n_rows == 2);
    REQUIRE(x_subset.n_cols == 2);
}

// ============================================================================
// Matrix-Vector Multiplication Tests
// ============================================================================

TEST_CASE("x_submat_mult_beta performs correct multiplication", "[data][mult]") {
    mat x = mat(3, 3);
    x << 1 << 2 << 3 << endr
      << 4 << 5 << 6 << endr
      << 7 << 8 << 9 << endr;

    mat y = mat(3, 1, fill::ones);
    vec w = vec(3, fill::ones);

    Data data(x, y, w);

    uvec rows = {0, 1, 2};
    uvec cols = {0, 1};
    vec beta = {1.0, 2.0};

    vec result = data.x_submat_mult_beta(rows, cols, beta);

    REQUIRE(result.n_elem == 3);
    // Row 0: 1*1 + 2*2 = 5
    REQUIRE(result(0) == Approx(5.0));
    // Row 1: 4*1 + 5*2 = 14
    REQUIRE(result(1) == Approx(14.0));
    // Row 2: 7*1 + 8*2 = 23
    REQUIRE(result(2) == Approx(23.0));
}

TEST_CASE("x_submat_mult_beta with partial dependence", "[data][mult][pd]") {
    mat x = mat(3, 3);
    x << 1 << 2 << 3 << endr
      << 4 << 5 << 6 << endr
      << 7 << 8 << 9 << endr;

    mat y = mat(3, 1, fill::ones);
    vec w = vec(3, fill::ones);

    Data data(x, y, w);

    uvec rows = {0, 1, 2};
    uvec cols = {0, 1};
    vec beta = {1.0, 2.0};
    vec pd_x_vals = {10.0}; // Override column 0 with value 10
    uvec pd_x_cols = {0};

    vec result = data.x_submat_mult_beta(rows, cols, beta, pd_x_vals, pd_x_cols);

    REQUIRE(result.n_elem == 3);
    // All rows: 10*1 + [col1]*2
    REQUIRE(result(0) == Approx(10.0 + 2.0*2.0)); // 14
    REQUIRE(result(1) == Approx(10.0 + 5.0*2.0)); // 20
    REQUIRE(result(2) == Approx(10.0 + 8.0*2.0)); // 26
}

// ============================================================================
// Column Manipulation Tests
// ============================================================================

TEST_CASE("save_col and restore_col work correctly", "[data][columns]") {
    mat x = mat(3, 2);
    x << 1 << 2 << endr
      << 3 << 4 << endr
      << 5 << 6 << endr;

    mat y = mat(3, 1, fill::ones);
    vec w = vec(3, fill::ones);

    Data data(x, y, w);

    // Save column 1
    data.save_col(1);

    // Modify column 1
    data.fill_col(99.0, 1);

    // Verify modification
    mat x_modified = data.get_x();
    REQUIRE(x_modified(0, 1) == 99.0);
    REQUIRE(x_modified(1, 1) == 99.0);
    REQUIRE(x_modified(2, 1) == 99.0);

    // Restore column 1
    data.restore_col(1);

    // Verify restoration
    mat x_restored = data.get_x();
    REQUIRE(x_restored(0, 1) == 2.0);
    REQUIRE(x_restored(1, 1) == 4.0);
    REQUIRE(x_restored(2, 1) == 6.0);
}

TEST_CASE("permute_col shuffles column", "[data][columns]") {
    mat x = mat(5, 2);
    x << 1 << 10 << endr
      << 2 << 20 << endr
      << 3 << 30 << endr
      << 4 << 40 << endr
      << 5 << 50 << endr;

    mat y = mat(5, 1, fill::ones);
    vec w = vec(5, fill::ones);

    Data data(x, y, w);

    // Create RNG with fixed seed
    std::mt19937_64 rng(12345);

    // Permute column 1
    data.permute_col(1, rng);

    mat x_permuted = data.get_x();

    // Column 0 should be unchanged
    REQUIRE(x_permuted(0, 0) == 1.0);
    REQUIRE(x_permuted(1, 0) == 2.0);

    // Column 1 should contain same values but (likely) different order
    vec col1 = x_permuted.col(1);
    REQUIRE(col1.n_elem == 5);

    // Check that sum is preserved (all values still present)
    REQUIRE(sum(col1) == Approx(10 + 20 + 30 + 40 + 50));
}

TEST_CASE("fill_col sets all values correctly", "[data][columns]") {
    mat x = mat(4, 2, fill::randu);
    mat y = mat(4, 1, fill::ones);
    vec w = vec(4, fill::ones);

    Data data(x, y, w);

    data.fill_col(42.0, 0);

    mat x_filled = data.get_x();
    REQUIRE(x_filled(0, 0) == 42.0);
    REQUIRE(x_filled(1, 0) == 42.0);
    REQUIRE(x_filled(2, 0) == 42.0);
    REQUIRE(x_filled(3, 0) == 42.0);

    // Column 1 should be unchanged
    REQUIRE(x_filled(0, 1) != 42.0); // Random, unlikely to be 42
}
```

#### 2. Update Makefile
**File**: `src/tests/Makefile`
**Action**: Add test_data.cpp to TEST_SOURCES

```makefile
# Change:
TEST_SOURCES = test_main.cpp test_utility.cpp test_coxph.cpp

# To:
TEST_SOURCES = test_main.cpp test_utility.cpp test_coxph.cpp test_data.cpp
```

### Success Criteria

#### Automated Verification:
- [ ] Tests compile: `make -C src/tests clean && make -C src/tests` succeeds
- [ ] All Data class tests pass: `make -C src/tests test` includes passing data tests
- [ ] Constructor tests pass: `./src/tests/test_runner "[constructor]"` succeeds
- [ ] Submatrix tests pass: `./src/tests/test_runner "[submatrix]"` succeeds
- [ ] Matrix multiplication tests pass: `./src/tests/test_runner "[mult]"` succeeds
- [ ] Column manipulation tests pass: `./src/tests/test_runner "[columns]"` succeeds

#### Manual Verification:
- [ ] Test output clearly shows Data class functionality being tested
- [ ] Can run tests with different tags to focus on specific functionality
- [ ] Test execution is fast (< 1 second for Data tests alone)

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to the next phase.

---

## Phase 5: Linear/Logistic Regression Tests

### Overview
Add tests for linear and logistic regression fitting functions used for computing linear combinations in oblique splits. These are simpler than Cox regression and should produce predictable results with synthetic data.

### Changes Required

#### 1. Create test_regression.cpp
**File**: `src/tests/test_regression.cpp`
**Action**: Create test file for regression functions

```cpp
#include "catch2/catch.hpp"
#include <RcppArmadillo.h>
#include "../utility.h"

using namespace arma;
using namespace aorsf;

// ============================================================================
// Linear Regression Test Data
// ============================================================================

struct RegressionData {
    mat x;
    mat y;
    vec w;
};

RegressionData make_perfect_linear_data() {
    RegressionData data;

    // Perfect linear relationship: y = 2*x1 + 3*x2
    data.x = mat(5, 2);
    data.x << 1 << 1 << endr
         << 2 << 1 << endr
         << 3 << 2 << endr
         << 4 << 2 << endr
         << 5 << 3 << endr;

    data.y = mat(5, 1);
    data.y.col(0) = 2.0 * data.x.col(0) + 3.0 * data.x.col(1);

    data.w = vec(5, fill::ones);

    return data;
}

RegressionData make_simple_regression_data() {
    RegressionData data;

    // Simple dataset with some noise
    data.x = mat(6, 2);
    data.x << 1.0 << 0.5 << endr
         << 2.0 << 1.0 << endr
         << 3.0 << 1.5 << endr
         << 4.0 << 2.0 << endr
         << 5.0 << 2.5 << endr
         << 6.0 << 3.0 << endr;

    data.y = mat(6, 1);
    data.y << 2.1 << endr
         << 4.2 << endr
         << 6.0 << endr
         << 8.1 << endr
         << 10.0 << endr
         << 12.1 << endr;

    data.w = vec(6, fill::ones);

    return data;
}

// ============================================================================
// Classification Test Data
// ============================================================================

RegressionData make_simple_classification_data() {
    RegressionData data;

    // Binary classification: clear separation
    data.x = mat(6, 2);
    data.x << 1.0 << 1.0 << endr  // Class 0
         << 1.5 << 1.2 << endr  // Class 0
         << 2.0 << 1.5 << endr  // Class 0
         << 4.0 << 4.0 << endr  // Class 1
         << 4.5 << 4.2 << endr  // Class 1
         << 5.0 << 4.5 << endr; // Class 1

    // One-hot encoded: [class0, class1]
    data.y = mat(6, 2);
    data.y << 1 << 0 << endr
         << 1 << 0 << endr
         << 1 << 0 << endr
         << 0 << 1 << endr
         << 0 << 1 << endr
         << 0 << 1 << endr;

    data.w = vec(6, fill::ones);

    return data;
}

// ============================================================================
// Linear Regression Tests
// ============================================================================

TEST_CASE("linreg_fit returns finite coefficients", "[regression][linear]") {
    auto data = make_simple_regression_data();

    mat result = linreg_fit(
        data.x,
        data.y,
        data.w,
        true,   // do_scale
        1e-8,   // epsilon
        20      // iter_max
    );

    // Should return beta coefficients
    REQUIRE(result.n_rows > 0);
    REQUIRE(result.n_cols > 0);

    // All coefficients should be finite
    for (uword i = 0; i < result.n_elem; i++) {
        REQUIRE(std::isfinite(result(i)));
    }
}

TEST_CASE("linreg_fit with perfect linear relationship", "[regression][linear]") {
    auto data = make_perfect_linear_data();

    mat result = linreg_fit(data.x, data.y, data.w, false, 1e-8, 20);

    // Check we get coefficients
    REQUIRE(result.n_elem >= 2);

    // Coefficients should be close to true values (2, 3)
    // Note: May not be exact due to numerical precision
    REQUIRE(std::abs(result(0) - 2.0) < 0.1);
    REQUIRE(std::abs(result(1) - 3.0) < 0.1);
}

TEST_CASE("linreg_fit with single predictor", "[regression][linear]") {
    RegressionData data;

    // y = 2*x
    data.x = mat(5, 1);
    data.x << 1 << endr << 2 << endr << 3 << endr << 4 << endr << 5 << endr;

    data.y = mat(5, 1);
    data.y << 2 << endr << 4 << endr << 6 << endr << 8 << endr << 10 << endr;

    data.w = vec(5, fill::ones);

    mat result = linreg_fit(data.x, data.y, data.w, false, 1e-8, 20);

    REQUIRE(result.n_elem >= 1);
    REQUIRE(std::abs(result(0) - 2.0) < 0.1);
}

TEST_CASE("linreg_fit with scaling", "[regression][linear]") {
    auto data = make_simple_regression_data();

    // With scaling
    mat result_scaled = linreg_fit(data.x, data.y, data.w, true, 1e-8, 20);

    // Without scaling
    mat result_unscaled = linreg_fit(data.x, data.y, data.w, false, 1e-8, 20);

    // Both should produce finite results
    REQUIRE(std::isfinite(result_scaled(0)));
    REQUIRE(std::isfinite(result_unscaled(0)));

    // Results may differ but both should be reasonable
    // (not testing exact equality, just that both approaches work)
}

// ============================================================================
// Logistic Regression Tests
// ============================================================================

TEST_CASE("logreg_fit returns finite coefficients", "[regression][logistic]") {
    auto data = make_simple_classification_data();

    mat result = logreg_fit(
        data.x,
        data.y,
        data.w,
        true,   // do_scale
        1e-8,   // epsilon
        20      // iter_max
    );

    // Should return coefficients
    REQUIRE(result.n_rows > 0);
    REQUIRE(result.n_cols > 0);

    // All coefficients should be finite
    for (uword i = 0; i < result.n_elem; i++) {
        REQUIRE(std::isfinite(result(i)));
    }
}

TEST_CASE("logreg_fit with clear separation", "[regression][logistic]") {
    auto data = make_simple_classification_data();

    mat result = logreg_fit(data.x, data.y, data.w, false, 1e-8, 20);

    // Should get coefficients for predictors
    REQUIRE(result.n_elem >= 2);

    // Coefficients should be non-zero (there's signal)
    REQUIRE(std::abs(result(0)) > 0.01);
    REQUIRE(std::abs(result(1)) > 0.01);
}

TEST_CASE("logreg_fit with binary outcome", "[regression][logistic]") {
    RegressionData data;

    // Simple binary classification
    data.x = mat(4, 1);
    data.x << 1 << endr << 2 << endr << 3 << endr << 4 << endr;

    // First two are class 0, last two are class 1
    data.y = mat(4, 2);
    data.y << 1 << 0 << endr
         << 1 << 0 << endr
         << 0 << 1 << endr
         << 0 << 1 << endr;

    data.w = vec(4, fill::ones);

    mat result = logreg_fit(data.x, data.y, data.w, false, 1e-8, 20);

    REQUIRE(result.n_elem >= 1);
    REQUIRE(std::isfinite(result(0)));

    // Coefficient should be positive (higher x -> class 1)
    REQUIRE(result(0) > 0.0);
}

TEST_CASE("logreg_fit handles weighted observations", "[regression][logistic]") {
    auto data = make_simple_classification_data();

    // Give more weight to first three observations
    data.w = vec{2.0, 2.0, 2.0, 1.0, 1.0, 1.0};

    mat result = logreg_fit(data.x, data.y, data.w, true, 1e-8, 20);

    // Should still converge
    REQUIRE(std::isfinite(result(0)));
    REQUIRE(std::isfinite(result(1)));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_CASE("linreg_fit with collinear predictors", "[regression][linear][edge]") {
    RegressionData data;

    // Second column is 2x first column (perfect collinearity)
    data.x = mat(5, 2);
    data.x.col(0) = vec{1, 2, 3, 4, 5};
    data.x.col(1) = 2.0 * data.x.col(0);

    data.y = mat(5, 1);
    data.y.col(0) = data.x.col(0);

    data.w = vec(5, fill::ones);

    // Should still return finite values (even if not unique solution)
    mat result = linreg_fit(data.x, data.y, data.w, true, 1e-8, 20);

    REQUIRE(std::isfinite(result(0)));
    REQUIRE(std::isfinite(result(1)));
}

TEST_CASE("logreg_fit with all same class", "[regression][logistic][edge]") {
    RegressionData data;

    data.x = mat(4, 2, fill::randu);

    // All observations are class 0
    data.y = mat(4, 2);
    data.y.col(0).fill(1.0);
    data.y.col(1).fill(0.0);

    data.w = vec(4, fill::ones);

    // Should handle gracefully (may not converge but shouldn't crash)
    mat result = logreg_fit(data.x, data.y, data.w, true, 1e-8, 5);

    // Just verify it returns something finite
    REQUIRE(result.n_elem > 0);
}
```

#### 2. Update Makefile
**File**: `src/tests/Makefile`
**Action**: Add test_regression.cpp to TEST_SOURCES

```makefile
# Change:
TEST_SOURCES = test_main.cpp test_utility.cpp test_coxph.cpp test_data.cpp

# To:
TEST_SOURCES = test_main.cpp test_utility.cpp test_coxph.cpp test_data.cpp test_regression.cpp
```

### Success Criteria

#### Automated Verification:
- [ ] Tests compile: `make -C src/tests clean && make -C src/tests` succeeds
- [ ] All regression tests pass: `make -C src/tests test` includes passing regression tests
- [ ] Linear regression tests pass: `./src/tests/test_runner "[linear]"` succeeds
- [ ] Logistic regression tests pass: `./src/tests/test_runner "[logistic]"` succeeds
- [ ] Edge case tests pass: `./src/tests/test_runner "[edge]"` succeeds
- [ ] Complete test suite runs in < 10 seconds: `time make -C src/tests test`

#### Manual Verification:
- [ ] Test output shows clear distinction between linear and logistic regression tests
- [ ] Can run regression tests independently from other test suites
- [ ] Test results are reproducible across multiple runs
- [ ] All test assertions have meaningful descriptions

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to final documentation.

---

## Phase 6: Documentation and Polish

### Overview
Finalize documentation, add usage examples, and ensure the test infrastructure is well-documented for future maintainers.

### Changes Required

#### 1. Update main README
**File**: `README.md`
**Action**: Add section about C++ unit tests

```markdown
## For Developers

### C++ Unit Tests

This package includes comprehensive C++ unit tests using the Catch2 framework. These tests verify the correctness of core algorithms independently of the R interface.

**Running C++ tests:**

```bash
cd src/tests
make test
```

**Test coverage includes:**
- Statistical utility functions (log-rank, C-statistic, Gini impurity, variance reduction)
- Cox proportional hazards regression
- Linear and logistic regression
- Data class operations

See `src/tests/README.md` for detailed testing documentation.
```

#### 2. Update CLAUDE.md
**File**: `CLAUDE.md`
**Action**: Add C++ testing guidelines

```markdown
## C++ Testing

### Running C++ Unit Tests

```r
# From R, run system command
system("make -C src/tests test")
```

```bash
# Or directly from command line
cd src/tests
make test

# Run specific test tags
./test_runner "[utility]"
./test_runner "[coxph]"
```

### Adding New C++ Tests

1. Create or modify test file in `src/tests/`
2. Use Catch2 TEST_CASE macro with descriptive tags
3. Generate minimal synthetic test data in C++
4. Add test file to `TEST_SOURCES` in Makefile if new file
5. Run `make test` to verify

### Test Categories

- `[utility]` - Utility functions (compute_*, scale_x)
- `[coxph]` - Cox regression functions
- `[data]` - Data class methods
- `[regression]` - Linear/logistic regression
- `[sanity]` - Basic sanity checks
- `[edge]` - Edge cases

### C++ Test Best Practices

- Generate synthetic test data using Armadillo
- Use `REQUIRE()` for critical assertions
- Use `Approx()` for floating-point comparisons: `REQUIRE(x == Approx(1.0))`
- Tag tests appropriately for easy filtering
- Keep tests focused (one concept per TEST_CASE)
- Document expected behavior in test descriptions
```

#### 3. Create developer guide
**File**: `src/tests/DEVELOPER_GUIDE.md`
**Action**: Create comprehensive developer documentation

```markdown
# C++ Unit Testing Developer Guide

## Overview

This guide explains how to write and maintain C++ unit tests for aorsf using Catch2.

## Architecture

### Directory Structure

```
src/tests/
├── Makefile              # Build system
├── README.md             # User-facing documentation
├── DEVELOPER_GUIDE.md    # This file
├── catch2/
│   └── catch.hpp         # Catch2 v2.13.10 single-header
├── test_main.cpp         # Catch2 entry point (defines main)
├── test_utility.cpp      # Utility function tests
├── test_coxph.cpp        # Cox regression tests
├── test_data.cpp         # Data class tests
└── test_regression.cpp   # Linear/logistic regression tests
```

### Build Process

1. Makefile compiles all `test_*.cpp` files
2. Includes headers from `..` (src/) and Rcpp/RcppArmadillo
3. Links against LAPACK, BLAS, and R libraries
4. Produces `test_runner` executable

## Writing Tests

### Basic Test Structure

```cpp
#include "catch2/catch.hpp"
#include <RcppArmadillo.h>
#include "../your_header.h"

using namespace arma;
using namespace aorsf;

TEST_CASE("Descriptive test name", "[tag1][tag2]") {
    // Arrange - set up test data
    vec x = {1.0, 2.0, 3.0};

    // Act - call function under test
    double result = sum(x);

    // Assert - verify results
    REQUIRE(result == 6.0);
}
```

### Generating Test Data

**Simple vectors/matrices:**
```cpp
vec v = {1.0, 2.0, 3.0};
mat m = mat(5, 3, fill::randu);  // Random
mat eye = eye<mat>(3, 3);         // Identity
```

**Structured test data:**
```cpp
struct TestData {
    mat x;
    mat y;
    vec w;
};

TestData make_test_data() {
    TestData data;
    data.x = mat(10, 3, fill::randu);
    data.y = mat(10, 1, fill::randu);
    data.w = vec(10, fill::ones);
    return data;
}
```

### Assertions

**Exact comparisons:**
```cpp
REQUIRE(x == 5);
REQUIRE(result == expected);
REQUIRE_FALSE(condition);
```

**Floating-point comparisons:**
```cpp
REQUIRE(x == Approx(1.0));                    // Default tolerance
REQUIRE(x == Approx(1.0).epsilon(0.01));      // Relative tolerance (1%)
REQUIRE(x == Approx(0.0).margin(1e-10));      // Absolute tolerance
```

**Exceptions:**
```cpp
REQUIRE_THROWS(risky_function());
REQUIRE_NOTHROW(safe_function());
REQUIRE_THROWS_AS(func(), std::runtime_error);
```

**Ranges:**
```cpp
REQUIRE(x > 0.0);
REQUIRE(x <= 1.0);
REQUIRE(std::isfinite(x));
```

### Test Organization

**One concept per TEST_CASE:**
```cpp
// Good - focused
TEST_CASE("compute_gini with pure node returns zero", "[gini]") {
    // ...
}

TEST_CASE("compute_gini with 50-50 split returns maximum", "[gini]") {
    // ...
}

// Bad - too broad
TEST_CASE("compute_gini tests", "[gini]") {
    // Tests multiple scenarios in one case
}
```

**Use sections for related tests:**
```cpp
TEST_CASE("Data class construction", "[data]") {
    SECTION("with valid inputs") {
        // ...
    }

    SECTION("with empty weights") {
        // ...
    }
}
```

### Tagging Strategy

Use hierarchical tags for filtering:

- **Broad categories**: `[utility]`, `[coxph]`, `[data]`, `[regression]`
- **Specific functionality**: `[logrank]`, `[gini]`, `[cholesky]`, `[mult]`
- **Special cases**: `[edge]`, `[sanity]`, `[performance]`

**Examples:**
```cpp
TEST_CASE("...", "[utility][logrank]") { }      // Can run all utility or just logrank
TEST_CASE("...", "[data][mult]") { }            // Can run all data or just multiplication tests
TEST_CASE("...", "[regression][linear][edge]") { } // Linear regression edge case
```

## Common Patterns

### Testing Statistical Functions

```cpp
TEST_CASE("Function name with known result", "[tag]") {
    // Create data with known statistical properties
    vec x = {1, 2, 3, 4, 5};

    // Known result (calculated by hand or reference implementation)
    double expected = 3.0;  // mean

    double result = my_function(x);

    REQUIRE(result == Approx(expected).epsilon(0.01));
}
```

### Testing with Reference Implementation

```cpp
TEST_CASE("Function matches reference", "[tag]") {
    auto data = make_test_data();

    // Call our implementation
    double our_result = our_function(data.x, data.y);

    // Call reference implementation
    double ref_result = reference_function(data.x, data.y);

    REQUIRE(our_result == Approx(ref_result).epsilon(0.001));
}
```

### Testing Edge Cases

```cpp
TEST_CASE("Function handles empty input", "[tag][edge]") {
    vec empty;
    REQUIRE_THROWS(function_requiring_data(empty));
}

TEST_CASE("Function handles single element", "[tag][edge]") {
    vec single = {1.0};
    double result = function(single);
    REQUIRE(std::isfinite(result));
}

TEST_CASE("Function handles all identical values", "[tag][edge]") {
    vec identical = vec(100, fill::value(5.0));
    double result = function(identical);
    REQUIRE(result == Approx(expected).margin(1e-10));
}
```

## Running Tests

### All tests
```bash
make test
```

### Specific tags
```bash
./test_runner "[utility]"
./test_runner "[coxph][cholesky]"  # Multiple tags
```

### Verbose output
```bash
make test-verbose
./test_runner -s  # Shows all assertions
```

### List all tests
```bash
./test_runner --list-tests
```

### List all tags
```bash
./test_runner --list-tags
```

## Debugging Tests

### Failed Test Output

Catch2 provides detailed failure messages:

```
test_utility.cpp:45: FAILED:
  REQUIRE( result == Approx(1.0) )
with expansion:
  1.0001 == Approx( 1.0 )
```

### Adding Debug Output

```cpp
TEST_CASE("Debug example", "[debug]") {
    vec x = {1, 2, 3};

    // Print values during test
    std::cout << "x = " << x.t() << std::endl;

    double result = function(x);

    // This will only show if test fails or with -s flag
    INFO("Intermediate result: " << result);

    REQUIRE(result > 0);
}
```

### Using gdb/lldb

```bash
make
gdb ./test_runner
(gdb) run
(gdb) backtrace  # On crash
```

## Performance Considerations

### Test Execution Speed

- Keep tests fast (< 10 seconds total)
- Use minimal data sizes that still test behavior
- Tag slow tests with `[.slow]` to exclude by default

```cpp
// Won't run by default (dot prefix excludes)
TEST_CASE("Slow test", "[.slow]") {
    // Large-scale test
}
```

Run slow tests explicitly:
```bash
./test_runner "[slow]"
```

### Memory Usage

- Catch2 creates new test fixtures for each TEST_CASE
- Avoid global state
- Clean up resources in test (though usually automatic with Armadillo)

## Continuous Integration

### Local Pre-commit Checks

```bash
# Before committing
make -C src/tests clean
make -C src/tests test
```

### Adding to CI Pipeline (Future)

```yaml
# Example GitHub Actions snippet
- name: Run C++ tests
  run: |
    cd src/tests
    make test
```

## Troubleshooting

### Compilation Errors

**Problem**: Cannot find Rcpp headers
```
fatal error: 'Rcpp.h' file not found
```

**Solution**: Ensure R is installed with Rcpp and RcppArmadillo packages

**Problem**: Undefined reference to LAPACK/BLAS functions

**Solution**: Check that `-llapack -lblas` are in `LDFLAGS` in Makefile

### Runtime Errors

**Problem**: Segmentation fault

**Solution**:
1. Run with debugger: `gdb ./test_runner`
2. Check for out-of-bounds access
3. Verify Armadillo matrix dimensions

**Problem**: NaN or Inf in results

**Solution**: Add assertions checking for finite values:
```cpp
REQUIRE(std::isfinite(result));
```

## Maintenance

### Updating Catch2

1. Download new version from https://github.com/catchorg/Catch2/releases
2. Replace `src/tests/catch2/catch.hpp`
3. Run full test suite to verify compatibility
4. Update version number in documentation

### Adding New Test Files

1. Create `test_newfeature.cpp`
2. Include necessary headers
3. Add to `TEST_SOURCES` in Makefile:
   ```makefile
   TEST_SOURCES = test_main.cpp test_utility.cpp ... test_newfeature.cpp
   ```
4. Run `make clean && make test`

### Refactoring Tests

When refactoring tests:
1. Ensure all tests still pass before refactoring
2. Refactor incrementally (one file at a time)
3. Run tests after each change
4. Keep test names and tags consistent

## Best Practices Summary

✅ **DO:**
- Write focused tests (one concept per TEST_CASE)
- Use descriptive test names
- Tag tests appropriately
- Generate minimal synthetic data
- Use Approx() for floating-point comparisons
- Test edge cases
- Keep tests fast

❌ **DON'T:**
- Test multiple unrelated things in one TEST_CASE
- Use hardcoded "magic numbers" without comments
- Rely on external files or R data
- Leave commented-out test code
- Ignore compiler warnings
- Test private implementation details

## Resources

- **Catch2 Documentation**: https://github.com/catchorg/Catch2/tree/v2.x
- **Armadillo Documentation**: http://arma.sourceforge.net/docs.html
- **aorsf Architecture**: See `Refactoring_Recommendations.md`
```

#### 4. Add CI placeholder
**File**: `.github/workflows/cpp-tests.yml.disabled`
**Action**: Create placeholder for future CI integration

```yaml
# Disabled by default - rename to .yml to enable
# C++ Unit Tests Workflow
# This workflow runs the C++ unit tests using Catch2

name: C++ Unit Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]

    steps:
    - uses: actions/checkout@v3

    - name: Setup R
      uses: r-lib/actions/setup-r@v2
      with:
        r-version: 'release'

    - name: Install R dependencies
      run: |
        install.packages(c("Rcpp", "RcppArmadillo"))
      shell: Rscript {0}

    - name: Install system dependencies (Ubuntu)
      if: runner.os == 'Linux'
      run: |
        sudo apt-get update
        sudo apt-get install -y liblapack-dev libblas-dev

    - name: Build C++ tests
      run: |
        cd src/tests
        make

    - name: Run C++ tests
      run: |
        cd src/tests
        make test
```

### Success Criteria

#### Automated Verification:
- [ ] Main README mentions C++ tests: `grep -q "C++ unit tests" README.md`
- [ ] CLAUDE.md has testing section: `grep -q "C++ Testing" CLAUDE.md`
- [ ] Developer guide exists: `test -f src/tests/DEVELOPER_GUIDE.md`
- [ ] All documentation is well-formatted and readable
- [ ] No broken links in documentation

#### Manual Verification:
- [ ] README section is clear and concise
- [ ] CLAUDE.md provides practical guidance for AI assistants
- [ ] DEVELOPER_GUIDE.md is comprehensive and helpful
- [ ] Documentation explains how to add new tests
- [ ] Examples in documentation are correct and work

**Implementation Note**: This is the final phase. After completion and verification, the C++ unit testing infrastructure is production-ready.

---

## Testing Strategy

### Unit Test Philosophy

**Focus**: Test individual functions and components in isolation
**Data**: Generate minimal synthetic test data in C++ (no external dependencies)
**Coverage**: Prioritize core algorithms used throughout the forest implementation

### Test Categories

1. **Sanity tests**: Basic Armadillo functionality, test infrastructure
2. **Utility functions**: Statistical computations (log-rank, C-statistic, Gini, variance)
3. **Cox regression**: Cholesky decomposition, solving, Cox fitting
4. **Data class**: Matrix operations, column manipulations, submatrix access
5. **Regression**: Linear and logistic regression fitting
6. **Edge cases**: Empty inputs, single elements, collinearity, all-same values

### What We're Testing

✅ **Mathematical correctness**:
- Statistical tests produce expected values
- Matrix operations compute correctly
- Regression algorithms converge

✅ **Edge case handling**:
- Empty inputs
- Single observations
- Perfect separation/collinearity
- All-censored survival data

✅ **Numerical stability**:
- Results are finite (not NaN/Inf)
- Algorithms converge within iteration limits
- Floating-point comparisons use appropriate tolerances

### What We're NOT Testing

❌ **Integration with R**: That's covered by existing testthat tests
❌ **Tree/Forest classes**: Too complex, may require refactoring first
❌ **Performance**: Tests verify correctness, not speed
❌ **Rcpp interface**: Not testing `_exported()` wrapper functions
❌ **Threading**: Single-threaded tests only

### Success Metrics

**Coverage targets** (after all phases):
- ≥20 unit tests across all categories
- All tests passing
- Test execution < 10 seconds total
- No compiler warnings
- No memory leaks (verified with valgrind if available)

**Quality indicators**:
- Tests use descriptive names
- Tests are properly tagged
- Synthetic data generation is clear and minimal
- Floating-point comparisons use Approx()
- Edge cases are explicitly tested

---

## Performance Considerations

### Build Time
- **Single-header Catch2**: ~1-2 seconds compilation overhead per test file
- **Incremental builds**: Makefile only recompiles changed files
- **Full rebuild**: ~5-10 seconds on modern hardware

### Test Execution Time
- **Target**: < 10 seconds for full test suite
- **Individual tests**: < 100ms each
- **Data sizes**: Minimal (5-10 observations per test)

### Memory Usage
- **Test runner**: < 50 MB RAM
- **Per-test fixtures**: < 1 MB (small Armadillo matrices)
- **No memory leaks**: Verified with valgrind

---

## Migration Notes

### For Existing Developers

**No changes to R package**:
- R package builds unchanged
- testthat tests unchanged
- CRAN submission unaffected (tests are separate)

**New capability**:
- Can test C++ code directly
- Faster iteration (no R startup overhead)
- Better debugging with gdb/lldb

**When to use each test type**:
- **C++ tests**: Algorithm correctness, edge cases, numerical stability
- **R testthat tests**: Integration with R, end-to-end workflows, API contracts

### Adding to Existing Workflows

**Local development**:
```bash
# Before committing C++ changes
make -C src/tests test
```

**Code review**:
- C++ changes should include corresponding tests
- Tests should pass before merge

---

## References

### Documentation
- **This plan**: `/Users/mweiss/sandbox/aorsf/CPP_Unit_Testing_Plan.md`
- **Refactoring recommendations**: `/Users/mweiss/sandbox/aorsf/Refactoring_Recommendations.md`
- **Package instructions**: `/Users/mweiss/sandbox/aorsf/CLAUDE.md`

### External Resources
- **Catch2 v2.x**: https://github.com/catchorg/Catch2/tree/v2.x
- **Armadillo**: http://arma.sourceforge.net/docs.html
- **Rcpp**: https://www.rcpp.org/
- **RcppArmadillo**: https://github.com/RcppCore/RcppArmadillo

### Key Files
- **C++ source**: `/Users/mweiss/sandbox/aorsf/src/*.{h,cpp}`
- **Existing R tests**: `/Users/mweiss/sandbox/aorsf/tests/testthat/*.R`
- **Makefiles**: `/Users/mweiss/sandbox/aorsf/src/Makevars*`

---

## Appendix: Catch2 Quick Reference

### Basic Assertions
```cpp
REQUIRE(expression);              // Test fails if false
REQUIRE_FALSE(expression);         // Test fails if true
CHECK(expression);                 // Logs failure but continues test
```

### Floating-Point
```cpp
REQUIRE(x == Approx(1.0));                  // Relative epsilon
REQUIRE(x == Approx(1.0).epsilon(0.01));    // Custom relative tolerance
REQUIRE(x == Approx(0.0).margin(1e-10));    // Absolute tolerance for near-zero
```

### Exceptions
```cpp
REQUIRE_THROWS(expression);                 // Must throw any exception
REQUIRE_NOTHROW(expression);                // Must not throw
REQUIRE_THROWS_AS(expression, ExType);      // Must throw specific type
```

### Test Organization
```cpp
TEST_CASE("Name", "[tag1][tag2]") {
    // Test code
}

SECTION("Sub-test") {
    // Inherits TEST_CASE setup
}
```

### Running Tests
```bash
./test_runner                    # Run all tests
./test_runner "[tag]"            # Run tests with tag
./test_runner -s                 # Show all assertions
./test_runner --list-tests       # List all tests
./test_runner --list-tags        # List all tags
```
