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
