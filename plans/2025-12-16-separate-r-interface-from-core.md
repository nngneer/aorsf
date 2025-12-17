# Separate R Interface from Core Logic - Implementation Plan

## Overview

Refactor the aorsf package to create a standalone C++ library (libaorsf) that can be used independently of R, with a thin Rcpp wrapper layer for the R package. This enables the C++ core to be called from Python, Julia, or other languages.

## Current State Analysis

The aorsf package currently has:
- **R Layer** (`R/`): R6 classes (`ObliqueForest*`), data validation, user interface
- **C++ Layer** (`src/`): Core computation (Tree/Forest hierarchies), tightly coupled with Rcpp

### Key Entanglement Points

| Location | Rcpp Dependency | Purpose |
|----------|-----------------|---------|
| `Tree.h:355-358` | `Rcpp::RObject` member variables | Store R callback functions |
| `Tree.cpp:156,806` | `Rcpp::stop()` | Error handling |
| `Tree.cpp:814-835` | `Rcpp::wrap()`, `as<>` | Convert types for R function calls |
| `Tree.cpp:1370-1375` | `Rcpp::wrap()`, `as<>` | OOB evaluation with R function |
| `Forest.cpp:103-108,283-313,...` | `Rcpp::Rcout` | Progress output |
| `Forest.h:37,281` | `Rcpp::IntegerVector`, `Rcpp::RObject` | Seeds and callbacks |

### Key Discoveries

- `src/globals.h:1-112`: All enums and constants are already pure C++ (no Rcpp)
- `src/Data.h`: Data class uses only Armadillo types (pure C++)
- `src/utility.h/.cpp`: Utility functions are mostly pure C++
- `src/Coxph.h/.cpp`: Cox regression is pure C++ with Armadillo
- Recent refactoring (commit 9f6901b) consolidated R interface methods in base Tree class

## Desired End State

After implementation:
1. `src/core/` contains a pure C++ library with no Rcpp dependencies
2. `src/rcpp/` contains thin Rcpp wrappers that bridge R to the core library
3. Core library can compile and run standalone (testable with catch2/googletest)
4. R package functionality is 100% preserved with same API
5. Performance is maintained or improved

### Verification

```bash
# Core library compiles without R
cd src/core && make test

# R package works identically
Rcpp::compileAttributes()
devtools::test()
devtools::check()
```

## What We're NOT Doing

- Changing the R-level API (orsf(), predict(), etc.)
- Changing the public C++ class interfaces significantly
- Adding new features (pure refactoring)
- Removing R callback functionality (just abstracting it)
- Creating Python/Julia bindings (future work enabled by this refactoring)

## Implementation Approach

Use the **Dependency Injection** pattern to abstract R-specific functionality:
1. Define C++ interfaces for console output and callbacks
2. Core library uses these interfaces
3. Rcpp layer provides implementations that use R types
4. Standalone builds can provide alternative implementations

## Phase 1: Abstract Console Output

### Overview
Replace all `Rcpp::Rcout` usage with an abstract output interface that can be implemented differently for R vs standalone builds.

### Changes Required

#### 1. Create Output Interface
**File**: `src/core/Output.h` (new)
```cpp
#ifndef AORSF_OUTPUT_H_
#define AORSF_OUTPUT_H_

#include <string>
#include <memory>

namespace aorsf {

class OutputHandler {
public:
    virtual ~OutputHandler() = default;
    virtual void print(const std::string& msg) = 0;
    virtual void println(const std::string& msg) = 0;
};

// Global output handler (default is silent)
class OutputManager {
public:
    static void set_handler(std::shared_ptr<OutputHandler> handler);
    static OutputHandler& get();
private:
    static std::shared_ptr<OutputHandler> handler_;
};

// Convenience macros
#define AORSF_OUT OutputManager::get()

} // namespace aorsf

#endif
```

#### 2. Create Silent Default Implementation
**File**: `src/core/Output.cpp` (new)
```cpp
#include "Output.h"

namespace aorsf {

class SilentOutput : public OutputHandler {
public:
    void print(const std::string&) override {}
    void println(const std::string&) override {}
};

std::shared_ptr<OutputHandler> OutputManager::handler_ =
    std::make_shared<SilentOutput>();

void OutputManager::set_handler(std::shared_ptr<OutputHandler> h) {
    handler_ = h ? h : std::make_shared<SilentOutput>();
}

OutputHandler& OutputManager::get() {
    return *handler_;
}

} // namespace aorsf
```

#### 3. Update Forest.cpp
**File**: `src/Forest.cpp`
**Changes**: Replace `Rcpp::Rcout` with `AORSF_OUT`

```cpp
// Before:
Rcpp::Rcout << "Growing trees: " << round(100 * relative_progress) << "%." << std::endl;

// After:
AORSF_OUT.println("Growing trees: " + std::to_string(round(100 * relative_progress)) + "%.");
```

#### 4. Create Rcpp Output Adapter
**File**: `src/rcpp/RcppOutput.h` (new)
```cpp
#ifndef AORSF_RCPP_OUTPUT_H_
#define AORSF_RCPP_OUTPUT_H_

#include <RcppArmadillo.h>
#include "../core/Output.h"

namespace aorsf {

class RcppOutput : public OutputHandler {
public:
    void print(const std::string& msg) override {
        Rcpp::Rcout << msg;
    }
    void println(const std::string& msg) override {
        Rcpp::Rcout << msg << std::endl;
    }
};

} // namespace aorsf

#endif
```

### Success Criteria

#### Automated Verification:
- [x] Package compiles: `Rcpp::compileAttributes() && devtools::load_all()`
- [x] All tests pass: `devtools::test()`
- [ ] R CMD check passes: `devtools::check()`
- [x] No Rcpp::Rcout in core files: `grep -r "Rcpp::Rcout" src/*.cpp | grep -v rcpp/`

#### Manual Verification:
- [ ] Verbose progress output still works in R: `orsf(pbc_orsf, Surv(time, status) ~ ., verbose_progress = TRUE, n_tree = 10)`

---

## Phase 2: Abstract Error Handling

### Overview
Replace `Rcpp::stop()` with C++ exceptions in core library. Rcpp wrapper catches and converts to R errors.

### Changes Required

#### 1. Create Exception Types
**File**: `src/core/Exceptions.h` (new)
```cpp
#ifndef AORSF_EXCEPTIONS_H_
#define AORSF_EXCEPTIONS_H_

#include <stdexcept>
#include <string>

namespace aorsf {

class aorsf_error : public std::runtime_error {
public:
    explicit aorsf_error(const std::string& msg)
        : std::runtime_error(msg) {}
};

class invalid_argument_error : public aorsf_error {
public:
    explicit invalid_argument_error(const std::string& msg)
        : aorsf_error(msg) {}
};

class computation_error : public aorsf_error {
public:
    explicit computation_error(const std::string& msg)
        : aorsf_error(msg) {}
};

} // namespace aorsf

#endif
```

#### 2. Update Tree.cpp
**File**: `src/Tree.cpp`
**Changes**: Replace `Rcpp::stop()` and `stop()` with exceptions

```cpp
// Before:
stop("attempting to allocate oob memory with empty rows_oobag");

// After:
throw aorsf::computation_error("attempting to allocate oob memory with empty rows_oobag");
```

#### 3. Update Rcpp Interface
**File**: `src/rcpp/orsf_rcpp.cpp` (refactored from orsf_oop.cpp)
**Changes**: Wrap core calls with try-catch

```cpp
// [[Rcpp::export]]
List orsf_cpp(...) {
    try {
        // ... call core library ...
    } catch (const aorsf::aorsf_error& e) {
        Rcpp::stop(e.what());
    } catch (const std::exception& e) {
        Rcpp::stop(std::string("Internal error: ") + e.what());
    }
}
```

### Success Criteria

#### Automated Verification:
- [x] Package compiles: `Rcpp::compileAttributes() && devtools::load_all()`
- [x] All tests pass: `devtools::test()`
- [x] Error tests still work: `expect_error()` tests in testthat
- [x] No Rcpp::stop in core: `grep -r "Rcpp::stop\|stop(" src/*.cpp | grep -v rcpp/`

#### Manual Verification:
- [ ] Invalid input produces correct R error message

---

## Phase 3: Abstract R Function Callbacks

### Overview
Replace `Rcpp::RObject` storage of R functions with C++ function pointer interfaces. This is the most complex phase as it affects the Tree and Forest class hierarchies.

### Changes Required

#### 1. Define Callback Interfaces
**File**: `src/core/Callbacks.h` (new)
```cpp
#ifndef AORSF_CALLBACKS_H_
#define AORSF_CALLBACKS_H_

#include <armadillo>
#include <functional>

namespace aorsf {

// Linear combination callback: computes coefficients from node data
using LinCombCallback = std::function<arma::mat(
    const arma::mat& x,  // predictors at node
    const arma::mat& y,  // outcomes at node
    const arma::vec& w   // weights at node
)>;

// OOB evaluation callback: computes custom accuracy metric
using OobagEvalCallback = std::function<double(
    const arma::mat& y,  // true outcomes
    const arma::vec& w,  // weights
    const arma::vec& p   // predictions
)>;

} // namespace aorsf

#endif
```

#### 2. Update Tree Class
**File**: `src/core/Tree.h` (moved from src/Tree.h)
**Changes**: Replace RObject with callback types

```cpp
// Before (in protected section):
Rcpp::RObject lincomb_R_function;
Rcpp::RObject oobag_R_function;

// After:
LinCombCallback lincomb_callback;
OobagEvalCallback oobag_callback;
```

#### 3. Update Tree.cpp Methods
**File**: `src/core/Tree.cpp`
**Changes**: Use callbacks instead of R function calls

```cpp
// Before (user_fit method):
NumericMatrix xx = wrap(x_node);
NumericMatrix yy = wrap(y_to_fit);
NumericVector ww = wrap(w_node);
Function f_beta = as<Function>(lincomb_R_function);
NumericMatrix out = f_beta(xx, yy, ww);
return as<mat>(out);

// After:
if (lincomb_callback) {
    return lincomb_callback(x_node, y_to_fit, w_node);
}
throw computation_error("No linear combination callback provided");
```

#### 4. Update Forest Class
**File**: `src/core/Forest.h`
**Changes**: Replace RObject with callback types, update init signature

```cpp
// Before:
Rcpp::RObject lincomb_R_function;
Rcpp::RObject oobag_R_function;

// After:
LinCombCallback lincomb_callback;
OobagEvalCallback oobag_callback;
```

#### 5. Create Rcpp Callback Adapters
**File**: `src/rcpp/RcppCallbacks.h` (new)
```cpp
#ifndef AORSF_RCPP_CALLBACKS_H_
#define AORSF_RCPP_CALLBACKS_H_

#include <RcppArmadillo.h>
#include "../core/Callbacks.h"

namespace aorsf {

// Wrap R function as LinCombCallback
inline LinCombCallback make_lincomb_callback(Rcpp::RObject r_func) {
    if (r_func.isNULL()) return nullptr;

    Rcpp::Function f = Rcpp::as<Rcpp::Function>(r_func);
    return [f](const arma::mat& x, const arma::mat& y, const arma::vec& w) -> arma::mat {
        Rcpp::NumericMatrix xx = Rcpp::wrap(x);
        Rcpp::NumericMatrix yy = Rcpp::wrap(y);
        Rcpp::NumericVector ww = Rcpp::wrap(w);
        Rcpp::NumericMatrix result = f(xx, yy, ww);
        return Rcpp::as<arma::mat>(result);
    };
}

// Wrap R function as OobagEvalCallback
inline OobagEvalCallback make_oobag_callback(Rcpp::RObject r_func) {
    if (r_func.isNULL()) return nullptr;

    Rcpp::Function f = Rcpp::as<Rcpp::Function>(r_func);
    return [f](const arma::mat& y, const arma::vec& w, const arma::vec& p) -> double {
        Rcpp::NumericMatrix yy = Rcpp::wrap(y);
        Rcpp::NumericVector ww = Rcpp::wrap(w);
        Rcpp::NumericVector pp = Rcpp::wrap(p);
        Rcpp::NumericVector result = f(yy, ww, pp);
        return result[0];
    };
}

} // namespace aorsf

#endif
```

#### 6. Update orsf_cpp Interface
**File**: `src/rcpp/orsf_rcpp.cpp`
**Changes**: Convert R functions to callbacks before passing to core

```cpp
// In orsf_cpp():
LinCombCallback lc_callback = make_lincomb_callback(lincomb_R_function);
OobagEvalCallback oob_callback = make_oobag_callback(oobag_R_function);

forest->init(..., lc_callback, oob_callback, ...);
```

### Success Criteria

#### Automated Verification:
- [ ] Package compiles: `Rcpp::compileAttributes() && devtools::load_all()`
- [ ] All tests pass: `devtools::test()`
- [ ] Custom lincomb tests pass: tests with `control = orsf_control_*(method = user_function)`
- [ ] Custom oobag tests pass: tests with `oobag_fun = custom_function`
- [ ] No RObject in core: `grep -r "RObject" src/core/`

#### Manual Verification:
- [ ] Custom linear combination function works
- [ ] Custom OOB evaluation function works

---

## Phase 4: Reorganize Directory Structure

### Overview
Physically separate core C++ files from Rcpp interface files into distinct directories.

### Changes Required

#### 1. Create Directory Structure
```
src/
├── core/                    # Pure C++ library
│   ├── Data.h/.cpp
│   ├── Tree.h/.cpp
│   ├── TreeSurvival.h/.cpp
│   ├── TreeClassification.h/.cpp
│   ├── TreeRegression.h/.cpp
│   ├── Forest.h/.cpp
│   ├── ForestSurvival.h/.cpp
│   ├── ForestClassification.h/.cpp
│   ├── ForestRegression.h/.cpp
│   ├── Coxph.h/.cpp
│   ├── utility.h/.cpp
│   ├── globals.h
│   ├── Output.h/.cpp
│   ├── Exceptions.h
│   └── Callbacks.h
├── rcpp/                    # Rcpp interface layer
│   ├── orsf_rcpp.cpp        # Main Rcpp exports
│   ├── RcppOutput.h
│   ├── RcppCallbacks.h
│   └── RcppExports.cpp      # Auto-generated
├── Makevars                 # Updated include paths
└── Makevars.win
```

#### 2. Update Makevars
**File**: `src/Makevars`
```make
PKG_CXXFLAGS = -I./core -I./rcpp $(SHLIB_OPENMP_CXXFLAGS)
PKG_LIBS = $(SHLIB_OPENMP_CXXFLAGS) $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)
```

#### 3. Update Include Paths
All files need updated `#include` statements:
```cpp
// Before:
#include "Tree.h"

// After (in rcpp/):
#include "../core/Tree.h"
// Or (in core/):
#include "Tree.h"
```

### Success Criteria

#### Automated Verification:
- [ ] Package compiles: `Rcpp::compileAttributes() && devtools::load_all()`
- [ ] All tests pass: `devtools::test()`
- [ ] R CMD check passes: `devtools::check()`
- [ ] Core directory has no Rcpp: `grep -r "Rcpp" src/core/`

#### Manual Verification:
- [ ] Directory structure matches specification

---

## Phase 5: Add Standalone Build Support

### Overview
Create a Makefile and test harness for building and testing the core library without R.

### Changes Required

#### 1. Create Standalone Makefile
**File**: `src/core/Makefile` (new)
```make
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -I/usr/include
LDFLAGS = -larmadillo

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
LIB = libaorsf.a

all: $(LIB)

$(LIB): $(OBJECTS)
	ar rcs $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(LIB)

test: $(LIB)
	$(CXX) $(CXXFLAGS) -o test_runner ../tests/test_main.cpp -L. -laorsf $(LDFLAGS)
	./test_runner

.PHONY: all clean test
```

#### 2. Create Basic Test File
**File**: `src/tests/test_main.cpp` (new)
```cpp
#include <iostream>
#include <cassert>
#include "../core/Data.h"
#include "../core/Tree.h"
#include "../core/globals.h"

void test_data_creation() {
    arma::mat x(100, 5, arma::fill::randn);
    arma::mat y(100, 2, arma::fill::randn);
    arma::vec w(100, arma::fill::ones);

    aorsf::Data data(x, y, w);

    assert(data.n_rows == 100);
    assert(data.n_cols_x == 5);
    std::cout << "test_data_creation: PASSED" << std::endl;
}

void test_globals() {
    assert(aorsf::DEFAULT_N_TREE == 500);
    assert(aorsf::DEFAULT_LEAF_MIN_OBS == 5);
    std::cout << "test_globals: PASSED" << std::endl;
}

int main() {
    std::cout << "Running standalone aorsf tests..." << std::endl;

    test_globals();
    test_data_creation();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
```

### Success Criteria

#### Automated Verification:
- [ ] Core library builds standalone: `cd src/core && make`
- [ ] Standalone tests pass: `cd src/core && make test`
- [ ] R package still works: `devtools::test()`

#### Manual Verification:
- [ ] Can link against libaorsf.a from external C++ code

---

## Phase 6: Simplify R Layer

### Overview
Reduce R6 class complexity now that most logic is in C++. Keep R layer focused on user interface, data validation, and result formatting.

### Changes Required

#### 1. Review R6 Methods
**File**: `R/orsf_R6.R`
**Changes**: Identify methods that can be simplified or removed

Methods to keep in R:
- `initialize()`: Parameter validation and defaults
- `print()`: User-friendly output formatting
- `predict()`: User interface with argument handling
- Data preparation methods

Methods that may move to C++:
- Complex computation currently done in R
- Any duplicate logic between R and C++

#### 2. Update RcppExports
**File**: `R/RcppExports.R`
Ensure auto-generated file matches new Rcpp interface.

### Success Criteria

#### Automated Verification:
- [ ] All tests pass: `devtools::test()`
- [ ] R CMD check passes: `devtools::check()`
- [ ] No new warnings

#### Manual Verification:
- [ ] All user-facing functions work identically to before

---

## Testing Strategy

### Unit Tests (C++)
- Test core library functions in isolation
- Test Data class construction and methods
- Test Tree/Forest algorithms with synthetic data
- Test error handling (exception throwing)

### Integration Tests (R)
- All existing testthat tests must pass
- Add tests for edge cases discovered during refactoring
- Performance regression tests

### Manual Testing Steps
1. Run full test suite: `devtools::test()`
2. Run R CMD check: `devtools::check()`
3. Test with custom R functions for lincomb and oobag
4. Verify verbose output works correctly
5. Benchmark performance against pre-refactor version

## Performance Considerations

- Callback mechanism adds minimal overhead (function pointer call)
- Output abstraction has negligible impact (only called during progress)
- Exception handling is zero-cost when no exception thrown (C++ standard)
- Directory reorganization has no runtime impact

## Migration Notes

- No changes to R-level API
- No changes to saved forest format (backward compatible)
- C++ API changes are internal only
- Users of custom control functions unaffected

## References

- Current architecture: `src/*.h`, `src/*.cpp`
- R6 classes: `R/orsf_R6.R`
- Recent refactoring: commit 9f6901b
- Main Rcpp bridge: `src/orsf_oop.cpp`
