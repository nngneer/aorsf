# Python Wrapper (pyaorsf) Implementation Plan

## Overview

Create a Python package `pyaorsf` that wraps the aorsf C++ core library using nanobind for bindings, carma for Armadillo↔NumPy conversion, and provides a scikit-learn compatible API for oblique random forests.

## Current State Analysis

The aorsf C++ core has been refactored to be ~95% pure C++ (7,745 lines in `src/core/`). The R interface layer (`src/orsf_rcpp.cpp`, 705 lines) demonstrates the binding pattern we'll follow for Python.

### Key Discoveries:

- `src/core/Callbacks.h:26-47` - Abstract callback interfaces using `std::function` (ready for Python)
- `src/core/Output.h:23-28` - Abstract output interface (ready for Python)
- `src/core/Exceptions.h` - C++ exception hierarchy (can map to Python exceptions)
- `src/core/Forest.h:39,283` - **Still uses `Rcpp::IntegerVector`** for tree seeds (needs fix)
- `src/core/Forest.cpp:316,443` - **Still uses `Rcpp::checkUserInterrupt()`** (needs abstraction)

### Dependencies to Add:

| Dependency | Version | Purpose |
|------------|---------|---------|
| nanobind | >=2.0 | C++→Python bindings (lighter than pybind11) |
| carma | >=0.6 | Armadillo↔NumPy automatic conversion |
| scikit-build-core | >=0.5 | Modern Python build system for C++ |
| NumPy | >=1.20 | Array interface |
| Armadillo | >=10.0 | Linear algebra (already required) |

## Desired End State

After implementation:

1. `pip install pyaorsf` works on Linux, macOS, Windows
2. Users can train oblique random forests with scikit-learn API:
   ```python
   from pyaorsf import ObliqueForestClassifier
   clf = ObliqueForestClassifier(n_trees=500)
   clf.fit(X_train, y_train)
   predictions = clf.predict(X_test)
   ```
3. Survival, classification, and regression all supported
4. Custom Python callbacks work for linear combinations and OOB evaluation
5. Variable importance and partial dependence available
6. Performance matches R package (same C++ core)

### Verification:

```bash
# Python package installs
pip install -e ./python

# Tests pass
pytest python/tests/

# Scikit-learn compatibility
python -c "from sklearn.utils.estimator_checks import check_estimator; from pyaorsf import ObliqueForestClassifier; check_estimator(ObliqueForestClassifier())"
```

## What We're NOT Doing

- Changing the R package API or functionality
- Rewriting C++ algorithms (using existing core as-is)
- Supporting Python < 3.9
- Implementing survival analysis scikit-learn integration (no standard exists)
- Creating conda packages (future work)
- GPU acceleration (future work)

## Implementation Approach

Use a layered architecture:

```
┌─────────────────────────────────────────┐
│  Python API (scikit-learn compatible)   │  pyaorsf/*.py
├─────────────────────────────────────────┤
│  nanobind bindings + carma conversion   │  python/src/_pyaorsf.cpp
├─────────────────────────────────────────┤
│  Python-specific adapters               │  python/src/python/*.h
│  (PythonOutput, PythonCallbacks)        │
├─────────────────────────────────────────┤
│  Pure C++ Core (unchanged)              │  src/core/*.cpp
└─────────────────────────────────────────┘
```

---

## Phase 1: Fix Remaining Core Dependencies

### Overview

Remove the last Rcpp dependencies from `src/core/` to make it 100% standalone C++. This is a prerequisite for clean Python bindings.

### Changes Required:

#### 1. Replace Rcpp::IntegerVector with std::vector<int>

**File**: `src/core/Forest.h`
**Changes**: Replace `Rcpp::IntegerVector` with `std::vector<int>`

```cpp
// Line 39 - In init() signature, change:
// Before:
void init(std::unique_ptr<Data> input_data,
          Rcpp::IntegerVector& tree_seeds,
          ...

// After:
void init(std::unique_ptr<Data> input_data,
          std::vector<int>& tree_seeds,
          ...
```

```cpp
// Line 283 - In member variables, change:
// Before:
Rcpp::IntegerVector tree_seeds;

// After:
std::vector<int> tree_seeds;
```

**File**: `src/core/Forest.cpp`
**Changes**: Update init() signature to match

```cpp
// Line 15 - Change parameter type:
void Forest::init(std::unique_ptr<Data> input_data,
                  std::vector<int>& tree_seeds,
                  ...
```

#### 2. Abstract User Interrupt Checking

**File**: `src/core/Interrupts.h` (new)
```cpp
#ifndef AORSF_INTERRUPTS_H_
#define AORSF_INTERRUPTS_H_

#include <functional>
#include <memory>
#include <atomic>

namespace aorsf {

/**
 * @brief Abstract interface for checking user interrupts.
 *
 * This allows the core library to check for interrupts without
 * depending on R/Rcpp. Different implementations can be provided
 * for R (using Rcpp::checkUserInterrupt) or Python (using PyErr_CheckSignals).
 */
class InterruptHandler {
public:
    virtual ~InterruptHandler() = default;

    /**
     * @brief Check if the user has requested an interrupt.
     * @return true if interrupted, false otherwise
     */
    virtual bool check() = 0;
};

/**
 * @brief Default interrupt handler that never interrupts.
 */
class NoInterrupt : public InterruptHandler {
public:
    bool check() override { return false; }
};

/**
 * @brief Global interrupt manager for accessing the current handler.
 */
class InterruptManager {
public:
    static void set_handler(std::shared_ptr<InterruptHandler> handler) {
        get_handler_ptr() = handler ? handler : std::make_shared<NoInterrupt>();
    }

    static bool check() {
        return get_handler_ptr()->check();
    }

private:
    static std::shared_ptr<InterruptHandler>& get_handler_ptr() {
        static std::shared_ptr<InterruptHandler> handler = std::make_shared<NoInterrupt>();
        return handler;
    }
};

// Convenience macro
#define AORSF_CHECK_INTERRUPT() if (aorsf::InterruptManager::check()) return

} // namespace aorsf

#endif // AORSF_INTERRUPTS_H_
```

**File**: `src/core/Forest.cpp`
**Changes**: Replace `Rcpp::checkUserInterrupt()` with abstract check

```cpp
// Add include at top:
#include "Interrupts.h"

// Line 316 - Replace:
// Before:
Rcpp::checkUserInterrupt();

// After:
if (InterruptManager::check()) {
    throw aorsf::computation_error("User interrupt");
}

// Line 443 - Same replacement
```

#### 3. Update R Interface to Use New Types

**File**: `src/orsf_rcpp.cpp`
**Changes**: Convert R types at boundary

```cpp
// Before calling forest->init(), convert IntegerVector to std::vector<int>:
std::vector<int> seeds_vec(tree_seeds.begin(), tree_seeds.end());

forest->init(std::move(data),
             seeds_vec,  // Changed from tree_seeds
             ...
```

**File**: `src/rcpp/RcppInterrupts.h` (new)
```cpp
#ifndef AORSF_RCPP_INTERRUPTS_H_
#define AORSF_RCPP_INTERRUPTS_H_

#include <RcppArmadillo.h>
#include "../core/Interrupts.h"

namespace aorsf {

class RcppInterrupt : public InterruptHandler {
public:
    bool check() override {
        try {
            Rcpp::checkUserInterrupt();
            return false;
        } catch (...) {
            return true;
        }
    }
};

inline void init_r_interrupt() {
    InterruptManager::set_handler(std::make_shared<RcppInterrupt>());
}

} // namespace aorsf

#endif // AORSF_RCPP_INTERRUPTS_H_
```

**File**: `src/orsf_rcpp.cpp`
**Changes**: Initialize R interrupt handler

```cpp
// Add include:
#include "rcpp/RcppInterrupts.h"

// In orsf_cpp(), after init_r_output():
init_r_output();
init_r_interrupt();  // Add this line
```

#### 4. Remove Rcpp includes from core headers

**File**: `src/core/Forest.h`
**Changes**: Remove Rcpp include

```cpp
// Remove this line if present:
// #include <RcppArmadillo.h>

// Add if needed:
#include <vector>
```

### Success Criteria:

#### Automated Verification:
- [ ] No Rcpp in core: `grep -r "Rcpp::" src/core/ | grep -v "^src/core/.*:.*//"`
- [ ] R package compiles: `Rcpp::compileAttributes() && devtools::load_all()`
- [ ] All R tests pass: `devtools::test()`
- [ ] R CMD check passes: `devtools::check()`

#### Manual Verification:
- [ ] Verify interrupt handling works: run long forest, press Ctrl+C in R
- [ ] Confirm tree seeds produce reproducible results

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to Phase 2.

---

## Phase 2: Standalone Build System

### Overview

Add CMake build system to compile the C++ core as a standalone static library, verifying it has no R dependencies.

### Changes Required:

#### 1. Create Core CMakeLists.txt

**File**: `src/core/CMakeLists.txt` (new)
```cmake
cmake_minimum_required(VERSION 3.15)
project(aorsf_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Find OpenMP (optional)
find_package(OpenMP)

# Core library sources
set(AORSF_CORE_SOURCES
    Coxph.cpp
    Forest.cpp
    ForestClassification.cpp
    ForestRegression.cpp
    ForestSurvival.cpp
    Tree.cpp
    TreeClassification.cpp
    TreeRegression.cpp
    TreeSurvival.cpp
    utility.cpp
)

# Core library headers
set(AORSF_CORE_HEADERS
    Callbacks.h
    Coxph.h
    Data.h
    Exceptions.h
    Forest.h
    ForestClassification.h
    ForestRegression.h
    ForestSurvival.h
    globals.h
    Interrupts.h
    Output.h
    Tree.h
    TreeClassification.h
    TreeRegression.h
    TreeSurvival.h
    utility.h
)

# Create static library
add_library(aorsf_core STATIC ${AORSF_CORE_SOURCES})

target_include_directories(aorsf_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/aorsf>
    PRIVATE
        ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(aorsf_core
    PUBLIC
        ${ARMADILLO_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(aorsf_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Install rules
install(TARGETS aorsf_core
    EXPORT aorsf_coreTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${AORSF_CORE_HEADERS}
    DESTINATION include/aorsf
)

# Export for find_package
install(EXPORT aorsf_coreTargets
    FILE aorsf_coreTargets.cmake
    NAMESPACE aorsf::
    DESTINATION lib/cmake/aorsf_core
)
```

#### 2. Create Test to Verify Standalone Compilation

**File**: `src/core/tests/test_standalone.cpp` (new)
```cpp
/**
 * @brief Standalone compilation test for aorsf core.
 *
 * This test verifies that the core library can be compiled and linked
 * without any R/Rcpp dependencies.
 */

#include <iostream>
#include <cassert>

#include "Data.h"
#include "Forest.h"
#include "ForestSurvival.h"
#include "ForestClassification.h"
#include "ForestRegression.h"
#include "globals.h"
#include "Exceptions.h"
#include "Output.h"
#include "Interrupts.h"

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

void test_forest_creation() {
    // Test that forest classes can be instantiated
    ForestSurvival fs(1.0, 1.0, arma::vec({30, 60, 90}));
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
    test_forest_creation();

    std::cout << "======================================" << std::endl;
    std::cout << "All standalone tests passed!" << std::endl;

    return 0;
}
```

#### 3. Add Test CMakeLists.txt

**File**: `src/core/tests/CMakeLists.txt` (new)
```cmake
# Standalone test executable
add_executable(test_standalone test_standalone.cpp)
target_link_libraries(test_standalone PRIVATE aorsf_core)

# Enable testing
enable_testing()
add_test(NAME standalone_test COMMAND test_standalone)
```

#### 4. Update Core CMakeLists.txt for Tests

**File**: `src/core/CMakeLists.txt`
**Changes**: Add at end of file

```cmake
# Optional: Build tests
option(BUILD_TESTS "Build standalone tests" OFF)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

### Success Criteria:

#### Automated Verification:
- [ ] Core compiles standalone:
  ```bash
  cd src/core
  mkdir -p build && cd build
  cmake .. -DBUILD_TESTS=ON
  make
  ```
- [ ] Standalone tests pass: `./tests/test_standalone`
- [ ] No R symbols in library: `nm libaorsf_core.a | grep -i rcpp` (should be empty)
- [ ] R package still works: `devtools::test()`

#### Manual Verification:
- [ ] Verify library can be linked from external C++ project

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to Phase 3.

---

## Phase 3: Python Package Structure

### Overview

Create the pyaorsf Python package structure with scikit-build-core for building C++ extensions with nanobind.

### Changes Required:

#### 1. Create Python Package Directory Structure

```
python/
├── pyproject.toml
├── CMakeLists.txt
├── README.md
├── src/
│   ├── _pyaorsf.cpp           # nanobind bindings
│   └── python/
│       ├── PythonOutput.h     # Python output handler
│       ├── PythonCallbacks.h  # Python callback wrappers
│       └── PythonInterrupts.h # Python interrupt handler
├── pyaorsf/
│   ├── __init__.py
│   ├── _version.py
│   ├── forest.py              # Scikit-learn compatible classes
│   ├── survival.py            # Survival-specific classes
│   └── utils.py               # Utilities
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_classification.py
    ├── test_regression.py
    └── test_survival.py
```

#### 2. Create pyproject.toml

**File**: `python/pyproject.toml` (new)
```toml
[build-system]
requires = ["scikit-build-core>=0.5", "nanobind>=2.0"]
build-backend = "scikit_build_core.build"

[project]
name = "pyaorsf"
version = "0.1.0"
description = "Accelerated Oblique Random Forests for Python"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Byron C. Jaeger", email = "bjaeger@wakehealth.edu"}
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "scikit-learn>=1.0",  # For testing sklearn compatibility
]
docs = [
    "sphinx",
    "sphinx-rtd-theme",
]

[project.urls]
Homepage = "https://github.com/ropensci/aorsf"
Documentation = "https://docs.ropensci.org/aorsf/"
Repository = "https://github.com/ropensci/aorsf"

[tool.scikit-build]
cmake.minimum-version = "3.15"
cmake.build-type = "Release"
wheel.packages = ["pyaorsf"]

[tool.scikit-build.cmake.define]
BUILD_PYTHON = "ON"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

#### 3. Create Python CMakeLists.txt

**File**: `python/CMakeLists.txt` (new)
```cmake
cmake_minimum_required(VERSION 3.15)
project(pyaorsf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Find OpenMP (optional)
find_package(OpenMP)

# Include carma for Armadillo <-> NumPy conversion
include(FetchContent)
FetchContent_Declare(
    carma
    GIT_REPOSITORY https://github.com/RUrlus/carma.git
    GIT_TAG v0.6.7
)
FetchContent_MakeAvailable(carma)

# Path to aorsf core
set(AORSF_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/core")

# Core library sources (compile directly, don't require pre-built library)
set(AORSF_CORE_SOURCES
    ${AORSF_CORE_DIR}/Coxph.cpp
    ${AORSF_CORE_DIR}/Forest.cpp
    ${AORSF_CORE_DIR}/ForestClassification.cpp
    ${AORSF_CORE_DIR}/ForestRegression.cpp
    ${AORSF_CORE_DIR}/ForestSurvival.cpp
    ${AORSF_CORE_DIR}/Tree.cpp
    ${AORSF_CORE_DIR}/TreeClassification.cpp
    ${AORSF_CORE_DIR}/TreeRegression.cpp
    ${AORSF_CORE_DIR}/TreeSurvival.cpp
    ${AORSF_CORE_DIR}/utility.cpp
)

# Python binding sources
set(PYAORSF_SOURCES
    src/_pyaorsf.cpp
)

# Create the Python module
nanobind_add_module(_pyaorsf
    ${AORSF_CORE_SOURCES}
    ${PYAORSF_SOURCES}
)

target_include_directories(_pyaorsf PRIVATE
    ${AORSF_CORE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/python
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(_pyaorsf PRIVATE
    ${ARMADILLO_LIBRARIES}
    carma::carma
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_pyaorsf PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install the module
install(TARGETS _pyaorsf LIBRARY DESTINATION pyaorsf)
```

#### 4. Create Python Package __init__.py

**File**: `python/pyaorsf/__init__.py` (new)
```python
"""
pyaorsf: Accelerated Oblique Random Forests for Python
======================================================

A Python interface to the aorsf C++ library for fitting oblique random forests.
Provides scikit-learn compatible estimators for classification, regression,
and survival analysis.

Main Classes
------------
ObliqueForestClassifier
    Oblique random forest for classification tasks.
ObliqueForestRegressor
    Oblique random forest for regression tasks.
ObliqueForestSurvival
    Oblique random survival forest for time-to-event data.

Examples
--------
>>> from pyaorsf import ObliqueForestClassifier
>>> clf = ObliqueForestClassifier(n_trees=100)
>>> clf.fit(X_train, y_train)
>>> predictions = clf.predict(X_test)
"""

from ._version import __version__
from .forest import ObliqueForestClassifier, ObliqueForestRegressor
from .survival import ObliqueForestSurvival

__all__ = [
    "__version__",
    "ObliqueForestClassifier",
    "ObliqueForestRegressor",
    "ObliqueForestSurvival",
]
```

#### 5. Create Version File

**File**: `python/pyaorsf/_version.py` (new)
```python
__version__ = "0.1.0"
```

#### 6. Create Placeholder Binding File

**File**: `python/src/_pyaorsf.cpp` (new)
```cpp
/**
 * @brief nanobind bindings for pyaorsf
 *
 * This file creates Python bindings for the aorsf C++ core library
 * using nanobind and carma for Armadillo <-> NumPy conversion.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

// Placeholder - will be expanded in Phase 4
namespace nb = nanobind;

NB_MODULE(_pyaorsf, m) {
    m.doc() = "Python bindings for aorsf C++ core library";

    // Version info
    m.attr("__version__") = "0.1.0";

    // Placeholder - bindings will be added in Phase 4
}
```

#### 7. Create Python README

**File**: `python/README.md` (new)
```markdown
# pyaorsf

Python interface to Accelerated Oblique Random Forests.

## Installation

```bash
pip install pyaorsf
```

## Quick Start

```python
from pyaorsf import ObliqueForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train model
clf = ObliqueForestClassifier(n_trees=100, n_threads=4)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
```

## Features

- Oblique (linear combination) splits for better decision boundaries
- Survival analysis with accelerated Cox regression
- Variable importance (negation, permutation, ANOVA methods)
- Partial dependence plots
- Full scikit-learn compatibility
- Multi-threaded training and prediction

## Requirements

- Python >= 3.9
- NumPy >= 1.20
- Armadillo (system library)
- OpenMP (optional, for parallelization)
```

### Success Criteria:

#### Automated Verification:
- [ ] Package structure created: `ls python/pyaorsf/__init__.py`
- [ ] pyproject.toml is valid: `pip install build && python -m build --sdist python/`
- [ ] Python imports work (after Phase 4): `python -c "import pyaorsf"`

#### Manual Verification:
- [ ] Directory structure matches specification above

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to Phase 4.

---

## Phase 4: Core Bindings

### Overview

Create the nanobind bindings for Forest classes, Data container, and utility functions. This is the main C++→Python bridge.

### Changes Required:

#### 1. Create Python Output Handler

**File**: `python/src/python/PythonOutput.h` (new)
```cpp
#ifndef PYAORSF_PYTHON_OUTPUT_H_
#define PYAORSF_PYTHON_OUTPUT_H_

#include <nanobind/nanobind.h>
#include "Output.h"
#include <iostream>

namespace nb = nanobind;

namespace aorsf {

/**
 * @brief Python output handler using sys.stdout
 */
class PythonOutput : public OutputHandler {
public:
    void print(const std::string& msg) override {
        nb::gil_scoped_acquire guard;
        nb::module_ sys = nb::module_::import_("sys");
        nb::object stdout = sys.attr("stdout");
        stdout.attr("write")(msg);
        stdout.attr("flush")();
    }

    void println(const std::string& msg) override {
        print(msg + "\n");
    }
};

/**
 * @brief Silent output for when verbose=False
 */
inline void init_python_output(bool verbose) {
    if (verbose) {
        OutputManager::set_handler(std::make_shared<PythonOutput>());
    } else {
        OutputManager::set_handler(nullptr);  // Silent
    }
}

} // namespace aorsf

#endif // PYAORSF_PYTHON_OUTPUT_H_
```

#### 2. Create Python Interrupt Handler

**File**: `python/src/python/PythonInterrupts.h` (new)
```cpp
#ifndef PYAORSF_PYTHON_INTERRUPTS_H_
#define PYAORSF_PYTHON_INTERRUPTS_H_

#include <nanobind/nanobind.h>
#include "Interrupts.h"

namespace nb = nanobind;

namespace aorsf {

/**
 * @brief Python interrupt handler using PyErr_CheckSignals
 */
class PythonInterrupt : public InterruptHandler {
public:
    bool check() override {
        nb::gil_scoped_acquire guard;
        if (PyErr_CheckSignals() != 0) {
            PyErr_Clear();  // Clear the error, we'll handle it
            return true;
        }
        return false;
    }
};

inline void init_python_interrupt() {
    InterruptManager::set_handler(std::make_shared<PythonInterrupt>());
}

} // namespace aorsf

#endif // PYAORSF_PYTHON_INTERRUPTS_H_
```

#### 3. Create Python Callback Wrappers

**File**: `python/src/python/PythonCallbacks.h` (new)
```cpp
#ifndef PYAORSF_PYTHON_CALLBACKS_H_
#define PYAORSF_PYTHON_CALLBACKS_H_

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <carma>
#include "Callbacks.h"

namespace nb = nanobind;

namespace aorsf {

/**
 * @brief Create a LinCombCallback from a Python callable.
 *
 * @param py_func Python function taking (x, y, w) numpy arrays
 * @return LinCombCallback that wraps the Python function
 */
inline LinCombCallback make_python_lincomb_callback(nb::callable py_func) {
    if (py_func.is_none()) {
        return nullptr;
    }

    return [py_func](const arma::mat& x, const arma::mat& y, const arma::vec& w) -> arma::mat {
        nb::gil_scoped_acquire guard;

        // Convert Armadillo to NumPy using carma
        nb::ndarray<double> x_np = carma::mat_to_arr(x);
        nb::ndarray<double> y_np = carma::mat_to_arr(y);
        nb::ndarray<double> w_np = carma::col_to_arr(w);

        // Call Python function
        nb::object result = py_func(x_np, y_np, w_np);

        // Convert result back to Armadillo
        return carma::arr_to_mat<double>(nb::cast<nb::ndarray<double>>(result));
    };
}

/**
 * @brief Create an OobagEvalCallback from a Python callable.
 *
 * @param py_func Python function taking (y, w, p) numpy arrays
 * @return OobagEvalCallback that wraps the Python function
 */
inline OobagEvalCallback make_python_oobag_callback(nb::callable py_func) {
    if (py_func.is_none()) {
        return nullptr;
    }

    return [py_func](const arma::mat& y, const arma::vec& w, const arma::vec& p) -> double {
        nb::gil_scoped_acquire guard;

        // Convert Armadillo to NumPy using carma
        nb::ndarray<double> y_np = carma::mat_to_arr(y);
        nb::ndarray<double> w_np = carma::col_to_arr(w);
        nb::ndarray<double> p_np = carma::col_to_arr(p);

        // Call Python function
        nb::object result = py_func(y_np, w_np, p_np);

        // Return scalar result
        return nb::cast<double>(result);
    };
}

} // namespace aorsf

#endif // PYAORSF_PYTHON_CALLBACKS_H_
```

#### 4. Create Main Binding File

**File**: `python/src/_pyaorsf.cpp` (replace placeholder)
```cpp
/**
 * @brief nanobind bindings for pyaorsf
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <carma>

#include "globals.h"
#include "Data.h"
#include "Forest.h"
#include "ForestSurvival.h"
#include "ForestClassification.h"
#include "ForestRegression.h"
#include "Exceptions.h"

#include "python/PythonOutput.h"
#include "python/PythonInterrupts.h"
#include "python/PythonCallbacks.h"

namespace nb = nanobind;
using namespace aorsf;

// Helper to convert numpy array to arma::mat
arma::mat numpy_to_mat(nb::ndarray<double, nb::c_contig> arr) {
    return carma::arr_to_mat<double>(arr);
}

// Helper to convert numpy array to arma::vec
arma::vec numpy_to_vec(nb::ndarray<double, nb::c_contig> arr) {
    return carma::arr_to_col<double>(arr);
}

// Helper to convert arma::mat to numpy
nb::ndarray<nb::numpy, double> mat_to_numpy(const arma::mat& m) {
    return carma::mat_to_arr(m);
}

/**
 * @brief Low-level forest training function
 */
nb::dict fit_forest(
    nb::ndarray<double, nb::c_contig> x_np,
    nb::ndarray<double, nb::c_contig> y_np,
    nb::ndarray<double, nb::c_contig> w_np,
    int tree_type,
    std::vector<int> tree_seeds,
    int n_tree,
    int mtry,
    bool sample_with_replacement,
    double sample_fraction,
    int vi_type,
    double vi_max_pvalue,
    double leaf_min_events,
    double leaf_min_obs,
    int split_rule,
    double split_min_events,
    double split_min_obs,
    double split_min_stat,
    int split_max_cuts,
    int split_max_retry,
    int lincomb_type,
    double lincomb_eps,
    int lincomb_iter_max,
    bool lincomb_scale,
    double lincomb_alpha,
    int lincomb_df_target,
    int lincomb_ties_method,
    nb::object lincomb_py_function,
    std::vector<double> pred_horizon,
    int pred_type,
    bool oobag,
    int oobag_eval_type,
    int oobag_eval_every,
    nb::object oobag_py_function,
    int n_thread,
    bool verbose
) {
    // Initialize Python output and interrupt handlers
    init_python_output(verbose);
    init_python_interrupt();

    try {
        // Convert numpy arrays to Armadillo
        arma::mat x = numpy_to_mat(x_np);
        arma::mat y = numpy_to_mat(y_np);
        arma::vec w = numpy_to_vec(w_np);

        // Create data object
        auto data = std::make_unique<Data>(x, y, w);
        arma::uword n_obs = data->get_n_rows();

        // Handle thread count
        if (n_thread == 0) {
            n_thread = std::thread::hardware_concurrency();
        }

        // Create forest based on type
        std::unique_ptr<Forest> forest;
        TreeType tt = static_cast<TreeType>(tree_type);

        switch (tt) {
            case TREE_SURVIVAL: {
                arma::vec horizon(pred_horizon);
                forest = std::make_unique<ForestSurvival>(
                    leaf_min_events, split_min_events, horizon
                );
                break;
            }
            case TREE_CLASSIFICATION:
                forest = std::make_unique<ForestClassification>(data->n_cols_y);
                break;
            case TREE_REGRESSION:
                forest = std::make_unique<ForestRegression>();
                break;
            default:
                throw invalid_argument_error("Unknown tree type");
        }

        // Create callbacks
        LinCombCallback lc_callback = nullptr;
        if (lincomb_type == LC_R_FUNCTION && !lincomb_py_function.is_none()) {
            lc_callback = make_python_lincomb_callback(
                nb::cast<nb::callable>(lincomb_py_function)
            );
        }

        OobagEvalCallback oob_callback = nullptr;
        if (oobag_eval_type == EVAL_R_FUNCTION && !oobag_py_function.is_none()) {
            oob_callback = make_python_oobag_callback(
                nb::cast<nb::callable>(oobag_py_function)
            );
        }

        // Empty partial dependence containers
        std::vector<arma::mat> pd_x_vals;
        std::vector<arma::uvec> pd_x_cols;
        arma::vec pd_probs;

        // Initialize forest
        forest->init(
            std::move(data),
            tree_seeds,
            n_tree,
            mtry,
            sample_with_replacement,
            sample_fraction,
            true,  // grow_mode
            static_cast<VariableImportance>(vi_type),
            vi_max_pvalue,
            leaf_min_obs,
            static_cast<SplitRule>(split_rule),
            split_min_obs,
            split_min_stat,
            split_max_cuts,
            split_max_retry,
            static_cast<LinearCombo>(lincomb_type),
            lincomb_eps,
            lincomb_iter_max,
            lincomb_scale,
            lincomb_alpha,
            lincomb_df_target,
            lincomb_ties_method,
            lc_callback,
            static_cast<PredType>(pred_type),
            false,  // pred_mode
            true,   // pred_aggregate
            PD_NONE,
            pd_x_vals,
            pd_x_cols,
            pd_probs,
            oobag,
            static_cast<EvalType>(oobag_eval_type),
            oobag_eval_every,
            oob_callback,
            n_thread,
            verbose ? 1 : 0
        );

        // Run forest
        forest->run(oobag);

        // Build result dictionary
        nb::dict result;

        // Predictions
        if (oobag) {
            result["oob_predictions"] = mat_to_numpy(forest->get_predictions());
        }

        // OOB evaluation
        result["oob_eval"] = mat_to_numpy(forest->get_oobag_eval());

        // Variable importance
        if (vi_type != VI_NONE) {
            arma::vec vi;
            if (vi_type == VI_ANOVA) {
                arma::uvec denom = forest->get_vi_denom();
                arma::uvec zeros = arma::find(denom == 0);
                if (zeros.n_elem > 0) denom(zeros).fill(1);
                vi = forest->get_vi_numer() / arma::conv_to<arma::vec>::from(denom);
            } else {
                vi = forest->get_vi_numer() / n_tree;
            }
            result["importance"] = carma::col_to_arr(vi);
        }

        // Forest structure for serialization
        nb::dict forest_data;
        forest_data["n_obs"] = n_obs;
        forest_data["n_tree"] = n_tree;
        forest_data["cutpoint"] = forest->get_cutpoint();
        forest_data["child_left"] = forest->get_child_left();
        forest_data["coef_values"] = forest->get_coef_values();
        forest_data["coef_indices"] = forest->get_coef_indices();
        forest_data["leaf_summary"] = forest->get_leaf_summary();
        forest_data["rows_oobag"] = forest->get_rows_oobag();
        forest_data["oobag_denom"] = carma::col_to_arr(forest->get_oobag_denom());

        // Type-specific data
        if (tt == TREE_SURVIVAL) {
            auto& fs = dynamic_cast<ForestSurvival&>(*forest);
            forest_data["leaf_pred_indx"] = fs.get_leaf_pred_indx();
            forest_data["leaf_pred_prob"] = fs.get_leaf_pred_prob();
            forest_data["leaf_pred_chaz"] = fs.get_leaf_pred_chaz();
        } else if (tt == TREE_CLASSIFICATION) {
            auto& fc = dynamic_cast<ForestClassification&>(*forest);
            forest_data["leaf_pred_prob"] = fc.get_leaf_pred_prob();
        } else if (tt == TREE_REGRESSION) {
            auto& fr = dynamic_cast<ForestRegression&>(*forest);
            forest_data["leaf_pred_prob"] = fr.get_leaf_pred_prob();
        }

        result["forest"] = forest_data;

        return result;

    } catch (const aorsf_error& e) {
        throw std::runtime_error(e.what());
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Internal error: ") + e.what());
    }
}

/**
 * @brief Low-level prediction function
 */
nb::ndarray<nb::numpy, double> predict_forest(
    nb::ndarray<double, nb::c_contig> x_np,
    nb::dict forest_data,
    int tree_type,
    std::vector<double> pred_horizon,
    int pred_type,
    bool pred_aggregate,
    int n_thread
) {
    init_python_output(false);
    init_python_interrupt();

    try {
        arma::mat x = numpy_to_mat(x_np);
        arma::uword n_obs = x.n_rows;

        // Create dummy y and w for prediction
        arma::mat y(n_obs, 2, arma::fill::zeros);
        arma::vec w(n_obs, arma::fill::ones);

        auto data = std::make_unique<Data>(x, y, w);

        // Handle threads
        if (n_thread == 0) {
            n_thread = std::thread::hardware_concurrency();
        }

        // Create forest
        std::unique_ptr<Forest> forest;
        TreeType tt = static_cast<TreeType>(tree_type);

        switch (tt) {
            case TREE_SURVIVAL: {
                arma::vec horizon(pred_horizon);
                forest = std::make_unique<ForestSurvival>(1.0, 1.0, horizon);
                break;
            }
            case TREE_CLASSIFICATION: {
                int n_class = nb::cast<int>(forest_data["n_class"]);
                forest = std::make_unique<ForestClassification>(n_class);
                break;
            }
            case TREE_REGRESSION:
                forest = std::make_unique<ForestRegression>();
                break;
            default:
                throw invalid_argument_error("Unknown tree type");
        }

        // Initialize for prediction mode
        std::vector<int> dummy_seeds;
        std::vector<arma::mat> pd_x_vals;
        std::vector<arma::uvec> pd_x_cols;
        arma::vec pd_probs;

        forest->init(
            std::move(data),
            dummy_seeds,
            nb::cast<int>(forest_data["n_tree"]),
            1,  // mtry (unused in pred mode)
            true, 0.632,  // sampling params (unused)
            false,  // grow_mode = false (prediction mode)
            VI_NONE, 0.0,
            1.0,  // leaf_min_obs
            SPLIT_LOGRANK, 1.0, 0.0, 5, 3,
            LC_GLM, 1e-9, 20, true, 0.5, 4, 1,
            nullptr,  // no callback needed
            static_cast<PredType>(pred_type),
            true,  // pred_mode
            pred_aggregate,
            PD_NONE, pd_x_vals, pd_x_cols, pd_probs,
            false, EVAL_NONE, 0, nullptr,
            n_thread, 0
        );

        // Load forest from dictionary
        // (Detailed loading code would extract cutpoint, child_left, etc.)
        // This is simplified - full implementation needs forest->load()

        forest->run(false);

        return mat_to_numpy(forest->get_predictions());

    } catch (const aorsf_error& e) {
        throw std::runtime_error(e.what());
    }
}

NB_MODULE(_pyaorsf, m) {
    m.doc() = "Python bindings for aorsf C++ core library";
    m.attr("__version__") = "0.1.0";

    // Expose enums
    nb::enum_<TreeType>(m, "TreeType")
        .value("CLASSIFICATION", TREE_CLASSIFICATION)
        .value("REGRESSION", TREE_REGRESSION)
        .value("SURVIVAL", TREE_SURVIVAL);

    nb::enum_<VariableImportance>(m, "VariableImportance")
        .value("NONE", VI_NONE)
        .value("NEGATE", VI_NEGATE)
        .value("PERMUTE", VI_PERMUTE)
        .value("ANOVA", VI_ANOVA);

    nb::enum_<SplitRule>(m, "SplitRule")
        .value("LOGRANK", SPLIT_LOGRANK)
        .value("CONCORD", SPLIT_CONCORD)
        .value("GINI", SPLIT_GINI)
        .value("VARIANCE", SPLIT_VARIANCE);

    nb::enum_<LinearCombo>(m, "LinearCombo")
        .value("GLM", LC_GLM)
        .value("RANDOM", LC_RANDOM_COEFS)
        .value("GLMNET", LC_GLMNET)
        .value("CUSTOM", LC_R_FUNCTION);

    nb::enum_<PredType>(m, "PredType")
        .value("NONE", PRED_NONE)
        .value("RISK", PRED_RISK)
        .value("SURVIVAL", PRED_SURVIVAL)
        .value("CHAZ", PRED_CHAZ)
        .value("MORTALITY", PRED_MORTALITY)
        .value("MEAN", PRED_MEAN)
        .value("PROBABILITY", PRED_PROBABILITY)
        .value("CLASS", PRED_CLASS);

    // Main functions
    m.def("fit_forest", &fit_forest,
          "Fit an oblique random forest",
          nb::arg("x"), nb::arg("y"), nb::arg("w"),
          nb::arg("tree_type"), nb::arg("tree_seeds"),
          nb::arg("n_tree"), nb::arg("mtry"),
          nb::arg("sample_with_replacement"), nb::arg("sample_fraction"),
          nb::arg("vi_type"), nb::arg("vi_max_pvalue"),
          nb::arg("leaf_min_events"), nb::arg("leaf_min_obs"),
          nb::arg("split_rule"),
          nb::arg("split_min_events"), nb::arg("split_min_obs"),
          nb::arg("split_min_stat"), nb::arg("split_max_cuts"),
          nb::arg("split_max_retry"),
          nb::arg("lincomb_type"), nb::arg("lincomb_eps"),
          nb::arg("lincomb_iter_max"), nb::arg("lincomb_scale"),
          nb::arg("lincomb_alpha"), nb::arg("lincomb_df_target"),
          nb::arg("lincomb_ties_method"), nb::arg("lincomb_py_function"),
          nb::arg("pred_horizon"), nb::arg("pred_type"),
          nb::arg("oobag"), nb::arg("oobag_eval_type"),
          nb::arg("oobag_eval_every"), nb::arg("oobag_py_function"),
          nb::arg("n_thread"), nb::arg("verbose"));

    m.def("predict_forest", &predict_forest,
          "Generate predictions from a fitted forest",
          nb::arg("x"), nb::arg("forest_data"),
          nb::arg("tree_type"), nb::arg("pred_horizon"),
          nb::arg("pred_type"), nb::arg("pred_aggregate"),
          nb::arg("n_thread"));
}
```

### Success Criteria:

#### Automated Verification:
- [ ] Python module compiles: `pip install -e python/`
- [ ] Module imports: `python -c "from pyaorsf import _pyaorsf; print(_pyaorsf.__version__)"`
- [ ] Enums accessible: `python -c "from pyaorsf._pyaorsf import TreeType; print(TreeType.SURVIVAL)"`

#### Manual Verification:
- [ ] Low-level fit_forest function works with test data
- [ ] Low-level predict_forest function works with fitted forest

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to Phase 5.

---

## Phase 5: Scikit-learn API

### Overview

Implement the high-level Python classes with scikit-learn compatible interface: `ObliqueForestClassifier`, `ObliqueForestRegressor`, and `ObliqueForestSurvival`.

### Changes Required:

#### 1. Create Base Forest Class

**File**: `python/pyaorsf/forest.py` (new)
```python
"""
Scikit-learn compatible oblique random forest estimators.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Callable, Union, Literal
from numpy.typing import ArrayLike, NDArray

from . import _pyaorsf


class BaseObliqueForest:
    """Base class for oblique random forest estimators."""

    _tree_type: int = None  # Set by subclasses

    def __init__(
        self,
        n_trees: int = 500,
        mtry: Optional[int] = None,
        sample_fraction: float = 0.632,
        sample_with_replacement: bool = True,
        leaf_min_obs: int = 5,
        split_min_obs: int = 10,
        split_min_stat: float = 0.0,
        split_max_cuts: int = 5,
        split_max_retry: int = 3,
        lincomb_type: Literal["glm", "random", "custom"] = "glm",
        lincomb_eps: float = 1e-9,
        lincomb_iter_max: int = 20,
        lincomb_scale: bool = True,
        lincomb_function: Optional[Callable] = None,
        importance: Optional[Literal["none", "negate", "permute", "anova"]] = None,
        oobag: bool = True,
        oobag_eval_every: int = 0,
        oobag_function: Optional[Callable] = None,
        n_threads: int = 0,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_trees = n_trees
        self.mtry = mtry
        self.sample_fraction = sample_fraction
        self.sample_with_replacement = sample_with_replacement
        self.leaf_min_obs = leaf_min_obs
        self.split_min_obs = split_min_obs
        self.split_min_stat = split_min_stat
        self.split_max_cuts = split_max_cuts
        self.split_max_retry = split_max_retry
        self.lincomb_type = lincomb_type
        self.lincomb_eps = lincomb_eps
        self.lincomb_iter_max = lincomb_iter_max
        self.lincomb_scale = lincomb_scale
        self.lincomb_function = lincomb_function
        self.importance = importance
        self.oobag = oobag
        self.oobag_eval_every = oobag_eval_every
        self.oobag_function = oobag_function
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes (set during fit)
        self.forest_: Optional[dict] = None
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[NDArray] = None
        self.oob_score_: Optional[float] = None

    def _validate_data(
        self, X: ArrayLike, y: Optional[ArrayLike] = None, reset: bool = True
    ) -> tuple[NDArray, Optional[NDArray]]:
        """Validate input data."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if reset:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        if y is not None:
            y = np.asarray(y)
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"X has {X.shape[0]} samples, y has {y.shape[0]}"
                )

        return X, y

    def _get_lincomb_type_code(self) -> int:
        """Convert lincomb_type string to enum code."""
        mapping = {
            "glm": _pyaorsf.LinearCombo.GLM,
            "random": _pyaorsf.LinearCombo.RANDOM,
            "custom": _pyaorsf.LinearCombo.CUSTOM,
        }
        return int(mapping.get(self.lincomb_type, _pyaorsf.LinearCombo.GLM))

    def _get_importance_code(self) -> int:
        """Convert importance string to enum code."""
        if self.importance is None:
            return int(_pyaorsf.VariableImportance.NONE)
        mapping = {
            "none": _pyaorsf.VariableImportance.NONE,
            "negate": _pyaorsf.VariableImportance.NEGATE,
            "permute": _pyaorsf.VariableImportance.PERMUTE,
            "anova": _pyaorsf.VariableImportance.ANOVA,
        }
        return int(mapping.get(self.importance, _pyaorsf.VariableImportance.NONE))

    def _generate_seeds(self, n: int) -> list[int]:
        """Generate random seeds for trees."""
        rng = np.random.default_rng(self.random_state)
        return rng.integers(0, 2**31, size=n).tolist()

    def _prepare_y(self, y: NDArray) -> NDArray:
        """Prepare y for the C++ core. Override in subclasses."""
        raise NotImplementedError

    def _get_split_rule(self) -> int:
        """Get split rule code. Override in subclasses."""
        raise NotImplementedError

    def _get_pred_type(self) -> int:
        """Get prediction type code. Override in subclasses."""
        raise NotImplementedError


class ObliqueForestClassifier(BaseObliqueForest):
    """
    Oblique random forest classifier.

    Uses linear combinations of features for splits, providing smoother
    decision boundaries than axis-aligned random forests.

    Parameters
    ----------
    n_trees : int, default=500
        Number of trees in the forest.
    mtry : int, optional
        Number of features to consider at each split. Default is sqrt(n_features).
    sample_fraction : float, default=0.632
        Fraction of samples to use for each tree.
    sample_with_replacement : bool, default=True
        Whether to sample with replacement (bootstrap).
    leaf_min_obs : int, default=5
        Minimum observations in a leaf node.
    split_min_obs : int, default=10
        Minimum observations required to attempt a split.
    lincomb_type : {"glm", "random", "custom"}, default="glm"
        Method for computing linear combinations.
    importance : {"none", "negate", "permute", "anova"}, optional
        Variable importance method.
    n_threads : int, default=0
        Number of threads. 0 means use all available.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print progress messages.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances (if importance was requested).
    oob_score_ : float
        Out-of-bag accuracy score.

    Examples
    --------
    >>> from pyaorsf import ObliqueForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20)
    >>> clf = ObliqueForestClassifier(n_trees=100, random_state=42)
    >>> clf.fit(X, y)
    >>> clf.score(X, y)
    0.95
    """

    _tree_type = int(_pyaorsf.TreeType.CLASSIFICATION)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes_: Optional[NDArray] = None
        self.n_classes_: Optional[int] = None

    def _prepare_y(self, y: NDArray) -> NDArray:
        """Convert labels to one-hot encoded matrix."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Create label mapping
        label_map = {label: i for i, label in enumerate(self.classes_)}
        y_mapped = np.array([label_map[yi] for yi in y])

        # One-hot encode
        y_onehot = np.zeros((len(y), self.n_classes_), dtype=np.float64)
        y_onehot[np.arange(len(y)), y_mapped] = 1.0

        return y_onehot

    def _get_split_rule(self) -> int:
        return int(_pyaorsf.SplitRule.GINI)

    def _get_pred_type(self) -> int:
        return int(_pyaorsf.PredType.PROBABILITY)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ObliqueForestClassifier":
        """
        Fit the oblique random forest classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : ObliqueForestClassifier
            Fitted estimator.
        """
        X, y = self._validate_data(X, y, reset=True)
        y_prepared = self._prepare_y(y)

        # Determine mtry
        mtry = self.mtry or max(1, int(np.sqrt(self.n_features_in_)))

        # Sample weights (uniform)
        w = np.ones(X.shape[0], dtype=np.float64)

        # Generate tree seeds
        tree_seeds = self._generate_seeds(self.n_trees)

        # Fit forest
        result = _pyaorsf.fit_forest(
            x=X,
            y=y_prepared,
            w=w,
            tree_type=self._tree_type,
            tree_seeds=tree_seeds,
            n_tree=self.n_trees,
            mtry=mtry,
            sample_with_replacement=self.sample_with_replacement,
            sample_fraction=self.sample_fraction,
            vi_type=self._get_importance_code(),
            vi_max_pvalue=0.01,
            leaf_min_events=1.0,
            leaf_min_obs=float(self.leaf_min_obs),
            split_rule=self._get_split_rule(),
            split_min_events=1.0,
            split_min_obs=float(self.split_min_obs),
            split_min_stat=self.split_min_stat,
            split_max_cuts=self.split_max_cuts,
            split_max_retry=self.split_max_retry,
            lincomb_type=self._get_lincomb_type_code(),
            lincomb_eps=self.lincomb_eps,
            lincomb_iter_max=self.lincomb_iter_max,
            lincomb_scale=self.lincomb_scale,
            lincomb_alpha=0.5,
            lincomb_df_target=4,
            lincomb_ties_method=1,
            lincomb_py_function=self.lincomb_function,
            pred_horizon=[],
            pred_type=self._get_pred_type(),
            oobag=self.oobag,
            oobag_eval_type=0 if self.oobag_function is None else 2,
            oobag_eval_every=self.oobag_eval_every or self.n_trees,
            oobag_py_function=self.oobag_function,
            n_thread=self.n_threads,
            verbose=self.verbose,
        )

        # Store results
        self.forest_ = result["forest"]
        self.forest_["n_class"] = self.n_classes_

        if "importance" in result:
            self.feature_importances_ = np.asarray(result["importance"])

        if self.oobag and "oob_eval" in result:
            oob_eval = np.asarray(result["oob_eval"])
            if oob_eval.size > 0:
                self.oob_score_ = float(oob_eval[-1, 0])

        return self

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if self.forest_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X, _ = self._validate_data(X, reset=False)

        proba = _pyaorsf.predict_forest(
            x=X,
            forest_data=self.forest_,
            tree_type=self._tree_type,
            pred_horizon=[],
            pred_type=int(_pyaorsf.PredType.PROBABILITY),
            pred_aggregate=True,
            n_thread=self.n_threads,
        )

        return np.asarray(proba)

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return mean accuracy on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        return np.mean(self.predict(X) == y)


class ObliqueForestRegressor(BaseObliqueForest):
    """
    Oblique random forest regressor.

    Uses linear combinations of features for splits, providing smoother
    predictions than axis-aligned random forests.

    Parameters
    ----------
    n_trees : int, default=500
        Number of trees in the forest.
    mtry : int, optional
        Number of features to consider at each split. Default is n_features/3.
    [... same parameters as classifier ...]

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances (if importance was requested).
    oob_score_ : float
        Out-of-bag R² score.

    Examples
    --------
    >>> from pyaorsf import ObliqueForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=20)
    >>> reg = ObliqueForestRegressor(n_trees=100, random_state=42)
    >>> reg.fit(X, y)
    >>> reg.score(X, y)
    0.92
    """

    _tree_type = int(_pyaorsf.TreeType.REGRESSION)

    def _prepare_y(self, y: NDArray) -> NDArray:
        """Reshape y to 2D matrix."""
        y = y.astype(np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def _get_split_rule(self) -> int:
        return int(_pyaorsf.SplitRule.VARIANCE)

    def _get_pred_type(self) -> int:
        return int(_pyaorsf.PredType.MEAN)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ObliqueForestRegressor":
        """Fit the oblique random forest regressor."""
        X, y = self._validate_data(X, y, reset=True)
        y_prepared = self._prepare_y(y)

        mtry = self.mtry or max(1, self.n_features_in_ // 3)
        w = np.ones(X.shape[0], dtype=np.float64)
        tree_seeds = self._generate_seeds(self.n_trees)

        result = _pyaorsf.fit_forest(
            x=X,
            y=y_prepared,
            w=w,
            tree_type=self._tree_type,
            tree_seeds=tree_seeds,
            n_tree=self.n_trees,
            mtry=mtry,
            sample_with_replacement=self.sample_with_replacement,
            sample_fraction=self.sample_fraction,
            vi_type=self._get_importance_code(),
            vi_max_pvalue=0.01,
            leaf_min_events=1.0,
            leaf_min_obs=float(self.leaf_min_obs),
            split_rule=self._get_split_rule(),
            split_min_events=1.0,
            split_min_obs=float(self.split_min_obs),
            split_min_stat=self.split_min_stat,
            split_max_cuts=self.split_max_cuts,
            split_max_retry=self.split_max_retry,
            lincomb_type=self._get_lincomb_type_code(),
            lincomb_eps=self.lincomb_eps,
            lincomb_iter_max=self.lincomb_iter_max,
            lincomb_scale=self.lincomb_scale,
            lincomb_alpha=0.5,
            lincomb_df_target=4,
            lincomb_ties_method=1,
            lincomb_py_function=self.lincomb_function,
            pred_horizon=[],
            pred_type=self._get_pred_type(),
            oobag=self.oobag,
            oobag_eval_type=0 if self.oobag_function is None else 2,
            oobag_eval_every=self.oobag_eval_every or self.n_trees,
            oobag_py_function=self.oobag_function,
            n_thread=self.n_threads,
            verbose=self.verbose,
        )

        self.forest_ = result["forest"]

        if "importance" in result:
            self.feature_importances_ = np.asarray(result["importance"])

        if self.oobag and "oob_eval" in result:
            oob_eval = np.asarray(result["oob_eval"])
            if oob_eval.size > 0:
                self.oob_score_ = float(oob_eval[-1, 0])

        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict target values."""
        if self.forest_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X, _ = self._validate_data(X, reset=False)

        pred = _pyaorsf.predict_forest(
            x=X,
            forest_data=self.forest_,
            tree_type=self._tree_type,
            pred_horizon=[],
            pred_type=int(_pyaorsf.PredType.MEAN),
            pred_aggregate=True,
            n_thread=self.n_threads,
        )

        return np.asarray(pred).ravel()

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return R² score on the given test data."""
        y_pred = self.predict(X)
        y_true = np.asarray(y).ravel()

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
```

#### 2. Create Survival Forest Class

**File**: `python/pyaorsf/survival.py` (new)
```python
"""
Oblique random survival forest for time-to-event data.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Callable, Literal, Union
from numpy.typing import ArrayLike, NDArray

from . import _pyaorsf
from .forest import BaseObliqueForest


class ObliqueForestSurvival(BaseObliqueForest):
    """
    Oblique random survival forest.

    Uses accelerated Cox proportional hazards regression for linear
    combinations, providing efficient oblique splits for survival data.

    Parameters
    ----------
    n_trees : int, default=500
        Number of trees in the forest.
    mtry : int, optional
        Number of features to consider at each split. Default is sqrt(n_features).
    leaf_min_events : int, default=1
        Minimum events required in a leaf node.
    split_min_events : int, default=5
        Minimum events required to attempt a split.
    split_rule : {"logrank", "concord"}, default="logrank"
        Split criterion for survival trees.
    pred_horizon : array-like, optional
        Time points for survival predictions. Default is unique event times.
    pred_type : {"risk", "survival", "chaz", "mortality"}, default="risk"
        Type of prediction to return.
    [... other parameters from base class ...]

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances (if importance was requested).
    oob_score_ : float
        Out-of-bag concordance index (C-statistic).
    event_times_ : ndarray
        Unique event times from training data.

    Examples
    --------
    >>> from pyaorsf import ObliqueForestSurvival
    >>> import numpy as np
    >>> # Create survival data (time, status)
    >>> n = 500
    >>> X = np.random.randn(n, 10)
    >>> time = np.random.exponential(10, n)
    >>> status = np.random.binomial(1, 0.7, n)
    >>> osf = ObliqueForestSurvival(n_trees=100, random_state=42)
    >>> osf.fit(X, time, status)
    >>> risk_scores = osf.predict(X)
    """

    _tree_type = int(_pyaorsf.TreeType.SURVIVAL)

    def __init__(
        self,
        n_trees: int = 500,
        mtry: Optional[int] = None,
        leaf_min_events: int = 1,
        split_min_events: int = 5,
        split_rule: Literal["logrank", "concord"] = "logrank",
        pred_horizon: Optional[ArrayLike] = None,
        pred_type: Literal["risk", "survival", "chaz", "mortality"] = "risk",
        **kwargs
    ):
        super().__init__(n_trees=n_trees, mtry=mtry, **kwargs)
        self.leaf_min_events = leaf_min_events
        self.split_min_events = split_min_events
        self.split_rule = split_rule
        self.pred_horizon = pred_horizon
        self.pred_type = pred_type

        # Fitted attributes
        self.event_times_: Optional[NDArray] = None

    def _get_split_rule(self) -> int:
        mapping = {
            "logrank": _pyaorsf.SplitRule.LOGRANK,
            "concord": _pyaorsf.SplitRule.CONCORD,
        }
        return int(mapping.get(self.split_rule, _pyaorsf.SplitRule.LOGRANK))

    def _get_pred_type_code(self) -> int:
        mapping = {
            "risk": _pyaorsf.PredType.RISK,
            "survival": _pyaorsf.PredType.SURVIVAL,
            "chaz": _pyaorsf.PredType.CHAZ,
            "mortality": _pyaorsf.PredType.MORTALITY,
        }
        return int(mapping.get(self.pred_type, _pyaorsf.PredType.RISK))

    def _get_pred_type(self) -> int:
        return self._get_pred_type_code()

    def _prepare_y(self, time: NDArray, status: NDArray) -> NDArray:
        """Prepare survival outcome as 2-column matrix [time, status]."""
        time = np.asarray(time, dtype=np.float64)
        status = np.asarray(status, dtype=np.float64)

        if time.shape != status.shape:
            raise ValueError("time and status must have the same shape")

        # Store unique event times
        self.event_times_ = np.sort(np.unique(time[status == 1]))

        return np.column_stack([time, status])

    def fit(
        self,
        X: ArrayLike,
        time: ArrayLike,
        status: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "ObliqueForestSurvival":
        """
        Fit the oblique random survival forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        time : array-like of shape (n_samples,)
            Observed times (event or censoring).
        status : array-like of shape (n_samples,)
            Event indicator (1 = event, 0 = censored).
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self : ObliqueForestSurvival
            Fitted estimator.
        """
        X, _ = self._validate_data(X, reset=True)
        y = self._prepare_y(time, status)

        mtry = self.mtry or max(1, int(np.sqrt(self.n_features_in_)))

        if sample_weight is None:
            w = np.ones(X.shape[0], dtype=np.float64)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)

        tree_seeds = self._generate_seeds(self.n_trees)

        # Prediction horizon
        if self.pred_horizon is not None:
            horizon = list(np.asarray(self.pred_horizon, dtype=np.float64))
        else:
            # Use quantiles of event times
            horizon = list(np.quantile(self.event_times_, [0.25, 0.5, 0.75]))

        result = _pyaorsf.fit_forest(
            x=X,
            y=y,
            w=w,
            tree_type=self._tree_type,
            tree_seeds=tree_seeds,
            n_tree=self.n_trees,
            mtry=mtry,
            sample_with_replacement=self.sample_with_replacement,
            sample_fraction=self.sample_fraction,
            vi_type=self._get_importance_code(),
            vi_max_pvalue=0.01,
            leaf_min_events=float(self.leaf_min_events),
            leaf_min_obs=float(self.leaf_min_obs),
            split_rule=self._get_split_rule(),
            split_min_events=float(self.split_min_events),
            split_min_obs=float(self.split_min_obs),
            split_min_stat=self.split_min_stat,
            split_max_cuts=self.split_max_cuts,
            split_max_retry=self.split_max_retry,
            lincomb_type=self._get_lincomb_type_code(),
            lincomb_eps=self.lincomb_eps,
            lincomb_iter_max=self.lincomb_iter_max,
            lincomb_scale=self.lincomb_scale,
            lincomb_alpha=0.5,
            lincomb_df_target=4,
            lincomb_ties_method=1,
            lincomb_py_function=self.lincomb_function,
            pred_horizon=horizon,
            pred_type=self._get_pred_type(),
            oobag=self.oobag,
            oobag_eval_type=0 if self.oobag_function is None else 2,
            oobag_eval_every=self.oobag_eval_every or self.n_trees,
            oobag_py_function=self.oobag_function,
            n_thread=self.n_threads,
            verbose=self.verbose,
        )

        self.forest_ = result["forest"]
        self._pred_horizon = horizon

        if "importance" in result:
            self.feature_importances_ = np.asarray(result["importance"])

        if self.oobag and "oob_eval" in result:
            oob_eval = np.asarray(result["oob_eval"])
            if oob_eval.size > 0:
                self.oob_score_ = float(oob_eval[-1, 0])

        return self

    def predict(
        self,
        X: ArrayLike,
        pred_type: Optional[str] = None,
        pred_horizon: Optional[ArrayLike] = None,
    ) -> NDArray:
        """
        Generate predictions from the fitted survival forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        pred_type : {"risk", "survival", "chaz", "mortality"}, optional
            Type of prediction. Default uses value from initialization.
        pred_horizon : array-like, optional
            Time points for predictions. Required for survival/chaz/mortality.

        Returns
        -------
        predictions : ndarray
            Predicted values. Shape depends on pred_type:
            - "risk": (n_samples,) - risk scores
            - "survival": (n_samples, n_times) - survival probabilities
            - "chaz": (n_samples, n_times) - cumulative hazard
            - "mortality": (n_samples,) - mortality at last horizon
        """
        if self.forest_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X, _ = self._validate_data(X, reset=False)

        ptype = pred_type or self.pred_type
        horizon = pred_horizon if pred_horizon is not None else self._pred_horizon
        horizon = list(np.asarray(horizon, dtype=np.float64))

        pred_type_code = {
            "risk": _pyaorsf.PredType.RISK,
            "survival": _pyaorsf.PredType.SURVIVAL,
            "chaz": _pyaorsf.PredType.CHAZ,
            "mortality": _pyaorsf.PredType.MORTALITY,
        }.get(ptype, _pyaorsf.PredType.RISK)

        pred = _pyaorsf.predict_forest(
            x=X,
            forest_data=self.forest_,
            tree_type=self._tree_type,
            pred_horizon=horizon,
            pred_type=int(pred_type_code),
            pred_aggregate=True,
            n_thread=self.n_threads,
        )

        pred = np.asarray(pred)

        # Return appropriate shape
        if ptype == "risk" or ptype == "mortality":
            return pred.ravel()
        return pred

    def predict_survival_function(
        self, X: ArrayLike, times: Optional[ArrayLike] = None
    ) -> NDArray:
        """
        Predict survival function S(t) = P(T > t).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        times : array-like, optional
            Time points. Default uses event_times_.

        Returns
        -------
        survival : ndarray of shape (n_samples, n_times)
            Survival probabilities at each time point.
        """
        if times is None:
            times = self.event_times_
        return self.predict(X, pred_type="survival", pred_horizon=times)

    def predict_cumulative_hazard(
        self, X: ArrayLike, times: Optional[ArrayLike] = None
    ) -> NDArray:
        """
        Predict cumulative hazard function H(t).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        times : array-like, optional
            Time points. Default uses event_times_.

        Returns
        -------
        chaz : ndarray of shape (n_samples, n_times)
            Cumulative hazard at each time point.
        """
        if times is None:
            times = self.event_times_
        return self.predict(X, pred_type="chaz", pred_horizon=times)

    def score(
        self,
        X: ArrayLike,
        time: ArrayLike,
        status: ArrayLike,
    ) -> float:
        """
        Return concordance index (C-statistic) on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        time : array-like of shape (n_samples,)
            Observed times.
        status : array-like of shape (n_samples,)
            Event indicators.

        Returns
        -------
        score : float
            Concordance index (0.5 = random, 1.0 = perfect discrimination).
        """
        risk = self.predict(X, pred_type="risk")

        # Compute concordance index
        time = np.asarray(time)
        status = np.asarray(status)

        concordant = 0
        discordant = 0

        for i in range(len(time)):
            if status[i] == 0:
                continue
            for j in range(len(time)):
                if time[i] < time[j]:
                    if risk[i] > risk[j]:
                        concordant += 1
                    elif risk[i] < risk[j]:
                        discordant += 1

        total = concordant + discordant
        return concordant / total if total > 0 else 0.5
```

### Success Criteria:

#### Automated Verification:
- [ ] Classification works:
  ```python
  from pyaorsf import ObliqueForestClassifier
  from sklearn.datasets import make_classification
  X, y = make_classification(100, 10)
  clf = ObliqueForestClassifier(n_trees=10).fit(X, y)
  assert clf.score(X, y) > 0.5
  ```
- [ ] Regression works:
  ```python
  from pyaorsf import ObliqueForestRegressor
  from sklearn.datasets import make_regression
  X, y = make_regression(100, 10)
  reg = ObliqueForestRegressor(n_trees=10).fit(X, y)
  assert reg.score(X, y) > 0.0
  ```
- [ ] Survival works:
  ```python
  from pyaorsf import ObliqueForestSurvival
  import numpy as np
  X = np.random.randn(100, 5)
  time = np.random.exponential(10, 100)
  status = np.random.binomial(1, 0.7, 100)
  osf = ObliqueForestSurvival(n_trees=10).fit(X, time, status)
  assert osf.predict(X).shape == (100,)
  ```

#### Manual Verification:
- [ ] API feels natural and scikit-learn-like
- [ ] Variable importance is computed correctly
- [ ] OOB scores are reasonable

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to Phase 6.

---

## Phase 6: Python Callbacks

### Overview

Ensure Python callback functions work correctly for custom linear combinations and OOB evaluation, including proper GIL handling and error propagation.

### Changes Required:

#### 1. Update Callback Wrappers with Error Handling

**File**: `python/src/python/PythonCallbacks.h`
**Changes**: Add proper error handling and GIL management

```cpp
// Add to PythonCallbacks.h

/**
 * @brief Wrapper that catches Python exceptions and converts to C++ exceptions
 */
inline LinCombCallback make_python_lincomb_callback_safe(nb::callable py_func) {
    if (py_func.is_none()) {
        return nullptr;
    }

    return [py_func](const arma::mat& x, const arma::mat& y, const arma::vec& w) -> arma::mat {
        nb::gil_scoped_acquire guard;

        try {
            nb::ndarray<double> x_np = carma::mat_to_arr(x);
            nb::ndarray<double> y_np = carma::mat_to_arr(y);
            nb::ndarray<double> w_np = carma::col_to_arr(w);

            nb::object result = py_func(x_np, y_np, w_np);

            // Validate result
            if (result.is_none()) {
                throw aorsf::computation_error(
                    "Linear combination function returned None"
                );
            }

            auto arr = nb::cast<nb::ndarray<double>>(result);
            arma::mat coeffs = carma::arr_to_mat<double>(arr);

            // Validate dimensions
            if (coeffs.n_rows != x.n_cols) {
                throw aorsf::computation_error(
                    "Linear combination function returned wrong number of coefficients. "
                    "Expected " + std::to_string(x.n_cols) +
                    ", got " + std::to_string(coeffs.n_rows)
                );
            }

            return coeffs;

        } catch (nb::python_error& e) {
            // Convert Python exception to C++ exception
            throw aorsf::computation_error(
                std::string("Python error in lincomb function: ") + e.what()
            );
        }
    };
}

/**
 * @brief Safe OOB callback wrapper with validation
 */
inline OobagEvalCallback make_python_oobag_callback_safe(nb::callable py_func) {
    if (py_func.is_none()) {
        return nullptr;
    }

    return [py_func](const arma::mat& y, const arma::vec& w, const arma::vec& p) -> double {
        nb::gil_scoped_acquire guard;

        try {
            nb::ndarray<double> y_np = carma::mat_to_arr(y);
            nb::ndarray<double> w_np = carma::col_to_arr(w);
            nb::ndarray<double> p_np = carma::col_to_arr(p);

            nb::object result = py_func(y_np, w_np, p_np);

            if (result.is_none()) {
                throw aorsf::computation_error(
                    "OOB evaluation function returned None"
                );
            }

            double value = nb::cast<double>(result);

            if (!std::isfinite(value)) {
                throw aorsf::computation_error(
                    "OOB evaluation function returned non-finite value"
                );
            }

            return value;

        } catch (nb::python_error& e) {
            throw aorsf::computation_error(
                std::string("Python error in oobag function: ") + e.what()
            );
        }
    };
}
```

#### 2. Add Callback Examples and Tests

**File**: `python/tests/test_callbacks.py` (new)
```python
"""Tests for Python callback functionality."""

import numpy as np
import pytest
from pyaorsf import ObliqueForestClassifier, ObliqueForestRegressor


def custom_lincomb_random(x, y, w):
    """Simple random coefficients for testing."""
    n_features = x.shape[1]
    coeffs = np.random.randn(n_features, 1)
    return coeffs


def custom_lincomb_pca(x, y, w):
    """Use first principal component direction."""
    # Center data
    x_centered = x - np.average(x, axis=0, weights=w)
    # Compute covariance
    cov = np.cov(x_centered.T, aweights=w)
    # Get first eigenvector
    _, vecs = np.linalg.eigh(cov)
    coeffs = vecs[:, -1].reshape(-1, 1)
    return coeffs


def custom_oobag_accuracy(y, w, p):
    """Custom accuracy metric."""
    # For classification, y is one-hot, p is probabilities
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(p.reshape(-1, y.shape[1]), axis=1) if p.ndim > 1 else (p > 0.5).astype(int)
    return np.average(y_true == y_pred, weights=w)


class TestLinCombCallbacks:
    """Test custom linear combination functions."""

    def test_random_lincomb(self):
        """Test that random coefficients callback works."""
        X, y = make_classification_data()

        clf = ObliqueForestClassifier(
            n_trees=10,
            lincomb_type="custom",
            lincomb_function=custom_lincomb_random,
            random_state=42,
        )
        clf.fit(X, y)

        # Should fit without error
        assert clf.forest_ is not None
        assert clf.predict(X).shape == (len(y),)

    def test_pca_lincomb(self):
        """Test PCA-based coefficients."""
        X, y = make_classification_data()

        clf = ObliqueForestClassifier(
            n_trees=10,
            lincomb_type="custom",
            lincomb_function=custom_lincomb_pca,
            random_state=42,
        )
        clf.fit(X, y)

        assert clf.score(X, y) > 0.5

    def test_callback_error_handling(self):
        """Test that callback errors are properly propagated."""
        def bad_callback(x, y, w):
            raise ValueError("Intentional error")

        X, y = make_classification_data()

        clf = ObliqueForestClassifier(
            n_trees=10,
            lincomb_type="custom",
            lincomb_function=bad_callback,
        )

        with pytest.raises(RuntimeError, match="Python error"):
            clf.fit(X, y)

    def test_callback_wrong_shape(self):
        """Test that wrong coefficient shape is caught."""
        def wrong_shape_callback(x, y, w):
            return np.array([[1, 2, 3]])  # Wrong shape

        X, y = make_classification_data(n_features=10)

        clf = ObliqueForestClassifier(
            n_trees=10,
            lincomb_type="custom",
            lincomb_function=wrong_shape_callback,
        )

        with pytest.raises(RuntimeError, match="wrong number"):
            clf.fit(X, y)


class TestOobagCallbacks:
    """Test custom OOB evaluation functions."""

    def test_custom_accuracy(self):
        """Test custom accuracy metric."""
        X, y = make_classification_data()

        clf = ObliqueForestClassifier(
            n_trees=10,
            oobag=True,
            oobag_function=custom_oobag_accuracy,
            random_state=42,
        )
        clf.fit(X, y)

        assert clf.oob_score_ is not None
        assert 0 <= clf.oob_score_ <= 1


def make_classification_data(n_samples=200, n_features=10):
    """Helper to create classification data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y
```

### Success Criteria:

#### Automated Verification:
- [ ] Callback tests pass: `pytest python/tests/test_callbacks.py -v`
- [ ] Custom lincomb function produces valid forest
- [ ] Errors in Python callbacks are properly caught and reported

#### Manual Verification:
- [ ] Custom PCA-based linear combination works correctly
- [ ] Progress interruption (Ctrl+C) works during training with callbacks

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to Phase 7.

---

## Phase 7: Testing & Documentation

### Overview

Create comprehensive tests, type hints, and basic documentation for the Python package.

### Changes Required:

#### 1. Create Test Configuration

**File**: `python/tests/conftest.py` (new)
```python
"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 500, 20

    X = rng.standard_normal((n_samples, n_features))
    # Create informative features
    y = ((X[:, 0] + X[:, 1] + 0.5 * X[:, 2]) > 0).astype(int)

    return X, y


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 500, 20

    X = rng.standard_normal((n_samples, n_features))
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + rng.standard_normal(n_samples) * 0.5

    return X, y


@pytest.fixture
def survival_data():
    """Generate survival dataset."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 500, 20

    X = rng.standard_normal((n_samples, n_features))

    # Generate survival times with hazard depending on X
    hazard = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1])
    time = rng.exponential(10 / hazard)

    # Random censoring
    censor_time = rng.exponential(15, n_samples)
    status = (time <= censor_time).astype(int)
    time = np.minimum(time, censor_time)

    return X, time, status
```

#### 2. Create Classification Tests

**File**: `python/tests/test_classification.py` (new)
```python
"""Tests for ObliqueForestClassifier."""

import numpy as np
import pytest
from pyaorsf import ObliqueForestClassifier


class TestClassifierBasic:
    """Basic functionality tests."""

    def test_fit_predict(self, classification_data):
        """Test basic fit and predict."""
        X, y = classification_data

        clf = ObliqueForestClassifier(n_trees=50, random_state=42)
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert predictions.shape == y.shape
        assert set(predictions).issubset(set(y))

    def test_predict_proba(self, classification_data):
        """Test probability predictions."""
        X, y = classification_data

        clf = ObliqueForestClassifier(n_trees=50, random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_score(self, classification_data):
        """Test accuracy score."""
        X, y = classification_data

        clf = ObliqueForestClassifier(n_trees=50, random_state=42)
        clf.fit(X, y)

        score = clf.score(X, y)
        assert 0 <= score <= 1
        assert score > 0.7  # Should be reasonably accurate

    def test_oob_score(self, classification_data):
        """Test out-of-bag score."""
        X, y = classification_data

        clf = ObliqueForestClassifier(
            n_trees=100, oobag=True, random_state=42
        )
        clf.fit(X, y)

        assert clf.oob_score_ is not None
        assert 0 <= clf.oob_score_ <= 1


class TestClassifierImportance:
    """Variable importance tests."""

    def test_negate_importance(self, classification_data):
        """Test negation-based importance."""
        X, y = classification_data

        clf = ObliqueForestClassifier(
            n_trees=50, importance="negate", random_state=42
        )
        clf.fit(X, y)

        assert clf.feature_importances_ is not None
        assert len(clf.feature_importances_) == X.shape[1]

    def test_permute_importance(self, classification_data):
        """Test permutation-based importance."""
        X, y = classification_data

        clf = ObliqueForestClassifier(
            n_trees=50, importance="permute", random_state=42
        )
        clf.fit(X, y)

        assert clf.feature_importances_ is not None
        # First few features should be most important
        top_features = np.argsort(clf.feature_importances_)[-3:]
        assert 0 in top_features or 1 in top_features


class TestClassifierParameters:
    """Parameter handling tests."""

    def test_mtry(self, classification_data):
        """Test mtry parameter."""
        X, y = classification_data

        clf = ObliqueForestClassifier(n_trees=10, mtry=5, random_state=42)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.5

    def test_n_threads(self, classification_data):
        """Test multi-threading."""
        X, y = classification_data

        clf = ObliqueForestClassifier(n_trees=50, n_threads=4, random_state=42)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.5

    def test_reproducibility(self, classification_data):
        """Test that random_state produces reproducible results."""
        X, y = classification_data

        clf1 = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf2 = ObliqueForestClassifier(n_trees=10, random_state=42)

        clf1.fit(X, y)
        clf2.fit(X, y)

        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))


class TestClassifierEdgeCases:
    """Edge case tests."""

    def test_single_feature(self):
        """Test with single feature."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 1))
        y = (X[:, 0] > 0).astype(int)

        clf = ObliqueForestClassifier(n_trees=10, random_state=42)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.5

    def test_multiclass(self):
        """Test multiclass classification."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 10))
        y = (X[:, 0] + X[:, 1]).astype(int) % 3

        clf = ObliqueForestClassifier(n_trees=50, random_state=42)
        clf.fit(X, y)

        assert clf.n_classes_ == 3
        assert clf.predict_proba(X).shape == (300, 3)

    def test_unfitted_error(self):
        """Test that unfitted model raises error."""
        clf = ObliqueForestClassifier()

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(np.array([[1, 2, 3]]))
```

#### 3. Create Survival Tests

**File**: `python/tests/test_survival.py` (new)
```python
"""Tests for ObliqueForestSurvival."""

import numpy as np
import pytest
from pyaorsf import ObliqueForestSurvival


class TestSurvivalBasic:
    """Basic functionality tests."""

    def test_fit_predict_risk(self, survival_data):
        """Test basic fit and risk prediction."""
        X, time, status = survival_data

        osf = ObliqueForestSurvival(n_trees=50, random_state=42)
        osf.fit(X, time, status)

        risk = osf.predict(X)
        assert risk.shape == (len(time),)

    def test_predict_survival(self, survival_data):
        """Test survival probability prediction."""
        X, time, status = survival_data

        osf = ObliqueForestSurvival(n_trees=50, random_state=42)
        osf.fit(X, time, status)

        times = [10, 20, 30]
        surv = osf.predict_survival_function(X, times)

        assert surv.shape == (len(X), len(times))
        assert (surv >= 0).all() and (surv <= 1).all()
        # Survival should decrease over time
        assert (surv[:, 0] >= surv[:, -1]).all()

    def test_score_cindex(self, survival_data):
        """Test concordance index score."""
        X, time, status = survival_data

        osf = ObliqueForestSurvival(n_trees=50, random_state=42)
        osf.fit(X, time, status)

        score = osf.score(X, time, status)
        assert 0 <= score <= 1
        assert score > 0.55  # Should be better than random

    def test_oob_score(self, survival_data):
        """Test out-of-bag C-index."""
        X, time, status = survival_data

        osf = ObliqueForestSurvival(
            n_trees=100, oobag=True, random_state=42
        )
        osf.fit(X, time, status)

        assert osf.oob_score_ is not None
        assert 0.5 <= osf.oob_score_ <= 1.0


class TestSurvivalPredTypes:
    """Test different prediction types."""

    def test_cumulative_hazard(self, survival_data):
        """Test cumulative hazard prediction."""
        X, time, status = survival_data

        osf = ObliqueForestSurvival(n_trees=50, random_state=42)
        osf.fit(X, time, status)

        times = [10, 20, 30]
        chaz = osf.predict_cumulative_hazard(X, times)

        assert chaz.shape == (len(X), len(times))
        assert (chaz >= 0).all()
        # Cumulative hazard should increase over time
        assert (chaz[:, 0] <= chaz[:, -1]).all()

    def test_mortality(self, survival_data):
        """Test mortality prediction."""
        X, time, status = survival_data

        osf = ObliqueForestSurvival(
            n_trees=50, pred_type="mortality", random_state=42
        )
        osf.fit(X, time, status)

        mort = osf.predict(X)
        assert mort.shape == (len(X),)


class TestSurvivalSplitRules:
    """Test different split rules."""

    def test_logrank_split(self, survival_data):
        """Test log-rank split rule."""
        X, time, status = survival_data

        osf = ObliqueForestSurvival(
            n_trees=50, split_rule="logrank", random_state=42
        )
        osf.fit(X, time, status)
        assert osf.score(X, time, status) > 0.5

    def test_concord_split(self, survival_data):
        """Test concordance split rule."""
        X, time, status = survival_data

        osf = ObliqueForestSurvival(
            n_trees=50, split_rule="concord", random_state=42
        )
        osf.fit(X, time, status)
        assert osf.score(X, time, status) > 0.5
```

#### 4. Create Type Stubs

**File**: `python/pyaorsf/py.typed` (new)
```
# Marker file for PEP 561
```

**File**: `python/pyaorsf/_pyaorsf.pyi` (new)
```python
"""Type stubs for _pyaorsf C++ extension."""

from typing import Dict, List, Optional, Callable, Any
import numpy as np
from numpy.typing import NDArray

class TreeType:
    CLASSIFICATION: int
    REGRESSION: int
    SURVIVAL: int

class VariableImportance:
    NONE: int
    NEGATE: int
    PERMUTE: int
    ANOVA: int

class SplitRule:
    LOGRANK: int
    CONCORD: int
    GINI: int
    VARIANCE: int

class LinearCombo:
    GLM: int
    RANDOM: int
    GLMNET: int
    CUSTOM: int

class PredType:
    NONE: int
    RISK: int
    SURVIVAL: int
    CHAZ: int
    MORTALITY: int
    MEAN: int
    PROBABILITY: int
    CLASS: int

def fit_forest(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    tree_type: int,
    tree_seeds: List[int],
    n_tree: int,
    mtry: int,
    sample_with_replacement: bool,
    sample_fraction: float,
    vi_type: int,
    vi_max_pvalue: float,
    leaf_min_events: float,
    leaf_min_obs: float,
    split_rule: int,
    split_min_events: float,
    split_min_obs: float,
    split_min_stat: float,
    split_max_cuts: int,
    split_max_retry: int,
    lincomb_type: int,
    lincomb_eps: float,
    lincomb_iter_max: int,
    lincomb_scale: bool,
    lincomb_alpha: float,
    lincomb_df_target: int,
    lincomb_ties_method: int,
    lincomb_py_function: Optional[Callable],
    pred_horizon: List[float],
    pred_type: int,
    oobag: bool,
    oobag_eval_type: int,
    oobag_eval_every: int,
    oobag_py_function: Optional[Callable],
    n_thread: int,
    verbose: bool,
) -> Dict[str, Any]: ...

def predict_forest(
    x: NDArray[np.float64],
    forest_data: Dict[str, Any],
    tree_type: int,
    pred_horizon: List[float],
    pred_type: int,
    pred_aggregate: bool,
    n_thread: int,
) -> NDArray[np.float64]: ...

__version__: str
```

### Success Criteria:

#### Automated Verification:
- [ ] All tests pass: `pytest python/tests/ -v`
- [ ] Type checking passes: `mypy python/pyaorsf/`
- [ ] Test coverage > 80%: `pytest --cov=pyaorsf python/tests/`

#### Manual Verification:
- [ ] Documentation renders correctly
- [ ] Examples in docstrings work when copy-pasted

**Implementation Note**: After completing this phase, the Python wrapper should be fully functional and ready for use.

---

## Testing Strategy

### Unit Tests (Python)
- Test each estimator class independently
- Test edge cases: single feature, single sample, all same class
- Test parameter validation
- Test reproducibility with random_state

### Integration Tests
- Test full fit→predict→score workflow
- Test serialization (pickle)
- Test multi-threading
- Test custom callbacks

### Compatibility Tests
- Run scikit-learn's `check_estimator()` for classifier/regressor
- Test NumPy array and list inputs
- Test different dtypes

### Performance Tests
- Compare timing with R package on same data
- Verify memory usage is reasonable
- Test parallel scaling

## Performance Considerations

- nanobind has lower overhead than pybind11
- carma provides zero-copy conversion where possible
- GIL is released during C++ computation
- Multi-threading works at C++ level (no Python GIL issues)
- Callback overhead is minimal (only called at split points)

## Migration Notes

- No changes to R package
- Python package is independent installation
- Same C++ core ensures identical results
- Serialized forests are not compatible between R and Python (different formats)

## References

- Current R interface: `src/orsf_rcpp.cpp`
- Core separation plan: `plans/2025-12-16-separate-r-interface-from-core.md`
- nanobind documentation: https://nanobind.readthedocs.io/
- carma documentation: https://github.com/RUrlus/carma
- scikit-learn estimator guide: https://scikit-learn.org/stable/developers/develop.html
