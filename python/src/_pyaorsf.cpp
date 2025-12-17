/**
 * @brief nanobind bindings for pyaorsf
 *
 * This file creates Python bindings for the aorsf C++ core library
 * using nanobind and carma for Armadillo <-> NumPy conversion.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

// Include aorsf core headers
#include "arma_config.h"
#include "globals.h"
#include "Exceptions.h"

namespace nb = nanobind;

NB_MODULE(_pyaorsf, m) {
    m.doc() = "Python bindings for aorsf C++ core library";

    // Version info
    m.attr("__version__") = "0.1.0";

    // Expose constants from globals.h
    m.attr("DEFAULT_N_TREE") = aorsf::DEFAULT_N_TREE;
    m.attr("DEFAULT_LEAF_MIN_OBS") = aorsf::DEFAULT_LEAF_MIN_OBS;
    m.attr("DEFAULT_SPLIT_MIN_OBS") = aorsf::DEFAULT_SPLIT_MIN_OBS;

    // TODO: Phase 4 will add:
    // - Data class bindings
    // - Forest class bindings (ForestClassification, ForestRegression, ForestSurvival)
    // - Tree class bindings
    // - Utility function bindings
    // - Python-specific adapters (PythonOutput, PythonCallbacks, PythonInterrupts)
}
