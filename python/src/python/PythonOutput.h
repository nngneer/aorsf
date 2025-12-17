/*-----------------------------------------------------------------------------
 Python output handler for pyaorsf.
 Redirects C++ output to Python's sys.stdout.
#----------------------------------------------------------------------------*/

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
        try {
            nb::module_ sys = nb::module_::import_("sys");
            nb::object stdout_obj = sys.attr("stdout");
            stdout_obj.attr("write")(msg);
            stdout_obj.attr("flush")();
        } catch (...) {
            // Fallback to std::cout if Python stdout fails
            std::cout << msg << std::flush;
        }
    }

    void println(const std::string& msg) override {
        print(msg + "\n");
    }
};

/**
 * @brief Initialize Python output handler.
 * @param verbose If true, use PythonOutput; if false, use silent handler.
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
